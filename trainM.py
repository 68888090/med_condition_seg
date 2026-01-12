import argparse
from omegaconf import OmegaConf
from gc import enable
import os
from time import time

import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
import wandb

from losses.loss import Loss1
from unetM import UnetM
from optimizers.lr_scheduler import WarmupCosineSchedule
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from utils.data_process import get_loader
from models.text_processor import LanguageProcessor
from losses.detect_loss import NoduleDetectionLoss

JOSN_FILE = '/home/lhr/dataset/CSTPLung/data2.json'

def get_config():
    # 1. 设置命令行参数，只保留 config 路径和命令行覆盖功能
    parser = argparse.ArgumentParser(description="PyTorch Training with YAML")
    parser.add_argument("-c", "--config", default="config.yaml", help="Path to config file")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER, 
                        help="Modify config options from command line (e.g., training.epochs=200)")
    args = parser.parse_args()

    # 2. 加载 YAML
    conf = OmegaConf.load(args.config)

    # 3. 合并命令行参数 (允许 python train.py training.batch_size=4 覆盖 yaml)
    if args.opts:
        cli_conf = OmegaConf.from_cli(args.opts)
        conf = OmegaConf.merge(conf, cli_conf)

    # 4. [可选但推荐] 将结构化配置扁平化，为了兼容旧代码
    # 这样你可以继续使用 args.epochs 而不需要改成 args.training.epochs
    # 如果你希望代码更现代化，可以去掉这一步，然后修改后面代码的调用方式
    args_flat = flatten_config(conf)
    
    return args_flat

def flatten_config(conf):
    """
    将嵌套的 OmegaConf 配置展平成一层 argparse.Namespace。
    """
    # === 关键修复步骤 ===
    # 1. 将 OmegaConf 对象转换为标准的 Python 字典
    # resolve=True 确保如果是 ${...} 这种引用变量，会被计算出真实值
    conf_dict = OmegaConf.to_container(conf, resolve=True)

    flat_args = argparse.Namespace()

    def _extract_values(d):
        for key, value in d.items():
            # 现在只需要判断是否为标准 dict 即可
            if isinstance(value, dict):
                _extract_values(value)
            else:
                # 写入 Namespace
                setattr(flat_args, key, value)

    _extract_values(conf_dict)
    return flat_args

def main():
    
    def save_checkpoint(state, filename='checkpoint.pth.tar'):
        torch.save(state, filename)

    def train(args, epoch, global_step, train_loader, val_loader, best_loss, scaler,no_improve_count, patience, text_processor=None):
        
        # 【关键】设置 epoch，保证每个 epoch 随机数种子不同，shuffle 结果不同
        if args.distributed and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        model.train()
        loss_train = []
        loss_train_cross = []

        # 定义早停标志 (0: 继续, 1: 停止)
        stop_signal = False

        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            t1 = time()
            # image1, image2, text,label = batch["ctImage1"], batch["ctImage2"], batch["text"], batch["label_image"]
            image1, image2, text, heatmap_label, offset_label, size_label = batch["ctImage1"], batch["ctImage2"], batch["text"], batch["heatmap_label"], batch["offset_label"], batch["size_label"]
            with autocast(enabled = args.amp):
                # image1, image2, text, label = image1.to(args.device), image2.to(args.device), text.to(args.device), label.to(args.device)
                # print(f'text.shape:{text.shape}')
                image1 = image1.as_subclass(torch.Tensor)
                image2 = image2.as_subclass(torch.Tensor)
                text = text.as_subclass(torch.Tensor)
                image1, image2, text, heatmap_label, offset_label, size_label = image1.to(args.device), image2.to(args.device), text.to(args.device), heatmap_label.to(args.device), offset_label.to(args.device), size_label.to(args.device)
                # print(f'heatmap_label.shape:{heatmap_label.shape}')
                # print(f'image1.shape:{image1.shape}')
                # image1 = image1.view(-1, *image1.shape[2:])
                # image2 = image2.view(-1, *image2.shape[2:])
                # heatmap_label = heatmap_label.view(-1, *heatmap_label.shape[2:])
                # offset_label = offset_label.view(-1, *offset_label.shape[2:])
                # size_label = size_label.view(-1, *size_label.shape[2:])
                # text = text.view(-1, text.shape[-1])
                # with torch.no_grad():
                #     text_embedding = text_processor(text).to(args.device)
                # text_embedding = text_embedding.unsqueeze(1).repeat(1, args.sw_batch_size, 1, 1)
                # print(f'text.shape:{text_embedding.shape}')
                # print(f'image1.shape:{image1.shape}')
                # print(f'image2.shape:{image2.shape}')
                
                (pre_hm, pre_offset, pre_size), alphas = model(image1, image2, text)
                # alpha越大，对应的图像特征部分占比越大 , 第一个是最深层的，而后面的是逐渐浅层

                # output = model(image1, image2, text_embedding)
                # print(f'out_put.shape:{out_put.shape}') 
                # print(f'label_image.shape:{label_image.shape}')
                # 因为有掩码输出，先尝试使用通用的交叉熵跟dice
                label_batch = {'heatmap': heatmap_label, 'offset': offset_label, 'size': size_label}
                # total_loss, (hm_loss,_) = loss_function(output, label)
                total_loss, loss_dict = loss_function(pre_hm, pre_offset, pre_size, label_batch)
                hm_max = pre_hm.max().item()
                hm_min = pre_hm.min().item()

                # =================== 【修改后的防爆显存版本】 ===================
                if torch.isnan(total_loss) or torch.isinf(total_loss) or total_loss > 60000:
                    print(f"Warning: Bad batch detected at Global Step {global_step}!")
                    print(f"  Total Loss: {total_loss.item()}")
                    
                    # 1. 必须先将优化器的梯度清零（虽然还没backward，但习惯性清理是个好习惯）
                    optimizer.zero_grad()
                    
                    # 2. 删除持有计算图的所有变量
                    # 这些变量如果不删，计算图就一直挂在显存里
                    del total_loss, loss_dict, pre_hm, pre_offset, pre_size, alphas
                    del image1, image2, text, heatmap_label, offset_label, size_label
                    
                    # 3. 强制触发显存清理
                    import gc
                    gc.collect() # 清理 Python 垃圾
                    torch.cuda.empty_cache() # 清理 PyTorch 显存缓存
                    
                    print("  Skipping batch and cleared GPU memory.")
                    continue 
                # =============================================================
            loss_train.append(total_loss.item())
            loss_train_cross.append(loss_dict['hm_loss'].item() )
            if args.amp:
                scaler.scale(total_loss).backward()
                if args.grad_clip:
                # 1. 先把梯度从 scaler 中解包出来 (Unscale)
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        print(f"Warning: NaN/Inf gradients detected (Scaler={scaler.get_scale()})! Skipping step.")
                        optimizer.zero_grad() 
                        # 不执行 scaler.step，强制 scaler 减小
                        # 实际上 scaler.step(opt) 内部发现 nan 会自动 skip 并减小 scaler，
                        # 但为了日志清晰，你可以手动打印一下

                    # 2. 进行梯度裁剪 (Clip Gradient)
                    # max_norm 通常设为 1.0, 也可以尝试 0.5 或 2.0
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                if args.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
            
            if args.lrdecay and args.lr_schedule == "warmup_cosine":
                scheduler.step()
            


            if args.distributed:
                if dist.get_rank() == 0:
                    if global_step % args.log_num == 0:
                        wandb.log({
                        "train/total_loss": total_loss.item(),
                        "train/hm_loss": loss_dict['hm_loss'].item(),
                        "train/offset_loss": loss_dict['offset_loss'].item(),
                        "train/size_loss": loss_dict['size_loss'].item(),
                        "train/lr": optimizer.param_groups[0]['lr'],
                        "train/step_time": time() - t1,
                        "train/alpha5": alphas[0].mean().item(),
                        "train/alpha4": alphas[1].mean().item(),
                        "train/alpha3": alphas[2].mean().item(),
                        "train/alpha2": alphas[3].mean().item(),
                        "train/alpha1": alphas[4].mean().item(),
                        "train/hm_max": hm_max,
                        "train/hm_min": hm_min,
                        "global_step": global_step  # 显式传入 step，对齐横坐标
                    })
                    print("Step:{}/{}, Loss:{:.4f}, Time:{:.4f}".format(global_step, args.num_steps, total_loss, time() - t1))
            else:
                if global_step % args.log_num == 0:
                        wandb.log({
                        "train/total_loss": total_loss.item(),
                        "train/hm_loss": loss_dict['hm_loss'].item(),
                        "train/offset_loss": loss_dict['offset_loss'].item(),
                        "train/size_loss": loss_dict['size_loss'].item(),
                        "train/lr": optimizer.param_groups[0]['lr'],
                        "train/step_time": time() - t1,
                        "train/alpha5": alphas[0].mean().item(),
                        "train/alpha4": alphas[1].mean().item(),
                        "train/alpha3": alphas[2].mean().item(),
                        "train/alpha2": alphas[3].mean().item(),
                        "train/alpha1": alphas[4].mean().item(),
                        "train/hm_max": hm_max,
                        "train/hm_min": hm_min,
                        "global_step": global_step  # 显式传入 step，对齐横坐标
                    })
                print("Step:{}/{}, Loss:{:.4f}, Time:{:.4f}".format(global_step, args.num_steps, total_loss, time() - t1))

            global_step += 1
            # =================== 【早停核心修改区域】 ===================
            # 1. 判断是否是验证步 (所有 Rank 必须达成共识)
            is_val_step = (global_step % args.eval_num == 0)

            if is_val_step:
                # 2. 定义一个 Tensor 用于进程间通信，0表示继续，1表示停止
                should_stop = torch.tensor(0).to(args.device)

                # 3. 只有 Rank 0 进行验证和决策
                if args.rank == 0:
                    loss_val, loss_val_cross = validation(args, val_loader)
                    print(f"Validation Loss:{loss_val:.4f} (Best: {best_loss:.4f})")
                    
                    # 记录日志
                    wandb.log({"val/total_loss": loss_val, "val/hm_loss": loss_val_cross, "global_step": global_step})
                    writer.add_scalar("Loss/Validation", loss_val, global_step)

                    # --- 早停逻辑判断 ---
                    if loss_val < best_loss:
                        best_loss = loss_val
                        no_improve_count = 0  # 损失下降，计数器清零
                        
                        # 保存最佳模型
                        save_checkpoint({
                            'step': global_step,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                        }, filename=os.path.join(args.output_dir, args.name+'_checkpoint_best.pth.tar'))
                        print(f"Model saved! Best Val Loss: {best_loss:.4f}")
                    else:
                        no_improve_count += 1 # 损失没降，计数器+1
                        print(f"Validation Loss did not improve. Counter: {no_improve_count}/{patience}")
                        
                        # 检查是否达到阈值
                        if no_improve_count >= patience:
                            print(f"Early Stopping Triggered! No improvement for {patience} consecutive checks.")
                            should_stop += 1 # 标记为停止

                # 4. 分布式同步：Rank 0 把决定告诉大家
                if args.distributed:
                    # 把 should_stop 从 src(0) 广播到所有其他进程
                    dist.broadcast(should_stop, src=0)
                
                # 5. 检查是否停止
                if should_stop.item() == 1:
                    stop_signal = True
                    break # 跳出 DataLoader 循环
        if args.lrdecay and args.lr_schedule == "poly":
            scheduler.step()
        return global_step, total_loss, best_loss, no_improve_count, stop_signal

    def validation(args, test_loader):
        model.eval()
        loss_val = []
        loss_val_cross = []
        with torch.no_grad():
            for step, batch in enumerate(test_loader):
                # image1, image2, text, label = batch["ctImage1"], batch["ctImage2"], batch["text"], batch["label_image"]
                image1, image2, text, heatmap_label, offset_label, size_label = batch["ctImage1"], batch["ctImage2"], batch["text"], batch["heatmap_label"], batch["offset_label"], batch["size_label"]
                with autocast(enabled = args.amp):
                    # image1, image2, text, label = image1.to(args.device), image2.to(args.device), text.to(args.device), label.to(args.device)
                    image1, image2, text, heatmap_label, offset_label, size_label = image1.to(args.device), image2.to(args.device), text.to(args.device), heatmap_label.to(args.device), offset_label.to(args.device), size_label.to(args.device)
                    image1 = image1.as_subclass(torch.Tensor)
                    image2 = image2.as_subclass(torch.Tensor)
                    text = text.as_subclass(torch.Tensor)
                    # image1 = image1.view(-1, *image1.shape[2:])
                    # image2 = image2.view(-1, *image2.shape[2:])
                    # heatmap_label = heatmap_label.view(-1, *heatmap_label.shape[2:])
                    # offset_label = offset_label.view(-1, *offset_label.shape[2:])
                    # size_label = size_label.view(-1, *size_label.shape[2:])
                    # text = text.view(-1, text.shape[-1])

                    # 首先对应的text在这里是(B,maxlen),然后通过text_processor(text)获取对应的(B,maxlen,embedding_dim)，最后再repeat(B, num_samples, maxlen, embedding_dim)
                    # text_embedding = text_processor(text).to(args.device)
                    # text_embedding = text_embedding.unsqueeze(1).repeat(1, args.sw_batch_size, 1, 1)
                    (pre_hm, pre_offset, pre_size),_ = model(image1, image2, text)
                    # output = model(image1, image2, text_embedding)
                    # print(f'out_put.shape:{out_put.shape}') 
                    # print(f'label_image.shape:{label_image.shape}')
                    # 因为有掩码输出，先尝试使用通用的交叉熵跟dice
                    label_batch = {'heatmap': heatmap_label, 'offset': offset_label, 'size': size_label}
                    # total_loss, (hm_loss,_) = loss_function(output, label)
                    total_loss, loss_dict = loss_function(pre_hm, pre_offset, pre_size, label_batch)
                loss_val.append(total_loss.item())
                loss_val_cross.append(loss_dict['hm_loss'].item() )
                print("Validation step:{}, Loss:{:.4f}, Loss HeatMap:{:.4f}".format(step, total_loss, loss_dict['hm_loss'].item()))            
        return np.mean(loss_val), np.mean(loss_val_cross)

    args = get_config()
    



    print(args)

    print("*"*50)
    print('设置环境')
    print("*"*50)
    logdir = "./runs/" + args.logdir
    args.amp = not args.noamp
    torch.backends.cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(True)

    if "LOCAL_RANK" in os.environ:
        args.local_rank = int(os.environ["LOCAL_RANK"])
    else:
        args.local_rank = 0 # 默认为 0
    if "WORLD_SIZE" in os.environ:
        args.distributed = int(os.environ["WORLD_SIZE"]) > 1
    # args.device = "cuda:0"
    args.world_size = 1
    args.rank = 0        

    if args.distributed:
        args.device = "cuda:%d" % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method=args.dist_url)
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        print(
            "Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d."
            % (args.rank, args.world_size)
        )
    else:
        print("Training with a single process on 1 GPUs.")
    assert args.rank >= 0
    if args.rank == 0:
    # 初始化 wandb
        wandb.init(
            project="Lung-Nodule-Seg",  # 项目名称，自己起
            name=f"run_{args.name}",  # 实验名称
            config=args,                # 自动记录所有的超参数
            # mode="offline"            # 如果服务器没网，开启这个模式
        )
    print("*"*50)
    print('设置显卡')
    print("*"*50)

    if args.rank == 0:
        os.makedirs(logdir, exist_ok=True)
        writer = SummaryWriter(logdir)
    else:
        writer = None
    print("*"*50)
    print('加载模型')
    print("*"*50)
    text_processor = LanguageProcessor().to(args.device)
    model = UnetM(text_processor, batch_size=args.batch_size*args.sw_batch_size)
    model.to(args.device)
    # 可选：让 wandb 监控模型的梯度和参数分布
    # model (或 model.module) 需要是 nn.Module
    # wandb.watch(model, log="all", log_freq=100)
    print("*"*50)
    print('设置优化器')
    print("*"*50)
    if args.opt == "adam":
        optimizer = optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.decay)

    elif args.opt == "adamw":
        optimizer = optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.decay)

    elif args.opt == "sgd":
        optimizer = optim.SGD(params=model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.decay)

    if args.resume:
        model_pth = args.resume
        model_dict = torch.load(model_pth)
        model.load_state_dict(model_dict["state_dict"])
        model.epoch = model_dict["epoch"]
        model.optimizer = model_dict["optimizer"]
    print("*"*50)
    print('设置调度器')
    print("*"*50)
    if args.lrdecay:
        if args.lr_schedule == "warmup_cosine":
            scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=args.num_steps)

        elif args.lr_schedule == "poly":

            def lambdas(epoch):
                return (1 - float(epoch) / float(args.epochs)) ** 0.9

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdas)
    print("*"*50)
    print('设置损失')
    print("*"*50)
    # loss_function = Loss1()
    # 这里使用回归高斯热力图预测损失
    loss_function = NoduleDetectionLoss(1,0,10)


    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(model, device_ids=[args.local_rank],output_device=args.local_rank, find_unused_parameters=True)
    print("*"*50)
    print('设置数据加载器')
    print("*"*50)
    train_loader, test_loader = get_loader(args, text_processor)

    global_step = 0
    # === 初始化早停相关变量 ===
    best_val = 1e8
    no_improve_count = 0 # 连续未提升次数
    patience_limit = 5   # 早停阈值：5次
    if args.amp:
        scaler = GradScaler()
    else:
        scaler = None
    print("*"*50)
    print('开始训练')
    print("*"*50)
    # for epoch in range(args.epochs):
    #     if global_step >= args.num_steps:
    #         break
    #     global_step, loss, best_val = train(args, epoch, global_step, train_loader, test_loader, best_val, scaler, text_processor)
    epoch = 0
    while global_step < args.num_steps:
        # 这里的 epoch 变量只是为了传给 train 函数做 log 或者 shuffle 用
        global_step, loss, best_val , no_improve ,stop_signal = train(args, epoch, global_step, train_loader, test_loader, best_val, scaler, no_improve_count, patience_limit, text_processor)
        # 检查是否触发了早停
        if stop_signal:
            print(f"Training stopped early at Global Step {global_step}")
            break # 跳出 while 循环，结束训练
        epoch += 1
    checkpoint = {"epoch": args.epochs, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}

    if args.distributed:
        if dist.get_rank() == 0:
            torch.save(model.state_dict(), logdir + "final_model.pth")
        dist.destroy_process_group()
    else:
        torch.save(model.state_dict(), logdir + "final_model.pth")
    save_checkpoint(checkpoint, logdir + f"/{args.name}_model_final_epoch.pt")

if __name__ == "__main__":
    main()