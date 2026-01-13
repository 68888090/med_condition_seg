import json # 新增
from sklearn.model_selection import KFold # 新增
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
# ... (保留原有的 imports, get_config, flatten_config, train, validation 函数不变) ...

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



# 假设这是你的原始 JSON 路径
ORIGINAL_JSON_FILE = '/home/lhr/dataset/CSTPLung/data2.json' 

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
    
    # === 1. 环境初始化 (只执行一次) ===
    print("*"*50)
    print('设置环境')
    print("*"*50)
    logdir_base = "./runs/" + args.logdir # 基础日志路径
    args.amp = not args.noamp
    torch.backends.cudnn.benchmark = True
    
    if "LOCAL_RANK" in os.environ:
        args.local_rank = int(os.environ["LOCAL_RANK"])
    else:
        args.local_rank = 0 
    
    if "WORLD_SIZE" in os.environ:
        args.distributed = int(os.environ["WORLD_SIZE"]) > 1
    
    args.world_size = 1
    args.rank = 0        

    if args.distributed:
        args.device = "cuda:%d" % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method=args.dist_url)
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
    else:
        print("Training with a single process on 1 GPUs.")
    
    # === 2. 准备交叉验证数据 (关键修改) ===
    # 读取原始 JSON 数据
    with open(ORIGINAL_JSON_FILE, 'r') as f:
        full_data = json.load(f)
        train_data_list = full_data['training'] # 获取列表部分
        val_data_list = full_data['validation']
    data_list = train_data_list + val_data_list
    
    # 定义 4折
    n_folds = args.n_folds
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # 只需要加载一次 text_processor
    text_processor = LanguageProcessor().to(args.device)

    # === 3. 开始 K-Fold 循环 ===
    # enumerate 返回 (折数, (训练索引, 验证索引))
    for fold_idx, (train_indices, val_indices) in enumerate(kfold.split(data_list)):
        
        print(f"\n{'='*20} Start Fold {fold_idx + 1} / {n_folds} {'='*20}")
        
        # --- A. 动态生成本折的 JSON 文件 ---
        # 只让 Rank 0 负责写文件，避免多进程冲突
        fold_train_file = f"temp_data_fold{fold_idx}_train.json"
        fold_val_file = f"temp_data_fold{fold_idx}_val.json"
        
        if args.rank == 0:
            train_subset = [data_list[i] for i in train_indices]
            val_subset = [data_list[i] for i in val_indices]
            
            # 构造符合 dataset 要求的字典结构
            with open(fold_train_file, 'w') as f:
                json.dump({'training': train_subset}, f)
            with open(fold_val_file, 'w') as f:
                json.dump({'training': val_subset}, f)
            print(f"[Fold {fold_idx+1}] Created temporary json files.")

        # 分布式同步：等待 Rank 0 写完文件
        if args.distributed:
            dist.barrier()

        # --- B. 更新 args 以指向新的临时文件 ---
        # 【注意】你需要确保你的 get_loader 函数使用 args.train_path 和 args.val_path
        # 如果你之前的 get_loader 是硬编码读取 JOSN_FILE，请务必去修改 get_loader 
        # 让它接受 args.train_path 参数
        args.train_path = fold_train_file 
        args.val_path = fold_val_file
        
        # 更新日志目录，区分不同 Fold
        current_logdir = os.path.join(logdir_base, f"fold_{fold_idx+1}/")
        if args.rank == 0:
            os.makedirs(current_logdir, exist_ok=True)
            # 重新初始化 Tensorboard
            writer = SummaryWriter(current_logdir)
            
            # 重新初始化 WandB (使用 group 将不同 fold 归为一组)
            # 如果是第一次循环，初始化；后续循环，重新初始化
            if fold_idx > 0: wandb.finish() 
            wandb.init(
                project="Lung-Nodule-Seg-CV", 
                name=f"{args.name}_fold_{fold_idx+1}",
                group=f"{args.name}_CV", # 将4折归为同一个组
                config=args,
                reinit=True
            )
        else:
            writer = None

        # --- C. 重新初始化模型、优化器、调度器、Scaler (防止权重泄漏) ---
        print(f'[Fold {fold_idx+1}] Re-initializing Model and Optimizer...')
        model = UnetM(text_processor, batch_size=args.batch_size*args.sw_batch_size)
        model.to(args.device)
        
        # 【重要】在这里加入 Bias 初始化 (之前讨论过的)
        # prior_prob = 0.01
        # bias_value = -math.log((1 - prior_prob) / prior_prob)
        # model.heatmap_head.bias.data.fill_(bias_value)
        
        if args.opt == "adam":
            optimizer = optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.decay)
        elif args.opt == "adamw":
            optimizer = optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.decay)
        elif args.opt == "sgd":
            optimizer = optim.SGD(params=model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.decay)

        # 调度器重置
        if args.lrdecay:
            if args.lr_schedule == "warmup_cosine":
                scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=args.num_steps)
            elif args.lr_schedule == "poly":
                def lambdas(epoch):
                    return (1 - float(epoch) / float(args.epochs)) ** 0.9
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdas)
        
        # 损失函数重置
        loss_function = NoduleDetectionLoss(1,0,10)

        # DDP 包装重置
        if args.distributed:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

        # 数据加载器重置 (读取新的 json 文件)
        # ！！！关键：请确保 get_loader 内部读取的是 args.train_path 和 args.val_path
        train_loader, test_loader = get_loader(args, text_processor)

        # Scaler 重置
        if args.amp:
            scaler = GradScaler()
        else:
            scaler = None

        # --- D. 重置训练状态变量 ---
        global_step = 0
        best_val = 1e8
        no_improve_count = 0 # 连续未提升次数
        patience_limit = 5   # 早停阈值：5次
        epoch = 0

        # --- E. 开始训练本折 (复制原来的 while 循环) ---
        while global_step < args.num_steps:
            global_step, loss, best_val , no_improve ,stop_signal = train(
                args, epoch, global_step, train_loader, test_loader, 
                best_val, scaler, no_improve_count, patience_limit, text_processor
            )
            
            if stop_signal:
                print(f"[Fold {fold_idx+1}] Early stopping at step {global_step}")
                break
            epoch += 1
        
        # --- F. 本折结束后的清理 ---
        # 保存本折的最佳模型和最终模型
        if args.distributed:
            if dist.get_rank() == 0:
                torch.save(model.state_dict(), os.path.join(current_logdir, "final_model.pth"))
        else:
            torch.save(model.state_dict(), os.path.join(current_logdir, "final_model.pth"))
        
        # 显存清理，防止 OOM
        del model, optimizer, scheduler, scaler, train_loader, test_loader
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[Fold {fold_idx+1}] Finished. GPU cache cleared.")

    # 4折全部结束
    if args.distributed:
        dist.destroy_process_group()
    
    if args.rank == 0:
        wandb.finish()

if __name__ == "__main__":
    main()