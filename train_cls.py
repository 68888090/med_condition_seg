import json 
from sklearn.model_selection import KFold 
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

# === 【修改 1】导入新的 Loss ===
# 假设你把刚才那个损失函数保存为了 losses/hybrid_loss.py
from losses.text_condition_loss import TextGuidedHybridLoss 

from cls_UnetM import UnetM
from optimizers.lr_scheduler import WarmupCosineSchedule
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from utils.mask_data_process import get_loader
from utils.metrics import compute_segmentation_metrics
from models.text_processor import LanguageProcessor

TEMP_JSON_PATH = "/home/lhr/item/Swin-UNETR/temp_json/"
def get_config():
    parser = argparse.ArgumentParser(description="PyTorch Training with YAML")
    parser.add_argument("-c", "--config", default="config_cls.yaml", help="Path to config file")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER, 
                        help="Modify config options from command line")
    args = parser.parse_args()
    conf = OmegaConf.load(args.config)
    if args.opts:
        cli_conf = OmegaConf.from_cli(args.opts)
        conf = OmegaConf.merge(conf, cli_conf)
    args_flat = flatten_config(conf)
    return args_flat

def flatten_config(conf):
    conf_dict = OmegaConf.to_container(conf, resolve=True)
    flat_args = argparse.Namespace()
    def _extract_values(d):
        for key, value in d.items():
            if isinstance(value, dict):
                _extract_values(value)
            else:
                setattr(flat_args, key, value)
    _extract_values(conf_dict)
    return flat_args



# 假设这是你的原始 JSON 路径
ORIGINAL_JSON_FILE = '/home/lhr/dataset/Union_for_lx_Rigid_lung_mask_Crop_64mm_Rigid/dataset.json' 

def main():
    def save_checkpoint(state, filename='checkpoint.pth.tar'):
        torch.save(state, filename)

    # =================================================================================
    # Train Loop
    # =================================================================================
    def train(args, epoch, global_step, train_loader, val_loader, best_loss, best_dice, scaler, no_improve_count, patience, text_processor=None):
        
        if args.distributed and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        model.train()
        loss_train = []
        loss_main_list = []
        loss_aux_list = []

        stop_signal = False

        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            t1 = time()
            
            # === 【修改 2】读取新 Dataset 的 Batch ===
            # batch keys: ctImage1, ctImage2, label, text, satisfy(meta_data中获取或者dataset直接返回)
            # 注意：MONAI dataset 可能会把非图像字段放在 meta_dict 里，或者直接作为 key
            # 假设你的 dataset json 里有 "satisfy": true/false
            
            ct1 = batch["CT1_path"].to(args.device)
            ct2 = batch["CT2_path"].to(args.device)
            target_mask = batch["label"].to(args.device) # CT2 的真实 Mask
            text = batch["text"].to(args.device)
            
            # 处理 satisfy 字段
            # 如果 json 里写的是 boolean，dataloader 出来可能是 tensor([True, False])
            # 需要转为 float 0.0/1.0
            if "satisfy" in batch:
                is_text_match = batch["satisfy"].to(args.device).float()
            else:
                # 如果没有这个字段，默认全是正样本
                is_text_match = torch.ones(ct1.shape[0], device=args.device)

            with autocast(enabled=args.amp):
                # === 【修改 3】模型前向传播 ===
                # model 返回: (main_logits, aux_logits, offset, size, dummy_alphas)
                outputs = model(ct1, ct2, text)
                
                # 解包以便监控 (offset 和 size 在这里用于 loss 计算，但在 log 里我们只看 mask loss)
                # main_logits, aux_logits, pred_offset, pred_size = outputs["main_logits"], outputs["aux_logits"], outputs["offset"], outputs["diameter"]
                
                # === 【修改 4】计算 Loss ===
                # 目前没有 Offset/Size 的 GT，所以传入 None，Loss 会自动忽略回归部分
                # 如果你有 GT，请在这里传入 (gt_offset, gt_size)
                gt_reg = None 
                
                loss_dict = loss_function(outputs, target_mask, is_text_match, gt_reg)
                # ================= [Debug 探针 START] =================
                # if step % 10 == 0 and args.rank == 0:
                #     print(f"\n====== DEBUG STEP {step} ======")
                    
                #     # 1. 检查输入范围 (Swin 极其敏感!)
                #     print(f"[Input Data] Shape: {ct1.shape}")
                #     print(f"   CT1: min={ct1.min().item():.2f}, max={ct1.max().item():.2f}, mean={ct1.mean().item():.2f}")
                #     # 正常情况应该是 min>=0, max<=1 (或者 -1 到 1)。如果看到 -1000，必死无疑。

                #     # 2. 检查 Label 是否有内容 (防止空跑)
                #     mask_sum = target_mask.sum().item()
                #     print(f"[Label] Sum (Pixels): {mask_sum:.2f}")
                #     if mask_sum == 0:
                #         print("   ⚠️ WARNING: This batch has NO nodules (All background)!")

                #     # 3. 检查 Aux Head 的输出 (看模型是否“活着”)
                #     # 检查输出的 Logits 是否全是一样的数
                #     aux_out = outputs['aux_logits']
                #     print(f"[Model Output] Aux Logits: min={aux_out.min().item():.4f}, max={aux_out.max().item():.4f}")
                    
                #     # 4. 检查梯度 (看是否断开)
                #     # 此时还没有 backward，我们需要手动试一下 backward (仅测试用)
                #     # 注意：这会轻微影响这次迭代的计算图，正式跑的时候去掉下面这几行
                #     # test_loss = loss_dict['loss']
                #     # test_loss.backward(retain_graph=True)
                #     # if model.module.nodule_detection_head.aux_seg_head.weight.grad is not None:
                #     #     grad_norm = model.module.nodule_detection_head.aux_seg_head.weight.grad.norm().item()
                #     #     print(f"[Gradient] Aux Head Grad Norm: {grad_norm:.6f}")
                #     # else:
                #     #     print("   ⚠️ ERROR: No gradient on Aux Head!")
                #     # optimizer.zero_grad() # 清空测试梯度
                # # ================= [Debug 探针 END] =================
                total_loss = loss_dict["loss"]
                
                # =================== 防爆显存逻辑 ===================
                if torch.isnan(total_loss) or torch.isinf(total_loss) or total_loss > 10000:
                    print(f"Warning: Bad batch at Step {global_step}! Loss: {total_loss.item()}")
                    optimizer.zero_grad()
                    del total_loss, loss_dict, outputs, ct1, ct2, target_mask, text
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                    continue 
                # ===================================================

            loss_train.append(total_loss.item())
            loss_main_list.append(loss_dict['loss_main'].item())
            loss_aux_list.append(loss_dict['loss_aux'].item())
            
            # Backward
            if args.amp:
                scaler.scale(total_loss).backward()
                if args.grad_clip:
                    scaler.unscale_(optimizer)
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

            # Logging
            if global_step % args.log_num == 0:
                log_data = {
                    "train/total_loss": total_loss.item(),
                    "train/main_loss": loss_dict['loss_main'].item(),
                    "train/aux_loss": loss_dict['loss_aux'].item(),
                    "train/reg_loss": loss_dict['loss_reg'].item(), # 如果没有GT，这里是0
                    "train/lr": optimizer.param_groups[0]['lr'],
                    "global_step": global_step
                }
                if args.distributed:
                    if dist.get_rank() == 0:
                        wandb.log(log_data)
                        print(f"Step:{global_step}/{args.num_steps}, Loss:{total_loss:.4f}, Main:{loss_dict['loss_main']:.4f}")
                else:
                    wandb.log(log_data)
                    print(f"Step:{global_step}/{args.num_steps}, Loss:{total_loss:.4f}, Main:{loss_dict['loss_main']:.4f}")

            global_step += 1

            # =================== Validation Check ===================
            is_val_step = (global_step % args.eval_num == 0)
            if is_val_step:
                should_stop = torch.tensor(0).to(args.device)
                
                if args.rank == 0:
                    val_metrics = validation(args, val_loader)
                    
                    loss_val = val_metrics["all_loss"]
                    
                    # 【修改点 1】获取关键指标 Gated Dice
                    # 如果没有 gated 指标 (比如 aux 任务)，退化为 all_dice
                    dice_gated = val_metrics.get("gated_dice", val_metrics["all_dice"])
                    dice_pos = val_metrics.get("pos_dice", 0.0)
                    
                    print(f"\n[Validation] Loss: {loss_val:.4f} | Gated Dice: {dice_gated:.4f} (Best: {best_dice:.4f})")
                    
                    # WandB Logging
                    wandb_log_dict = {f"val/{k}": v for k, v in val_metrics.items()}
                    wandb_log_dict["global_step"] = global_step
                    wandb.log(wandb_log_dict)

                    # === 早停逻辑 (依然基于 Loss，防止过拟合) ===
                    # 通常早停看 Loss 比较稳，但保存模型看 Dice
                    if loss_val < best_loss:
                        best_loss = loss_val
                        no_improve_count = 0
                        # 保存最佳 Loss 模型
                        save_checkpoint({
                            'step': global_step,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'metric': loss_val
                        }, filename=os.path.join(args.output_dir, args.name+'_best_loss.pth.tar'))
                        print(f"Saved Best Loss Model!")
                    else:
                        no_improve_count += 1
                        print(f"No improvement in Loss. Counter: {no_improve_count}/{patience}")
                        if no_improve_count >= patience:
                            print(f"Early Stopping Triggered!")
                            should_stop += 1
                    
                    # === 【修改点 2】保存最佳 Gated Dice 模型 ===
                    if dice_gated > best_dice:
                        best_dice = dice_gated
                        save_checkpoint({
                            'step': global_step,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'metric': best_dice
                        }, filename=os.path.join(args.output_dir, args.name+'_best_gated_dice.pth.tar'))
                        print(f"!!! Saved Best Gated Dice Model! ({best_dice:.4f}) !!!")

                if args.distributed:
                    dist.broadcast(should_stop, src=0)
                
                if should_stop.item() == 1:
                    stop_signal = True
                    break 

        # 【修改点 3】返回值增加 best_dice
        return global_step, np.mean(loss_train), best_loss, best_dice, no_improve_count, stop_signal

    # =================================================================================
    # Validation Loop
    # =================================================================================
    def validation(args, test_loader):
        model.eval()
        
        # 1. 定义累加器：增加 'gated' 组
        meters = {
            "all": {k: [] for k in ["loss", "dice", "iou", "recall", "precision"]},
            "pos": {k: [] for k in ["dice", "iou", "recall", "precision"]}, # 正样本指标
            "gated": {k: [] for k in ["dice", "iou", "recall", "precision"]} # 【新增】门控指标
        }
        
        # 获取验证目标 (默认 main)
        val_target = getattr(args, 'val_target', 'main')

        with torch.no_grad():
            for step, batch in enumerate(test_loader):
                # 1. 数据准备
                ct1 = batch["CT1_path"].to(args.device)
                ct2 = batch["CT2_path"].to(args.device)
                target_mask = batch["label"].to(args.device)
                text = batch["text"].to(args.device)
                
                if "satisfy" in batch:
                    is_text_match = batch["satisfy"].to(args.device).float()
                else:
                    is_text_match = torch.ones(ct1.shape[0], device=args.device)

                with autocast(enabled=args.amp):
                    outputs = model(ct1, ct2, text)
                    
                    # 2. 解包输出 (根据你的模型返回是字典还是元组，这里保持和你输入一致的字典风格)
                    # 务必确保你的 forward 函数返回了 match_logits
                    main_logits = outputs["main_logits"]
                    aux_logits = outputs["aux_logits"]
                    match_logits = outputs["match_logits"] # 【新增】获取分类头输出
                    
                    # 计算 Loss
                    gt_reg = None
                    loss_dict = loss_function(outputs, target_mask, is_text_match, gt_reg)
                
                # 3. 确定用于计算指标的 Logits 和 GT
                if val_target == 'aux':
                    pred_logits = aux_logits
                    gt_final = target_mask # Aux 任务永远看真实结节
                else:
                    pred_logits = main_logits
                    B = target_mask.shape[0]
                    condition_weight = is_text_match.view(B, 1, 1, 1, 1).type_as(target_mask)
                    gt_final = target_mask * condition_weight # 文本不符则全黑

                # =================== 【指标计算 A: 全体样本 (Raw)】 ===================
                metrics_all = compute_segmentation_metrics(pred_logits, gt_final)
                
                meters["all"]["loss"].append(loss_dict['loss'].item())
                for k in metrics_all:
                    meters["all"][k].append(metrics_all[k])

                # =================== 【指标计算 B: 门控推理 (Gated)】 ===================
                # 只有在验证 Main 任务时，Gating 才有意义
                if val_target == 'main':
                    # A. 计算分类概率
                    match_prob = torch.sigmoid(match_logits) # [B, 1]
                    
                    # B. 生成门控信号 (阈值 0.5)
                    # 如果概率 > 0.5，gate=1 (保留)；否则 gate=0 (抑制)
                    B = main_logits.shape[0]
                    gate = (match_prob > 0.5).float().view(B, 1, 1, 1, 1)
                    
                    # C. 应用门控
                    # 如果 gate=1，保持原值；如果 gate=0，设为极小负数 (-1e9)，Sigmoid后即为0
                    gated_logits = main_logits * gate + (1 - gate) * -1.0e9
                    
                    # D. 计算指标 (使用 Gated Logits 和 真实的 GT)
                    metrics_gated = compute_segmentation_metrics(gated_logits, gt_final)
                    
                    for k in metrics_gated:
                        meters["gated"][k].append(metrics_gated[k])
                else:
                    # 如果是 aux 任务，gated 指标没意义，可以直接填 0 或者复制 all
                    for k in meters["gated"]: meters["gated"][k].append(0.0)

                # =================== 【指标计算 C: 仅正样本 (Pos Only)】 ===================
                # 筛选出 is_text_match == 1 的索引
                pos_indices = (is_text_match > 0.5)
                
                if pos_indices.sum() > 0: # 只有当 Batch 里有正样本时才计算
                    # 切片 (Slicing)
                    pred_pos = pred_logits[pos_indices]
                    gt_pos = gt_final[pos_indices]
                    
                    metrics_pos = compute_segmentation_metrics(pred_pos, gt_pos)
                    
                    for k in metrics_pos:
                        meters["pos"][k].append(metrics_pos[k])
                
                # 打印日志 (每10步)
                if step % 10 == 0:
                    curr_pos_dice = meters["pos"]["dice"][-1] if len(meters["pos"]["dice"]) > 0 else 0.0
                    curr_gated_dice = meters["gated"]["dice"][-1] if len(meters["gated"]["dice"]) > 0 else 0.0
                    print(f"Val Step {step} [{val_target.upper()}]: Loss={loss_dict['loss']:.4f} | Raw_Dice={metrics_all['dice']:.4f} | Gated_Dice={curr_gated_dice:.4f} | Pos_Dice={curr_pos_dice:.4f}")

        # 4. 汇总平均值
        avg_metrics = {}
        
        # 遍历所有组 (all, pos, gated)
        for group in meters:
            for k, v in meters[group].items():
                key_name = f"{group}_{k}"
                if len(v) > 0:
                    avg_metrics[key_name] = np.mean(v)
                else:
                    avg_metrics[key_name] = 0.0 # 防止空列表报错

        return avg_metrics
    # =================================================================================
    # Main Setup
    # =================================================================================
    args = get_config()
    
    # Environment Setup
    print("*"*50)
    print('Setting up Environment')
    print("*"*50)
    logdir_base = "./runs/" + args.logdir
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
    
    # K-Fold Data Prep
    with open(args.jsonlist, 'r') as f:
        # 假设 JSON 结构是 list of dicts，或者有 'data' 键
        # 根据你给的例子，它看起来是一个 list
        # 如果是 list: full_data = json.load(f)
        # 如果是 {'training': [...], 'validation': [...]}:
        full_data_dict = json.load(f)
        if isinstance(full_data_dict, list):
            data_list = full_data_dict
        else:
             # 如果原始json分了train/val，我们合并它们重新做KFold
            data_list = full_data_dict.get('training', []) + full_data_dict.get('validation', [])
    
    n_folds = args.n_folds
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    text_processor = LanguageProcessor().to(args.device)

    # K-Fold Loop
    for fold_idx, (train_indices, val_indices) in enumerate(kfold.split(data_list)):
        
        print(f"\n{'='*20} Start Fold {fold_idx + 1} / {n_folds} {'='*20}")
        
        # Temp Files Generation
        fold_train_file = f"{args.name}_temp_data_fold{fold_idx}_train.json"
        fold_val_file = f"{args.name}_temp_data_fold{fold_idx}_val.json"
        
        # 确保目录存在
        os.makedirs(TEMP_JSON_PATH, exist_ok=True)
        fold_train_file = os.path.join(TEMP_JSON_PATH, fold_train_file)
        fold_val_file = os.path.join(TEMP_JSON_PATH, fold_val_file)

        if args.rank == 0:
            train_subset = [data_list[i] for i in train_indices]
            val_subset = [data_list[i] for i in val_indices]
            with open(fold_train_file, 'w') as f:
                json.dump({'training': train_subset}, f) # 必须包装成 {'training': []} 格式供 load_decathlon_datalist 读取
            with open(fold_val_file, 'w') as f:
                json.dump({'val': val_subset}, f)
            print(f"[Fold {fold_idx+1}] Created temporary json files.")

        if args.distributed:
            dist.barrier()

        args.train_path = fold_train_file 
        args.val_path = fold_val_file
        
        # Logging Setup
        current_logdir = os.path.join(logdir_base, f"fold_{fold_idx+1}/")
        if args.rank == 0:
            os.makedirs(current_logdir, exist_ok=True)
            writer = SummaryWriter(current_logdir)
            if fold_idx > 0: wandb.finish() 
            wandb.init(
                project=args.project, 
                name=f"{args.name}_fold_{fold_idx+1}",
                group=f"{args.name}_CV",
                config=args,
                reinit=True
            )

        # Model Init
        print(f'[Fold {fold_idx+1}] Re-initializing Model...')
        # 注意：这里不需要传入 sw_batch_size 了，因为没有 Crop，batch_size 就是 args.batch_size
        model = UnetM(text_processor, batch_size=args.batch_size, dropout_rate=args.dropout_rate, fusion_mode=args.fusion_mode)
        model.to(args.device)
        
        if args.opt == "adam":
            optimizer = optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.decay)
        elif args.opt == "adamw":
            optimizer = optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.decay)
        
        if args.lrdecay and args.lr_schedule == "warmup_cosine":
            scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=args.num_steps)
        
        # === 【修改 5】 Loss Init ===
        loss_function = TextGuidedHybridLoss(
            lambda_main=args.main_loss_weight, 
            lambda_aux=args.aux_loss_weight, 
            lambda_reg=args.reg_loss_weight, # 暂时不训练回归，设为 0
            lambda_cls=args.cls_loss_weight
        ).to(args.device)

        if args.distributed:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

        train_loader, test_loader = get_loader(args, text_processor)

        if args.amp:
            scaler = GradScaler()
        else:
            scaler = None

        global_step = 0
        best_val = 1e8
        no_improve_count = 0 
        best_dice = -1.0
        patience_limit = args.patience if hasattr(args, 'patience') else 10
        epoch = 0

        while global_step < args.num_steps:
            global_step, loss, best_val ,best_dice ,no_improve_count ,stop_signal = train(
                args, epoch, global_step, train_loader, test_loader, 
                best_val, best_dice, scaler, no_improve_count, patience_limit, text_processor
            )
            
            if stop_signal:
                print(f"[Fold {fold_idx+1}] Early stopping at step {global_step}")
                break
            epoch += 1
        
        # Cleanup
        if args.distributed:
            if dist.get_rank() == 0:
                torch.save(model.state_dict(), os.path.join(current_logdir, args.name+"_final_model.pth"))
        else:
            torch.save(model.state_dict(), os.path.join(current_logdir, args.name+"_final_model.pth"))
        
        del model, optimizer, scheduler, scaler, train_loader, test_loader
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[Fold {fold_idx+1}] Finished.")

    if args.distributed:
        dist.destroy_process_group()
    if args.rank == 0:
        wandb.finish()

if __name__ == "__main__":
    main()