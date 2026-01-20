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

from losses.step_train_loss import TextGuidedHybridLoss 

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
    parser.add_argument("-c", "--config", default="config_step.yaml", help="Path to config file")
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

def set_training_phase(model, phase, args):
    if isinstance(model, DistributedDataParallel):
        real_model = model.module
    else:
        real_model = model

    print(f"rank {args.rank}: Switching to Phase {phase}...")

    if phase == 1:
        # === 阶段一 ===
        for name, param in real_model.named_parameters():
            # 冻结融合和主头
            if "fusion_module" in name or "main_seg_head" in name or "match_head" in name:
                param.requires_grad = False
            
            # 解冻 Decoder相关、Aux头、以及 【shared_conv】
            # 注意：shared_conv 在 detection head 里，名字通常包含 "shared_conv"
            elif "decoder" in name or "encoder10" in name or "aux_seg_head" in name or "shared_conv" in name:
                param.requires_grad = True
            
            

    elif phase == 2:
        # === 阶段二 ===
        for name, param in real_model.named_parameters():
            # 冻结 Decoder, Aux, Backbone, 以及 【shared_conv】
            # 这样保证 Aux 分支完全静止，只作为参考
            if "decoder" in name or "encoder10" in name or "aux_seg_head" in name or "feature_extractor" in name or "shared_conv" in name:
                param.requires_grad = False
            
            # 解冻融合和主头
            if "fusion_module" in name or "main_seg_head" in name or "match_head" in name:
                param.requires_grad = True

    trainable_params = sum(p.numel() for p in real_model.parameters() if p.requires_grad)
    print(f"rank {args.rank}: Phase {phase} set. Trainable params: {trainable_params}")
    return model
    
def update_loss_weights(loss_function, phase, args):
    """
    根据阶段调整 Loss 权重
    """
    if phase == 1:
        # 阶段一：只看 Aux Loss (纯分割)
        loss_function.lambda_main = 0.0
        loss_function.lambda_aux = 1.0  # 设为 1.0 或保持原样
        loss_function.lambda_reg = 0.0
        loss_function.lambda_cls = 0.0
    else:
        # 阶段二：恢复所有 Loss
        loss_function.lambda_main = args.main_loss_weight
        loss_function.lambda_aux = args.aux_loss_weight
        loss_function.lambda_reg = args.reg_loss_weight
        loss_function.lambda_cls = args.cls_loss_weight
    
    print(f"Loss weights updated for Phase {phase}: Main={loss_function.lambda_main}, Aux={loss_function.lambda_aux}")



# 假设这是你的原始 JSON 路径
ORIGINAL_JSON_FILE = '/home/lhr/dataset/Union_for_lx_Rigid_lung_mask_Crop_64mm_Rigid/dataset.json' 

def main():
    def save_checkpoint(state, filename='checkpoint.pth.tar'):
        torch.save(state, filename)

    # =================================================================================
    # Train Loop
    # =================================================================================
    def train(args, epoch, global_step, train_loader, val_loader, best_loss, best_dice, scaler, no_improve_count, patience, text_processor=None, current_mode="pure_seg", current_phase = 1):
        
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
                outputs = model(ct1, ct2, text, current_mode)
                
                # 解包以便监控 (offset 和 size 在这里用于 loss 计算，但在 log 里我们只看 mask loss)
                # main_logits, aux_logits, pred_offset, pred_size = outputs["main_logits"], outputs["aux_logits"], outputs["offset"], outputs["diameter"]
                
                # === 【修改 4】计算 Loss ===
                # 目前没有 Offset/Size 的 GT，所以传入 None，Loss 会自动忽略回归部分
                # 如果你有 GT，请在这里传入 (gt_offset, gt_size)
                gt_reg = None 
                
                loss_dict = loss_function(outputs, target_mask, is_text_match, gt_reg)
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
                    val_metrics = validation(args, val_loader, current_phase)
                    
                    loss_val = val_metrics["all_loss"]
                    
                    # 【修改点 1】获取关键指标 Gated Dice
                    # 如果没有 gated 指标 (比如 aux 任务)，退化为 all_dice
                    # === 【修改】根据阶段决定这一轮 "Best Dice" 看谁 ===
                    if current_phase == 1:
                        # Phase 1: 关注 all_dice (即 Aux Head 的能力)
                        current_main_metric = val_metrics["all_dice"]
                        metric_name = "Aux Dice"
                    else:
                        # Phase 2: 关注 gated_dice (最终能力)
                        current_main_metric = val_metrics.get("gated_dice", val_metrics["all_dice"])
                        metric_name = "Gated Dice"
                    
                    print(f"\n[Validation Phase {current_phase}] Loss: {loss_val:.4f} | {metric_name}: {current_main_metric:.4f} (Best: {best_dice:.4f})")
                    
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
                    if current_main_metric > best_dice:
                        best_dice = current_main_metric
                        save_filename = f"{args.name}_phase{current_phase}_best_dice.pth.tar"
                        save_checkpoint({
                            'step': global_step,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'metric': best_dice,
                            'phase': current_phase # 记录一下阶段
                        }, filename=os.path.join(args.output_dir, save_filename))
                        
                        print(f"!!! Saved Best {metric_name} Model! ({best_dice:.4f}) !!!")

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
    def validation(args, test_loader, current_phase):
        model.eval()
        
        # 1. 确定验证目标
        if current_phase == 1:
            val_target = 'aux'
        else:
            val_target = getattr(args, 'val_target', 'main')

        # 2. 定义累加器：增加 'cls' 组用于存放分类指标
        meters = {
            "all": {k: [] for k in ["loss", "dice", "iou", "recall", "precision"]},
            "pos": {k: [] for k in ["dice", "iou", "recall", "precision"]}, 
            "gated": {k: [] for k in ["dice", "iou", "recall", "precision"]},
            "cls": {"acc": [], "pred_prob": []} # 【新增】分类指标
        }

        with torch.no_grad():
            for step, batch in enumerate(test_loader):
                ct1 = batch["CT1_path"].to(args.device)
                ct2 = batch["CT2_path"].to(args.device)
                target_mask = batch["label"].to(args.device)
                text = batch["text"].to(args.device)
                
                if "satisfy" in batch:
                    is_text_match = batch["satisfy"].to(args.device).float()
                else:
                    is_text_match = torch.ones(ct1.shape[0], device=args.device)

                with autocast(enabled=args.amp):
                    # 这里的 mode 跟随 phase
                    val_mode = 'pure_seg' if current_phase == 1 else 'train_multimodal'
                    outputs = model(ct1, ct2, text, val_mode)
                    
                    main_logits = outputs["main_logits"]
                    aux_logits = outputs["aux_logits"]
                    
                    # 获取匹配头的输出 (Phase 1 可能是 None)
                    match_logits = outputs.get("match_logits", None) 
                    
                    # 计算 Loss
                    loss_dict = loss_function(outputs, target_mask, is_text_match, None)
                
                # ====================================================
                # 【新增】计算 Satisfy 分类准确率 (仅在 Phase 2)
                # ====================================================
                if current_phase == 2 and match_logits is not None:
                    # 1. 计算概率
                    probs = torch.sigmoid(match_logits).view(-1) # [B]
                    preds = (probs > 0.5).float()                # [B] 0.0 or 1.0
                    
                    # 2. 计算准确率
                    # is_text_match 也要 view(-1) 确保维度对齐
                    gt_cls = is_text_match.view(-1)
                    correct = (preds == gt_cls).float().sum()
                    acc = correct / gt_cls.numel()
                    
                    meters["cls"]["acc"].append(acc.item())
                    meters["cls"]["pred_prob"].append(probs.mean().item()) # 记录一下平均预测概率，观察是否总是预测为0或1
                else:
                    meters["cls"]["acc"].append(0.0) # Phase 1 占位

                # ====================================================
                # 原有的分割指标计算
                # ====================================================
                if val_target == 'aux':
                    pred_logits = aux_logits
                    gt_final = target_mask 
                else:
                    pred_logits = main_logits
                    B = target_mask.shape[0]
                    condition_weight = is_text_match.view(B, 1, 1, 1, 1).type_as(target_mask)
                    gt_final = target_mask * condition_weight 

                metrics_all = compute_segmentation_metrics(pred_logits, gt_final)
                meters["all"]["loss"].append(loss_dict['loss'].item())
                for k in metrics_all: meters["all"][k].append(metrics_all[k])

                # Gated Dice 计算
                if current_phase == 2 and val_target == 'main' and match_logits is not None:
                    # 复用上面算的 probs
                    probs = torch.sigmoid(match_logits) 
                    B = main_logits.shape[0]
                    gate = (probs > 0.5).float().view(B, 1, 1, 1, 1) # 这里其实就是preds
                    
                    gated_logits = main_logits * gate + (1 - gate) * -1.0e9
                    metrics_gated = compute_segmentation_metrics(gated_logits, gt_final)
                    for k in metrics_gated: meters["gated"][k].append(metrics_gated[k])
                    
                    pos_indices = (is_text_match > 0.5)
                    if pos_indices.sum() > 0:
                        metrics_pos = compute_segmentation_metrics(pred_logits[pos_indices], gt_final[pos_indices])
                        for k in metrics_pos: meters["pos"][k].append(metrics_pos[k])
                else:
                    for k in meters["gated"]: meters["gated"][k].append(0.0)
                    for k in meters["pos"]: meters["pos"][k].append(0.0)

                # ====================================================
                # 打印日志：增加 Cls Acc
                # ====================================================
                if step % 10 == 0:
                    if current_phase == 1:
                        print(f"Val Step {step} [Phase 1 - AUX]: Loss={loss_dict['loss']:.4f} | Raw_Dice={metrics_all['dice']:.4f}")
                    else:
                        curr_gated_dice = meters["gated"]["dice"][-1] if len(meters["gated"]["dice"]) > 0 else 0.0
                        curr_cls_acc = meters["cls"]["acc"][-1] if len(meters["cls"]["acc"]) > 0 else 0.0
                        # 打印 Acc
                        print(f"Val Step {step} [Phase 2 - MAIN]: Loss={loss_dict['loss']:.4f} | Dice={curr_gated_dice:.4f} | Match_Acc={curr_cls_acc:.4f}")

        # 汇总平均值
        avg_metrics = {}
        for group in meters:
            for k, v in meters[group].items():
                key_name = f"{group}_{k}"
                avg_metrics[key_name] = np.mean(v) if len(v) > 0 else 0.0

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
        current_phase = 1
        if current_phase == 1:
            current_mode = 'pure_seg'
        else:
            current_mode = 'train_multimodal' # 或者 args.mode
        # 如果 args 里没写 stage1_steps，默认跑 0 步(即直接阶段二)，或者设个默认值
        stage1_steps = getattr(args, 'stage1_steps', 50000)
        model = set_training_phase(model, 1, args)

        # 2. DDP 包装 (如果在 Phase 1 设置后包装，DDP 会自动识别 requires_grad)
        if args.distributed:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
        
        # 3. 初始化 Loss
        loss_function = TextGuidedHybridLoss(...).to(args.device)
        update_loss_weights(loss_function, 1, args) # 设置 Phase 1 权重

        # 4. 初始化 Optimizer (针对 Phase 1 的参数)
        # 注意：必须过滤掉 requires_grad=False 的参数，否则 Adam 会报错或浪费显存
        params_to_optimize = list(filter(lambda p: p.requires_grad, model.parameters()))
        if args.rank == 0:
            print(f"Phase 1 Optimizer Params Check: Found {len(params_to_optimize)} tensors to train.")
            if len(params_to_optimize) == 0:
                raise RuntimeError("CRITICAL ERROR: No parameters to optimize! Check set_training_phase.")
        if args.opt == "adamw":
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                                    lr=args.stage1_lr, weight_decay=args.decay)
        elif args.opt == "adam":
            optimizer = optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.decay)
        
        if args.lrdecay and args.lr_schedule == "warmup_cosine":
            scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=args.num_steps)
        
       

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

        # current_mode = args.mode

        while global_step < args.num_steps:
            # === 【修改 3】阶段切换逻辑检测 ===
            # 如果当前是 Phase 1，且步数超过了设定值 -> 切换到 Phase 2
            if current_phase == 1 and global_step >= stage1_steps:
                if args.rank == 0:
                    print(f"\n{'!'*20} Reached {global_step} steps. Switching to Phase 2 (Multimodal Fine-tuning) {'!'*20}\n")
                
                current_phase = 2
                current_mode = 'train_multimodal'
                
                # 1. 调整模型参数冻结状态
                model = set_training_phase(model, 2, args)
                
                # 2. 调整 Loss 权重 (开启 Main Loss 和 Match Loss)
                update_loss_weights(loss_function, 2, args)
                
                # 3. 【核心】重建 Optimizer
                # 因为可训练参数变了，旧的 optimizer 里的 param_groups 已经失效
                del optimizer
                if args.opt == "adamw":
                    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                                            lr=args.stage2_lr, weight_decay=args.decay) # 使用阶段二的学习率
                
                # 4. 重建 Scheduler (可选，如果你希望阶段二重新热身)
                if args.lrdecay:
                    scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=args.num_steps - global_step)

                # 5. 为了确保 DDP 状态同步，建议做一次 barrier
                if args.distributed:
                    dist.barrier()

            # 执行训练 (传入 current_mode)
            global_step, avg_loss, best_val, best_dice, no_improve_count, stop_signal = train(
                args, epoch, global_step, train_loader, test_loader, 
                best_val, best_dice, scaler, no_improve_count, patience_limit, text_processor,
                current_mode=current_mode, # <--- 传入模式
                current_phase=current_phase
            )
            
            if stop_signal:
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