import os
import argparse
import time
import numpy as np
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.nn as nn
import json
import random
from time import time
# === 导入你的模块 ===
# 确保 cls_UnetM.py 中是那个只接受 img2 的简化版 UnetM
from UnetM_seg import UnetM 
from utils.mask_data_process import get_loader
from utils.metrics import compute_segmentation_metrics
# 不需要 text_processor，但如果 get_loader 依赖它，传个 None 或伪造一个
from models.text_processor import LanguageProcessor
from losses.seg_loss import PureSegLoss



class PureConfig:
    def __init__(self):
        # --- 训练参数 ---
        self.lr = 4e-4              # 学习率
        self.weight_decay = 0.01    # 【关键】绝对不能是 0.1
        self.epochs = 200
        self.batch_size = 32        # 根据显存调整
        self.eval_freq = 50         # 每多少个 step 验证一次
        
        # --- 数据参数 ---
        self.jsonlist = "/home/lhr/dataset/Union_for_lx_Rigid_lung_mask_Crop_64mm_Rigid/all_train_dataloader.json"
        self.temp_dir = "./temp_split_json"
        self.roi_x = 64
        self.roi_y = 64
        self.roi_z = 64
        self.a_min = -1000.0
        self.a_max = 500.0
        self.b_min = 0.0
        self.b_max = 1.0
        self.space_x = 1.0
        self.space_y = 1.0
        self.space_z = 1.0
        self.smartcache_dataset = False
        self.cache_dataset = False
        self.workers = 8
        
        # --- 系统参数 ---
        self.device = "cuda"
        self.output_dir = "./runs/pure_seg_best_dice"
        self.distributed = False # 单卡跑，简单直接
        self.amp = True          # 开启混合精度

def split_data(original_json_path, temp_dir, val_ratio=0.1):
    """
    读取原始 json，随机切分为 train 和 val，保存为临时文件
    """
    os.makedirs(temp_dir, exist_ok=True)
    
    print(f"Loading data from {original_json_path}...")
    with open(original_json_path, 'r') as f:
        data = json.load(f)
    
    # 兼容不同的 json 格式
    if isinstance(data, dict):
        # 如果是 {'training': [...], 'validation': [...]} 这种格式
        full_list = data.get('training', []) + data.get('validation', [])
    else:
        # 如果直接是一个 list [...]
        full_list = data
    
    print(f"Total samples: {len(full_list)}")
    
    # 随机打乱
    random.seed(42)
    random.shuffle(full_list)
    
    # 切分
    val_size = int(len(full_list) * val_ratio)
    train_list = full_list[val_size:]
    val_list = full_list[:val_size]
    
    # 保存临时文件 (必须包装成字典格式，因为你的 get_loader 可能期望 'training'/'val' 键)
    train_path = os.path.join(temp_dir, "pure_train.json")
    val_path = os.path.join(temp_dir, "pure_val.json")
    
    with open(train_path, 'w') as f:
        json.dump({'training': train_list}, f)
        
    with open(val_path, 'w') as f:
        json.dump({'val': val_list}, f) # 注意这里 key 可能是 'validation' 或 'val'，根据你的 loader 调整
        
    print(f"Split done: {len(train_list)} train, {len(val_list)} val")
    print(f"Saved to {train_path} and {val_path}")
    
    return train_path, val_path
# ==============================================================================
# 3. 验证循环
# ==============================================================================
def validate(model, val_loader, device):
    model.eval()
    dice_list = []
    
    with torch.no_grad():
        for batch in val_loader:
            # 只取图像和Mask
            ct2 = batch["CT2_path"].to(device)
            target = batch["label"].to(device)
            
            with autocast():
                # 模型只接受 img2
                outputs = model(ct2)
                logits = outputs["aux_logits"]
            
            # 计算 Dice (使用你的工具函数)
            metrics = compute_segmentation_metrics(logits, target)
            dice_list.append(metrics["dice"])
            
    return np.mean(dice_list)

# ==============================================================================
# 4. 主函数
# ==============================================================================
def main():
    args = PureConfig()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Output Dir: {args.output_dir}")

    # --- 模型 ---
    print("Initializing UnetM (Pure Seg Mode)...")
    # 注意：text_processor 传 None，因为不需要处理文本
    model = UnetM(text_processor=None, batch_size=args.batch_size, dropout_rate=0.2)
    model.to(args.device)
    
    # 打印参数状态，确保 Backone 是解冻的
    for name, param in model.named_parameters():
        if "feature_extractor" in name:
            param.requires_grad = True # 确保 Backbone 参与训练
            
    # --- 优化器 ---
    # 过滤掉不需要梯度的参数（如果有的话）
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    
    # --- Loss ---
    criterion = PureSegLoss(bce_weight=0.5, dice_weight=1.0).to(args.device)
    scaler = GradScaler()
    
    # --- 数据 ---
    # 为了兼容 get_loader，我们需要把 PureConfig 转成 argparse.Namespace
    train_json_path, val_json_path = split_data(args.jsonlist, args.temp_dir)
    args_namespace = argparse.Namespace(**args.__dict__)
    # 传入空的 LanguageProcessor，防止 get_loader 报错（如果它依赖的话）
    args_namespace.train_path = train_json_path
    args_namespace.val_path = val_json_path

    dummy_lp = LanguageProcessor() 
    train_loader, val_loader = get_loader(args_namespace, dummy_lp)
    
    print("Start Training...")
    
    global_step = 0
    best_dice = 0.0
    
    for epoch in range(args.epochs):
        model.train()
        for step, batch in enumerate(train_loader):
            t0 = time()
            
            ct2 = batch["CT2_path"].to(args.device)
            target = batch["label"].to(args.device)
            
            optimizer.zero_grad()
            
            with autocast(enabled=args.amp):
                # 前向传播：只传入 ct2
                outputs = model(ct2)
                
                # 计算 Loss
                loss = criterion(outputs, target)
            
            # 检查 NaN
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Epoch {epoch} Step {step}: Loss is NaN! Skipping...")
                optimizer.zero_grad()
                continue
                
            # 反向传播
            scaler.scale(loss).backward()
            
            # 梯度裁剪
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            # 日志
            if global_step % 10 == 0:
                print(f"Epoch {epoch} | Step {global_step} | Loss: {loss.item():.4f} | Time: {time()-t0:.2f}s")
                
            # === 验证 & 保存最佳 Dice 模型 ===
            if global_step > 0 and global_step % args.eval_freq == 0:
                print(">>> Running Validation...")
                val_dice = validate(model, val_loader, args.device)
                print(f">>> Val Dice: {val_dice:.4f} (Best: {best_dice:.4f})")
                
                if val_dice > best_dice:
                    best_dice = val_dice
                    save_path = os.path.join(args.output_dir, "0120_3_best_dice_model.pth")
                    torch.save(model.state_dict(), save_path)
                    print(f"!!! New Best Model Saved to {save_path} !!!")
            
            global_step += 1

if __name__ == "__main__":
    main()