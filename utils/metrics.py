import torch
import numpy as np
from sklearn.metrics import roc_auc_score

def compute_segmentation_metrics(pred_logits, gt_mask, threshold=0.5):
    """
    计算分割指标 (Dice, IoU, Recall, Precision)
    pred_logits: 模型输出的 logits [B, 1, D, H, W] (未经过 sigmoid)
    gt_mask: 真实的 mask [B, 1, D, H, W] (0 或 1)
    """
    # 1. 预处理
    probs = torch.sigmoid(pred_logits)
    pred_mask = (probs > threshold).float()
    gt_mask = gt_mask.float()
    
    # 2. 计算交集和并集 (按 Batch 维度求和，保持 B 维度，用于计算平均值)
    # Flatten: [B, -1]
    pred_flat = pred_mask.view(pred_mask.size(0), -1)
    gt_flat = gt_mask.view(gt_mask.size(0), -1)
    
    intersection = (pred_flat * gt_flat).sum(dim=1)
    pred_sum = pred_flat.sum(dim=1)
    gt_sum = gt_flat.sum(dim=1)
    union = pred_sum + gt_sum - intersection
    
    # 3. 计算指标 (加 smooth 防止除以 0)
    smooth = 1e-5
    
    # Dice = 2*I / (P + G)
    dice = (2. * intersection + smooth) / (pred_sum + gt_sum + smooth)
    
    # IoU = I / U
    iou = (intersection + smooth) / (union + smooth)
    
    # Recall = I / G (注意：如果 GT 全为 0，Recall 无意义或定义为 1)
    # 这里我们简单处理：如果 GT 有值，正常计算；如果 GT 全 0 且 Pred 全 0，Recall=1；否则 Recall=0
    # 为简单起见，使用 smooth 处理分母为 0 的情况
    recall = (intersection + smooth) / (gt_sum + smooth)
    
    # Precision = I / P
    precision = (intersection + smooth) / (pred_sum + smooth)
    
    return {
        "dice": dice.mean().item(),
        "iou": iou.mean().item(),
        "recall": recall.mean().item(),
        "precision": precision.mean().item()
    }