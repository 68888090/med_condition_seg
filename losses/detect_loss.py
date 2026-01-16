import torch
import torch.nn as nn
import torch.nn.functional as F

class HeatmapDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pred_sigmoid, gt_hm):
        # pred_sigmoid: (B, 1, D, H, W) 已经经过 sigmoid
        # gt_hm: (B, 1, D, H, W) 
        
        intersection = (pred_sigmoid * gt_hm).sum()
        union = pred_sigmoid.sum() + gt_hm.sum()
        
        # 加上 smooth 防止除 0
        dice = (2. * intersection + 1e-5) / (union + 1e-5)
        
        return 1.0 - dice


class NoduleDetectionLoss(nn.Module):
    def __init__(self, weight_heatmap=1.0, weight_offset=1, weight_size=0.1):
        super().__init__()
        self.w_hm = weight_heatmap
        self.w_off = weight_offset
        self.w_size = weight_size
        
        # 使用 SmoothL1Loss 替代 L1Loss，beta=1.0 是通用设置
        self.smooth_l1 = nn.SmoothL1Loss(reduction='sum', beta=1.0)

    def forward(self, pred_hm, pred_off, pred_size, batch_targets):
        """
        输入:
        pred_hm:   (B, 1, D, H, W) - Logits (未Sigmoid)
        pred_off:  (B, 3, D, H, W) - Sigmoid后的偏移
        pred_size: (B, 1, D, H, W) - 直径预测值 (建议模型端输出 raw logits，不要 softplus)
        """
        # --- 1. 强制数值稳定转换 ---
        # 无论外面是否用 AMP，Loss 内部强制 float32 计算
        pred_hm = pred_hm.float()
        pred_off = pred_off.float()
        pred_size = pred_size.float()
        
        gt_hm = batch_targets['heatmap'].float()
        # print(torch.unique(gt_hm))
        gt_off = batch_targets['offset'].float()
        gt_size = batch_targets['size'].float()
        pred_hm = torch.clamp(pred_hm, min=-10.0, max=10.0)
        # --- 2. Heatmap Loss (Safe Version) ---
        pred_hm_sigmoid = torch.sigmoid(pred_hm)
        # 严格限制范围，防止 log(0) 或 log(1)
        pred_hm_sigmoid = torch.clamp(pred_hm_sigmoid, min=1e-4, max=1-1e-4)
        loss_hm = self.modified_focal_loss(pred_hm_sigmoid, gt_hm)

        # --- 3. Regression Loss (Masked) ---
        mask = gt_hm.gt(0.97) # 严格中心点
        mask_s = mask.squeeze(1)
        num_pos = mask_s.sum()

        if num_pos > 0:
            # 提取正样本位置
            pred_off_pos = pred_off.permute(0, 2, 3, 4, 1)[mask_s]
            gt_off_pos = gt_off.permute(0, 2, 3, 4, 1)[mask_s]
            
            pred_size_pos = pred_size.permute(0, 2, 3, 4, 1)[mask_s]
            gt_size_pos = gt_size.permute(0, 2, 3, 4, 1)[mask_s]

            # --- 改进点 A: Offset 使用 Smooth L1 ---
            loss_off = self.smooth_l1(pred_off_pos, gt_off_pos) / num_pos

            # --- 改进点 B: Size 使用 Log 空间回归 ---
            # 如果你的模型输出已经是 softplus 后的正数：
            # 我们对 预测值 和 真值 都取 log，在对数空间做回归
            # 加上 1e-6 防止 log(0)
            # d = 1e-10
            # pred_log_size = torch.log(pred_size_pos + d)
            # gt_log_size = torch.log(gt_size_pos + d)
            # 预测效果不好的原因可能是在于不在像素角度去进行一个预测导致的的问题
            # loss_size = self.smooth_l1(pred_log_size, gt_log_size) / num_pos
            norm_factor = 5.0 #数据统计值 

            loss_size = self.smooth_l1(pred_size_pos / norm_factor, 
                           gt_size_pos / norm_factor) / num_pos
        else:
            loss_off = torch.tensor(0.0, device=pred_hm.device)
            loss_size = torch.tensor(0.0, device=pred_hm.device)

        # --- 4. 总 Loss 聚合 ---
        total_loss = self.w_hm * loss_hm + self.w_off * loss_off + self.w_size * loss_size
        
        # --- 5. 【核武器】梯度截断钩子 (Gradient Clamp Hook) ---
        # 这行代码会注册一个钩子，在反向传播开始的第一时间，
        # 强行把 total_loss 传回网络的梯度限制在 [-1, 1] 之间。
        # 这样无论 Loss 算出来多大，回传给网络的“力”都不会导致权重爆炸。
        if total_loss.requires_grad:
            total_loss.register_hook(lambda grad: torch.clamp(grad, -1.0, 1.0))

        return total_loss, {"hm_loss": loss_hm, "offset_loss": loss_off, "size_loss": loss_size}

    def modified_focal_loss(self, pred, gt):
        """
        更加数值安全的 Focal Loss 实现
        """
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()
        
        neg_weights = torch.pow(1 - gt, 4)
        pred_sigmoid = pred
        loss = 0

        # 正样本 term
        # 既然 pred 已经 clamp 过了，直接计算 log 是安全的
        pos_loss = torch.log(pred_sigmoid) * torch.pow(1 - pred_sigmoid, 2) * pos_inds
        
        # 负样本 term
        neg_loss = torch.log(1 - pred_sigmoid) * torch.pow(pred_sigmoid, 2) * neg_weights * neg_inds

        num_pos = pos_inds.sum()
        
        # 单独求和，避免显存中大张量相加导致的精度损失
        pos_loss_sum = pos_loss.sum()
        neg_loss_sum = neg_loss.sum()

        if num_pos == 0:
            loss = loss -neg_loss_sum
        else:
            pos_weights = 100.0
            loss = -(pos_weights * pos_loss_sum + neg_loss_sum) / num_pos
            
        return loss