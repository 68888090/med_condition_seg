import torch
import torch.nn as nn
import torch.nn.functional as F

class PureSegLoss(nn.Module):
    def __init__(self, 
                 bce_weight=0.5, 
                 dice_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        
        # 数值稳定性参数
        self.smooth = 1e-6 
        self.epsilon = 1e-7 

    def forward(self, model_outputs, target_mask):
        """
        Args:
            model_outputs: 字典，必须包含 "aux_logits"
            target_mask: 真实的分割掩码 [B, 1, D, H, W]
        """
        # 1. 拆包 & 类型转换
        # 你的简化版模型只返回 aux_logits，我们把它作为唯一的预测输出
        pred_logits = model_outputs["aux_logits"].float()
        target = target_mask.float()
        
        # 2. 数值保护 (Logits 截断)
        # 防止 sigmoid/exp 计算溢出，导致 NaN
        pred_logits = torch.clamp(pred_logits, min=-15.0, max=15.0)

        # 3. 计算混合损失 (Dice + Focal)
        loss = self.compute_combo_loss(pred_logits, target)

        # 4. 安全检查 (防 NaN)
        if torch.isnan(loss) or torch.isinf(loss):
            print("[Warning] PureSegLoss is NaN or Inf! Returning zero loss.")
            # 返回带梯度的 0，防止 DDP 报错
            return torch.tensor(0.0, device=target.device, requires_grad=True)

        return loss

    def compute_combo_loss(self, logits, target):
        """
        计算单路分割的 Focal + Dice Loss
        """
        # --- A. Focal Loss (基于 BCE) ---
        # reduction='none' 使得我们可以手动加权或处理
        bce_loss = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
        
        # 防止 exp(-bce) 下溢
        bce_loss = torch.clamp(bce_loss, min=0, max=100) 
        pt = torch.exp(-bce_loss)
        
        # Focal 公式: (1-pt)^2 * log(pt)
        focal_loss = ((1 - pt) ** 2 * bce_loss).mean()

        # --- B. Dice Loss ---
        probs = torch.sigmoid(logits)
        
        # Flatten [B, C, D, H, W] -> [B, -1]
        probs_flat = probs.view(probs.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        
        # 计算 Intersection & Union
        intersection = (probs_flat * target_flat).sum(dim=1)
        
        # 分母增加 epsilon 防止除以 0
        denominator = probs_flat.sum(dim=1) + target_flat.sum(dim=1) + self.smooth
        denominator = torch.clamp(denominator, min=self.epsilon)
        
        dice_score = (2. * intersection + self.smooth) / denominator
        dice_loss = 1. - dice_score.mean()

        # --- C. 加权求和 ---
        return self.bce_weight * focal_loss + self.dice_weight * dice_loss