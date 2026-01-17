import torch
import torch.nn as nn
import torch.nn.functional as F

class TextGuidedHybridLoss(nn.Module):
    def __init__(self, 
                 lambda_main=1.0, 
                 lambda_aux=0.4, 
                 lambda_reg=0.1, 
                 bce_weight=0.5, 
                 dice_weight=0.5):
        """
        Args:
            lambda_main: 主任务（文本条件分割）的权重
            lambda_aux: 辅助任务（基础分割）的权重
            lambda_reg: 回归任务（Offset/Size）的权重
            bce_weight: 组合Loss中 Focal/BCE 的占比
            dice_weight: 组合Loss中 Dice 的占比
        """
        super().__init__()
        self.lambda_main = lambda_main
        self.lambda_aux = lambda_aux
        self.lambda_reg = lambda_reg
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = 1e-5

    def forward(self, model_outputs, target_mask, is_text_match, gt_reg=None):
        """
        Args:
            model_outputs: (main_logits, aux_logits, pred_offset, pred_size, _) 来自模型的输出
            target_mask: [B, 1, D, H, W]  CT2 的真实结节 Mask (0或1)
            is_text_match: [B]  布尔值或0/1，表示该样本的文本描述是否匹配 (正样本对=1, 负样本对=0)
            gt_reg: (Optional) [gt_offset, gt_size] 如果你需要训练回归头，需要提供这两个GT
                    gt_offset: [B, 3, D, H, W]
                    gt_size: [B, 1, D, H, W]
        """
        # 1. 拆包模型输出
        main_logits, aux_logits, pred_offset, pred_size = model_outputs["main_logits"], model_outputs["aux_logits"], model_outputs["offset"], model_outputs["diameter"]

        # 2. 准备 Ground Truth
        # target_object: 永远是 CT2 里真实的结节 (用于 Aux Loss)
        target_object = target_mask.float()
        
        # target_final: 如果文本不匹配，Mask 置为全 0 (用于 Main Loss)
        # 需调整 shape 以便广播: [B] -> [B, 1, 1, 1, 1]
        B = target_mask.shape[0]
        condition_weight = is_text_match.view(B, 1, 1, 1, 1).type_as(target_mask)
        target_final = target_object * condition_weight

        # --------------------------------------------
        # 3. 计算分割损失 (Dice + Focal)
        # --------------------------------------------
        
        # A. Main Loss (条件分割)
        # 即使 target_final 全为 0，Focal Loss 也能提供强有效的梯度压制背景
        loss_main_seg = self.compute_combo_loss(main_logits, target_final)
        
        # B. Aux Loss (基础分割)
        # 帮助 Encoder 学习结节特征，不受文本干扰
        loss_aux_seg = self.compute_combo_loss(aux_logits, target_object)

        # --------------------------------------------
        # 4. 计算回归损失 (Offset & Size) - Masked Loss
        # --------------------------------------------
        loss_reg = torch.tensor(0.0, device=target_mask.device)
        
        # 只有提供了 GT 并且当前图像确实有结节时，才计算回归损失
        if gt_reg is not None and target_object.sum() > 0:
            gt_offset, gt_size = gt_reg
            
            # 生成 Mask (只在结节内部计算回归损失，背景的预测不重要)
            # 扩展 mask 维度以匹配 offset [B, 1, D, H, W] -> [B, 3, D, H, W]
            mask_3d = target_object.expand_as(pred_offset) > 0.5
            mask_1d = target_object > 0.5

            # 计算 Offset Loss (Smooth L1)
            # 只取 Mask 内的元素计算 mean
            if mask_3d.sum() > 0:
                loss_offset = F.smooth_l1_loss(pred_offset[mask_3d], gt_offset[mask_3d], reduction='mean')
            else:
                loss_offset = 0.0

            # 计算 Size Loss (Smooth L1)
            if mask_1d.sum() > 0:
                loss_size = F.smooth_l1_loss(pred_size[mask_1d], gt_size[mask_1d], reduction='mean')
            else:
                loss_size = 0.0
            
            loss_reg = loss_offset + loss_size

        # --------------------------------------------
        # 5. 总损失聚合
        # --------------------------------------------
        total_loss = (self.lambda_main * loss_main_seg + 
                      self.lambda_aux * (loss_aux_seg + self.lambda_reg * loss_reg))

        return {
            "loss": total_loss,
            "loss_main": loss_main_seg,
            "loss_aux": loss_aux_seg,
            "loss_reg": loss_reg
        }

    def compute_combo_loss(self, logits, target):
        """
        组合 Focal Loss 和 Dice Loss
        logits: [B, 1, D, H, W] (未经过 Sigmoid)
        target: [B, 1, D, H, W]
        """
        # --- Focal Loss (BCE Variant) ---
        # 使用 BCEWithLogitsLoss 更加数值稳定
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
        pt = torch.exp(-bce) # p_t is the probability of the correct class
        focal_loss = (1 - pt) ** 2 * bce  # gamma=2
        focal_loss = focal_loss.mean()

        # --- Dice Loss ---
        probs = torch.sigmoid(logits)
        
        # Flatten [B, C, D, H, W] -> [B, -1]
        probs_flat = probs.view(probs.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        
        intersection = (probs_flat * target_flat).sum(dim=1)
        denominator = probs_flat.sum(dim=1) + target_flat.sum(dim=1)
        
        # Add smooth term to prevent division by zero
        dice_score = (2. * intersection + self.smooth) / (denominator + self.smooth)
        dice_loss = 1. - dice_score.mean()

        return self.bce_weight * focal_loss + self.dice_weight * dice_loss