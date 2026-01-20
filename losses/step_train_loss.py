import torch
import torch.nn as nn
import torch.nn.functional as F

class TextGuidedHybridLoss(nn.Module):
    def __init__(self, 
                 lambda_main=1.0, 
                 lambda_aux=0.4, 
                 lambda_reg=0.1, 
                 lambda_cls=0.2, 
                 bce_weight=0.5, 
                 dice_weight=0.5):
        super().__init__()
        # 这些权重会在训练过程中通过 update_loss_weights 函数动态修改
        self.lambda_main = lambda_main
        self.lambda_aux = lambda_aux
        self.lambda_reg = lambda_reg
        self.lambda_cls = lambda_cls
        
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = 1e-6 
        self.epsilon = 1e-7 
        
        self.cls_criterion = nn.BCEWithLogitsLoss()

    def forward(self, model_outputs, target_mask, is_text_match, gt_reg=None):
        """
        model_outputs: 字典, 包含 main_logits, aux_logits, match_logits 等
        """
        # 准备设备
        device = target_mask.device
        
        # 初始化各项 Loss 为 0
        loss_main_seg = torch.tensor(0.0, device=device)
        loss_aux_seg = torch.tensor(0.0, device=device)
        loss_reg = torch.tensor(0.0, device=device)
        loss_cls = torch.tensor(0.0, device=device)

        # --- 1. 准备 Ground Truth ---
        target_object = target_mask.float()
        B = target_mask.shape[0]
        
        # 确保 is_text_match 是 float 类型
        condition_weight = is_text_match.view(B, 1, 1, 1, 1).float()
        target_final = target_object * condition_weight

        # ==========================================================
        # Part A: Aux Seg Loss (Stage 1 & 2 通常都需要)
        # ==========================================================
        if self.lambda_aux > 0:
            aux_logits = model_outputs.get("aux_logits")
            if aux_logits is not None:
                aux_logits = torch.clamp(aux_logits.float(), min=-15.0, max=15.0)
                loss_aux_seg = self.compute_combo_loss(aux_logits, target_object)

        # ==========================================================
        # Part B: Main Seg Loss (通常仅 Stage 2)
        # ==========================================================
        if self.lambda_main > 0:
            main_logits = model_outputs.get("main_logits")
            if main_logits is not None:
                main_logits = torch.clamp(main_logits.float(), min=-15.0, max=15.0)
                loss_main_seg = self.compute_combo_loss(main_logits, target_final)

        # ==========================================================
        # Part C: Classification Loss (通常仅 Stage 2)
        # ==========================================================
        # 只有当 match_logits 存在且 lambda_cls > 0 时才计算
        match_logits = model_outputs.get("match_logits")
        
        if self.lambda_cls > 0 and match_logits is not None:
            match_logits = torch.clamp(match_logits.float(), min=-15.0, max=15.0)
            loss_cls = self.cls_criterion(match_logits.view(-1), is_text_match.float())

        # ==========================================================
        # Part D: Regression Loss (如果有 GT 且 Lambda > 0)
        # ==========================================================
        if self.lambda_reg > 0 and gt_reg is not None and target_object.sum() > 0:
            pred_offset = model_outputs.get("offset")
            pred_size = model_outputs.get("diameter")
            
            if pred_offset is not None and pred_size is not None:
                pred_offset = pred_offset.float()
                pred_size = pred_size.float()
                
                gt_offset, gt_size = gt_reg
                
                # 扩展 mask 维度
                mask_3d = target_object.expand_as(pred_offset) > 0.5
                mask_1d = target_object > 0.5

                # NaN 检查
                if torch.isnan(pred_offset).any() or torch.isinf(pred_offset).any():
                    loss_offset = torch.tensor(0.0, device=device)
                elif mask_3d.sum() > 0:
                    loss_offset = F.smooth_l1_loss(pred_offset[mask_3d], gt_offset[mask_3d], reduction='mean')
                else:
                    loss_offset = torch.tensor(0.0, device=device)

                if torch.isnan(pred_size).any() or torch.isinf(pred_size).any():
                    loss_size = torch.tensor(0.0, device=device)
                elif mask_1d.sum() > 0:
                    loss_size = F.smooth_l1_loss(pred_size[mask_1d], gt_size[mask_1d], reduction='mean')
                else:
                    loss_size = torch.tensor(0.0, device=device)
                
                loss_reg = loss_offset + loss_size

        # ==========================================================
        # 总损失聚合
        # ==========================================================
        total_loss = (self.lambda_main * loss_main_seg + 
                      self.lambda_aux * loss_aux_seg + 
                      self.lambda_reg * loss_reg +
                      self.lambda_cls * loss_cls)

        # 安全检查
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print("[Warning] HybridLoss is NaN or Inf! Returning zero loss.")
            # 使用 aux_logits 或 main_logits 保持梯度图连接（如果有的话），或者直接返回 0
            # 这里返回 0.0 requires_grad=True 是为了防止 DDP 报错 (unused parameters)
            return {
                "loss": torch.tensor(0.0, device=device, requires_grad=True),
                "loss_main": loss_main_seg,
                "loss_aux": loss_aux_seg,
                "loss_reg": loss_reg,
                "loss_cls": loss_cls
            }

        if total_loss.requires_grad:
            total_loss.register_hook(lambda grad: torch.clamp(grad, -1.0, 1.0))

        return {
            "loss": total_loss,
            "loss_main": loss_main_seg,
            "loss_aux": loss_aux_seg,
            "loss_reg": loss_reg,
            "loss_cls": loss_cls
        }

    def compute_combo_loss(self, logits, target):
        # 保持你原来的逻辑不变
        bce_loss = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
        bce_loss = torch.clamp(bce_loss, min=0, max=100) 
        pt = torch.exp(-bce_loss)
        focal_loss = ((1 - pt) ** 2 * bce_loss).mean()

        probs = torch.sigmoid(logits)
        probs_flat = probs.view(probs.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        
        intersection = (probs_flat * target_flat).sum(dim=1)
        denominator = probs_flat.sum(dim=1) + target_flat.sum(dim=1) + self.smooth
        denominator = torch.clamp(denominator, min=self.epsilon)
        
        dice_score = (2. * intersection + self.smooth) / denominator
        dice_loss = 1. - dice_score.mean()

        return self.bce_weight * focal_loss + self.dice_weight * dice_loss