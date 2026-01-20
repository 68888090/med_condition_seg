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
        self.lambda_main = lambda_main
        self.lambda_aux = lambda_aux
        self.lambda_reg = lambda_reg
        self.lambda_cls = lambda_cls
        
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = 1e-6 # 稍微减小一点，防止影响精度
        self.epsilon = 1e-7 # 防止 log(0) 的极小值
        
        # BCEWithLogitsLoss 内部已经做了数值稳定优化（LogSumExp技巧）
        self.cls_criterion = nn.BCEWithLogitsLoss()

    def forward(self, model_outputs, target_mask, is_text_match, gt_reg=None):
        """
        增加输入检查和数值保护
        """
        # --- 1. 拆包 & 类型安全转换 ---
        main_logits = model_outputs["main_logits"].float()
        aux_logits = model_outputs["aux_logits"].float()
        pred_offset = model_outputs["offset"].float()
        pred_size = model_outputs["diameter"].float()
        match_logits = model_outputs["match_logits"].float()

        # --- 2. 准备 Ground Truth ---
        target_object = target_mask.float()
        B = target_mask.shape[0]
        
        # 确保 is_text_match 是 float 类型且在 [0, 1] 之间
        condition_weight = is_text_match.view(B, 1, 1, 1, 1).float()
        target_final = target_object * condition_weight

        # --------------------------------------------
        # 3. 计算分割损失 (增加 Logits 截断)
        # --------------------------------------------
        # 【防崩溃 1】Logits 截断：防止 sigmoid/exp 计算溢出
        # 范围 [-15, 15] 足以覆盖 sigmoid 的 [0.0000003, 0.9999997]
        main_logits = torch.clamp(main_logits, min=-15.0, max=15.0)
        aux_logits = torch.clamp(aux_logits, min=-15.0, max=15.0)

        loss_main_seg = self.compute_combo_loss(main_logits, target_final)
        loss_aux_seg = self.compute_combo_loss(aux_logits, target_object)

        # --------------------------------------------
        # 4. 计算回归损失 (增加 NaN 过滤)
        # --------------------------------------------
        loss_reg = torch.tensor(0.0, device=target_mask.device)
        
        if gt_reg is not None and target_object.sum() > 0:
            gt_offset, gt_size = gt_reg
            
            # 扩展 mask 维度
            mask_3d = target_object.expand_as(pred_offset) > 0.5
            mask_1d = target_object > 0.5

            # 【防崩溃 2】检查回归预测值是否包含 NaN/Inf，如果有则忽略该样本
            # 这种情况虽然少见，但如果网络发散，回归头最先炸
            if torch.isnan(pred_offset).any() or torch.isinf(pred_offset).any():
                loss_offset = torch.tensor(0.0, device=target_mask.device)
            elif mask_3d.sum() > 0:
                loss_offset = F.smooth_l1_loss(pred_offset[mask_3d], gt_offset[mask_3d], reduction='mean')
            else:
                loss_offset = torch.tensor(0.0, device=target_mask.device)

            if torch.isnan(pred_size).any() or torch.isinf(pred_size).any():
                loss_size = torch.tensor(0.0, device=target_mask.device)
            elif mask_1d.sum() > 0:
                loss_size = F.smooth_l1_loss(pred_size[mask_1d], gt_size[mask_1d], reduction='mean')
            else:
                loss_size = torch.tensor(0.0, device=target_mask.device)
            
            loss_reg = loss_offset + loss_size

        # --------------------------------------------
        # 5. 计算分类损失
        # --------------------------------------------
        # 【防崩溃 3】同样对分类 Logits 截断
        match_logits = torch.clamp(match_logits, min=-15.0, max=15.0)
        loss_cls = self.cls_criterion(match_logits.view(-1), is_text_match.float())

        # --------------------------------------------
        # 6. 总损失聚合 & 最终安全检查
        # --------------------------------------------
        total_loss = (self.lambda_main * loss_main_seg + 
                      self.lambda_aux * loss_aux_seg + 
                      self.lambda_reg * loss_reg +
                      self.lambda_cls * loss_cls)

        # 【防崩溃 4】核武器：如果 Total Loss 是 NaN，返回 0 并打印警告
        # 这样不会因为一个坏 Batch 导致整个训练崩溃
        # if torch.isnan(total_loss) or torch.isinf(total_loss):
        #     print("[Warning] HybridLoss is NaN or Inf! Returning zero loss to skip update.")
        #     # 返回带梯度的 0，防止报错 "variable not in computation graph"
        #     # 也可以选择 return total_loss 让外层处理，视训练框架而定
        #     total_loss = main_logits.sum() * 0.0 

        # 【防崩溃 5】梯度裁剪钩子：防止反向传播时梯度爆炸
        # if total_loss.requires_grad:
        #     total_loss.register_hook(lambda grad: torch.clamp(grad, -1.0, 1.0))

        return {
            "loss": total_loss,
            "loss_main": loss_main_seg,
            "loss_aux": loss_aux_seg,
            "loss_reg": loss_reg,
            "loss_cls": loss_cls
        }

    def compute_combo_loss(self, logits, target):
        """
        logits: 已经经过 clamp 的 logits
        target: 0 or 1
        """
        # --- Focal Loss (BCE part) ---
        # binary_cross_entropy_with_logits 相对稳定，但为了 Focal 计算 pt 仍需小心
        bce_loss = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
        
        # 【防崩溃 6】防止 exp(-bce) 在 bce 极大时下溢（虽然 risk 较小，但为了保险）
        # bce_loss 理论上非负，但在混合精度下可能出现极小负数，clamp 一下
        # bce_loss = torch.clamp(bce_loss, min=0, max=100) 
        pt = torch.exp(-bce_loss)
        
        focal_loss = (1 - pt) ** 2 * bce_loss
        focal_loss = focal_loss.mean()

        # --- Dice Loss ---
        probs = torch.sigmoid(logits) # Logits 已经 clamp 过了，这里是安全的
        
        # Flatten
        probs_flat = probs.view(probs.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        
        intersection = (probs_flat * target_flat).sum(dim=1)
        
        # 【防崩溃 7】分母增加 epsilon，且限制最小值
        # 即使 smooth 设为 0，epsilon 也能保底
        denominator = probs_flat.sum(dim=1) + target_flat.sum(dim=1) + self.smooth
        denominator = torch.clamp(denominator, min=self.epsilon)
        
        dice_score = (2. * intersection + self.smooth) / denominator
        dice_loss = 1. - dice_score.mean()

        return self.bce_weight * focal_loss + self.dice_weight * dice_loss