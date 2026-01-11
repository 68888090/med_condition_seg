# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch.nn import functional as F
from monai.losses.dice import DiceLoss

class Contrast(torch.nn.Module):
    def __init__(self, args, batch_size, temperature=0.5):
        super().__init__()
        device = torch.device(f"cuda:{args.local_rank}")
        self.batch_size = batch_size
        self.register_buffer("temp", torch.tensor(temperature).to(torch.device(f"cuda:{args.local_rank}")))
        self.register_buffer("neg_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device)).float())

    def forward(self, x_i, x_j):
        z_i = F.normalize(x_i, dim=1)
        z_j = F.normalize(x_j, dim=1)
        z = torch.cat([z_i, z_j], dim=0)
        sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
        sim_ij = torch.diag(sim, self.batch_size)
        sim_ji = torch.diag(sim, -self.batch_size)
        pos = torch.cat([sim_ij, sim_ji], dim=0)
        nom = torch.exp(pos / self.temp)
        denom = self.neg_mask * torch.exp(sim / self.temp)
        return torch.sum(-torch.log(nom / torch.sum(denom, dim=1))) / (2 * self.batch_size)


class Loss(torch.nn.Module):
    def __init__(self, batch_size, args):
        super().__init__()
        self.rot_loss = torch.nn.CrossEntropyLoss().cuda()
        self.recon_loss = torch.nn.L1Loss().cuda()
        self.contrast_loss = Contrast(args, batch_size).cuda()
        self.alpha1 = 1.0
        self.alpha2 = 1.0
        self.alpha3 = 1.0

    def __call__(self, output_rot, target_rot, output_contrastive, target_contrastive, output_recons, target_recons):
        rot_loss = self.alpha1 * self.rot_loss(output_rot, target_rot)
        contrast_loss = self.alpha2 * self.contrast_loss(output_contrastive, target_contrastive)
        recon_loss = self.alpha3 * self.recon_loss(output_recons, target_recons)
        total_loss = rot_loss + contrast_loss + recon_loss

        return total_loss, (rot_loss, contrast_loss, recon_loss)


import torch
import torch.nn as nn
from monai.losses import DiceLoss

class Loss1(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        # 1. Dice Loss 配置
        # to_onehot_y=True: 会自动把 target (B, 1, ...) 转成 One-Hot (B, C, ...)
        # softmax=True: 会先对 output 做 softmax 再算 dice
        # reduction='mean': 默认就是 mean，会在 batch 维度求平均，保证 loss 不受 batch size 大小影响
        self.dice_loss = DiceLoss(
            to_onehot_y=True, 
            softmax=True, 
            reduction="mean" 
        )
        
        # 2. CrossEntropy 配置
        # reduction='mean': 同样在 batch 上求平均
        self.cross_loss = nn.CrossEntropyLoss(reduction="mean")
        
        # 3. 权重
        self.alpha1 = 1.0 # CE weight
        self.alpha2 = 1.0 # Contrastive weight (暂未启用)
        self.alpha3 = 1.0 # Dice weight

    def forward(self, output, target):
        """
        output: (B, C, H, W, D) - 模型的直接输出 (Logits)
        target: (B, 1, H, W, D) - 标签
        """
        
        # --- 处理 CrossEntropy Loss ---
        # CE Loss 要求 target 是 long 类型，且不能有 Channel 维度 (B, H, W, D)
        # 所以这里使用 squeeze(1) 去掉 Channel=1 的维度
        if target.shape[1] == 1:
            target_ce = target.squeeze(1).long()
        else:
            target_ce = target.long()
            
        rot_loss = self.alpha1 * self.cross_loss(output, target_ce)

        # --- 处理 Dice Loss ---
        # DiceLoss 在初始化时设置了 to_onehot_y=True，所以直接传原始 target 即可
        dice_loss = self.alpha3 * self.dice_loss(output, target)

        # --- 处理 Contrastive Loss (如果需要) ---
        # 你的原代码里没有计算 output_contrastive，这里先设为 0 以防报错
        contrast_loss = 0.0 
        # contrast_loss = self.alpha2 * self.contrast_loss(output_contrastive, target_contrastive)

        # 总损失
        total_loss = rot_loss + contrast_loss + dice_loss

        return total_loss, (rot_loss, dice_loss)
