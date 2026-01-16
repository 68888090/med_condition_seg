import torch
import torch.nn as nn
import torch.nn.functional as F

class TextGatedDiffFusion(nn.Module):
    """
    【All-in-One 融合模块】
    功能：
    1. 计算两张图的特征差分 (捕捉变化)
    2. 利用文本生成门控信号，筛选重要的变化特征
    3. 将筛选后的变化特征融合回 Target Image (Img2)
    4. 输出保持原始空间分辨率，供 UNet 解码使用
    """
    def __init__(self, 
                 feature_channels=[48, 96, 192, 384, 768], 
                 text_dim=768,
                 active_layers=[4, 3, 2, 1, 0], 
                 dropout_rate=0.1):
        super().__init__()
        
        self.active_layers = active_layers
        self.fusion_layers = nn.ModuleList()
        
        # 为每一层构建一个融合块
        # 如果某一层不在 active_layers 里，就用 Identity 占位
        for i in range(len(feature_channels)):
            if i in active_layers:
                self.fusion_layers.append(
                    SingleLayerGateFusion(feature_channels[i], text_dim, dropout_rate)
                )
            else:
                self.fusion_layers.append(nn.Identity())

    def forward(self, x_out1, x_out2, text_embedding):
        """
        x_out1, x_out2: List of [B, C, D, H, W]
        text_embedding: [B, Seq_Len, text_dim]
        """
        # 1. 文本池化：[B, L, C] -> [B, C]
        # 我们只需要文本的全局语义来指导"关注什么类型的变化"
        if text_embedding.dim() == 3:
            text_global = text_embedding.mean(dim=1) 
        else:
            text_global = text_embedding

        fused_outputs = []
        
        # 2. 逐层融合
        for i, (f1, f2) in enumerate(zip(x_out1, x_out2)):
            if i in self.active_layers:
                # 调用单层融合
                fused = self.fusion_layers[i](f1, f2, text_global)
                fused_outputs.append(fused)
            else:
                # 不融合层直接返回 Img2 (Target)
                fused_outputs.append(f2)
                
        return fused_outputs

class SingleLayerGateFusion(nn.Module):
    def __init__(self, channels, text_dim, dropout_rate):
        super().__init__()
        
        # A. 文本门控生成器
        self.text_gate = nn.Sequential(
            nn.Linear(text_dim, channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(channels, channels),
            nn.Sigmoid() # 输出 0-1 权重
        )
        
        # B. 差分特征提取 (简单卷积)
        self.diff_conv = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(channels), # 小 Batch 用 IN 更稳
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout_rate)
        )
        
        # C. 最终融合 (Channel Attention 后的加法)
        self.fusion_conv = nn.Sequential(
            nn.Conv3d(channels * 2, channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, f1, f2, text_vec):
        # 1. 计算原始差分
        diff = f2 - f1
        
        # 2. 生成文本权重
        # [B, C] -> [B, C, 1, 1, 1]
        gate = self.text_gate(text_vec).view(diff.shape[0], diff.shape[1], 1, 1, 1)
        
        # 3. 文本加权：筛选有意义的差分
        weighted_diff = diff * gate
        
        # 4. 提取空间特征
        diff_feat = self.diff_conv(weighted_diff)
        
        # 5. 融合回主干 (Img2)
        # 将"变化信息"注入到"当前图像特征"中

        #----------------------------------------
        # 注入方式：简单的 Channel-wise 拼接，应该还有优化空间
        #----------------------------------------

        combined = torch.cat([f2, diff_feat], dim=1)
        out = self.fusion_conv(combined)
        
        return out