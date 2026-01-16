import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class DensePixelTextLayer(nn.Module):
    """
    【深层专用：像素级文本注意力】
    适用于处理 "Growth +26%" 这种复杂条件。
    机制：让图像的每个空间位置 (Query) 去查询文本序列的所有 Token (Key/Value)。
    这样，图像中"增长了26%"的那个区域，就能强烈响应文本中的"26%" Token。
    """
    def __init__(self, img_dim, text_dim, dropout_rate=0.1):
        super().__init__()
        
        # 1. 投影层：把图像和文本映射到相同维度
        self.img_proj = nn.Conv3d(img_dim, img_dim, kernel_size=1)
        self.text_proj = nn.Linear(text_dim, img_dim)
        
        # 2. 差分提取
        self.diff_conv = nn.Sequential(
            nn.Conv3d(img_dim, img_dim, kernel_size=3, padding=1),
            nn.InstanceNorm3d(img_dim),
            nn.ReLU(inplace=True)
        )

        # 3. Cross Attention
        # Query: Image Pixels
        # Key/Value: Text Sequence (保留由数字组成的序列)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=img_dim, 
            num_heads=4,  # 头数不宜过多，防止小样本过拟合
            batch_first=True, 
            dropout=dropout_rate
        )
        self.norm = nn.LayerNorm(img_dim)
        
        # 4. 融合输出
        self.fusion_conv = nn.Sequential(
            nn.Conv3d(img_dim * 2, img_dim, kernel_size=1),
            nn.InstanceNorm3d(img_dim),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout_rate)
        )

    def forward(self, f1, f2, text_seq):
        """
        f1, f2: [B, C, D, H, W]
        text_seq: [B, Seq_Len, Text_Dim] (保留序列，不要池化!)
        """
        B, C, D, H, W = f1.shape
        
        # A. 计算并编码差分特征 (Query)
        # 我们让"变化"去寻找"文本描述"
        raw_diff = f2 - f1
        diff_feat = self.diff_conv(raw_diff) # [B, C, D, H, W]
        
        # B. 准备 Attention 输入
        # Query: Flatten Image Diff -> [B, N_pixels, C]
        q = rearrange(self.img_proj(diff_feat), 'b c d h w -> b (d h w) c')
        
        # Key/Value: Projected Text -> [B, Seq_Len, C]
        k = v = self.text_proj(text_seq)
        
        # C. 执行 Attention
        # 这一步模型会计算：图像位置 i 的变化特征，与文本 token j (比如 "26%") 的相似度
        attn_out, _ = self.cross_attn(query=q, key=k, value=v)
        
        # 残差 + Norm
        q = self.norm(q + attn_out)
        
        # D. 还原空间维度
        # [B, N, C] -> [B, C, D, H, W]
        attn_feat = rearrange(q, 'b (d h w) c -> b c d h w', d=D, h=H, w=W)
        
        # E. 融合回 Target Image
        return self.fusion_conv(torch.cat([f2, attn_feat], dim=1))


class SimpleGateLayer(nn.Module):
    """
    【浅层专用：全局门控】
    之前的简单版本，用于浅层，减少显存占用，防止过拟合。
    """
    def __init__(self, channels, text_dim, dropout_rate):
        super().__init__()
        self.text_gate = nn.Sequential(
            nn.Linear(text_dim, channels),
            nn.ReLU(),
            nn.Linear(channels, channels),
            nn.Sigmoid()
        )
        self.diff_conv = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(channels),
            nn.ReLU(inplace=True)
        )
        self.fusion_conv = nn.Sequential(
            nn.Conv3d(channels * 2, channels, kernel_size=1),
            nn.InstanceNorm3d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, f1, f2, text_global):
        diff = f2 - f1
        # [B, C] -> [B, C, 1, 1, 1]
        gate = self.text_gate(text_global).view(diff.shape[0], diff.shape[1], 1, 1, 1)
        weighted_diff = self.diff_conv(diff * gate)
        return self.fusion_conv(torch.cat([f2, weighted_diff], dim=1))


class HybridFusionModule(nn.Module):
    """
    【混合融合策略】
    Deep Layers (4, 3): 使用 DensePixelTextLayer (理解"26%"这种复杂数字)
    Shallow Layers (2, 1, 0): 使用 SimpleGateLayer (恢复边缘细节)
    """
    def __init__(self, 
                 feature_channels=[48, 96, 192, 384, 768], 
                 text_dim=768,
                 active_layers=[4, 3, 2, 1, 0],
                 dropout_rate=0.2):
        super().__init__()
        
        self.active_layers = active_layers
        self.fusion_layers = nn.ModuleList()
        
        # 定义哪些层属于"深层" (需要复杂 Attention)
        # 通常 feature_channels[4] (768) 和 [3] (384) 是深层
        self.deep_layer_indices = [3, 4] 
        
        for i in range(len(feature_channels)):
            if i in active_layers:
                if i in self.deep_layer_indices:
                    # 深层：使用 Pixel-Text Attention
                    self.fusion_layers.append(
                        DensePixelTextLayer(feature_channels[i], text_dim, dropout_rate)
                    )
                else:
                    # 浅层：使用 Simple Gating
                    self.fusion_layers.append(
                        SimpleGateLayer(feature_channels[i], text_dim, dropout_rate)
                    )
            else:
                self.fusion_layers.append(nn.Identity())

    def forward(self, x_out1, x_out2, text_embedding):
        """
        x_out1, x_out2: 特征列表
        text_embedding: [B, Seq_Len, 768] 注意保留序列维度
        """
        # 1. 准备全局文本向量 (给浅层用)
        text_global = text_embedding.mean(dim=1)
        
        fused_outputs = []
        
        for i, (f1, f2) in enumerate(zip(x_out1, x_out2)):
            if i in self.active_layers:
                layer_module = self.fusion_layers[i]
                
                # 根据层类型传入不同的文本形式
                if i in self.deep_layer_indices:
                    # 深层：传入序列文本 [B, Seq, C]
                    fused = layer_module(f1, f2, text_embedding)
                else:
                    # 浅层：传入全局文本 [B, C]
                    fused = layer_module(f1, f2, text_global)
                    
                fused_outputs.append(fused)
            else:
                fused_outputs.append(f2)
                
        return fused_outputs