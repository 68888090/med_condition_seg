import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleConsistencyDenoise(nn.Module):
    """
    专门用于生成去噪后的差分特征 (Key的来源)
    利用深层语义差异作为Mask，过滤浅层噪声
    """
    def __init__(self, channels_list):
        super().__init__()
        # 为了调整Mask的通道数以匹配上一层，我们需要简单的1x1卷积
        self.mask_alignments = nn.ModuleList()
        # 从 x4 -> x3, x3 -> x2 ... 依次对齐
        # channels_list 顺序为 [48, 96, 192, 384, 768]
        # 倒序处理：768->384, 384->192 ...
        for i in range(len(channels_list) - 1, 0, -1):
            deep_c = channels_list[i]
            shallow_c = channels_list[i-1]
            self.mask_alignments.append(
                nn.Conv3d(deep_c, shallow_c, kernel_size=1)
            )

    def forward(self, feat_a_list, feat_b_list):
        # 1. 计算原始差分
        raw_diffs = [fa - fb for fa, fb in zip(feat_a_list, feat_b_list)]
        
        cleaned_diffs = [None] * len(raw_diffs)
        
        # 2. 最深层 (x4) 假设是语义上最纯净的，直接作为基准
        cleaned_diffs[-1] = raw_diffs[-1] 
        
        # 3. 自顶向下 (Top-down) 过滤
        # 循环顺序：4 -> 3 -> 2 -> 1 (处理生成 3, 2, 1, 0)
        # align_mod 索引对应：0:(x4->x3), 1:(x3->x2)...
        for i in range(len(raw_diffs) - 1, 0, -1):
            deep_diff = cleaned_diffs[i]
            shallow_diff = raw_diffs[i-1]
            
            # 生成 Mask: Sigmoid激活，使其变为 [0, 1] 概率图
            mask = torch.sigmoid(deep_diff)
            
            # 调整通道数 (deep_channels -> shallow_channels)
            # 注意 self.mask_alignments 是按 append 顺序存储的，需要正确索引
            # align_index 0 对应 x4->x3
            align_index = (len(raw_diffs) - 1) - i 
            mask = self.mask_alignments[align_index](mask)
            
            # 上采样到浅层分辨率
            mask = F.interpolate(mask, size=shallow_diff.shape[2:], mode='trilinear', align_corners=False)
            
            # 过滤浅层噪声 (Hard 乘法 或 Residual 乘法均可，这里用直接乘法)
            cleaned_diffs[i-1] = shallow_diff * mask
            
        return cleaned_diffs

class CrossScaleFusion(nn.Module):
    """
    Q: Text
    K: Image Diff (Denoised)
    V: Image Sum
    """
    def __init__(self, img_channels_list, text_dim=768):
        super().__init__()
        self.denoiser = MultiScaleConsistencyDenoise(img_channels_list)
        
        self.fusion_layers = nn.ModuleList()
        
        for in_c in img_channels_list:
            self.fusion_layers.append(
                SingleScaleInteraction(img_dim=in_c, text_dim=text_dim)
            )

    def forward(self, feat_a_list, feat_b_list, text_embedding):
        """
        feat_a_list: list of [B, C, D, H, W]
        text_embedding: [B, Seq_Len, Text_Dim] (e.g. [1, 77, 768])
        """
        # 1. 获取去噪后的差分 (Keys)
        clean_diffs = self.denoiser(feat_a_list, feat_b_list)
        
        # 2. 计算和特征 (Values)
        sums = [fa + fb for fa, fb in zip(feat_a_list, feat_b_list)]
        
        # 3. 逐层注入特征到文本
        # 策略：从深层(x4)到浅层(x0)，让文本先学语义，再补细节
        # 或者反过来，取决于任务。通常 Deep-to-Shallow 较好。
        text_embeddings = []
        curr_text = text_embedding
        
        # 倒序遍历索引: 4, 3, 2, 1, 0
        for i in range(len(clean_diffs) - 1, -1, -1):
            k_img = clean_diffs[i]
            v_img = sums[i]
            
            # 调用单层交互模块
            curr_text = self.fusion_layers[i](curr_text, k_img, v_img)
            text_embeddings.append(curr_text)
            
        return text_embeddings

class SingleScaleInteraction(nn.Module):
    def __init__(self, img_dim, text_dim):
        super().__init__()
        # 投影层：将图像维度映射到文本维度
        self.k_proj = nn.Conv3d(img_dim, text_dim, kernel_size=1)
        self.v_proj = nn.Conv3d(img_dim, text_dim, kernel_size=1)
        
        # 交叉注意力
        self.cross_attn = nn.MultiheadAttention(embed_dim=text_dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(text_dim)
        self.ffn = nn.Sequential(
            nn.Linear(text_dim, text_dim * 4),
            nn.GELU(),
            nn.Linear(text_dim * 4, text_dim)
        )
        self.norm_ffn = nn.LayerNorm(text_dim)

    def forward(self, text, k_img, v_img):
        # text: [B, L, C_text]
        # k_img, v_img: [B, C_img, D, H, W]
        
        B = text.shape[0]
        
        # 1. 降低计算量的关键：对于浅层大尺寸特征，建议池化
        # 简单的自适应池化，或者步长卷积。这里为了演示直接用，实际中如果是x0可能需要Downsample
        if k_img.shape[-1] > 32: # 如果分辨率大于32，池化一下减小计算量
             k_img = F.adaptive_avg_pool3d(k_img, output_size=(16, 16, 16))
             v_img = F.adaptive_avg_pool3d(v_img, output_size=(16, 16, 16))
        
        # 2. 投影并展平 [B, C, D, H, W] -> [B, C_text, N_voxels] -> [B, N_voxels, C_text]
        K = self.k_proj(k_img).flatten(2).transpose(1, 2)
        V = self.v_proj(v_img).flatten(2).transpose(1, 2)
        
        # 3. Cross Attention (Q=Text, K=ImgDiff, V=ImgSum)
        # 注入：Text从图像差异中查询信息，并由图像背景填充
        # print(text.shape, K.shape, V.shape)
        attn_out, _ = self.cross_attn(query=text, key=K, value=V)
        
        # 4. 残差连接 + FFN
        text = self.norm(text + attn_out)
        text = self.norm_ffn(text + self.ffn(text))
        
        return text

# 模拟输入测试
if __name__ == "__main__":
    # 模拟5个尺度的特征图 (B=1)
    shapes = [
        (1, 48, 64, 64, 64),
        (1, 96, 32, 32, 32),
        (1, 192, 16, 16, 16),
        (1, 384, 8, 8, 8),
        (1, 768, 4, 4, 4)
    ]
    feat_a = [torch.randn(s) for s in shapes]
    feat_b = [torch.randn(s) for s in shapes]
    text_emb = torch.randn(1, 77, 768) # 假设CLIP输出
    
    model = CrossScaleFusion(img_channels_list=[48, 96, 192, 384, 768], text_dim=768)
    output_text = model(feat_a, feat_b, text_emb)
    
    print("Output Text Shape:", output_text.shape) # 应为 [1, 77, 768]