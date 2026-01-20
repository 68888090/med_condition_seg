import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# ==========================================
# 1. 文本分层编码器 (核心新增)
# ==========================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class HierarchicalTextEncoder(nn.Module):
    def __init__(self, input_dim=768, text_dims=[128, 256, 384, 512, 768]):
        """
        Args:
            input_dim: 输入文本特征维度 (如 BERT=768)
            text_dims: 每一层想要保留的文本维度列表 (非对称设计的核心)
                       例如: [128, 256, 384, 512, 768]
        """
        super().__init__()
        self.layers = nn.ModuleList()
        
        # 1. 初始投影 (Projection)
        # 负责将 BERT 768 维 -> Layer 0 的维度 (如 128)
        # 这一步比直接压到 48 要温和得多，保留了更多词法细节
        self.input_proj = nn.Sequential(
            nn.Conv1d(input_dim, text_dims[0], kernel_size=1),
            nn.ReLU(inplace=True)
        )
        
        current_dim = text_dims[0]
        
        # 2. 构建下采样路径
        # 生成 Layer 1 到 Layer 4
        for i in range(len(text_dims) - 1):
            next_dim = text_dims[i+1]
            self.layers.append(
                nn.Sequential(
                    # 语义提取 + 通道调整
                    nn.Conv1d(current_dim, next_dim, kernel_size=3, padding=1),
                    nn.InstanceNorm1d(next_dim),
                    nn.ReLU(inplace=True),
                    # 时序下采样 (Sequence Length / 2)
                    nn.MaxPool1d(kernel_size=2)
                )
            )
            current_dim = next_dim

    def forward(self, x):
        # x: [B, Seq_Len, Dim] -> [B, Dim, Seq_Len]
        x = x.transpose(1, 2)
        
        # Layer 0 特征
        curr = self.input_proj(x)
        features = [curr] 
        
        # 逐层处理
        for layer in self.layers:
            curr = layer(curr)
            features.append(curr)
            
        # features 列表内容示例 (假设 text_dims=[128, 256...]):
        # [0]: [B, 128, L]
        # [1]: [B, 256, L/2]
        # ...
        
        # 转回 [B, Seq, Dim] 以适配 Attention 接口
        return [f.transpose(1, 2) for f in features]
# ==========================================
# 2. 你的 Fusion Layers (微调构造函数)
# ==========================================

class SimpleGateLayer(nn.Module):
    def __init__(self, channels, current_text_dim, dropout_rate):
        super().__init__()
        # 【关键点】这里实现了"非对称"的对接
        # 输入是 current_text_dim (如 128)，输出映射到 channels (如 48)
        self.text_gate = nn.Sequential(
            nn.Linear(current_text_dim, channels), 
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

    def forward(self, f1, f2, text_seq):
        # text_seq: [B, Seq, 128]
        # text_global: [B, 128]
        text_global = text_seq.mean(dim=1) 
        
        diff = f2 - f1
        # self.text_gate 接收 128，输出 48，完成对齐
        gate = self.text_gate(text_global).view(diff.shape[0], diff.shape[1], 1, 1, 1)
        weighted_diff = self.diff_conv(diff * gate)
        return self.fusion_conv(torch.cat([f2, weighted_diff], dim=1))


class DensePixelTextLayer(nn.Module):
    def __init__(self, img_dim, current_text_dim, dropout_rate=0.1):
        super().__init__()
        
        self.img_proj = nn.Conv3d(img_dim, img_dim, kernel_size=1)
        self.text_proj = nn.Linear(current_text_dim, img_dim)
        
        self.diff_conv = nn.Sequential(
            nn.Conv3d(img_dim, img_dim, kernel_size=3, padding=1),
            nn.InstanceNorm3d(img_dim),
            nn.ReLU(inplace=True)
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=img_dim, 
            num_heads=4, 
            batch_first=True, 
            dropout=dropout_rate
        )
        self.norm = nn.LayerNorm(img_dim)
        
        self.fusion_conv = nn.Sequential(
            nn.Conv3d(img_dim * 2, img_dim, kernel_size=1),
            nn.InstanceNorm3d(img_dim),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout_rate)
        )

    def forward(self, f1, f2, text_seq):
        """
        f1, f2: [B, C, D, H, W]
        text_seq: [B, Seq_Len, C_text]
        """
        B, C, D, H, W = f1.shape
        raw_diff = f2 - f1
        diff_feat = self.diff_conv(raw_diff)
        
        # 1. 准备 Query (Image)
        q = rearrange(self.img_proj(diff_feat), 'b c d h w -> b (d h w) c')
        
        # 2. 准备 Key/Value (Text)
        k = v = self.text_proj(text_seq)
        
        # --- 【关键修正】强制 FP32 计算 Attention ---
        # 防止 FP16 下 exp() 溢出导致 NaN
        with torch.cuda.amp.autocast(enabled=False):
            # 必须手动转为 float32
            q_fp32 = q.float()
            k_fp32 = k.float()
            v_fp32 = v.float()
            
            attn_out, _ = self.cross_attn(query=q_fp32, key=k_fp32, value=v_fp32)
            
            # LayerNorm 也建议在 FP32 下做，或者转回来
            # 这里为了安全，先加残差再 Norm，全部在 FP32
            q_out = self.norm(q_fp32 + attn_out)
            
        # 3. 转回原来的 dtype (可能是 FP16) 以兼容后续网络
        # 这里的 q_out 是 FP32，但外面期望的是 AMP 上下文的类型
        q = q_out.to(diff_feat.dtype) 
        
        # 4. 恢复空间维度
        attn_feat = rearrange(q, 'b (d h w) c -> b c d h w', d=D, h=H, w=W)
        
        return self.fusion_conv(torch.cat([f2, attn_feat], dim=1))
# ==========================================
# 3. 集成模块 (Integrated Module)
# ==========================================

class HybridFusionModule(nn.Module):
    def __init__(self, 
                 # 1. 图像塔配置 (比较窄，为了省显存)
                 feature_channels=[48, 96, 192, 384, 768], 
                 
                 # 2. 文本塔配置 (非对称设计：保持较宽，为了留住信息)
                 # 注意：列表长度必须与 feature_channels 一致
                 text_hierarchy_dims=[128, 256, 384, 512, 768], 
                 
                 text_input_dim=768,
                 active_layers=[4, 3, 2, 1, 0],
                 dropout_rate=0.2):
        super().__init__()
        
        assert len(feature_channels) == len(text_hierarchy_dims), \
            "图像通道列表和文本通道列表的长度必须一致！"
            
        self.active_layers = active_layers
        self.deep_layer_indices = [3, 4]
        
        # 初始化文本编码器 (使用宽的配置)
        self.text_encoder = HierarchicalTextEncoder(
            input_dim=text_input_dim,
            text_dims=text_hierarchy_dims
        )
        
        self.fusion_layers = nn.ModuleList()
        
        # 遍历每一层，分别取出 img_ch 和 text_ch 进行配对
        for i in range(len(feature_channels)):
            if i in active_layers:
                img_ch = feature_channels[i]       # e.g., Layer 0 -> 48
                text_ch = text_hierarchy_dims[i]   # e.g., Layer 0 -> 128 (更宽!)
                
                if i in self.deep_layer_indices:
                    self.fusion_layers.append(
                        DensePixelTextLayer(img_ch, text_ch, dropout_rate)
                    )
                else:
                    self.fusion_layers.append(
                        SimpleGateLayer(img_ch, text_ch, dropout_rate)
                    )
            else:
                self.fusion_layers.append(nn.Identity())

    def forward(self, x_out1, x_out2, text_embedding):
        # 1. 获取"宽"的文本特征列表
        text_feats = self.text_encoder(text_embedding)
        
        fused_outputs = []
        for i, (f1, f2) in enumerate(zip(x_out1, x_out2)):
            if i in self.active_layers:
                # 传入对应层级的宽文本特征，Fusion Layer 内部会自动处理维度对齐
                fused = self.fusion_layers[i](f1, f2, text_feats[i])
                fused_outputs.append(fused)
            else:
                fused_outputs.append(f2)
                
        return fused_outputs