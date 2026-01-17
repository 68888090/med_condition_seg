import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PriorGuidedGatedAttentionLayer(nn.Module):
    """
    【深层专用：先验引导的门控注意力】
    (Prior-Guided Gated Attention)
    
    机制：
    不再是简单的 Query-Key 匹配，而是让 F2 (当前图像) 与 F_diff (差异特征) 
    在文本意图的指挥下进行"竞争"。
    
    公式：
    alpha = Sigmoid( (E_app - E_diff) * scale + ln(k) )
    Output = alpha * F2 + (1 - alpha) * F_diff
    """
    def __init__(self, img_dim, text_dim, initial_k=1.0):
        super().__init__()
        
        # 1. 差分特征提取 (保持原来的卷积提取能力)
        self.diff_conv = nn.Sequential(
            nn.Conv3d(img_dim, img_dim, kernel_size=3, padding=1),
            nn.InstanceNorm3d(img_dim),
            nn.ReLU(inplace=True)
        )
        
        # 2. 能量投影层 (Energy Projections)
        # 用于计算 Q * K
        self.proj_f2 = nn.Conv3d(img_dim, img_dim, kernel_size=1)   # K_app
        self.proj_diff = nn.Conv3d(img_dim, img_dim, kernel_size=1) # K_diff
        self.proj_text = nn.Linear(text_dim, img_dim)               # Q_text
        
        # 3. 可学习的参数
        # (A) 温度系数 scale: 初始化为 1/sqrt(2*dim) 以适应差分方差
        theoretical_scale = (2 * img_dim) ** -0.5
        self.log_scale = nn.Parameter(torch.tensor(math.log(theoretical_scale)))
        
        # (B) 宏观偏向 k (Macro Bias): 控制全局倾向于 F2 还是 F_diff
        # 初始为 0 (即 k=1, 无偏向)，让网络自己学
        self.k_bias = nn.Parameter(torch.tensor(math.log(initial_k)))
        
        # 4. 融合后的平滑处理
        self.fusion_norm = nn.InstanceNorm3d(img_dim)
        self.fusion_act = nn.ReLU(inplace=True)

    def forward(self, f1, f2, text_seq):
        """
        f1, f2: [B, C, D, H, W]
        text_seq: [B, Seq_Len, Text_Dim]
        """
        B, C, D, H, W = f1.shape
        
        # --- A. 准备特征 ---
        # 1. 提取差异特征 F_diff
        raw_diff = f2 - f1
        f_diff = self.diff_conv(raw_diff) # [B, C, D, H, W]
        
        # 2. 准备文本 Query
        # 我们需要一个明确的"意图向量"来做二选一决策
        # 这里在深层内部做 Pooling，提取整句的意图 (如 "Show me the new nodule")
        text_emb = text_seq.mean(dim=1)   # [B, Text_Dim]
        q_text = self.proj_text(text_emb).view(B, C, 1, 1, 1) # [B, C, 1, 1, 1] Broadcastable
        
        # --- B. 计算能量场 (Energy Fields) ---
        # 3. 动态调整 Scale (确保为正)
        scale = self.log_scale.exp()
        
        # 4. 计算 F2 的外观势能 (Appearance Energy)
        # E = (Feature * W) dot Q
        k_f2 = self.proj_f2(f2)
        e_app = torch.sum(k_f2 * q_text, dim=1, keepdim=True) * scale # [B, 1, D, H, W]
        
        # 5. 计算 F_diff 的差异势能 (Difference Energy)
        k_diff = self.proj_diff(f_diff)
        e_diff = torch.sum(k_diff * q_text, dim=1, keepdim=True) * scale # [B, 1, D, H, W]
        
        # --- C. 竞争与门控 (Competition & Gating) ---
        # 6. 计算 Logits = (E_app - E_diff) + bias
        # 物理含义：若 E_diff 很大 (差异符合文本)，则 delta_e 为负，alpha 变小，(1-alpha) 变大 -> 选 F_diff
        delta_e = e_app - e_diff
        logits = delta_e + self.k_bias
        
        # 7. 生成互补门控
        alpha = torch.sigmoid(logits) # F2 的权重
        beta = 1 - alpha              # F_diff 的权重
        
        # --- D. 动态融合 ---
        f_fused = alpha * f2 + beta * f_diff
        
        return self.fusion_act(self.fusion_norm(f_fused))

# -------------------------------------------------------------------
# 下面是保持不变的浅层模块和整合后的混合模块
# -------------------------------------------------------------------

class SimpleGateLayer(nn.Module):
    """
    【浅层专用：全局门控】(保持不变)
    """
    def __init__(self, channels, text_dim, dropout_rate=0.1):
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
        gate = self.text_gate(text_global).view(diff.shape[0], diff.shape[1], 1, 1, 1)
        weighted_diff = self.diff_conv(diff * gate)
        return self.fusion_conv(torch.cat([f2, weighted_diff], dim=1))


class HybridFusionModule(nn.Module):
    """
    【混合融合策略 - 升级版】
    Deep Layers: 使用 PriorGuidedGatedAttentionLayer (我们讨论的竞争机制)
    Shallow Layers: 使用 SimpleGateLayer (简单的通道注意力)
    """
    def __init__(self, 
                 feature_channels=[48, 96, 192, 384, 768], 
                 text_dim=768,
                 active_layers=[4, 3, 2, 1, 0],
                 dropout_rate=0.2):
        super().__init__()
        
        self.active_layers = active_layers
        self.fusion_layers = nn.ModuleList()
        
        # 定义深层索引
        self.deep_layer_indices = [3, 4] 
        
        for i in range(len(feature_channels)):
            if i in active_layers:
                if i in self.deep_layer_indices:
                    # 【核心修改】：替换为我们讨论的"竞争性注意力"模块
                    self.fusion_layers.append(
                        PriorGuidedGatedAttentionLayer(
                            img_dim=feature_channels[i], 
                            text_dim=text_dim,
                            initial_k=1.0 # 初始设为公平竞争
                        )
                    )
                else:
                    # 浅层保持简单
                    self.fusion_layers.append(
                        SimpleGateLayer(feature_channels[i], text_dim, dropout_rate)
                    )
            else:
                self.fusion_layers.append(nn.Identity())

    def forward(self, x_out1, x_out2, text_embedding):
        """
        x_out1, x_out2: 特征列表
        text_embedding: [B, Seq_Len, 768]
        """
        # 准备全局文本向量 (给浅层用)
        text_global = text_embedding.mean(dim=1)
        
        fused_outputs = []
        
        for i, (f1, f2) in enumerate(zip(x_out1, x_out2)):
            if i in self.active_layers:
                layer_module = self.fusion_layers[i]
                
                if i in self.deep_layer_indices:
                    # 深层：传入序列文本 (模块内部会提取 Query 意图)
                    fused = layer_module(f1, f2, text_embedding)
                else:
                    # 浅层：传入全局文本
                    fused = layer_module(f1, f2, text_global)
                    
                fused_outputs.append(fused)
            else:
                fused_outputs.append(f2)
                
        return fused_outputs