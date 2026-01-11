import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_dim=768, out_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim * 4),
            nn.GELU(),
            nn.Linear(out_dim * 4, out_dim)
        )
        self.norm = nn.LayerNorm(out_dim)
    
    def forward(self, in_seq):
        out = self.mlp(in_seq)
        out = self.norm(out)
        
        return out
    
class catQ_T(nn.Module):
    def __init__(self, feature_dim=256):
        super().__init__()
        self.feature_dim = feature_dim
        self.alpha_generator = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, feature_dim),
            nn.Sigmoid() # 限制在 0~1 之间
        )
        


    def forward(self, text, query):
        '''
        text: (B, text_counts, feature_dim)
        query: (B, query_counts, feature_dim)
        return: (B, seq_len, feature_dim)
        '''
        
        # b = text.shape[0]
        # print(text.shape, query.shape)
        # text_global = text.mean(dim=1)
        # query_global = query.mean(dim=1)
        # 生成全局注意力权重
        # descriptors = self.alpha_generator(torch.cat((text_global, query_global), dim=1))
        # alpha = descriptors.view(b, 1, -1)

        # query_weights = alpha * query
        # text_weights = (1 - alpha) * text

        # fused_seq = torch.cat([query_weights, text_weights], dim=1)
        fused_seq = torch.cat([text, query], dim=1)
        # 查看alpha的大小，通过均值的方式
        # mean_alpha = alpha.mean(dim=-1)
        # return fused_seq, mean_alpha
        return fused_seq, 0.5

class cat_to_3D(nn.Module):
    '''
    从最开始获得的text与query开始到最后的链接
    '''
    def __init__(self,text_dim=768,query_dim=256,text_counts = 60,query_counts = 300, spatial_size = 4,layer = 0, depth = 4):
        super().__init__()
        self.seq_len = text_counts + query_counts
        self.toQueryMlp = MLP(in_dim=text_dim, out_dim=query_dim)
        self.Cat = catQ_T(feature_dim=query_dim)
        self.layer = depth - layer
        self.spatial_size = spatial_size*(2**self.layer)
        # 变成的维度需要根据对应的layer与spatial来决定
        self.spatial_projector = nn.Linear(self.seq_len, (self.spatial_size ** 3))
        self.norm_spatial = nn.LayerNorm(query_dim)
        self.act = nn.GELU()
        self.channel = 768//(2**self.layer)
        self.alignChannel = nn.Conv3d(query_dim, self.channel, kernel_size=1)
        # 需要归一化或者激活吗？



    def forward(self, text, query):
        '''
        text: (B, text_counts, feature_dim)
        query: (B, query_counts, feature_dim)
        return: (B, feature_dim, D, H, W)
        '''
        b = text.shape[0]
        text_emb = self.toQueryMlp(text) # (B, text_counts, feature_dim)
        seq_emb, alpha = self.Cat(text_emb, query) # (B, seq_len, feature_dim)

        seq_emb = seq_emb.transpose(1, 2) # (B, feature_dim, seq_len)

        spatial_emb = self.spatial_projector(seq_emb) # (B, feature_dim, D*H*W)
        spatial_flat = self.norm_spatial(spatial_emb.transpose(1, 2)).transpose(1, 2) # (B, feature_dim, D*H*W)
        res = spatial_flat.view(b, -1, self.spatial_size, self.spatial_size, self.spatial_size) # (B, feature_dim, D, H, W)
        res = self.act(res)
        res = self.alignChannel(res)

        return res, alpha # (B, channel, D, H, W)

def main():
    print("========== 开始测试多模态特征融合与重塑模块 ==========\n")
    
    # 1. 定义模拟输入参数
    B = 2               # Batch Size
    text_dim = 768      # 原始文本维度
    query_dim = 256     # 图像/对齐后的维度
    text_counts = 60    # 文本序列长度
    query_counts = 300  # Query序列长度
    
    # 2. 创建随机输入数据
    print(f"生成输入数据: Batch={B}, Text=[{text_counts}, {text_dim}], Query=[{query_counts}, {query_dim}]")
    text_input = torch.randn(B, text_counts, text_dim)
    query_input = torch.randn(B, query_counts, query_dim)
    
    # 3. 测试两种不同的 Layer 配置 (模拟 SwinUNETR 的不同层级)
    
    # --- Case A: Layer 0 (最深层/Bottleneck) ---
    # 预期: Spatial=4, Channel=768 (x4_out)
    print("\n[测试 Case 1] Layer = 0 (Bottleneck)")
    model_l0 = cat_to_3D(
        text_dim=text_dim, 
        query_dim=query_dim, 
        text_counts=text_counts, 
        query_counts=query_counts, 
        spatial_size=4, # 基础大小
        layer=4,        # 第0层 (feature_size * 16)
        depth=4
    )
    
    output_l0 = model_l0(text_input, query_input)
    print(f"期望输出形状: [{B}, 768, 4, 4, 4]")
    print(f"实际输出形状: {output_l0.shape}")
    
    assert output_l0.shape == (B, 768, 4, 4, 4), "Layer 0 输出维度不匹配！"
    print(">>> Layer 0 测试通过 √")
    
    # --- Case B: Layer 2 (中间层) ---
    # 预期: Spatial = 4 * (2^2) = 16, Channel = 768 / (2^2) = 192 (x2_out)
    print("\n[测试 Case 2] Layer = 2 (Middle Layer)")
    model_l2 = cat_to_3D(
        text_dim=text_dim, 
        query_dim=query_dim, 
        text_counts=text_counts, 
        query_counts=query_counts, 
        spatial_size=4,
        layer=2,
        depth=4
    )
    
    output_l2 = model_l2(text_input, query_input)
    print(f"期望输出形状: [{B}, 192, 16, 16, 16]")
    print(f"实际输出形状: {output_l2.shape}")
    
    assert output_l2.shape == (B, 192, 16, 16, 16), "Layer 2 输出维度不匹配！"
    print(">>> Layer 2 测试通过 √")

    # 4. 检查是否有梯度（简单的反向传播检查）
    print("\n[测试 Case 3] 反向传播检查")
    loss = output_l2.sum()
    loss.backward()
    print("反向传播成功，梯度正常生成。")
    print(">>> 梯度检查通过 √")
    
    print("\n========== 所有测试完成，模块运行正常 ==========")

if __name__ == "__main__":
    main()