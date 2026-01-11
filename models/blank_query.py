import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from models.image_backbone import SwinTransformerBackbone


import argparse
from monai.transforms import (
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Compose,
    Spacingd,
    Resized,
    EnsureTyped,
)
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, DistributedSampler, SmartCacheDataset, load_decathlon_datalist


class MultiScaleQueryFusion(nn.Module):
    def __init__(self, 
                 num_queries=300,   # 空查询的数量 (你可以理解为"探测器"的数量)
                 embed_dim=256,     # 统一的特征维度
                 feature_channels=[48, 96, 192, 384, 768] # SwinUNETR各层的通道数
                 ):
        super().__init__()
        
        self.num_queries = num_queries
        self.embed_dim = embed_dim
        
        # 1. [空查询] 定义可学习的 Query (Null Queries)
        # 形状: (1, num_queries, embed_dim)
        self.learnable_queries = nn.Parameter(torch.randn(1, num_queries, embed_dim))
        
        # 2. [多尺度投影层] 将不同层级的特征投影到相同的 embed_dim
        # 我们需要为每一层定义一个独立的投影层
        self.projections = nn.ModuleList([
            nn.Conv3d(in_channels, embed_dim, kernel_size=1) 
            for in_channels in feature_channels
        ])
        
        # 3. [核心融合层] Cross-Attention
        # Q 来自 learnable_queries
        # K, V 来自 图像特征
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=8, 
            batch_first=True
        )
        
        # 4. [后处理] FFN (Feed Forward Network)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x_out1, x_out2):
        """
        输入:
            x_out1: List of tensors [x0, x1, x2, x3, x4] 来自图像A
            x_out2: List of tensors [x0, x1, x2, x3, x4] 来自图像B
        """
        batch_size = x_out1[0].shape[0]
        
        # --- 步骤 A: 多尺度特征扁平化与对齐 ---
        all_keys = []
        
        # 遍历每一层 (建议只取后三层 x2, x3, x4 以节省显存，x0太大了)
        # 这里为了演示，假设我们用 x2, x3, x4 (索引 2, 3, 4)
        active_levels = [2, 3, 4] 
        
        for i in active_levels:
            # 1. 取出两张图对应层的特征
            f1 = x_out1[i] # (B, C_i, D_i, H_i, W_i)
            f2 = x_out2[i] 
            
            # 2. 投影到统一维度 embed_dim
            f1 = self.projections[i](f1) # (B, 256, D_i, H_i, W_i)
            f2 = self.projections[i](f2)
            
            # 3. 加上位置编码 (Positional Embedding) - *非常重要，这里简化省略，实际建议加*
            
            # 4. 展平空间维度 (B, 256, N_pixels) -> (B, N_pixels, 256)
            f1_flat = rearrange(f1, 'b c d h w -> b (d h w) c')
            f2_flat = rearrange(f2, 'b c d h w -> b (d h w) c')
            
            # 5. 将两张图的特征拼接作为这一层的 Key/Value
            # 此时不仅仅融合了不同层，还融合了两个不同的图像源
            all_keys.append(f1_flat)
            all_keys.append(f2_flat)
            
        # 将所有层级、所有图像的特征拼成一个超级长的序列
        # shape: (B, Total_Pixels_from_all_levels, embed_dim)
        memory = torch.cat(all_keys, dim=1)
        
        # --- 步骤 B: 空查询登场 ---
        
        # 复制 query 到当前 batch 大小
        queries = repeat(self.learnable_queries, '1 n d -> b n d', b=batch_size)
        
        # --- 步骤 C: Cross-Attention 融合 ---
        # Query 也就是你的"空查询"，主动去 memory 里找它感兴趣的结节特征
        # memory 包含了来自两张图、不同尺度的所有信息
        attn_out, _ = self.cross_attn(
            query=queries, 
            key=memory, 
            value=memory
        )
        
        # 残差连接 + Norm
        queries = self.norm(queries + attn_out)
        
        # FFN
        queries = queries + self.ffn(queries)
        
        return queries  # 输出: (B, num_queries, embed_dim)

class SingleScaleFusionLayer(nn.Module):
    """
    单尺度融合层：处理特定一个尺度的特征融合
    结构：[Proj -> Flatten -> Concat] -> [CrossAttn] -> [FFN]
    """
    def __init__(self, in_channels, embed_dim=256, num_heads=8):
        super().__init__()
        
        # 1. 特征投影：将不同通道数 (48/96/...) 统一映射到 embed_dim
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=1)
        
        # 2. Cross-Attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            batch_first=True,
            dropout=0.1
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # 3. FFN (Feed Forward Network)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, query, feat_a, feat_b):
        """
        query:  (B, num_queries, embed_dim) - 上一层传下来的 Query
        feat_a: (B, in_channels, D, H, W)   - 当前尺度的图像A特征
        feat_b: (B, in_channels, D, H, W)   - 当前尺度的图像B特征
        """
        
        # --- A. 图像特征处理 ---
        # 1. 投影到统一维度: (B, C_in, D, H, W) -> (B, embed, D, H, W)
        f1 = self.proj(feat_a)
        f2 = self.proj(feat_b)
        
        # 2. 展平: (B, embed, D, H, W) -> (B, embed, N) -> (B, N, embed)
        f1_flat = rearrange(f1, 'b c d h w -> b (d h w) c')
        f2_flat = rearrange(f2, 'b c d h w -> b (d h w) c')
        
        # 3. 拼接两张图的特征作为 Key/Value
        # (B, N*2, embed)
        kv_memory = torch.cat([f1_flat, f2_flat], dim=1)
        
        # --- B. Transformer 交互 (残差连接) ---
        # Query 去 "查询" 这一层的图像特征
        attn_out, _ = self.cross_attn(query=query, key=kv_memory, value=kv_memory)
        
        # 残差连接 1
        query = self.norm1(query + attn_out)
        
        # --- C. FFN 更新 (残差连接) ---
        # 残差连接 2
        query = self.norm2(query + self.ffn(query))
        
        return query

class HierarchicalQueryFusion(nn.Module):
    def __init__(self, 
                 num_queries=300, 
                 embed_dim=256, 
                 feature_channels=[48, 96, 192, 384, 768], # 对应 x0 到 x4
                 active_layers=[4, 3, 2], # 我们可以选择只用深层，或者全部 [4,3,2,1,0]
                 batch_size: int = 1,
                 ):
        super().__init__()
        
        self.num_queries = num_queries
        self.embed_dim = embed_dim
        self.active_layers = active_layers # 保存通过的层级顺序
        self.batch_size = batch_size
        # 1. 初始的可学习 Query
        self.learnable_queries = nn.Parameter(torch.randn(1, num_queries, embed_dim))
        
        # 2. 构建级联层 (ModuleList)
        # 注意：我们需要根据 active_layers 来构建对应通道的 FusionLayer
        self.fusion_layers = nn.ModuleList()
        
        for layer_idx in active_layers:
            in_c = feature_channels[layer_idx]
            self.fusion_layers.append(
                SingleScaleFusionLayer(in_channels=in_c, embed_dim=embed_dim)
            )
            
        # 3. 最终输出归一化
        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(self, x_out1, x_out2):
        """
        x_out1/2: List of [x0, x1, x2, x3, x4]
        """
        batch_size = x_out1[0].shape[0]
        
        # 1. 初始化 Query
        # (1, 300, 256) -> (B, 300, 256)
        curr_query = repeat(self.learnable_queries, '1 n d -> b n d', b=batch_size)
        query_array = []
        # 2. 级联交互 (由深到浅)
        # active_layers 比如是 [4, 3, 2]，我们依次遍历
        for i, layer_module in enumerate(self.fusion_layers):
            layer_idx = self.active_layers[i] # 获取对应的特征层索引 (如 4)
            
            # 取出对应层的图像特征
            feat_a = x_out1[layer_idx]
            feat_b = x_out2[layer_idx]
            
            # 执行单层交互，更新 Query
            # query_new = layer(query_old, feat_a, feat_b)
            curr_query = layer_module(curr_query, feat_a, feat_b)
            curr_query = self.final_norm(curr_query)
            query_array.append(curr_query)
            
        # 3. 最终输出
        
        
        return query_array  # [layer_nums, B, num_queries, embed_dim]


def main1():
    from torch.profiler import profile, record_function, ProfilerActivity
    # 用来测试backbone加上对应的空查询的内存占用与效果运行
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()

    parser.add_argument('-o', '--model_out_channels', type=int,  help="model out channels", default=2)
    parser.add_argument('-w', '--model_weights_path', type=str,  help="model weights path", default="/home/lhr/dataset/checkpoints/swin-unetr/model_swinvit_fixed.pt")
    args = parser.parse_args()
    # 初始化
    feature_extractor = SwinTransformerBackbone(args.model_weights_path, args)
    feature_extractor.to(device)
    feature_extractor.eval()
    train_transforms = Compose([
        # 1. 修改读取方式：指定 reader="ITKReader"
        # ITKReader 可以完美读取 .mhd 和 .nii.gz，所以你可以放心地混合使用
        LoadImaged(
            keys=["image"], 
            reader="ITKReader", 
            image_only=False  # 保留元数据以便进行方向校正
        ),

        # 2. 确保通道在前 (Channel First)
        # 无论是 3D 还是 2D，MONAI 推荐格式是 (Channel, Spatial...)
        EnsureChannelFirstd(keys=["image"]),

        # 3. 统一方向为 RAS (这是让 mhd 表现得像 nii.gz 的核心)
        # 这一步会自动处理坐标轴的翻转和重排
        Orientationd(keys=["image"], axcodes="RAS"),

        # 4. (可选但推荐) 统一体素间距
        # mhd 和 nii 的 spacing 定义方式略有不同，这一步能强制统一物理尺度
        Spacingd(
            keys=["image"],
            pixdim=(1.0, 1.0, 1.0), # 根据你的需求设定，例如 (1.0, 1.0, 1.0) mm
            mode=("bilinear")
        ),
        Resized(
                keys=["image"],
                spatial_size=(128, 128, 128),
                mode=("trilinear"),
            ),
        EnsureTyped(keys=["image"], device=device, track_meta=False),
        # ... 你原来的其他变换 (ScaleIntensity, RandCrop 等) ...
    ])

    jsonfile_path = "/home/lhr/dataset/CSTPLung/data.json"
    image_path = "/home/lhr/dataset/CSTPLung/data"
    datalist = load_decathlon_datalist(jsonfile_path, True, "train")
    # print(datalist[0])
    train_ds = CacheDataset(
        data=datalist,
        transform=train_transforms,
        cache_rate=0.5,
        num_workers=4,
        
    )

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True) as prof:
    
    # 这里的 record_function 是为了让输出中明显看到这段代码的标签
        with record_function("model_inference"):        
    # 提取
            with torch.no_grad():
                input_tensor1 = torch.unsqueeze(train_ds[0]["image"], 1).to(device)
                input_tensor2 = torch.unsqueeze(train_ds[1]["image"], 1).to(device)

                x_out1 = feature_extractor(input_tensor1)
                x_out2 = feature_extractor(input_tensor2)
                
                # 提取不同层的特征
                # x0_out1, x1_out1, x2_out1, x3_out1, x4_out1 = x_out1
                # x0_out2, x1_out2, x2_out2, x3_out2, x4_out2 = x_out2
            
            blank_fusion = MultiScaleQueryFusion()
            blank_fusion.to(device)
            out = blank_fusion(x_out1, x_out2)
            print(out.shape)
    print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))

if __name__ == "__main__":
    main1()