import torch
import torch.nn as nn
import numpy as np
from collections.abc import Sequence
# ==========================================
# Part 1: 模拟依赖库 (Mock Dependencies)
# 这一部分是为了让代码在没有外部文件的情况下也能跑通
# ==========================================

from monai.networks.blocks import UnetrUpBlock, UnetOutBlock, UnetrBasicBlock
from monai.networks.blocks.dynunet_block import UnetBasicBlock, UnetResBlock, get_conv_layer

class UnetrUpDecoder(nn.Module):
    """
    An upsampling module that can be used for UNETR: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int] | int,
        upsample_kernel_size: Sequence[int] | int,
        norm_name: tuple | str,
        res_block: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
            res_block: bool argument to determine if residual block is used.

        """

        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )

        if res_block:
            self.conv_block = UnetResBlock(
                spatial_dims,
                out_channels ,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )
        else:
            self.conv_block = UnetBasicBlock(  # type: ignore
                spatial_dims,
                out_channels ,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )

    def forward(self, inp):
        # number of channels for skip should equals to out_channels
        out = self.transp_conv(inp)
        out = self.conv_block(out)
        return out

class DummySwinBackbone(nn.Module):
    """模拟 SwinTransformerBackbone 输出五层特征"""
    def __init__(self, weights_path, out_channels):
        super().__init__()
    def forward(self, x):
        # 假设输入是 [B, 1, 128, 128, 128]
        # x0: [B, 48, 64, 64, 64]
        # x1: [B, 96, 32, 32, 32]
        # x2: [B, 192, 16, 16, 16]
        # x3: [B, 384, 8, 8, 8]
        # x4: [B, 768, 4, 4, 4]
        B = x.shape[0]
        return [
            torch.randn(B, 48, 64, 64, 64).to(x.device),
            torch.randn(B, 96, 32, 32, 32).to(x.device),
            torch.randn(B, 192, 16, 16, 16).to(x.device),
            torch.randn(B, 384, 8, 8, 8).to(x.device),
            torch.randn(B, 768, 4, 4, 4).to(x.device)
        ]

class DummyLanguageProcessor(nn.Module):
    """模拟文本编码器"""
    def __init__(self, bert_type, num_texts):
        super().__init__()
        self.proj = nn.Linear(10, 768) # 简单模拟
    def forward(self, text):
        # 假设返回 [B, 77, 768]
        return torch.randn(1, 60, 768).cuda() if torch.cuda.is_available() else torch.randn(1, 60, 768)

class DummyHierarchicalQueryFusion(nn.Module):
    """模拟层级查询融合，返回每一层的Query状态列表"""
    def __init__(self, num_queries, query_dim, channels, layers):
        super().__init__()
    def forward(self, x1_list, x2_list):
        B = x1_list[0].shape[0]
        # 返回5个层级的 query list，对应 [x4, x3, x2, x1, x0] 或者反过来
        # 你的代码使用了 queries_fusions[0]...[4]，所以这里返回5个
        return [torch.randn(B, 300, 256).to(x1_list[0].device) for _ in range(5)]

class DummyCrossScaleFusion(nn.Module):
    """模拟跨尺度融合，返回每一层的文本融合特征"""
    def __init__(self, channels, text_dim):
        super().__init__()
    def forward(self, x1, x2, text):
        B = x1[0].shape[0]
        # 返回5个层级的文本特征列表 [B, 77, 256]
        return [torch.randn(B, 60, 256).to(x1[0].device) for _ in range(5)]

class cat_to_3D(nn.Module):
    """
    为了测试，这里使用一个简化的 cat_to_3D
    模拟将 (Query + Text) -> 3D Feature Map
    """
    def __init__(self, text_dim, query_dim, text_counts, query_counts, spatial_size, layer, depth):
        super().__init__()
        self.spatial_size = spatial_size
        # 计算对应的输出通道数：layer 4 (deepest) -> 768, layer 0 -> 48
        # 注意：这里需要根据 feature_channels 列表反推
        channels_map = {4: 768, 3: 384, 2: 192, 1: 96, 0: 48} 
        self.out_c = channels_map[layer]
        
    def forward(self, text, query):
        B = query.shape[0]
        # 直接生成对应尺寸的特征图用于测试流程
        return torch.randn(B, self.out_c, self.spatial_size, self.spatial_size, self.spatial_size).to(query.device)


# ==========================================
# Part 2: 修正后的 UnetM 模型
# ==========================================

class UnetM(nn.Module):
    def __init__(self, 
        in_channels: int = 1, # [修正] 添加输入通道数
        num_queries: int = 300,
        query_dim: int = 256,
        feature_channels : list = [48,96,192,384,768],
        text_dim: int = 768,
        num_texts: int = 60,
        spatial_size:int = 4,
        out_channels: int = 2, 
        backbone_weights_path: str = '',
        bert_type: str = 'bert-base-uncased',
        norm_name: str = 'instance', # [修正] 添加 norm_name
        patch_size: int = 2, # [修正] 添加 patch_size
        device: str = 'cuda'
    ):
        super().__init__()
        self.num_queries = num_queries
        self.query_dim = query_dim
        self.feature_channels = feature_channels
        self.text_dim = text_dim
        self.num_texts = num_texts
        self.spatial_size = spatial_size
        self.out_channels = out_channels
        self.patch_size = patch_size # 用于检查尺寸
        
        # 使用模拟类 (实际使用时请替换回真实的类)
        self.bert = DummyLanguageProcessor(bert_type, self.num_texts)
        self.feature_extractor = DummySwinBackbone(backbone_weights_path, in_channels) # 传入输入通道
        self.blank_query_fusion = DummyHierarchicalQueryFusion(self.num_queries, self.query_dim, self.feature_channels, [4,3,2,1,0])
        self.cross_scale_fusion = DummyCrossScaleFusion(self.feature_channels, self.text_dim)
        
        # Decoder 块定义
        # Decoder 5: Input (768) -> Output (384)
        self.decoder5 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=self.feature_channels[4], # 768
            out_channels=self.feature_channels[3], # 384
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        # Decoder 4: Input (384) -> Output (192)
        self.decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=self.feature_channels[3],
            out_channels=self.feature_channels[2],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        
        # Decoder 3: Input (192) -> Output (96)
        self.decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=self.feature_channels[2],
            out_channels=self.feature_channels[1],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        
        # Decoder 2: Input (96) -> Output (48)
        self.decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=self.feature_channels[1],
            out_channels=self.feature_channels[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        
        # Decoder 1: Input (48) -> Output (48) (恢复到原图尺寸 128)
        self.decoder1 = UnetrUpDecoder(
            spatial_dims=3,
            in_channels=self.feature_channels[0],
            out_channels=self.feature_channels[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        
        # Skip Connections (Sequence -> Volume)
        # 注意 layer 参数对应 feature_channels 的索引
        self.skip5 = cat_to_3D(self.text_dim, self.query_dim, self.num_texts, self.num_queries, 4, layer=4, depth=4)
        self.skip4 = cat_to_3D(self.text_dim, self.query_dim, self.num_texts, self.num_queries, 8, layer=3, depth=4)
        self.skip3 = cat_to_3D(self.text_dim, self.query_dim, self.num_texts, self.num_queries, 16, layer=2, depth=4)
        self.skip2 = cat_to_3D(self.text_dim, self.query_dim, self.num_texts, self.num_queries, 32, layer=1, depth=4)
        self.skip1 = cat_to_3D(self.text_dim, self.query_dim, self.num_texts, self.num_queries, 64, layer=0, depth=4)

        # Bottleneck
        self.encoder10 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=self.feature_channels[4],
            out_channels=self.feature_channels[4],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.out = UnetOutBlock(spatial_dims=3, in_channels=self.feature_channels[0], out_channels=self.out_channels) 
        self.softmax = nn.Softmax(dim=1) 
        

    @torch.jit.unused
    def _check_input_size(self, spatial_shape):
        img_size = np.array(spatial_shape)
        remainder = (img_size % np.power(self.patch_size, 5)) > 0
        if remainder.any():
            wrong_dims = (np.where(remainder)[0] + 2).tolist()
            # 简单print warning，不中断测试
            print(f"Warning: spatial dimensions {wrong_dims} might not be divisible by patch_size**5.")

    def forward(self, img1, img2, text):
        if not torch.jit.is_scripting() and not torch.jit.is_tracing():
            self._check_input_size(img1.shape[2:])
            self._check_input_size(img2.shape[2:])
        combined = torch.cat([img1, img2], dim=0)
        with torch.no_grad():
            combined_out = self.feature_extractor(combined)
            text_out = self.bert(text) # [B, L, 768]
        # 1. 提取特征
        img1_out = []
        img2_out = []
        for feat in combined_out:
            # feat shape: [2*B, C_i, D_i, H_i, W_i]
            # torch.chunk 将张量切分成 2 份，在 dim=0 上切
            f1, f2 = torch.chunk(feat, chunks=2, dim=0)
            img1_out.append(f1)
            img2_out.append(f2)
        # text_out = self.bert(text) # [B, L, 768]
        
        # 2. 融合
        text_fusions = self.cross_scale_fusion(img1_out, img2_out, text_out) # list of 5
        queries_fusions = self.blank_query_fusion(img1_out, img2_out) # list of 5
        print(f"text_fusions,shape: {[t.shape for t in text_fusions]}")
        print(f"queries_fusions,shape: {[q.shape for q in queries_fusions]}")
        # 3. 逐层解码 (U-Net Logic)
        
        # Layer 4 (Bottleneck): x4 (4x4x4)
        # 将序列特征转为 3D 特征 [B, 768, 4, 4, 4]
        cat4 = self.skip5(text_fusions[0], queries_fusions[0]) 
        print(f"cat4,shape: {cat4.shape}")
        dec4 = self.encoder10(cat4) # Bottleneck 处理
        print(f"dec4,shape: {dec4.shape}")
        # Layer 3: x3 (8x8x8)
        cat3 = self.skip4(text_fusions[1], queries_fusions[1])
        print(f"cat3,shape: {cat3.shape}")
        # UnetrUpBlock(input, skip) -> 这里的 input 是上一层解码输出，skip 是当前层特征
        dec3 = self.decoder5(dec4, cat3) 
        print(f"dec3,shape: {dec3.shape}")
        
        # Layer 2: x2 (16x16x16)
        cat2 = self.skip3(text_fusions[2], queries_fusions[2])
        print(f"cat2,shape: {cat2.shape}")
        dec2 = self.decoder4(dec3, cat2)
        print(f"dec2,shape: {dec2.shape}")
        
        # Layer 1: x1 (32x32x32)
        cat1 = self.skip2(text_fusions[3], queries_fusions[3])
        print(f"cat1,shape: {cat1.shape}")
        dec1 = self.decoder3(dec2, cat1)
        print(f"dec1,shape: {dec1.shape}")
        
        # Layer 0: x0 (64x64x64)
        cat0 = self.skip1(text_fusions[4], queries_fusions[4])
        print(f"cat0,shape: {cat0.shape}")
        dec0 = self.decoder2(dec1, cat0)
        print(f"dec0,shape: {dec0.shape}")
        dec0 = self.decoder1(dec0)
        
        # Final Upsample to (128x128x128)
        # 这一步通常在 SwinUNETR 中用于从 64 恢复到 128
        # 如果不需要 skip connection，第二个参数可以传 None 或者调整 Block 定义
        # 这里的 decoder1 定义为 UnetrUpBlock，通常需要 skip。
        # 如果没有 x_minus_1 特征，我们可以只做上采样卷积。
        # 为了跑通，这里我们假设 decoder1 对 dec0 自上采样
        # final_feat = self.decoder1(dec0, dec0) # 这里有些 trick，实际可能直接用反卷积
        
        # Output Head
        logits = self.out(dec0)
        probs = self.softmax(logits)

        return probs

# ==========================================
# Part 3: 测试 Main 函数
# ==========================================

def main():
    # 检查 CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 初始化模型
    print("Initializing UnetM Model...")
    model = UnetM(
        in_channels=1, 
        feature_channels=[48, 96, 192, 384, 768],
        spatial_size=4,
        out_channels=2,
        device=str(device)
    ).to(device)
    
    # 2. 生成随机输入数据
    # 注意: SwinUNETR 期望 5D 张量 [Batch, Channel, D, H, W]
    print("Generating random input data...")
    img1 = torch.randn(2, 1, 64, 64, 64).to(device)
    img2 = torch.randn(2, 1, 64, 64, 64).to(device)
    
    # 文本输入通常是 token ids，这里模拟简单的 list 或 tensor
    # 实际上由于我们在 DummyLanguageProcessor 里处理了，传什么都行
    text_input = ["This is a random medical text description"] 
    
    # 3. 前向传播测试
    print("Starting Forward Pass...")
    try:
        output = model(img1, img2, text_input)
        
        print("\n=== Test Successful ===")
        print(f"Input Image Shape: {img1.shape}")
        print(f"Output Shape: {output.shape}")
        
        # 验证输出维度
        expected_shape = (2, 2, 128, 128, 128)
        if output.shape == expected_shape:
            print("Status: Output shape matches expected dimensions!")
        else:
            print(f"Status: Shape mismatch. Expected {expected_shape}, got {output.shape}")
            
    except Exception as e:
        print("\n=== Test Failed ===")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()