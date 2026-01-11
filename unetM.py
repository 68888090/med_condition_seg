import torch
import torch.nn as nn
from collections.abc import Sequence
import numpy as np
from models.blank_query import HierarchicalQueryFusion
from models.fuse_to_skip import cat_to_3D
from models.image_backbone import SwinTransformerBackbone
from models.text_fusion import CrossScaleFusion
from models.text_processor import LanguageProcessor

from monai.networks.blocks import UnetrUpBlock, UnetOutBlock, UnetrBasicBlock
from monai.networks.blocks.dynunet_block import UnetBasicBlock, UnetResBlock, get_conv_layer

import torch
import torch.nn as nn
from torch.cuda.amp import autocast

class NoduleDetectionHead(nn.Module):
    def __init__(self, in_channels=48, hidden_channels=64):
        super().__init__()
        
        # 1. 共享的特征提取层 (可选，用于进一步整合特征)
        self.shared_conv = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(hidden_channels),
            nn.ReLU(inplace=True)
        )
        
        # 2. 分类头 (Heatmap): 预测是否存在结节中心
        # 输出 (B, 1, D, H, W)
        self.heatmap_head = nn.Conv3d(hidden_channels, 1, kernel_size=1, bias=True)
        # 注意：这里先不加 Sigmoid，通常放在 Loss 计算或推理时加，为了数值稳定性
        
        # 3. 偏移头 (Offset): 预测 (delta_z, delta_y, delta_x)
        # 输出 (B, 3, D, H, W)
        self.offset_head = nn.Conv3d(hidden_channels, 3, kernel_size=1, bias=True)
        
        # 4. 尺寸头 (Diameter): 预测直径 d
        # 输出 (B, 1, D, H, W)
        self.size_head = nn.Conv3d(hidden_channels, 1, kernel_size=1, bias=True)
    @autocast(enabled=False)
    def forward(self, x):
        x = x.float()
        # 0. 检查输入是否正常
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("Error: Input x contains NaN or Inf!")

        feature = self.shared_conv(x)
        
        # 1. 检查共享特征层
        if torch.isnan(feature).any():
            print("Error: NaN generated in shared_conv")

        # --- Head 1: Probability ---
        heatmap = self.heatmap_head(feature)
        
        # --- Head 2: Center Offset ---
        offset_logits = self.offset_head(feature)
        offset = torch.sigmoid(offset_logits)
        
        # --- Head 3: Diameter ---
        size_logits = self.size_head(feature)
        # 2. 这里的 softplus 最容易导致数值问题
        size_logits = torch.clamp(size_logits, min=-20.0, max=20.0)
        diameter = torch.nn.functional.softplus(size_logits)

        # 3. 检查输出
        if torch.isnan(diameter).any() or torch.isinf(diameter).any():
            print("Error: NaN/Inf generated in diameter head")
            # 打印一下导致溢出的原始值
            print("Max value in size_logits:", size_logits.max())

        return heatmap, offset, diameter

# ==========================================
# 在你的主模型中使用方法：
# ==========================================

# 1. 在 __init__ 中定义 Head
# self.detection_head = NoduleDetectionHead(in_channels=48)

# 2. 在 forward 函数的最后修改逻辑
# 你的原始代码:
# dec0 = self.decoder1(dec0) (此时 shape: B, 48, D, H, W)

# --- 修改为 ---
# heatmap, offset, diameter = self.detection_head(dec0)

# return heatmap, offset, diameter

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

class UnetM(nn.Module):
    def __init__(self, 
        text_processor: LanguageProcessor = None,  
        num_queries: int = 300,
        query_dim: int = 256,
        feature_channels : list[int] = [48,96,192,384,768],
        text_dim: int = 768,
        num_texts: int = 60,
        spatial_size:int = 4,
        out_channels: int = 2, # 设置为2通道判断是前景还是后景，后续可以配合softmax函数与交叉熵损失与dice损失函数
        backbone_weights_path: str = '/home/lhr/dataset/checkpoints/swin-unetr/model_swinvit_fixed.pt',
        # bert_type: str = 'bert-base-uncased',
        device: str = 'cuda',
        norm_name: str = 'instance',
        patch_size: int = 2,
        batch_size: int = 1,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.query_dim = query_dim
        self.feature_channels = feature_channels
        self.text_dim = text_dim
        self.num_texts = num_texts
        self.spatial_size = spatial_size
        self.out_channels = out_channels
        self.backbone_weights_path = backbone_weights_path
        self.patch_size = patch_size
        self.batch_size = batch_size
        # self.bert_type = bert_type
        self.text_processor = LanguageProcessor()
        # self.bert = LanguageProcessor(bert_type, self.num_texts)
        self.feature_extractor = SwinTransformerBackbone(self.backbone_weights_path, self.out_channels)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        self.blank_query_fusion = HierarchicalQueryFusion(self.num_queries, self.query_dim, self.feature_channels,[4,3,2,1,0],self.batch_size) # 目前我们就选择五层，全部选择
        self.cross_scale_fusion = CrossScaleFusion(self.feature_channels, self.text_dim)
        
        self.decoder5 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=self.feature_channels[4],
            out_channels=self.feature_channels[3],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=self.feature_channels[3],
            out_channels=self.feature_channels[2],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        
        self.decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=self.feature_channels[2],
            out_channels=self.feature_channels[1],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        
        self.decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=self.feature_channels[1],
            out_channels=self.feature_channels[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        
        self.decoder1 = UnetrUpDecoder(
            spatial_dims=3,
            in_channels=self.feature_channels[0],
            out_channels=self.feature_channels[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        
        self.skip5 = cat_to_3D(
            text_dim = self.text_dim,
            query_dim = self.query_dim,
            text_counts=self.num_texts, 
            query_counts=self.num_queries, 
            spatial_size=2, # 基础大小
            layer=4,        # 第0层 (feature_size * 16)
            depth=len(self.feature_channels)-1        
        )

        self.encoder10 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=self.feature_channels[4],
            out_channels=self.feature_channels[4],
            kernel_size=3,
            stride = 1,
            norm_name=norm_name,
            res_block=True,
        )

        self.skip4 = cat_to_3D(
            text_dim = self.text_dim,
            query_dim = self.query_dim,
            text_counts=self.num_texts, 
            query_counts=self.num_queries, 
            spatial_size=2, # 基础大小
            layer=3,        # 第0层 (feature_size * 8)
            depth=len(self.feature_channels)-1        
        )

        self.skip3 = cat_to_3D(
            text_dim = self.text_dim,
            query_dim = self.query_dim,
            text_counts=self.num_texts, 
            query_counts=self.num_queries, 
            spatial_size=2, # 基础大小
            layer=2,        # 第0层 (feature_size * 4)
            depth=len(self.feature_channels)-1        
        )

        self.skip2 = cat_to_3D(
            text_dim = self.text_dim,
            query_dim = self.query_dim,
            text_counts=self.num_texts, 
            query_counts=self.num_queries, 
            spatial_size=2, #    基础大小
            layer=1,        # 第0层 (feature_size * 2)
            depth=len(self.feature_channels)-1        
        )

        self.skip1 = cat_to_3D(
            text_dim = self.text_dim,
            query_dim = self.query_dim,
            text_counts=self.num_texts, 
            query_counts=self.num_queries, 
            spatial_size=2, # 基础大小
            layer=0,        # 第0层 (feature_size)
            depth=len(self.feature_channels)-1        
        )
        
        # self.out = UnetOutBlock(spatial_dims=3,in_channels=self.feature_channels[0],out_channels=self.out_channels) #是一个卷积层，还需要一个softmax函数
        # self.softmax = nn.Softmax(dim=1) # 对通道维度进行softmax
        self.dropout = nn.Dropout3d(p=dropout_rate)
        self.nodule_detection_head = NoduleDetectionHead(self.feature_channels[0])

        

    @torch.jit.unused
    def _check_input_size(self, spatial_shape):
        img_size = np.array(spatial_shape)
        remainder = (img_size % np.power(self.patch_size, 5)) > 0
        if remainder.any():
            wrong_dims = (np.where(remainder)[0] + 2).tolist()
            raise ValueError(
                f"spatial dimensions {wrong_dims} of input image (spatial shape: {spatial_shape})"
                f" must be divisible by {self.patch_size}**5."
            )

    def forward(self, img1, img2, textout):

        if not torch.jit.is_scripting() and not torch.jit.is_tracing():
            self._check_input_size(img1.shape[2:])
            self._check_input_size(img2.shape[2:])
        combined = torch.cat([img1, img2], dim=0)
        with torch.no_grad():
            combined_out = self.feature_extractor(combined)
            # 并非需要这个encoder，在data_loader
            text = self.text_processor(textout)
        img1_out = []
        img2_out = []
        
        for feat in combined_out:
            # feat shape: [2*B, C_i, D_i, H_i, W_i]
            # torch.chunk 将张量切分成 2 份，在 dim=0 上切
            f1, f2 = torch.chunk(feat, chunks=2, dim=0)
            img1_out.append(f1)
            img2_out.append(f2)
            
        text_fusions = self.cross_scale_fusion(img1_out, img2_out, text)
        queries_fusions = self.blank_query_fusion(img1_out, img2_out)
        # print(f"text_fusions,shape: {[t.shape for t in text_fusions]}")
        # print(f"queries_fusions,shape: {[q.shape for q in queries_fusions]}")

        # 两个特征都是从深层开始的
        cat4, alpha5 = self.skip5(text_fusions[0], queries_fusions[0])
        # print(f"cat4,shape: {cat4.shape}")
        dec4 = self.encoder10(cat4)
        # print(f"dec4,shape: {dec4.shape}")
        cat3,alpha4 = self.skip4(text_fusions[1], queries_fusions[1])
        # print(f"cat3,shape: {cat3.shape}")
        dec3 = self.decoder5(dec4, cat3)
        # print(f"dec3,shape: {dec3.shape}")
        cat2, alpha3 = self.skip3(text_fusions[2], queries_fusions[2])
        # print(f"cat2,shape: {cat2.shape}")
        dec2 = self.decoder4(dec3, cat2)
        # print(f"dec2,shape: {dec2.shape}")
        cat1, alpha2 = self.skip2(text_fusions[3], queries_fusions[3])
        # print(f"cat1,shape: {cat1.shape}")
        dec1 = self.decoder3(dec2, cat1)
        # print(f"dec1,shape: {dec1.shape}")
        cat0, alpha1 = self.skip1(text_fusions[4], queries_fusions[4])
        # print(f"cat0,shape: {cat0.shape}")
        dec0 = self.decoder2(dec1, cat0)
        # print(f"dec0,shape: {dec0.shape}")
        dec0 = self.decoder1(dec0)
        # print(f"dec0,shape: {dec0.shape}")
        # out = self.out(dec0)
        # out = self.softmax(out) # 对通道维度进行softmax
        dec0 = self.dropout(dec0)
        out = self.nodule_detection_head(dec0)
        # print(f'out shape {out.shape}')

        return out, (alpha5, alpha4, alpha3, alpha2, alpha1)

       
