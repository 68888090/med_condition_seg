import torch
import torch.nn as nn
from collections.abc import Sequence
import numpy as np




from models.image_backbone import SwinTransformerBackbone
from models.text_processor import LanguageProcessor

from monai.networks.blocks import UnetrUpBlock, UnetOutBlock, UnetrBasicBlock
from monai.networks.blocks.dynunet_block import UnetBasicBlock, UnetResBlock, get_conv_layer

import math
import torch
import torch.nn as nn
from torch.cuda.amp import autocast

class TextMatchingHead(nn.Module):
    def __init__(self, in_channels, hidden_dim=256, dropout=0.1):
        super().__init__()
        # Global Average Pooling 将 [B, C, D, H, W] -> [B, C, 1, 1, 1]
        self.pool = nn.AdaptiveAvgPool3d(1)
        
        # MLP 分类器
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1) # 输出 Logits，不加 Sigmoid
        )

    def forward(self, x):
        x = self.pool(x)
        logits = self.mlp(x)
        return logits

class NoduleDetectionHead(nn.Module):
    def __init__(self, in_channels=48, hidden_channels=64, dropout_rate=0.1):
        super().__init__()
        
        # 1. 共享特征提取
        self.shared_conv = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=dropout_rate)
        )
        
        # ================== 修改点：双头输出 ==================
        
        # 2. Main Head (文本条件分割) -> 对应 label_final (可能全黑)
        self.main_seg_head = nn.Conv3d(hidden_channels, 1, kernel_size=1)
        
        # 3. Aux Head (基础目标分割) -> 对应 label_object (永远有结节)
        # 这是一个辅助分支，专门用来保证"不管文不文本，先给我把结节抠出来"
        self.aux_seg_head = nn.Conv3d(hidden_channels, 1, kernel_size=1)

        # 4. 其他检测头 (Offset, Diameter) 属于 Main 任务，只对 Final Mask 有效
        self.offset_head = nn.Conv3d(hidden_channels, 3, kernel_size=1)
        self.size_head = nn.Conv3d(hidden_channels, 1, kernel_size=1)

        # 初始化偏置 (防止训练初期 loss 爆炸)
        for head in [self.main_seg_head, self.aux_seg_head]:
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            head.bias.data.fill_(bias_value)

    def forward(self, x):
        feature = self.shared_conv(x)
        
        # 输出两个 Mask Logits
        main_logits = self.main_seg_head(feature) # 最终预测
        aux_logits = self.aux_seg_head(feature)   # 辅助预测
        
        # 检测任务通常只针对 Main 结果有效
        offset_logits = self.offset_head(feature)
        offset = torch.sigmoid(offset_logits)
        
        size_logits = self.size_head(feature)
        size_logits = torch.clamp(size_logits, min=-20.0, max=20.0)
        diameter = torch.nn.functional.softplus(size_logits)

        # 返回四个值：主Mask, 辅Mask, 偏移, 直径
        return main_logits, aux_logits, offset, diameter
        
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
        num_queries: int = 300, # (不再使用，但保留参数接口以防报错)
        query_dim: int = 256,   # (不再使用)
        feature_channels : list[int] = [48,96,192,384,768],
        text_dim: int = 768,
        num_texts: int = 60,
        spatial_size:int = 4,
        out_channels: int = 2, 
        backbone_weights_path: str = '/home/lhr/dataset/checkpoints/swin-unetr/model_swinvit_fixed.pt',
        device: str = 'cuda',
        norm_name: str = 'instance',
        patch_size: int = 2,
        batch_size: int = 1,
        dropout_rate: float = 0.2, # <--- 建议调高到 0.2

        # 下面是实验模块变换部分
        fusion_mode: str = "prior_guide"

    ):
        super().__init__()

        if fusion_mode == "prior_guide":
            from models.gate_modal_fusion import HybridFusionModule
        elif fusion_mode == "simple_gate":
            from models.new_spatial_fusion import HybridFusionModule
        elif fusion_mode == "attention":
            from models.spatial_fusion1 import HybridFusionModule



        self.feature_channels = feature_channels
        self.text_dim = text_dim
        self.patch_size = patch_size
        self.dropout_rate = dropout_rate
        
        self.text_processor = LanguageProcessor()
        self.feature_extractor = SwinTransformerBackbone(backbone_weights_path, out_channels)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        # text_channels = [128, 256, 384, 512, 768]
        # === 【核心修改 1】替换为混合融合模块 ===
        # 移除 MultiScaleQueryFusion 和 CrossScaleFusion
        self.fusion_module = HybridFusionModule(
            feature_channels = self.feature_channels,
            # text_hierarchy_dims = text_channels, # 传入这个宽列表
            # text_input_dim = 768,
            active_layers = [4, 3, 2, 1, 0]
        )
        
        # === Decoder 定义 (保持不变) ===
        self.decoder5 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_channels[4],
            out_channels=feature_channels[3],
            kernel_size=3, upsample_kernel_size=2, norm_name=norm_name, res_block=True
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_channels[3],
            out_channels=feature_channels[2],
            kernel_size=3, upsample_kernel_size=2, norm_name=norm_name, res_block=True
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_channels[2],
            out_channels=feature_channels[1],
            kernel_size=3, upsample_kernel_size=2, norm_name=norm_name, res_block=True
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_channels[1],
            out_channels=feature_channels[0],
            kernel_size=3, upsample_kernel_size=2, norm_name=norm_name, res_block=True
        )
        self.decoder1 = UnetrUpDecoder(
            spatial_dims=3,
            in_channels=feature_channels[0],
            out_channels=feature_channels[0],
            kernel_size=3, upsample_kernel_size=2, norm_name=norm_name, res_block=True
        )
        
        # BottleNeck 编码器
        self.encoder10 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=feature_channels[4],
            out_channels=feature_channels[4],
            kernel_size=3, stride=1, norm_name=norm_name, res_block=True
        )

        # === 【核心修改 2】全局 Dropout ===
        self.dropout = nn.Dropout3d(p=dropout_rate)
        
        # === Head ===
        # 确保 Detection Head 也接收 dropout_rate (需要你修改 Head 的 __init__)
        self.nodule_detection_head = NoduleDetectionHead(
            feature_channels[0], 
            dropout_rate=dropout_rate
        )

        self.match_head = TextMatchingHead(in_channels=feature_channels[4], dropout=dropout_rate)

    @torch.jit.unused
    def _check_input_size(self, spatial_shape):
        img_size = np.array(spatial_shape)
        remainder = (img_size % np.power(self.patch_size, 5)) > 0
        if remainder.any():
            wrong_dims = (np.where(remainder)[0] + 2).tolist()
            raise ValueError(f"spatial dimensions {wrong_dims} error")

    def forward(self, img1, img2, textout):
        if not torch.jit.is_scripting() and not torch.jit.is_tracing():
            self._check_input_size(img1.shape[2:])
            self._check_input_size(img2.shape[2:])
            
        # 1. 提取特征
        combined = torch.cat([img1, img2], dim=0)
        with torch.no_grad():
            combined_out = self.feature_extractor(combined)

            text = self.text_processor(textout) 

            
        img1_out = []
        img2_out = []
        for feat in combined_out:
            f1, f2 = torch.chunk(feat, chunks=2, dim=0)
            img1_out.append(f1)
            img2_out.append(f2)
            
        # === 【核心修改 3】 执行融合 ===
        # 这里返回的 fused_skips 直接对应 [cat0, cat1, cat2, cat3, cat4]
        # 且保留了原始空间分辨率 (D, H, W)

        fused_skips = self.fusion_module(img1_out, img2_out, text)

        # === Decoder 流 (加入逐层 Dropout) ===
        # cat4 对应最深层 features[4]
        cat4 = fused_skips[4] 
        
        # Bottleneck
        dec4 = self.encoder10(cat4)
        dec4 = self.dropout(dec4) # <--- Dropout

        # Up 1
        cat3 = fused_skips[3]
        dec3 = self.decoder5(dec4, cat3)
        dec3 = self.dropout(dec3) # <--- Dropout

        # Up 2
        cat2 = fused_skips[2]
        dec2 = self.decoder4(dec3, cat2)
        dec2 = self.dropout(dec2) # <--- Dropout

        # Up 3
        cat1 = fused_skips[1]
        dec1 = self.decoder3(dec2, cat1)
        dec1 = self.dropout(dec1) # <--- Dropout

        # Up 4
        cat0 = fused_skips[0]
        dec0 = self.decoder2(dec1, cat0)
        # dec0 这里可以不用立即 dropout，因为紧接着是最后一层上采样
        
        # Final Up
        dec0 = self.decoder1(dec0)
        dec0 = self.dropout(dec0) # <--- Dropout
        
        # Head
        main_logits, aux_logits, offset, diameter = self.nodule_detection_head(dec0) 
        match_logits = self.match_head(dec4)

        out = {
            "main_logits": main_logits,
            "aux_logits": aux_logits,
            "match_logits": match_logits,
            "offset": offset,
            "diameter": diameter
        }
        
        
        return out