import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR
import argparse
import pydicom



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
import json
import os 
import glob

# device应该是后面的时候通过配置文件来实现的
class SwinTransformerBackbone(nn.Module):
    def __init__(self, original_model_path, model_out_channels):
        super().__init__()
        # 1. 实例化完整的 SwinUNETR
        full_model = SwinUNETR(
            img_size=(96, 96, 96), # 这里的尺寸要和训练时一致，否则位置编码可能不匹配
            in_channels=1 ,
            out_channels=model_out_channels,
            feature_size=48,
            use_checkpoint=True
        )
        
        # 2. 加载权重
        print(f"Loading weights from {original_model_path}")
        full_model.load_state_dict(torch.load(original_model_path)["state_dict"], strict=False)
        
        # 3. 提取编码器 (SwinViT) 并注册为子模块
        # 这样会自动管理梯度和设备移动
        self.backbone = full_model.swinViT
        
        # 释放不需要的解码器显存 (可选，python GC 会处理，但手动删更保险)
        del full_model.encoder1
        del full_model.encoder2
        del full_model.encoder3
        del full_model.encoder4
        del full_model.decoder5
        del full_model.decoder4
        del full_model.decoder3
        del full_model.decoder2
        del full_model.decoder1
        del full_model.out

    def forward(self, x):
        # normalize=True 会在输出前进行 LayerNorm，这对分类任务很重要
        hidden_states = self.backbone(x, normalize=True)
        return hidden_states


# --- 使用方式 ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()

    parser.add_argument('-o', '--model_out_channels', type=int,  help="model out channels", default=2)
    parser.add_argument('-w', '--model_weights_path', type=str,  help="model weights path", default="/home/lhr/dataset/checkpoints/swin-unetr/model_swinvit_fixed.pt")
    args = parser.parse_args()
    # 初始化
    feature_extractor = SwinTransformerBackbone(args.model_weights_path, args.model_out_channels)
    feature_extractor.to(device)
    feature_extractor.eval()
    '''
    SwinTransformerBackbone(
    (backbone): SwinTransformer(
        (patch_embed): PatchEmbed(
        (proj): Conv3d(1, 48, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        )
        (pos_drop): Dropout(p=0.0, inplace=False)
        (layers1): ModuleList(
        (0): BasicLayer(
            (blocks): ModuleList(
            (0-1): 2 x SwinTransformerBlock(
                (norm1): LayerNorm((48,), eps=1e-05, elementwise_affine=True)
                (attn): WindowAttention(
                (qkv): Linear(in_features=48, out_features=144, bias=True)
                (attn_drop): Dropout(p=0.0, inplace=False)
                (proj): Linear(in_features=48, out_features=48, bias=True)
                (proj_drop): Dropout(p=0.0, inplace=False)
                (softmax): Softmax(dim=-1)
                )
                (drop_path): Identity()
                (norm2): LayerNorm((48,), eps=1e-05, elementwise_affine=True)
                (mlp): MLPBlock(
                (linear1): Linear(in_features=48, out_features=192, bias=True)
                (linear2): Linear(in_features=192, out_features=48, bias=True)
                (fn): GELU(approximate='none')
                (drop1): Dropout(p=0.0, inplace=False)
                (drop2): Dropout(p=0.0, inplace=False)
                )
            )
            )
            (downsample): PatchMerging(
            (reduction): Linear(in_features=384, out_features=96, bias=False)
            (norm): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            )
        )
        )
        (layers2): ModuleList(
        (0): BasicLayer(
            (blocks): ModuleList(
            (0-1): 2 x SwinTransformerBlock(
                (norm1): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
                (attn): WindowAttention(
                (qkv): Linear(in_features=96, out_features=288, bias=True)
                (attn_drop): Dropout(p=0.0, inplace=False)
                (proj): Linear(in_features=96, out_features=96, bias=True)
                (proj_drop): Dropout(p=0.0, inplace=False)
                (softmax): Softmax(dim=-1)
                )
                (drop_path): Identity()
                (norm2): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
                (mlp): MLPBlock(
                (linear1): Linear(in_features=96, out_features=384, bias=True)
                (linear2): Linear(in_features=384, out_features=96, bias=True)
                (fn): GELU(approximate='none')
                (drop1): Dropout(p=0.0, inplace=False)
                (drop2): Dropout(p=0.0, inplace=False)
                )
            )
            )
            (downsample): PatchMerging(
            (reduction): Linear(in_features=768, out_features=192, bias=False)
            (norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            )
        )
        )
        (layers3): ModuleList(
        (0): BasicLayer(
            (blocks): ModuleList(
            (0-1): 2 x SwinTransformerBlock(
                (norm1): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
                (attn): WindowAttention(
                (qkv): Linear(in_features=192, out_features=576, bias=True)
                (attn_drop): Dropout(p=0.0, inplace=False)
                (proj): Linear(in_features=192, out_features=192, bias=True)
                (proj_drop): Dropout(p=0.0, inplace=False)
                (softmax): Softmax(dim=-1)
                )
                (drop_path): Identity()
                (norm2): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
                (mlp): MLPBlock(
                (linear1): Linear(in_features=192, out_features=768, bias=True)
                (linear2): Linear(in_features=768, out_features=192, bias=True)
                (fn): GELU(approximate='none')
                (drop1): Dropout(p=0.0, inplace=False)
                (drop2): Dropout(p=0.0, inplace=False)
                )
            )
            )
            (downsample): PatchMerging(
            (reduction): Linear(in_features=1536, out_features=384, bias=False)
            (norm): LayerNorm((1536,), eps=1e-05, elementwise_affine=True)
            )
        )
        )
        (layers4): ModuleList(
        (0): BasicLayer(
            (blocks): ModuleList(
            (0-1): 2 x SwinTransformerBlock(
                (norm1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
                (attn): WindowAttention(
                (qkv): Linear(in_features=384, out_features=1152, bias=True)
                (attn_drop): Dropout(p=0.0, inplace=False)
                (proj): Linear(in_features=384, out_features=384, bias=True)
                (proj_drop): Dropout(p=0.0, inplace=False)
                (softmax): Softmax(dim=-1)
                )
                (drop_path): Identity()
                (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
                (mlp): MLPBlock(
                (linear1): Linear(in_features=384, out_features=1536, bias=True)
                (linear2): Linear(in_features=1536, out_features=384, bias=True)
                (fn): GELU(approximate='none')
                (drop1): Dropout(p=0.0, inplace=False)
                (drop2): Dropout(p=0.0, inplace=False)
                )
            )
            )
            (downsample): PatchMerging(
            (reduction): Linear(in_features=3072, out_features=768, bias=False)
            (norm): LayerNorm((3072,), eps=1e-05, elementwise_affine=True)
            )
        )
        )
    )
    )

    '''


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
    print(datalist[0])
    train_ds = CacheDataset(
        data=datalist,
        transform=train_transforms,
        cache_rate=0.5,
        num_workers=4,
        
    )


    # 提取
    with torch.no_grad():
        input_tensor = torch.unsqueeze(train_ds[0]["image"], 1).to(device)
        # 输出是一个元组,总共是4层
        '''变量名,物理含义,空间比例,"维度 (B, C, D, H, W)",作用,
        当输出图像为[128,128,128]时
        x0_out.shape: torch.Size([1, 48, 64, 64, 64])
        x1_out.shape: torch.Size([1, 96, 32, 32, 32])
        x2_out.shape: torch.Size([1, 192, 16, 16, 16])
        x3_out.shape: torch.Size([1, 384, 8, 8, 8])
        x4_out.shape: torch.Size([1, 768, 4, 4, 4])
        '''
        
        x_out = feature_extractor(input_tensor)
        # 提取不同层的特征
        x0_out, x1_out, x2_out, x3_out, x4_out = x_out
        
        for i in range(5):
            print(f"x{i}_out.shape: {eval(f'x{i}_out').shape}")

if __name__ == "__main__":
    main()