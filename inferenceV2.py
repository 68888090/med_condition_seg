import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import json
import re
import os
import SimpleITK as sitk
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, 
    Spacingd, ScaleIntensityRanged, ToTensord, SpatialPadd,
)
from monai.inferers import SlidingWindowInferer
from sklearn.metrics import roc_auc_score

# === 导入你的自定义模块 ===
try:
    from utils.transform import ProcessText
    from new_UnetM import UnetM  # 确保这是你最新的模型定义
    from models.text_processor import LanguageProcessor
except ImportError:
    print("Warning: 自定义模块导入失败，请检查路径。")

# ==============================================================================
# 1. 用户配置区域
# ==============================================================================
CONFIG = {
    # 输入的全肺图像路径 (支持 .mhd, .nii.gz)
    "image1_path": "/home/lhr/dataset/CSTPLung/data/bbox_data/fixed_94_20220407.mhd",
    "image2_path": "/home/lhr/dataset/CSTPLung/data/bbox_data/94_20221025.mhd",
    
    # 文本提示
    "text_prompt": "Which findings increased in size by 54?",
    
    # 包含标注信息的 JSON
    "test_json_data" : "/home/lhr/dataset/CSTPLung/data2.json",
    
    # 训练好的模型权重
    "checkpoint": "/home/lhr/dataset/checkpoints/swin-unetr/1_16_1_checkpoint_best.pth.tar",
    
    # 推理参数
    "roi_size": (64, 64, 64), # 必须与训练时的 Patch Size 一致
    "sw_batch_size": 4,       # 滑动窗口并行数
    "overlap": 0.5,           # 滑动窗口重叠率
    "threshold": 0.5,         # Mask 二值化阈值
    "use_gating": True,       # 是否启用分类头门控
    "device": "cuda:0"
}

# 预处理参数 (必须与训练时保持一致)
PREPROCESS_ARGS = {
    "pixdim": (1.0, 1.0, 1.0) # 强制重采样到 1mm
}

# ==============================================================================
# 2. 核心辅助函数
# ==============================================================================

def fill_sphere_ras(mask_tensor, center_radius):
    """
    在 3D Tensor 中画球 (生成 GT 用)
    center_radius: ((x, y, z), radius) - 坐标应为 Voxel 坐标
    """
    center, radius = center_radius
    # mask_tensor shape: [D, H, W] or [C, D, H, W]
    if mask_tensor.ndim == 4:
        d, h, w = mask_tensor.shape[1:]
    else:
        d, h, w = mask_tensor.shape
        
    cx, cy, cz = center
    rad = int(radius)
    
    # 定义包围盒加速计算
    x_min, x_max = max(0, int(cx - rad)), min(d, int(cx + rad + 1))
    y_min, y_max = max(0, int(cy - rad)), min(h, int(cy + rad + 1))
    z_min, z_max = max(0, int(cz - rad)), min(w, int(cz + rad + 1))
    
    if x_min >= x_max or y_min >= y_max or z_min >= z_max:
        return

    # 生成网格
    grid_x, grid_y, grid_z = torch.meshgrid(
        torch.arange(x_min, x_max, device=mask_tensor.device),
        torch.arange(y_min, y_max, device=mask_tensor.device),
        torch.arange(z_min, z_max, device=mask_tensor.device),
        indexing='ij'
    )
    
    dist_sq = (grid_x - cx)**2 + (grid_y - cy)**2 + (grid_z - cz)**2
    sphere_mask = dist_sq <= (radius ** 2)
    
    if mask_tensor.ndim == 4:
        mask_tensor[0, x_min:x_max, y_min:y_max, z_min:z_max] |= sphere_mask
    else:
        mask_tensor[x_min:x_max, y_min:y_max, z_min:z_max] |= sphere_mask

def compute_metrics(pred_logits, gt_mask, threshold=0.5):
    """
    计算基于分割 Mask 的指标
    """
    probs = torch.sigmoid(pred_logits)
    pred_mask = (probs > threshold).float()
    gt_mask = gt_mask.float()
    
    # Flatten
    pred_flat = pred_mask.view(-1)
    gt_flat = gt_mask.view(-1)
    probs_flat = probs.view(-1)
    
    intersection = (pred_flat * gt_flat).sum()
    union = pred_flat.sum() + gt_flat.sum() - intersection
    
    smooth = 1e-5
    dice = (2. * intersection + smooth) / (pred_flat.sum() + gt_flat.sum() + smooth)
    iou = (intersection + smooth) / (union + smooth)
    recall = (intersection + smooth) / (gt_flat.sum() + smooth)
    precision = (intersection + smooth) / (pred_flat.sum() + smooth)
    
    # AUC Calculation
    try:
        # 只有当 GT 不全为 0 时计算 AUC，否则无意义(或0.5)
        if gt_flat.sum() > 0:
            auc = roc_auc_score(gt_flat.cpu().numpy(), probs_flat.cpu().numpy())
        else:
            auc = 0.5 
    except:
        auc = 0.0

    return {
        "Dice": dice.item(),
        "IoU": iou.item(),
        "Recall": recall.item(),
        "Precision": precision.item(),
        "AUC": auc
    }

# ==============================================================================
# 3. GT 生成逻辑 (适配全肺 + Spacingd)
# ==============================================================================

def generate_gt_mask(json_data_path, text_prompt, target_shape, origin, spacing):
    """
    根据 JSON 和 文本提示，生成全肺的 GT Mask。
    注意：target_shape, origin, spacing 必须对应 重采样后 的图像空间。
    """
    # 1. 解析 Text Prompt 获取筛选条件 (Enlarged/Smaller/New)
    kind_list = ("enlarged", "smaller", "new")
    condition_nums = re.findall(r"-?\d+\.\d+|-?\d+", text_prompt)
    if not condition_nums:
        kind = "new"
    else:
        kind = "enlarged" if float(condition_nums[0]) > 0 else "smaller"
    
    print(f"GT筛选逻辑: Type={kind}, Threshold={condition_nums[0] if condition_nums else 'N/A'}")

    # 2. 读取 JSON
    with open(json_data_path, "r") as f:
        data = json.load(f)

    # 3. 找到当前图像对的标注
    labels_to_draw = []
    # 假设 JSON 结构是 list of dicts
    training_list = data.get("training", []) + data.get("validation", [])
    
    for item in training_list:
        # 简单的路径匹配 (建议实际使用 filename 匹配更稳健)
        if item["ctImage2"] == CONFIG["image2_path"] and item["ctImage1"] == CONFIG["image1_path"]:
            labels_raw = item["label"] # 可能是 list of lists
            
            # 标准化 label 格式
            if isinstance(labels_raw[0], list):
                candidates = labels_raw
            else:
                candidates = [labels_raw]

            # 遍历每个结节，看是否符合 text_prompt
            for label in candidates:
                # label 格式: [uuid, x, y, z, diameter, attribute_text]
                attr_text = label[-1]
                attr_nums = re.findall(r"-?\d+\.\d+|-?\d+", attr_text)
                
                is_match = False
                if kind == "new" and not attr_nums:
                    is_match = True
                elif kind != "new" and len(attr_nums) == 1 and len(condition_nums) == 1:
                    val = float(attr_nums[0])
                    thresh = float(condition_nums[0])
                    if kind == "enlarged" and val > thresh: is_match = True
                    if kind == "smaller" and val < thresh: is_match = True
                
                if is_match:
                    # 坐标转换：World (mm) -> Resampled Voxel
                    # 公式: (World - Origin) / New_Spacing
                    # 注意：Monai LoadImage 后的 Origin 是原始的，但 Spacingd 把 Spacing 变成了 (1,1,1)
                    # 这里的 spacing 参数应该传入 (1.0, 1.0, 1.0)
                    world_pos = np.array(label[1:4], dtype=float)
                    voxel_pos = (world_pos - origin) / np.array(spacing)
                    
                    diameter_mm = float(label[4])
                    radius_voxel = (diameter_mm / spacing[0]) / 2.0
                    
                    labels_to_draw.append((voxel_pos, radius_voxel))

    # 4. 绘制 Mask
    gt_tensor = torch.zeros(target_shape, dtype=torch.bool) # [C, D, H, W]
    print(f"找到 {len(labels_to_draw)} 个符合条件的真实结节。")
    
    for center, radius in labels_to_draw:
        fill_sphere_ras(gt_tensor, (center, radius))
        
    return gt_tensor

# ==============================================================================
# 4. 可视化函数 (Mask Overlay)
# ==============================================================================
def visualize_overlay(vol, pred_mask, gt_mask, save_name="result.png"):
    """
    寻找包含最大 GT 或 预测区域的切片进行可视化
    """
    # 转换为 Numpy
    if torch.is_tensor(vol): vol = vol.cpu().numpy()
    if torch.is_tensor(pred_mask): pred_mask = pred_mask.cpu().numpy()
    if torch.is_tensor(gt_mask): gt_mask = gt_mask.cpu().numpy()
    
    # 寻找最佳切片 (GT 面积最大的层，如果没有 GT 则找 Pred 最大的层)
    if gt_mask.sum() > 0:
        z_sums = gt_mask.sum(axis=(0, 1)) # [D, H, W] -> sum over H, W? No, input usually [D, H, W]
        # 假设输入是 [D, H, W] (Depth, Height, Width) -> Axial Slice is along D?
        # 取决于 Monai Orientation. RAS 通常 D是左右, H是前后, W是上下(Z)? 
        # 我们假设 dim 2 是 Z 轴 (Axial)
        z_sums = gt_mask.sum(axis=(0, 1))
        z_slice = np.argmax(z_sums)
    elif pred_mask.sum() > 0:
        z_sums = pred_mask.sum(axis=(0, 1))
        z_slice = np.argmax(z_sums)
    else:
        z_slice = vol.shape[2] // 2 # 中间层
        
    print(f"可视化切片位置: Z={z_slice}")

    img_slice = vol[:, :, z_slice]
    pred_slice = pred_mask[:, :, z_slice]
    gt_slice = gt_mask[:, :, z_slice]
    
    plt.figure(figsize=(12, 6))
    
    # 1. Image + GT
    plt.subplot(1, 2, 1)
    plt.imshow(img_slice.T, cmap='gray', origin='lower')
    # 叠加 GT (绿色)
    if gt_slice.sum() > 0:
        plt.imshow(np.ma.masked_where(gt_slice.T == 0, gt_slice.T), cmap='Greens', alpha=0.5, origin='lower')
    plt.title("Ground Truth (Green)")
    plt.axis('off')

    # 2. Image + Pred
    plt.subplot(1, 2, 2)
    plt.imshow(img_slice.T, cmap='gray', origin='lower')
    # 叠加 Pred (红色)
    if pred_slice.sum() > 0:
        plt.imshow(np.ma.masked_where(pred_slice.T == 0, pred_slice.T), cmap='autumn', alpha=0.5, origin='lower')
    plt.title(f"Prediction (Red) - Text Guided")
    plt.axis('off')
    
    plt.savefig(save_name)
    plt.close()
    print(f"结果已保存: {save_name}")

# ==============================================================================
# 5. 主程序
# ==============================================================================
def main():
    device = torch.device(CONFIG["device"])
    
    # --- A. 模型初始化 ---
    print(">>> 初始化模型...")
    text_processor = LanguageProcessor().to(device)
    processText = ProcessText(text_processor, keys=["text"], max_len=60)
    
    # 加载模型架构 (确保与训练一致)
    model = UnetM(text_processor, batch_size=1).to(device) 
    
    # 加载权重
    checkpoint = torch.load(CONFIG["checkpoint"], map_location=device)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'], strict=True)
    else:
        model.load_state_dict(checkpoint, strict=True)
    model.eval()

    # --- B. 数据加载与预处理 ---
    print(">>> 加载数据 (全肺)...")
    transforms = Compose([
        LoadImaged(keys=["img1", "img2"]),
        EnsureChannelFirstd(keys=["img1", "img2"]),
        Orientationd(keys=["img1", "img2"], axcodes="RAS"),
        # 【关键】强制重采样到 1mm，确保物理尺寸与训练 Patch (64mm=64pixel) 一致
        Spacingd(keys=["img1", "img2"], pixdim=PREPROCESS_ARGS["pixdim"], mode="bilinear"),
        ScaleIntensityRanged(keys=["img1", "img2"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True),
        processText,
        ToTensord(keys=["img1", "img2", "text"])
    ])

    raw_data = {
        "img1": CONFIG["image1_path"], 
        "img2": CONFIG["image2_path"],
        "text": CONFIG["text_prompt"]
    }
    data = transforms(raw_data)
    
    img1 = data["img1"].unsqueeze(0).to(device) # [1, 1, D, H, W]
    img2 = data["img2"].unsqueeze(0).to(device)
    text = data["text"].unsqueeze(0).to(device) # [1, Seq_Len]
    
    # 获取图像元数据用于 GT 生成 (Origin)
    # Monai 会把元数据存在 data['img1_meta_dict'] 或 data['img1'].meta
    # 这里我们简化处理，重新读一下以获取 Origin
    itk_img = sitk.ReadImage(CONFIG["image2_path"])
    origin = itk_img.GetOrigin() # 原始 Origin
    
    print(f"Image Shape (Resampled): {img1.shape}")

    # --- C. 滑动窗口推理 ---
    print(">>> 开始滑动窗口推理...")
    inferer = SlidingWindowInferer(
        roi_size=CONFIG["roi_size"],
        sw_batch_size=CONFIG["sw_batch_size"],
        overlap=CONFIG["overlap"],
        mode="gaussian",
        progress=True
    )

    # 包装器：处理多输出和门控逻辑
    def network_wrapper(inputs):
        # inputs: [B, 2, 64, 64, 64] (由 inferer 拼接)
        inp1 = inputs[:, 0:1, ...]
        inp2 = inputs[:, 1:2, ...]
        
        # 扩展 text 以匹配当前 window batch size
        curr_bs = inp1.shape[0]
        txt_batch = text.expand(curr_bs, -1)
        
        # 前向传播
        # outputs: (main, aux, offset, dia, match)
        outputs = model(inp1, inp2, txt_batch)
        main_logits = outputs[0]
        match_logits = outputs[4]
        
        # === 门控逻辑 (Gated Inference) ===
        if CONFIG["use_gating"]:
            match_prob = torch.sigmoid(match_logits).view(-1, 1, 1, 1, 1)
            gate = (match_prob > 0.5).float()
            # 如果不匹配，将 logits 设为极小值 (Sigmoid后为0)
            final_logits = main_logits * gate + (1 - gate) * -100.0
        else:
            final_logits = main_logits
            
        return final_logits

    with torch.no_grad():
        combined_input = torch.cat([img1, img2], dim=1) # [1, 2, D, H, W]
        # 推理结果是 Logits [1, 1, D, H, W]
        prediction_logits = inferer(combined_input, network_wrapper)

    # --- D. 生成 GT 并评估 ---
    print(">>> 生成全肺 GT 并计算指标...")
    
    # 1. 生成 GT Mask (在重采样后的空间)
    # 这里的 target_shape 需要只有空间维度 [D, H, W]
    spatial_shape = img1.shape[2:] 
    gt_mask_tensor = generate_gt_mask(
        CONFIG["test_json_data"], 
        CONFIG["text_prompt"], 
        spatial_shape, 
        origin, 
        PREPROCESS_ARGS["pixdim"]
    ).to(device)

    # 2. 计算指标
    metrics = compute_metrics(prediction_logits, gt_mask_tensor, threshold=CONFIG["threshold"])

    print("\n" + "="*40)
    print(f"定量分析报告 (Gated: {CONFIG['use_gating']})")
    print("-" * 40)
    print(f"Dice      : {metrics['Dice']:.4f}")
    print(f"IoU       : {metrics['IoU']:.4f}")
    print(f"Recall    : {metrics['Recall']:.4f}")
    print(f"Precision : {metrics['Precision']:.4f}")
    print(f"AUC       : {metrics['AUC']:.4f}")
    print("="*40 + "\n")

    # --- E. 可视化 ---
    pred_mask = (torch.sigmoid(prediction_logits) > CONFIG["threshold"]).float()
    
    # 这里的 img1[0,0] 是重采样后的 Tensor，直接可视化
    visualize_overlay(img1[0, 0], pred_mask[0, 0], gt_mask_tensor, save_name="inference_result.png")

if __name__ == "__main__":
    main()