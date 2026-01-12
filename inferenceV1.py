import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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

# === 假设这些是你的自定义模块，请确保路径正确 ===
# from utils.transform import ProcessText
# from unetM import UnetM
# from models.text_processor import LanguageProcessor
# 为了让代码跑通，我这里保留import，实际运行时请取消注释
try:
    from utils.transform import ProcessText
    from unetM import UnetM
    from models.text_processor import LanguageProcessor
except ImportError:
    print("Warning: 自定义模块导入失败，请检查路径。")

# ==============================================================================
# 1. 用户配置区域
# ==============================================================================
CONFIG = {
    "image1_path": "/home/lhr/dataset/CSTPLung/data/bbox_data/fixed_80_20190314.mhd",
    "image2_path": "/home/lhr/dataset/CSTPLung/data/bbox_data/80_20220905.mhd",
    "text_prompt": "Find nodules enlarged by 50",
    "test_json_data" : "/home/lhr/dataset/CSTPLung/data2.json",
    "checkpoint": "/home/lhr/dataset/checkpoints/swin-unetr/9_checkpoint_best.pth.tar",
    "roi_size": (64, 64, 64),
    "roi_x": 64, "roi_y": 64, "roi_z": 64,
    "sw_batch_size": 4,
    "overlap": 0.5,
    "threshold": 0.1,  # 验证时阈值设低一点，方便计算 Recall
    "device": "cuda:0"
}

PREPROCESS_ARGS = {
    "a_min": -1000, "a_max": 1000,
    "b_min": 0.0,   "b_max": 1.0,
    "pixdim": (1.0, 1.0, 1.0) # 重要：这是重采样后的 spacing
}

# ==============================================================================
# 2. 核心辅助函数 (球形填充与坐标变换)
# ==============================================================================

def fill_sphere_ras(mask_tensor, center_radius):
    """
    在 RAS 格式的 3D bool tensor 中填充一个球体。
    center_radius: ((r, a, s), radius)
    """
    center, radius = center_radius
    len_r, len_a, len_s = mask_tensor.shape[-3:]
    cr, ca, cs = center
    rad = int(radius)
    
    # 计算包围盒
    r_min, r_max = max(0, cr - rad), min(len_r, cr + rad + 1)
    a_min, a_max = max(0, ca - rad), min(len_a, ca + rad + 1)
    s_min, s_max = max(0, cs - rad), min(len_s, cs + rad + 1)
    
    if r_min >= r_max or a_min >= a_max or s_min >= s_max:
        return

    # 生成局部网格
    lr = torch.arange(r_min, r_max, device=mask_tensor.device, dtype=torch.float32)
    la = torch.arange(a_min, a_max, device=mask_tensor.device, dtype=torch.float32)
    ls = torch.arange(s_min, s_max, device=mask_tensor.device, dtype=torch.float32)
    
    grid_r, grid_a, grid_s = torch.meshgrid(lr, la, ls, indexing='ij')
    
    dist_sq = (grid_r - cr)**2 + (grid_a - ca)**2 + (grid_s - cs)**2
    sphere_mask = dist_sq <= (radius ** 2)
    
    # 填充
    if mask_tensor.ndim == 5: 
        mask_tensor[0, 0, r_min:r_max, a_min:a_max, s_min:s_max] |= sphere_mask
    elif mask_tensor.ndim == 3:
        mask_tensor[r_min:r_max, a_min:a_max, s_min:s_max] |= sphere_mask

def decode_and_print(heatmap, size, threshold=0.1):
    """从热力图中解析坐标，并生成预测的球形Mask"""
    # 1. 归一化与寻找峰值
    heatmap = torch.sigmoid(heatmap) 
    hmax = F.max_pool3d(heatmap, kernel_size=3, padding=1, stride=1)
    epsilon = 1e-4 
    
    keep = (heatmap >= (hmax - epsilon)) & (heatmap > threshold)
    indices = torch.nonzero(keep, as_tuple=False)
    
    # 初始化预测的 Mask
    sphere_mask_tensor = torch.zeros_like(heatmap, dtype=torch.bool)
    
    results = []
    print(f"检测到 {len(indices)} 个潜在结节 (Threshold={threshold})...")
    
    for idx in indices:
        b, c, x, y, z = idx.tolist() # 注意：这里假设 dim2=x(R), dim3=y(A), dim4=z(S)
        
        d = size[b, 0, x, y, z].item()
        score = heatmap[b, 0, x, y, z].item()
        
        # 填充预测 Mask
        center_radius = ((x, y, z), d/2)
        fill_sphere_ras(sphere_mask_tensor, center_radius)
        
        results.append({"z": z, "y": y, "x": x, "d": d, "score": score})
        # print(f"  -> Pos: ({x}, {y}, {z}), Dia: {d:.1f}, Score: {score:.2f}")

    return results, sphere_mask_tensor

# ==============================================================================
# 3. 数据解析与 GT 生成
# ==============================================================================

def find_suit_data(origin, spacing) -> list:
    """
    根据文本提示和文件名，从 JSON 中找到对应的 GT 标签，
    并将其世界坐标转换为当前的体素坐标。
    """
    if not os.path.exists(CONFIG["test_json_data"]):
        print("Error: JSON file not found.")
        return []

    with open(CONFIG["test_json_data"], "r") as f:
        data = json.load(f)
    
    # 1. 解析 Text Prompt 条件
    kind_list = ("enlarged", "smaller", "new")
    condition = re.findall(r"-?\d+\.\d+|-?\d+", CONFIG["text_prompt"])
    
    if len(condition) == 0:
        kind = kind_list[-1] # new
    else:
        # 如果 prompt 里有数字，正数为 enlarged, 负数为 smaller
        kind = kind_list[0] if float(condition[0]) > 0 else kind_list[1]
    
    print(f"筛选条件: Type={kind}, Threshold={condition[0] if condition else 'None'}")

    # 2. 筛选对应的 Image Pair
    label_raw = []
    for item in data["training"]:
        # 注意：这里做简单的字符串匹配，确保 JSON 里的路径和 CONFIG 里的路径完全一致
        if item["ctImage2"] == CONFIG["image2_path"] and item["ctImage1"] == CONFIG["image1_path"]:
            label_raw.append(item["label"])
    
    if not label_raw:
        print("Warning: 未在 JSON 中找到当前图像对的标注信息。")
        return []

    # 3. 筛选符合 prompt 描述的结节
    m_voxel_list = []
    
    for label_group in label_raw: # label_group 可能是多个结节的列表，或者本身就是一个列表
        # 这里假设 label_raw 是 [[label1], [label2]] 或者 [label1, label2] 
        # 根据你的数据结构，这里可能需要微调。假设 item["label"] 是一个结节列表
        
        # 如果 item["label"] 直接是列表的列表:
        labels_to_check = label_group if isinstance(label_group[0], list) else [label_group]

        for label in labels_to_check:
            # label 格式假设: [uuid, x, y, z, diameter, attribute_text]
            attr_text = label[-1]
            num = re.findall(r"-?\d+\.\d+|-?\d+", attr_text)
            
            is_match = False
            if len(num) == 0 and kind == "new":
                is_match = True
            elif len(num) == 1:
                val = float(num[0])
                thresh = float(condition[0])
                if kind == "enlarged" and val > thresh:
                    is_match = True
                elif kind == "smaller" and val < thresh:
                    is_match = True
            
            if is_match:
                # 4. 世界坐标转体素坐标 (World -> Voxel)
                # 假设 label[1:4] 是 (x, y, z) 世界坐标
                # 且 origin 是 (x, y, z), spacing 是 (sx, sy, sz)
                # RAS 坐标系下直接减
                world_coord = np.array(label[1:4], dtype=float)
                voxel_coord = np.round((world_coord - origin) / spacing).astype(int)
                
                # 注意：label[4] 是直径
                diameter = float(label[4])
                # 体素直径 = 物理直径 / 平均spacing (这里简化处理)
                voxel_dia = diameter / spacing[0] 
                
                m_voxel_list.append(((voxel_coord[0], voxel_coord[1], voxel_coord[2]), voxel_dia/2.0))
    
    print(f"找到符合条件的 GT 结节数量: {len(m_voxel_list)}")
    return m_voxel_list

# ==============================================================================
# 4. 指标计算
# ==============================================================================

def compute_Recall(m_voxel_list, pred_mask):
    """
    计算基于像素的指标。
    m_voxel_list: [(center, radius), ...]
    pred_mask: 预测的 boolean tensor
    """
    # 1. 生成 GT Mask
    gt_tensor = torch.zeros_like(pred_mask, dtype=torch.bool)
    for center_radius in m_voxel_list:
        fill_sphere_ras(gt_tensor, center_radius)
    
    pred = pred_mask.bool()
    gt = gt_tensor.bool()

    # 2. 计算交并集
    intersection = (pred & gt).sum().float()
    union = (pred | gt).sum().float()
    pred_area = pred.sum().float()
    gt_area = gt.sum().float()

    # 3. 计算指标
    iou = (intersection + 1e-6) / (union + 1e-6)
    dice = (2 * intersection + 1e-6) / (pred_area + gt_area + 1e-6)
    recall = (intersection + 1e-6) / (gt_area + 1e-6) # Sensitivity
    precision = (intersection + 1e-6) / (pred_area + 1e-6) # PPV

    metrics = {
        "IoU": iou.item(),
        "Dice": dice.item(),
        "Recall_Pixel": recall.item(),
        "Precision_Pixel": precision.item(),
        "AUC": 0.0 # 占位
    }
    return metrics, gt

def compute_AUC(heatmap, gt_tensor, metrics):
    """计算 AUC"""
    # 展平
    heatmap = torch.sigmoid(heatmap) # 确保是概率
    y_scores = heatmap.detach().cpu().numpy().flatten()
    y_true = gt_tensor.detach().cpu().numpy().flatten().astype(int)
    
    try:
        # 如果 GT 全是 0 (负样本)，AUC 未定义，通常返回 0 或 NaN
        if np.sum(y_true) == 0:
            metrics["AUC"] = 0.5 # 无 GT 时无法评估排序能力
        else:
            metrics["AUC"] = roc_auc_score(y_true, y_scores)
    except Exception as e:
        print(f"AUC计算出错: {e}")
        metrics["AUC"] = 0.0
        
    return metrics

def visualize_slice(vol, nodules, save_name="result.png"):
    """可视化置信度最高的结节切片"""
    if not nodules:
        return
    best_nodule = max(nodules, key=lambda x: x['score'])
    z_slice = int(round(best_nodule['z']))
    z_slice = max(0, min(z_slice, vol.shape[0]-1))

    plt.figure(figsize=(8, 8))
    # 注意：vol 通常是 RAS，所以 vol[x, y, z] -> 切片取 [:, :, z]
    # 如果 vol 是 [R, A, S]，imshow 需要展示 R-A 平面 (X-Y)
    # 这里的切片维度取决于你想看哪个面，通常看 axial 面是取 Z 轴 (dim 2)
    img_slice = vol[:, :, z_slice] 
    
    plt.imshow(img_slice.T, cmap='gray', origin='lower') # .T 和 origin='lower' 是为了符合解剖学观看习惯
    plt.title(f"Detection (Axial Slice Z={z_slice})")
    
    ax = plt.gca()
    for n in nodules:
        if abs(n['z'] - z_slice) < 2:
            # Matplotlib Circle (x, y) 对应 tensor 的 (dim0, dim1)
            circ = patches.Circle((n['x'], n['y']), n['d']/2, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(circ)
            ax.text(n['x'], n['y'], f"{n['score']:.2f}", color='yellow', fontsize=8)
    
    plt.savefig(save_name)
    plt.close()
    print(f"可视化已保存: {save_name}")

# ==============================================================================
# 5. MAIN 函数
# ==============================================================================
def main():
    device = torch.device(CONFIG["device"])
    
    # --- A. 加载模型 ---
    print(">>> 正在初始化模型...")
    text_processor = LanguageProcessor().to(device)
    processText = ProcessText(text_processor, keys=["text"], max_len=60)
    
    model = UnetM(text_processor, batch_size=1).to(device)
    checkpoint = torch.load(CONFIG["checkpoint"], map_location=device)
    state_dict = {k.replace("module.", ""): v for k, v in checkpoint['state_dict'].items()} if 'state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    
    inferer = SlidingWindowInferer(
        roi_size=[CONFIG["roi_x"], CONFIG["roi_y"], CONFIG["roi_z"]],
        sw_batch_size=CONFIG["sw_batch_size"],
        overlap=CONFIG["overlap"],
        mode="gaussian"
    )

    # --- B. 数据处理 ---
    print(">>> 正在加载与预处理数据...")
    transforms = Compose([
        LoadImaged(keys=["img1", "img2"]),
        EnsureChannelFirstd(keys=["img1", "img2"]),
        Orientationd(keys=["img1", "img2"], axcodes="RAS"),
        Spacingd(keys=["img1", "img2"], pixdim=PREPROCESS_ARGS["pixdim"], mode="bilinear"),
        ScaleIntensityRanged(keys=["img1", "img2"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True),
        processText,
        SpatialPadd(keys=["img1", "img2"], spatial_size=[CONFIG["roi_x"], CONFIG["roi_y"], CONFIG["roi_z"]]),
        ToTensord(keys=["img1", "img2", "text"])
    ])

    data_dict = {
        "img1": CONFIG["image1_path"], 
        "img2": CONFIG["image2_path"],
        "text": CONFIG["text_prompt"]
    }
    
    # 执行变换
    data = transforms(data_dict)
    def read_mhd_image(file_path):
        """
        读取 mhd 文件并返回图像数组及空间信息。
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件未找到: {file_path}")

        itk_image = sitk.ReadImage(file_path)
        numpy_image = sitk.GetArrayFromImage(itk_image)
        print(np.unique(numpy_image))
        # 去掉小于0的值
        numpy_image[numpy_image < 0] = 0

        # 获取空间元数据
        origin = itk_image.GetOrigin()       # (x, y, z)
        spacing = itk_image.GetSpacing()     # (x_sp, y_sp, z_sp)
        direction = itk_image.GetDirection() # 9个元素的元组，展平的方向矩阵

        return numpy_image, origin, spacing, direction
    
    # 【关键步骤】获取 Origin 以便对齐 GT
    # 变换后的 meta_dict 中 'affine' 矩阵的前三行第四列是 Origin
    # 如果没有 affine，这步会报错。Monai LoadImage 默认会有。
    _ , origin, spacing, direction = read_mhd_image(CONFIG["image2_path"])

    print(f"提取到图像 Origin (RAS): {origin}")


    # 准备 Tensor
    img1 = data["img1"].unsqueeze(0).to(device) # [1, C, R, A, S]
    img2 = data["img2"].unsqueeze(0).to(device)
    text = data["text"].unsqueeze(0).to(device)

    # --- C. 推理 ---
    print(">>> 正在推理...")
    with torch.no_grad():
        def model_wrapper(inputs):
            current_bs = inputs.shape[0]
            batch_text = text.expand(current_bs, -1)
            # 假设 model forward 接收 (img1, img2, text)
            # 注意：SlidingWindowInferer 会把 img1 和 img2 拼在一起送进来
            # inputs shape: [B, 2, R, A, S]
            return model(inputs[:, 0:1], inputs[:, 1:2], batch_text)[0] # 返回 tuple 第一个元素 heatmap

        combined_img = torch.cat([img1, img2], dim=1) 
        outputs = inferer(combined_img, model_wrapper)
        hm_pred, off_pred, size_pred = outputs

    # --- D. 后处理与评估 ---
    print(">>> 正在解码与评估...")
    
    # 1. 解码得到预测列表和预测 Mask
    nodule_results, pred_mask_tensor = decode_and_print(hm_pred, size_pred, threshold=CONFIG["threshold"])
    
    # 2. 从 JSON 寻找 GT，并转换坐标
    gt_voxel_list = find_suit_data(origin, spacing=PREPROCESS_ARGS["pixdim"])
    
    # 3. 计算指标
    metrics, gt_tensor = compute_Recall(gt_voxel_list, pred_mask_tensor)
    
    # 4. 计算 AUC (需要原始 heatmap)
    metrics = compute_AUC(hm_pred, gt_tensor, metrics)
    
    # --- E. 输出报告 ---
    print("\n" + "="*40)
    print(f"定量分析报告 (Text: {CONFIG['text_prompt']})")
    print("-" * 40)
    print(f"IoU (Voxel)      : {metrics['IoU']:.4f}")
    print(f"Dice (Voxel)     : {metrics['Dice']:.4f}")
    print(f"Recall (Pixel)   : {metrics['Recall_Pixel']:.4f}")
    print(f"Precision (Pixel): {metrics['Precision_Pixel']:.4f}")
    print(f"AUC              : {metrics['AUC']:.4f}")
    print("="*40 + "\n")

    # --- F. 可视化 ---
    vol_numpy = img1[0, 0].cpu().numpy() # [R, A, S]
    visualize_slice(vol_numpy, nodule_results, save_name="final_analysis.png")

if __name__ == "__main__":
    main()