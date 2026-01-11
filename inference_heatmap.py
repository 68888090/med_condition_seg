import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import SimpleITK as sitk
import os
import torch.nn.functional as F
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, 
    Spacingd, ScaleIntensityRanged, SpatialPadd, ToTensord, MapTransform, Lambda
)
from monai.inferers import SlidingWindowInferer

# === 引用你的模型 ===
from unetM import UnetM
from models.text_processor import LanguageProcessor

# ==============================================================================
# 0. 配置区域 (请修改这里)
# ==============================================================================
CONFIG = {
    # --- 输入图像 ---
    "image1_path": "/home/lhr/dataset/CSTPLung/moving_data/fixed_80_20190314.mhd", 
    "image2_path": "/home/lhr/dataset/CSTPLung/moving_data/80_20220905.mhd", 
    "text_prompt": "Find nodules enlarged by 50", 

    # --- 【新增】真值标签路径 (用于对比) ---
    # 如果你没有 offset 和 size 的 label 文件，可以填 None，代码会自动处理
    "label_heatmap_path": "/home/lhr/dataset/CSTPLung/label/1.25mm_3D_detection_mhd_3/label_mask/80_20220905_heatmap.nii.gz",  
    "label_offset_path": "/home/lhr/dataset/CSTPLung/label/1.25mm_3D_detection_mhd_3/label_mask/80_20220905_offset.nii.gz", # "/path/to/your/offset_label.nii.gz" (可选)
    "label_size_path": "/home/lhr/dataset/CSTPLung/label/1.25mm_3D_detection_mhd_3/label_mask/80_20220905_size.nii.gz",   # "/path/to/your/size_label.nii.gz" (可选)

    # --- 模型与参数 ---
    "checkpoint": "/home/lhr/dataset/checkpoints/swin-unetr/checkpoint_best.pth.tar", 
    "roi_x": 64, "roi_y": 64, "roi_z": 64,
    "sw_batch_size": 4,
    "overlap": 0.5,
    "threshold": 0.2, # 预测阈值
    "device": "cuda:0"
}

PREPROCESS_ARGS = {
    "a_min": -1000, "a_max": 1000,
    "b_min": 0.0,   "b_max": 1.0,
    "pixdim": (1.5, 1.5, 2.0)
}

# ==============================================================================
# 1. 辅助类与函数
# ==============================================================================
class ProcessText(MapTransform):
    def __init__(self, language_processor, keys: list, max_len: int = 60):
        super().__init__(keys)
        self.max_len = max_len
        self.language_processor = language_processor

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            tokens = self.language_processor.tokenizer.tokenize(d[key])
            ids = self.language_processor.tokenizer.convert_tokens_to_ids(tokens)
            ids = ids + [0] * (self.max_len - len(ids))
            ids = ids[:self.max_len]
            d[key] = np.array(ids, dtype=np.int64)
        return d

def decode_nodules(heatmap, offset=None, size=None, threshold=2.1e-1, is_ground_truth=False):
    """
    通用解码函数：适用于预测结果和 Label 数据
    
    Args:
        heatmap: (1, D, H, W)
        offset: (3, D, H, W) or None. 如果为 None，强制偏移为0
        size: (1, D, H, W) or None. 如果为 None，强制直径固定
        threshold: 阈值
    """
    # 1. 处理数值类型和归一化
    heatmap = heatmap.float()
    
    if not is_ground_truth:
        # 只有模型的输出需要 Sigmoid，Label 本身已经是 0~1 的高斯分布
        heatmap = torch.sigmoid(heatmap)
    print(f'heatmap unique values: {torch.unique(heatmap)}')
    # 2. 寻找峰值
    hmax = F.max_pool3d(heatmap, kernel_size=3, padding=1, stride=1)
    scale = 10000
    heatmap_int = (heatmap * scale).long()
    hmax_int = (hmax * scale).long()
    
    # 3. 严格相等判断 (Int 比较是绝对精确的)
    # 必须同时满足：
    # A. 它是局部最大值 (heatmap_int == hmax_int)
    # B. 它的值大于阈值
    # hmax = F.max_pool3d(heatmap, kernel_size=3, padding=1, stride=1)

    keep = (heatmap >= (hmax - 1.1e-4)) & (heatmap > threshold)
    # print(f'keep unique values: {torch.unique(keep)}')
    indices = torch.nonzero(keep, as_tuple=False)
    
    results = []
    for idx in indices:
        _, _, z, y, x = idx.tolist() # [batch, ch, z, y, x]
        
        # --- 核心逻辑修改：处理 Offset ---
        if offset is not None:
            off_z = offset[0, 0, z, y, x].item()
            off_y = offset[0, 1, z, y, x].item()
            off_x = offset[0, 2, z, y, x].item()
        else:
            # 如果没提供 Offset (或模型没训练)，强制为 0
            off_z, off_y, off_x = 0, 0, 0
            
        # --- 核心逻辑修改：处理 Size ---
        if size is not None:
            d = size[0, 0, z, y, x].item()
        else:
            # 如果没提供 Size (或模型没训练)，强制为固定值以便可视化
            d = 10.0 # 默认 10mm 圆圈
            
        final_z = z + off_z
        final_y = y + off_y
        final_x = x + off_x
        
        score = heatmap[0, 0, z, y, x].item()
        
        results.append({"z": final_z, "y": final_y, "x": final_x, "d": d, "score": score})
        
        tag = "GT" if is_ground_truth else "Pred"
        print(f"[{tag}] Found at: Z={final_z:.1f}, Y={final_y:.1f}, X={final_x:.1f}, Score={score:.2f}")
        
    return results

def visualize_comparison(vol, pred_nodules, gt_nodules, save_name="result.png"):
    """
    可视化对比：GT (红色) vs Pred (绿色)
    """
    # 收集所有需要绘制的层
    z_indices = set()
    for n in pred_nodules: z_indices.add(int(round(n['z'])))
    for n in gt_nodules: z_indices.add(int(round(n['z'])))
    
    if not z_indices:
        print("没有检测到任何结节 (GT 或 Pred)，绘制中间层。")
        z_indices.add(vol.shape[0] // 2)

    # 遍历每一层进行绘制
    for z_slice in z_indices:
        if z_slice < 0 or z_slice >= vol.shape[0]: continue

        plt.figure(figsize=(10, 10))
        plt.imshow(vol[z_slice], cmap='gray')
        plt.title(f"Comparison Result (Slice Z={z_slice})")
        
        ax = plt.gca()
        
        # 1. 画 Ground Truth (红色实线)
        gt_hit = False
        for n in gt_nodules:
            if abs(n['z'] - z_slice) < 2:
                # 
                circ = patches.Circle((n['x'], n['y']), n['d']/2, linewidth=2, edgecolor='r', facecolor='none', label='GT')
                ax.add_patch(circ)
                gt_hit = True

        # 2. 画 Prediction (绿色虚线)
        pred_hit = False
        for n in pred_nodules:
            if abs(n['z'] - z_slice) < 2:
                 # 
                circ = patches.Circle((n['x'], n['y']), n['d']/2, linewidth=2, edgecolor='#00FF00', linestyle='--', facecolor='none', label='Pred')
                ax.add_patch(circ)
                ax.text(n['x'], n['y']-n['d']/2-2, f"{n['score']:.2f}", color='#00FF00', fontsize=10, fontweight='bold')
                pred_hit = True
        
        # 处理图例 (去重)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='upper right')

        save_path = f"{save_name.replace('.png', '')}_slice{z_slice}.png"
        plt.savefig(save_path)
        print(f"已保存切片: {save_path}")
        plt.close()

# ==============================================================================
# 2. 主流程
# ==============================================================================
def main():
    device = torch.device(CONFIG["device"])
    
    # --- A. 加载模型 ---
    print("正在加载模型...")
    text_processor = LanguageProcessor().to(device)
    processText = ProcessText(text_processor, keys=["text"], max_len=60)
    
    model = UnetM(text_processor, batch_size=1).to(device)
    
    checkpoint = torch.load(CONFIG["checkpoint"], map_location=device)
    if 'state_dict' in checkpoint:
        state_dict = {k.replace("module.", ""): v for k, v in checkpoint['state_dict'].items()}
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    
    inferer = SlidingWindowInferer(
        roi_size=[CONFIG["roi_x"], CONFIG["roi_y"], CONFIG["roi_z"]],
        sw_batch_size=CONFIG["sw_batch_size"],
        overlap=CONFIG["overlap"],
        mode="gaussian"
    )

    # --- B. 数据预处理 (同时处理 Image 和 Label) ---
    print("正在加载和预处理数据...")
    
    # 1. 动态构建加载列表
    keys_to_load = ["img1", "img2"]
    if CONFIG["label_heatmap_path"]: keys_to_load.append("heatmap_label")
    if CONFIG["label_offset_path"]: keys_to_load.append("offset_label")
    if CONFIG["label_size_path"]: keys_to_load.append("size_label")
    
    # 2. 构建数据字典
    data_dict = {
        "img1": CONFIG["image1_path"], 
        "img2": CONFIG["image2_path"],
        "text": CONFIG["text_prompt"]
    }
    if CONFIG["label_heatmap_path"]: data_dict["heatmap_label"] = CONFIG["label_heatmap_path"]
    if CONFIG["label_offset_path"]: data_dict["offset_label"] = CONFIG["label_offset_path"]
    if CONFIG["label_size_path"]: data_dict["size_label"] = CONFIG["label_size_path"]

    # 3. 定义变换
    transforms = Compose([
        LoadImaged(keys=keys_to_load, allow_missing_keys=True),
        EnsureChannelFirstd(keys=keys_to_load, allow_missing_keys=True),
        
        # 你的 offset 清理逻辑
        Lambda(func=lambda d: {
            **d, 
            "offset_label": d["offset_label"].squeeze(-2) 
            if "offset_label" in d and d["offset_label"].shape[-2] == 1 
            else d.get("offset_label")
        }),
        # 如果 squeeze 导致 offset channel 跑没了 (变成3, D, H, W -> D, H, W)，需要再 EnsureChannelFirst
        # 这里的逻辑视具体数据情况而定，通常 EnsureChannelFirstd 在 Load 后就够了
        
        Orientationd(keys=keys_to_load, axcodes="RAS", allow_missing_keys=True),
        
        # Spacing: Image 用 bilinear, Label 用 bilinear (因为是热力图)
        Spacingd(keys=keys_to_load, pixdim=PREPROCESS_ARGS["pixdim"], mode="bilinear", allow_missing_keys=True),
        
        ScaleIntensityRanged(keys=["img1", "img2"], 
            a_min=PREPROCESS_ARGS["a_min"], a_max=PREPROCESS_ARGS["a_max"], 
            b_min=PREPROCESS_ARGS["b_min"], b_max=PREPROCESS_ARGS["b_max"], clip=True),
        
        processText,
        SpatialPadd(keys=keys_to_load, spatial_size=[CONFIG["roi_x"], CONFIG["roi_y"], CONFIG["roi_z"]]),
        ToTensord(keys=keys_to_load + ["text"], allow_missing_keys=True)
    ])

    data = transforms(data_dict)
    
    # --- C. 推理 (Inference) ---
    print("正在推理预测...")
    img1 = data["img1"].unsqueeze(0).to(device)
    img2 = data["img2"].unsqueeze(0).to(device)
    text = data["text"].unsqueeze(0).to(device)
    
    img1_pure = img1.as_tensor() if hasattr(img1, "as_tensor") else img1
    img2_pure = img2.as_tensor() if hasattr(img2, "as_tensor") else img2
    text_pure = text.as_tensor() if hasattr(text, "as_tensor") else text
    
    with torch.no_grad():
        def model_wrapper(inputs):
            current_bs = inputs.shape[0]
            batch_text = text_pure.expand(current_bs, -1)
            return model(inputs[:, 0:1], inputs[:, 1:2], batch_text)

        combined_img = torch.cat([img1_pure, img2_pure], dim=1)
        outputs = inferer(combined_img, model_wrapper)
        hm_pred, _, _ = outputs # 忽略 offset/size 输出

    # --- D. 解码与对比 ---
    print("正在解码结果...")

    # 1. 解码预测结果 (Prediction)
    # 强制不使用 Offset/Size，只看 Heatmap
    pred_nodules = decode_nodules(
        hm_pred, 
        offset=None, size=None, # 关键：传 None 强制为 0 和固定大小
        threshold=CONFIG["threshold"],
        is_ground_truth=False
    )
    
    # 2. 解码真值结果 (Ground Truth)
    gt_nodules = []
    if "heatmap_label" in data:
        print("发现 Label 数据，正在解析 Ground Truth...")
        gt_hm = data["heatmap_label"].unsqueeze(0).to(device)
        
        # 尝试获取 GT 的 offset 和 size，如果 CONFIG 里没填，就用 None
        gt_off = data["offset_label"].unsqueeze(0).to(device) if "offset_label" in data and data["offset_label"] is not None else None
        gt_size = data["size_label"].unsqueeze(0).to(device) if "size_label" in data and data["size_label"] is not None else None
        
        gt_nodules = decode_nodules(
            gt_hm, 
            offset=gt_off, size=gt_size, # 如果有 Label 文件就用，没有就用固定值
            threshold=0.9, # GT 的阈值通常高一点
            is_ground_truth=True
        )
    else:
        print("Warning: 未在 CONFIG 中配置 label_heatmap_path，无法绘制 Ground Truth。")

    # --- E. 绘图 ---
    print("正在绘图...")
    vol_numpy = img2_pure[0, 0].cpu().numpy()
    print (f'count of pred nodules: {len(pred_nodules)}')
    visualize_comparison(vol_numpy, pred_nodules, gt_nodules, save_name="viz_compare.png")

if __name__ == "__main__":
    main()