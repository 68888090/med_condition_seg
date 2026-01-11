import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import SimpleITK as sitk
import os
import torch.nn.functional as F
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, 
    Spacingd, ScaleIntensityRanged, ToTensord,SpatialPadd,
)
from monai.inferers import SlidingWindowInferer
from utils.transform import ProcessText
# 引入你的模型定义
from unetM import UnetM
from models.text_processor import LanguageProcessor

# ==============================================================================
# 1. 用户配置区域 (修改这里即可)
# ==============================================================================
CONFIG = {
    "image1_path": "/home/lhr/dataset/CSTPLung/data/bbox_data/fixed_80_20190314.mhd",  # 替换为你的 CT 影像路径
    "image2_path": "/home/lhr/dataset/CSTPLung/data/bbox_data/80_20220905.mhd",  # 替换为对比影像 (如果没有，填一样的)
    "text_prompt": "Find nodules enlarged by -50", # 替换为你想测试的文本
    "test_json_data" : "/home/lhr/dataset/CSTPLung/data2.json",
    "checkpoint": "/home/lhr/dataset/checkpoints/swin-unetr/9_checkpoint_best.pth.tar", # 权重路径
    "roi_size": (64, 64, 64),       # 必须和训练时的 Patch Size 一致
    "roi_x": 64,
    "roi_y": 64,
    "roi_z": 64,
    "sw_batch_size": 4,             # 滑动窗口并行度，显存大可以调大
    "overlap": 0.5,                 # 滑动窗口重叠率
    "threshold": 0.3,               # 预测概率阈值 (大于此值才算结节)
    "device": "cuda:0"              # 显卡
}

# 图像预处理参数 (必须和训练 args 保持一致)
PREPROCESS_ARGS = {
    "a_min": -1000, "a_max": 1000,
    "b_min": 0.0,   "b_max": 1.0,
    "pixdim": (1.0, 1.0, 1.0) # 训练时统一的 Spacing
}

# ==============================================================================
# 2. 辅助函数：解码与画图,填充球体
# ==============================================================================
def decode_and_print(heatmap, size, threshold=0.1):
    """从热力图中解析坐标"""
    heatmap = torch.sigmoid(heatmap) # 确保归一化
    print(heatmap.unique())
    hmax = F.max_pool3d(heatmap, kernel_size=3, padding=1, stride=1)
    print(hmax.unique())
    epsilon = 1.1e-4 
    
    # 逻辑：如果 (heatmap >= hmax - epsilon)，我们就认为它是峰值
    keep = (heatmap >= (hmax - epsilon)) & (heatmap > threshold)
    # 初始化 sphere_mask_tensor
    sphere_mask_tensor = torch.zeros_like(heatmap, dtype=torch.bool)
    indices = torch.nonzero(keep, as_tuple=False)
    
    results = []
    for idx in indices:
        _, _, x, y, z = idx.tolist() # batch, ch, z, y, x
        
        # 加上回归的偏移量
        # off_z = offset[0, 0, z, y, x].item() # 假设 offset 是 (1, 3, D, H, W)
        # off_y = offset[0, 1, z, y, x].item()
        # off_x = offset[0, 2, z, y, x].item()
        d = size[0, 0, x, y, z].item()
        score = heatmap[0, 0, x, y, z].item()
        
        # 最终坐标 (浮点数)
        final_z = z 
        final_y = y 
        final_x = x 
        center_radius = ((z, y, x), d/2)
        fill_sphere_ras(sphere_mask_tensor, center_radius)
        
        results.append({"z": final_z, "y": final_y, "x": final_x, "d": d, "score": score, "iz": z})
        print(f"检测到结节: Z={final_z:.1f}, Y={final_y:.1f}, X={final_x:.1f}, 直径={d:.1f}, 置信度={score:.2f}")
        # 获得了总填充球形体
    keep = sphere_mask_tensor
    return results , keep 

def visualize_slice(vol, nodules, save_name="result.png"):
    """只画出有结节的层"""
    if not nodules:
        print("未检测到结节，不生成图像。")
        return

    # 找出置信度最高的结节所在的层
    best_nodule = max(nodules, key=lambda x: x['score'])
    z_slice = int(round(best_nodule['z']))
    
    # 防止越界
    z_slice = max(0, min(z_slice, vol.shape[0]-1))

    plt.figure(figsize=(10, 10))
    plt.imshow(vol[z_slice], cmap='gray')
    plt.title(f"Detection Result (Slice Z={z_slice})")
    
    ax = plt.gca()
    for n in nodules:
        # 如果这个结节在当前层附近 (上下浮动2层)，就画出来
        if abs(n['z'] - z_slice) < 2:
            # 画圆
            circ = patches.Circle((n['x'], n['y']), n['d']/2, linewidth=2, edgecolor='g', facecolor='none')
            ax.add_patch(circ)
            ax.text(n['x'], n['y']-n['d']/2-5, f"{n['score']:.2f}", color='g', fontweight='bold')
    
    plt.savefig(save_name)
    print(f"可视化结果已保存至: {save_name}")
    plt.close()



def fill_sphere_ras(mask_tensor, center_radius):
    """
    在 RAS 格式的 3D bool tensor 中填充一个球体。
    
    参数:
        mask_tensor: [R, A, S] 或 [B, C, R, A, S] 的 bool tensor
        center: (r, a, s) 格式的中心点坐标 (即 x, y, z)
        radius: 球体的半径
    """
    center, radius = center_radius
    # 获取最后三个维度的长度：Right(x), Anterior(y), Superior(z)
    len_r, len_a, len_s = mask_tensor.shape[-3:]
    cr, ca, cs = center # center r, center a, center s
    rad = int(radius)
    
    # 1. 计算局部包围盒 (Local Bounding Box)
    # 维度 0: Right (X)
    r_min = max(0, cr - rad)
    r_max = min(len_r, cr + rad + 1)
    
    # 维度 1: Anterior (Y)
    a_min = max(0, ca - rad)
    a_max = min(len_a, ca + rad + 1)
    
    # 维度 2: Superior (Z)
    s_min = max(0, cs - rad)
    s_max = min(len_s, cs + rad + 1)
    
    # 边界检查
    if r_min >= r_max or a_min >= a_max or s_min >= s_max:
        return

    # 2. 生成局部网格 (Local Grid)
    # 注意这里生成的坐标轴顺序是 r, a, s
    lr = torch.arange(r_min, r_max, device=mask_tensor.device, dtype=torch.float32)
    la = torch.arange(a_min, a_max, device=mask_tensor.device, dtype=torch.float32)
    ls = torch.arange(s_min, s_max, device=mask_tensor.device, dtype=torch.float32)
    
    # meshgrid 生成 3D 网格 (indexing='ij' 保证维度顺序不乱)
    # grid_r 对应 dim0, grid_a 对应 dim1, grid_s 对应 dim2
    grid_r, grid_a, grid_s = torch.meshgrid(lr, la, ls, indexing='ij')
    
    # 3. 计算欧氏距离平方
    # (r - cr)^2 + (a - ca)^2 + (s - cs)^2
    dist_sq = (grid_r - cr)**2 + (grid_a - ca)**2 + (grid_s - cs)**2
    
    # 4. 生成 Mask
    sphere_mask = dist_sq <= (radius ** 2)
    
    # 5. 填充回原 Tensor
    # 对应顺序：[..., r_range, a_range, s_range]
    if mask_tensor.ndim == 5: # [B, C, R, A, S]
        mask_tensor[0, 0, r_min:r_max, a_min:a_max, s_min:s_max] |= sphere_mask
    elif mask_tensor.ndim == 3: # [R, A, S]
        mask_tensor[r_min:r_max, a_min:a_max, s_min:s_max] |= sphere_mask

# ==============================================================================
# 通过判断label里面是否有满足条件的结节
# ==============================================================================
def find_suit_data(origin) -> list:
    import json
    label_lists = []
    with open(CONFIG["test_json_data"], "r") as f:
        data = json.load(f)
    import re 
    kind_list = ("enlarged", "smaller", "new")
    condition = re.findall(r"-?\d+\.\d+|-?\d+", CONFIG["text_prompt"])
    if len(condition) == 0:
        kind = kind_list[-1]
    elif len(condition) == 1:
        kind = kind_list[0] if float(condition[0]) > 0 else kind_list[1]
    
    for item in data["training"]:
        if item["ctImage2"] == CONFIG["image2_path"] and item["ctImage1"] == CONFIG["image1_path"]:
            label_lists.append(item["label"])
    m_lable_lists = []
    for label in label_lists:
        num = re.findall(r"-?\d+\.\d+|-?\d+", label[-1])
        if len(num) == 0 and kind == kind_list[-1]:
            #筛选条件是没有数字，且是新结节
            m_lable_lists.append(label)
        elif len(num) == 1:
            if kind == kind_list[0] and float(num[0]) > float(condition[0]):
                # 筛选条件是有数字，且是 enlargement 结节，且数字大于阈值
                m_lable_lists.append(label)
            elif kind == kind_list[1] and float(num[0]) < float(condition[0]):
                # 筛选条件是有数字，且是 smaller 结节，且数字小于阈值
                m_lable_lists.append(label)
    m_voxel_list = []
    def extract_voxel_form_world(coodinate, origin, spacing):
        ids_x = int(np.round((coodinate[0] - origin[0]) / spacing[0]))
        ids_y = int(np.round((coodinate[1] - origin[1]) / spacing[1]))
        ids_z = int(np.round((coodinate[2] - origin[2]) / spacing[2]))
        return ids_x, ids_y, ids_z
    for m_item in m_lable_lists:
        voxel_coordinate = extract_voxel_form_world(
            m_item[1:4],
            origin,
            np.array([float(CONFIG["spacing_x"]), float(CONFIG["spacing_y"]), float(CONFIG["spacing_z"])]),
        )
        m_voxel_list.append((voxel_coordinate, m_item[4]))
        
    return m_voxel_list #返回一个带有体素的列表

# ==============================================================================
# 计算召回率
# ==============================================================================
def compute_Recall(m_lable_lists, data_keep):
    # 按照预测tensor填充对应的gttensor后续计算
    gt_tensor = torch.zeros_like(data_keep, dtype = torch.bool)
    for item in m_lable_lists:
        fill_sphere_ras(gt_tensor, item)
    
    pred = data_keep.bool()
    gt = gt_tensor.bool()

    # 2. 计算基本集合 (Intersection & Union)
    # TP (True Positive): 预测是1，真实也是1
    intersection = (pred & gt).sum().float()
    
    # Union: 预测是1 或者 真实是1
    union = (pred | gt).sum().float()

    pred_area = pred.sum().float()
    gt_area = gt.sum().float()

    # ---------------------------
    # 3. 计算指标
    # ---------------------------
    
    # IoU
    iou = (intersection + 1e-6) / (union + 1e-6)
    
    # Dice Coefficient
    dice = (2 * intersection + 1e-6) / (pred_area + gt_area + 1e-6)
    
    # Pixel-level Recall (Sensitivity)
    # 分母是 GT 的总像素数
    recall = (intersection + 1e-6) / (gt_area + 1e-6)
    
    # Pixel-level Precision
    # 分母是 Pred 的总像素数
    precision = (intersection + 1e-6) / (pred_area + 1e-6)
    
    metrics = {
        "IoU": iou.item(),
        "Dice": dice.item(),
        "Recall_Pixel": recall.item(),
        "Precision_Pixel": precision.item(),
        "AUC": None
    }

    return metrics, gt # 最后返回指标与后续继续参与AUC计算
    
# ==============================================================================
# 计算AUC
# ==============================================================================
def compute_AUC(heatmap, gt , metrics):
    if raw_heatmap is not None:
        # 必须把 3D tensor 展平为 1D 数组
        # 注意：这步操作在显存中可能非常大，建议 .cpu() 后处理
        
        # 确保 heatmap 和 mask 形状一致
        if raw_heatmap.shape != gt.shape:
            # 有时 heatmap 是 [1,1,D,H,W]，mask 是 [D,H,W]
            raw_heatmap = raw_heatmap.view(gt.shape)
        y_true = gt.detach().cpu().numpy().flatten()
        y_scores = raw_heatmap.detach().cpu().numpy().flatten()
        
        # 为了速度，可以只采样一部分背景像素 (因为背景太多了)
        # 但严谨计算需要全部像素
        try:
            val_auc = roc_auc_score(y_true, y_scores)
            metrics["AUC"] = val_auc
        except ValueError:
            # 如果 GT 全是 0 (没有结节)，AUC 无法计算
            metrics["AUC"] = 0.0
    return metrics

# ==============================================================================
# 3. 主流程
# ==============================================================================
def main():
    device = torch.device(CONFIG["device"])
    
    # --- A. 加载模型 ---
    print("正在加载模型...")
    text_processor = LanguageProcessor().to(device)
    # 初始化 ProcessText Transform
    processText = ProcessText(text_processor, keys=["text"], max_len=60)
    
    model = UnetM(text_processor, batch_size=1).to(device)
    
    checkpoint = torch.load(CONFIG["checkpoint"], map_location=device)
    # 兼容 DDP 和普通保存的权重
    if 'state_dict' in checkpoint:
        state_dict = {k.replace("module.", ""): v for k, v in checkpoint['state_dict'].items()}
    else:
        state_dict = checkpoint # 有时候直接保存的是 state_dict
        
    model.load_state_dict(state_dict)
    model.eval()
    
    inferer = SlidingWindowInferer(
        roi_size=[CONFIG["roi_x"], CONFIG["roi_y"], CONFIG["roi_z"]],
        sw_batch_size=CONFIG["sw_batch_size"],
        overlap=CONFIG["overlap"],
        mode="gaussian"
    )

    # --- B. 数据预处理 ---
    print("正在处理数据...")
    transforms = Compose([
        LoadImaged(keys=["img1", "img2"]),
        EnsureChannelFirstd(keys=["img1", "img2"]),
        Orientationd(keys=["img1", "img2"], axcodes="RAS"),
        # 建议开启 Spacingd 以匹配训练分布，如果你训练没开这行，这里也注释掉
        Spacingd(keys=["img1", "img2"], pixdim=PREPROCESS_ARGS["pixdim"], mode="bilinear"),
        ScaleIntensityRanged(
            keys=["img1", "img2"], 
            a_min=PREPROCESS_ARGS["a_min"], a_max=PREPROCESS_ARGS["a_max"], 
            b_min=PREPROCESS_ARGS["b_min"], b_max=PREPROCESS_ARGS["b_max"], 
            clip=True
        ),
        processText, # 调用上面定义的类
        SpatialPadd(keys=["img1", "img2"], spatial_size=[CONFIG["roi_x"], CONFIG["roi_y"], CONFIG["roi_z"]]),
        ToTensord(keys=["img1", "img2", "text"])
    ])

    data = {
        "img1": CONFIG["image1_path"], 
        "img2": CONFIG["image2_path"],
        "text": CONFIG["text_prompt"]
    }
    data = transforms(data)
    
    # 增加 Batch 维度 [C, D, H, W] -> [1, C, D, H, W]
    img1 = data["img1"].unsqueeze(0).to(device)
    img2 = data["img2"].unsqueeze(0).to(device)
    text = data["text"].unsqueeze(0).to(device) # [1, max_len]

    # --- C. 推理 (Inference) ---
    print("正在进行滑动窗口推理...")
    
    # 1. 剥离 MetaTensor (防止 SwinUNETR 报错)
    img1_pure = img1.as_tensor() if hasattr(img1, "as_tensor") else img1
    img2_pure = img2.as_tensor() if hasattr(img2, "as_tensor") else img2
    text_pure = text.as_tensor() if hasattr(text, "as_tensor") else text
    
    with torch.no_grad():
        # 2. 定义 Wrapper (修复了你之前的报错)
        def model_wrapper(inputs):
            # inputs shape: [sw_batch_size, 2, D, H, W] (这是滑动窗口切出来的 Batch)
            
            # 获取当前切片的 batch size (通常等于 sw_batch_size，但在边缘可能变小)
            current_bs = inputs.shape[0]
            
            # 【关键修复】将文本扩展到与图像 patch 相同的 batch size
            # text_pure: [1, max_len] -> [current_bs, max_len]
            batch_text = text_pure.expand(current_bs, -1)
            
            tuple1, tuple2 = model(inputs[:, 0:1], inputs[:, 1:2], batch_text)
            # 拆分通道送入模型
            return tuple1

        # 3. 拼接输入 (Hack for SlidingWindowInferer)
        combined_img = torch.cat([img1_pure, img2_pure], dim=1) # [1, 2, D, H, W]
        
        # 4. 执行推理
        outputs = inferer(combined_img, model_wrapper)
        
        hm_pred, off_pred, size_pred = outputs
    sigmoid_fun = nn.Sigmoid()
    hm_pred = sigmoid_fun(hm_pred)
    print(f'hm_pred unique: {hm_pred.unique()}')
    print(f'off_pred unique: {off_pred.unique()}')
    print(f'size_pred unique: {size_pred.unique()}')

    # --- D. 后处理 ---
    print("推理完成，正在解码...")
    nodules = decode_and_print(hm_pred, off_pred, size_pred, threshold=CONFIG["threshold"])
    
    # 取第一张图的第一个通道用于画图
    vol_numpy = img1_pure[0, 0].cpu().numpy()
    visualize_slice(vol_numpy, nodules, save_name="single_test_result.png")

if __name__ == "__main__":
    main()