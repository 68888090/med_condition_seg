import matplotlib.pyplot as plt
import numpy as np
import os
from models.text_processor import LanguageProcessor
import argparse
from omegaconf import OmegaConf
from utils.data_process import get_loader
import torch

def check_data_loader(loader, save_dir="./debug_vis", num_batches=3):
    """
    可视化 DataLoader 的前 N 个 Batch
    Args:
        loader: 数据加载器
        save_dir: 保存路径
        num_batches: 想要查看多少个 Batch (默认 3 次)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"[*] Starting visualization for {num_batches} batches...")

    # 使用 enumerate 循环获取数据
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= num_batches:
            break  # 达到指定次数，退出

        print(f"\n=== Processing Batch {batch_idx + 1}/{num_batches} ===")

        # ================= 核心：一键清洗所有 MetaTensor =================
        # 你的原代码中漏掉了 offset_label 的转换，这里统一处理，防止报错
        batch_data = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor) or hasattr(v, "as_tensor"):
                # 剥离 MONAI 元数据，转为纯 PyTorch Tensor
                batch_data[k] = v.as_tensor() if hasattr(v, "as_tensor") else v.as_subclass(torch.Tensor)
            else:
                batch_data[k] = v
        # ==============================================================

        # 2. 打印基本 Tensor 形状和统计信息
        keys_to_check = ["ctImage1", "ctImage2", "heatmap_label", "size_label", "offset_label"]
        for key in keys_to_check:
            if key in batch_data:
                data = batch_data[key]
                print(f"Key: {key}, Shape: {data.shape}, Type: {data.dtype}")
                # 打印统计信息时，确保转换为了 float 以防格式问题
                print(f"    Min: {data.min():.4f}, Max: {data.max():.4f}, Mean: {data.mean():.4f}")
                
        if "text" in batch_data:
            print(f"Text data batch size: {len(batch_data['text'])}")

        # 3. 可视化切片 (默认取每个 Batch 的第 0 个样本)
        sample_idx = 0
        
        # 转换为 numpy 用于绘图
        img_ct1 = batch_data["ctImage1"][sample_idx][0].numpy()
        img_ct2 = batch_data["ctImage2"][sample_idx][0].numpy()
        heatmap = batch_data["heatmap_label"][sample_idx][0].numpy()
        
        # 寻找 Heatmap 最亮的切片位置，如果没有热点则取中间
        if heatmap.max() > 0:
            # 沿 Z 轴找最大值 (假设 shape 是 D, H, W)
            z_slice = np.argmax(np.max(heatmap, axis=(1, 2))) 
            tag = "MaxHeatmap"
        else:
            z_slice = img_ct1.shape[0] // 2
            print(f"[WARNING] Batch {batch_idx} Sample {sample_idx} has EMPTY heatmap! Showing middle slice.")
            tag = "MiddleSlice"

        print(f"[*] Visualizing slice Z={z_slice} ({tag})")

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        # CT Image 1
        axes[0].imshow(img_ct1[z_slice], cmap="gray")
        axes[0].set_title(f"CT1 (Slice {z_slice})")
        axes[0].axis('off')
        
        # CT Image 2
        axes[1].imshow(img_ct2[z_slice], cmap="gray")
        axes[1].set_title(f"CT2 (Slice {z_slice})")
        axes[1].axis('off')
        
        # Heatmap Label
        axes[2].imshow(heatmap[z_slice], cmap="jet")
        axes[2].set_title(f"Heatmap (Slice {z_slice})\nMax Val: {heatmap.max():.4f}")
        axes[2].axis('off')
        
        # Overlay (CT2 + Heatmap)
        axes[3].imshow(img_ct2[z_slice], cmap="gray")
        axes[3].imshow(heatmap[z_slice], cmap="jet", alpha=0.5) # 半透明叠加
        axes[3].set_title("Overlay")
        axes[3].axis('off')
        
        # 保存图片，文件名带上 batch 编号
        file_name = f"batch_{batch_idx}_debug.png"
        save_path = os.path.join(save_dir, file_name)
        plt.suptitle(f"Batch {batch_idx} - Sample {sample_idx} - Slice {z_slice}")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        print(f"[*] Saved visualization to {save_path}")

    print(f"[*] Done checking {num_batches} batches.")

def get_config():
    # 1. 设置命令行参数，只保留 config 路径和命令行覆盖功能
    parser = argparse.ArgumentParser(description="PyTorch Training with YAML")
    parser.add_argument("-c", "--config", default="config.yaml", help="Path to config file")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER, 
                        help="Modify config options from command line (e.g., training.epochs=200)")
    args = parser.parse_args()

    # 2. 加载 YAML
    conf = OmegaConf.load(args.config)

    # 3. 合并命令行参数 (允许 python train.py training.batch_size=4 覆盖 yaml)
    if args.opts:
        cli_conf = OmegaConf.from_cli(args.opts)
        conf = OmegaConf.merge(conf, cli_conf)

    # 4. [可选但推荐] 将结构化配置扁平化，为了兼容旧代码
    # 这样你可以继续使用 args.epochs 而不需要改成 args.training.epochs
    # 如果你希望代码更现代化，可以去掉这一步，然后修改后面代码的调用方式
    args_flat = flatten_config(conf)
    
    return args_flat

def flatten_config(conf):
    """
    将嵌套的 OmegaConf 配置展平成一层 argparse.Namespace。
    """
    # === 关键修复步骤 ===
    # 1. 将 OmegaConf 对象转换为标准的 Python 字典
    # resolve=True 确保如果是 ${...} 这种引用变量，会被计算出真实值
    conf_dict = OmegaConf.to_container(conf, resolve=True)

    flat_args = argparse.Namespace()

    def _extract_values(d):
        for key, value in d.items():
            # 现在只需要判断是否为标准 dict 即可
            if isinstance(value, dict):
                _extract_values(value)
            else:
                # 写入 Namespace
                setattr(flat_args, key, value)

    _extract_values(conf_dict)
    return flat_args


# ======= 使用方法 =======
# args, processor = ... (你原本的初始化代码)
# train_loader, val_loader = get_loader(args, processor)
# check_data_loader(train_loader) 
# exit() # 调试完退出，或者注释掉继续训练
def main():
    args = get_config()
    text_processor = LanguageProcessor().to(args.device)
    train_loader, test_loader = get_loader(args, text_processor)
    check_data_loader(train_loader, num_batches=50)
    exit()

if __name__ == "__main__":
    main()