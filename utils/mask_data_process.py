'''
在这个数据处理文件中，是用来处理新的一批数据集的
数据集特点:   1.一个大小只有64*64*64mm^3    2.一个样本有一个结节    3.一个样本有对应的mask
模型输入要求: 1.首先是一个文本输入，有填充到对应的长度 2.然后是CT1的64*64*64的volume 3.然后是CT2的64*64*64的volume,
3.最后是CT2mask的64*64*64的volume

'''


from monai.data import CacheDataset, DataLoader, Dataset, DistributedSampler, SmartCacheDataset, load_decathlon_datalist
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    SpatialPadd,
    Resized,
    RandFlipd,
    RandRotate90d,
    ToTensord,
    Spacingd,
    CenterSpatialCropd,  
)
from utils.transform import ProcessText

def get_loader(args, language_processor):
    train_fold_jsonfile = args.train_path
    val_fold_jsonfile = args.val_path
    num_workers = args.workers
    
    # 加载 JSON 数据列表
    # 假设 JSON 中的 key 对应如下：
    # "ctImage1": CT1 路径
    # "ctImage2": CT2 路径
    # "label": CT2 Mask 路径 (如果是 mask，请确保 JSON key 是 label 或修改下方 keys)
    # "text": 文本描述
    datalist = load_decathlon_datalist(train_fold_jsonfile, True, "training")
    vali_datalist = load_decathlon_datalist(val_fold_jsonfile, True, "val")
    
    print("Dataset all training: number of data: {}".format(len(datalist)))
    print("Dataset all validation: number of data: {}".format(len(vali_datalist)))
    
    # 文本处理器 (保持不变)
    processText = ProcessText(language_processor, keys=["text"], max_len=256)
    
    # 定义目标尺寸 (64, 64, 64)
    target_roi_size = [args.roi_x, args.roi_y, args.roi_z] 
    # 或者直接用 args.roi_x 等，确保 args 传进来的是 64
    spacing = (args.space_x, args.space_y, args.space_z)

    # ================= 训练预处理 =================
    train_transforms = Compose(
        [
            LoadImaged(keys=["CT1_path", "CT2_path", "label"]),
            EnsureChannelFirstd(keys=["CT1_path", "CT2_path", "label"]),
            Orientationd(keys=["CT1_path", "CT2_path", "label"], axcodes="RAS"),

            # =================== 【核心修改：统一 Spacing】 ===================
            # 将所有数据的体素间距强制变为 1mm x 1mm x 1mm
            # 因为你的物理裁剪是 64mm，除以 1mm Spacing，结果自然就是 64个像素
            Spacingd(
                keys=["CT1_path", "CT2_path", "label"],
                pixdim=spacing, 
                # CT图像用双线性插值 (bilinear)，Label必须用最近邻插值 (nearest) 保持0/1整数
                mode=("bilinear", " bilinear", "nearest") 
            ),
            # ================================================================

            ScaleIntensityRanged(
                keys=["CT1_path", "CT2_path"], 
                a_min=args.a_min, a_max=args.a_max, 
                b_min=args.b_min, b_max=args.b_max, 
                clip=True
            ),
            processText,

            # =================== 【尺寸安全锁】 ===================
            # 重采样可能会因为浮点数计算出现 63 或 65 这种 1像素的误差
            # 所以一定要先 Pad 再 Crop，强制锁定为 64x64x64
            SpatialPadd(
                keys=["CT1_path", "CT2_path", "label"], 
                spatial_size=target_roi_size
            ),
            CenterSpatialCropd(
                keys=["CT1_path", "CT2_path", "label"],
                roi_size=target_roi_size
            ),
            # ====================================================

            # 数据增强
            RandFlipd(keys=["CT1_path", "CT2_path", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["CT1_path", "CT2_path", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["CT1_path", "CT2_path", "label"], prob=0.5, spatial_axis=2),
            RandRotate90d(keys=["CT1_path", "CT2_path", "label"], prob=0.5, max_k=3),
            
            ToTensord(keys=["CT1_path", "CT2_path", "label", "text", "satisfy"]),
        ]
    )
    
    # 验证集做同样的操作！
    val_transforms = Compose(
        [
            LoadImaged(keys=["CT1_path", "CT2_path", "label"]),
            EnsureChannelFirstd(keys=["CT1_path", "CT2_path", "label"]),
            Orientationd(keys=["CT1_path", "CT2_path", "label"], axcodes="RAS"),
            
            # 验证集也必须重采样，否则模型看不懂
            Spacingd(
                keys=["CT1_path", "CT2_path", "label"],
                pixdim=spacing,
                mode=("bilinear", "bilinear", "nearest")
            ),
            
            ScaleIntensityRanged(keys=["CT1_path", "CT2_path"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True), # 你的参数
            processText,
            
            # 安全锁
            SpatialPadd(keys=["CT1_path", "CT2_path", "label"], spatial_size=target_roi_size),
            CenterSpatialCropd(keys=["CT1_path", "CT2_path", "label"], roi_size=target_roi_size),
            
            ToTensord(keys=["CT1_path", "CT2_path", "label", "text", "satisfy"]),
        ]
    )

    # ================= Dataset & Loader (保持不变) =================
    if args.cache_dataset:
        print("Using MONAI Cache Dataset")
        train_ds = CacheDataset(data=datalist, transform=train_transforms, cache_rate=0.5, num_workers=num_workers)
    elif args.smartcache_dataset:
        print("Using MONAI SmartCache Dataset")
        train_ds = SmartCacheDataset(
            data=datalist,
            transform=train_transforms,
            replace_rate=1.0,
            cache_num=2 * args.batch_size * args.sw_batch_size, # 注意：如果没有 sw_batch_size，这里可能要改
        )
    else:
        print("Using generic dataset")
        train_ds = Dataset(data=datalist, transform=train_transforms)

    if args.distributed:
        train_sampler = DistributedSampler(dataset=train_ds, even_divisible=True, shuffle=True)
    else:
        train_sampler = None
    
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, num_workers=num_workers, sampler=train_sampler, drop_last=True
    )

    val_ds = Dataset(data=vali_datalist, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=num_workers, shuffle=False, drop_last=True)

    return train_loader, val_loader