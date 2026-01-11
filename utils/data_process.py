'''
必须在这里处理文本问答的数据集，因为在后面处理的话batch都连在一起了，
'''
from monai.data import CacheDataset, DataLoader, Dataset, DistributedSampler, SmartCacheDataset, load_decathlon_datalist
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandSpatialCropSamplesd,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
    ToTensord,
    Lambda,
    RandCropByPosNegLabeld,
    RandWeightedCropd
)
from utils.transform import ProcessText

def get_loader(args, language_processor):
    jsonfile = args.jsonlist
    num_workers = args.workers
    datalist = load_decathlon_datalist(jsonfile, True, "training")
    vali_datalist = load_decathlon_datalist(jsonfile, True, "validation")
    print("Dataset all training: number of data: {}".format(len(datalist)))
    print("Dataset all validation: number of data: {}".format(len(vali_datalist)))
    processText = ProcessText(language_processor, keys=["text"], max_len=60)
    train_transforms = Compose(
        [
            LoadImaged(keys=["ctImage1", "ctImage2", 'heatmap_label', 'offset_label', 'size_label']),
            # ================== 核心修复：清理幽灵维度 ==================
        # 针对 offset_label: 如果形状里有那个讨厌的 1 (例如 [D, H, W, 1, 3])，把它挤掉
            Lambda(func=lambda d: {
                **d, 
                "offset_label": d["offset_label"].squeeze(-2) if d["offset_label"].shape[-2] == 1 else d["offset_label"]
            }),
            # --- 调试步骤 1: 刚读进来长什么样？ ---
            # Lambda(func=lambda x: print(f"\n[DEBUG] Raw Loaded Shapes:") or x),
            # Lambda(func=lambda x: print(f"  Offset: {x['offset_label'].shape}") or x),
            # Lambda(func=lambda x: print(f"  Heatmap: {x['heatmap_label'].shape}") or x),
            EnsureChannelFirstd(keys=["ctImage1", "ctImage2", 'heatmap_label', 'size_label']),
            EnsureChannelFirstd(keys=['offset_label'], channel_dim=-1),
            # Lambda(func=lambda x: print(f"[DEBUG] Before Orientationd Shapes:") or x),
            # Lambda(func=lambda x: print(f"  Offset: {x['offset_label'].shape}") or x),
            # Lambda(func=lambda x: print(f"  Heatmap: {x['heatmap_label'].shape}") or x),
            Orientationd(keys=["ctImage1", "ctImage2", 'heatmap_label', 'offset_label', 'size_label'], axcodes="RAS"),

            ScaleIntensityRanged(
                keys=["ctImage1", "ctImage2"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            processText,
            SpatialPadd(keys=["ctImage1", "ctImage2", 'heatmap_label', 'offset_label', 'size_label'], spatial_size=[args.roi_x, args.roi_y, args.roi_z]),

            # CropForegroundd(keys=["ctImage1", "ctImage2", 'label_image'], source_key="ctImage1", k_divisible=[args.roi_x, args.roi_y, args.roi_z]),
            
            # RandSpatialCropSamplesd(
            #     keys=["ctImage1", "ctImage2", 'heatmap_label', 'offset_label', 'size_label'],
            #     roi_size=[args.roi_x, args.roi_y, args.roi_z],
            #     num_samples=args.sw_batch_size,
            #     random_center=True,
            #     random_size=False,
            # ),
            # RandCropByPosNegLabeld(
            #     keys=["ctImage1", "ctImage2", 'heatmap_label', 'offset_label', 'size_label'],
            #     label_key="heatmap_label",     # 根据 heatmap 来判断哪里是正样本
            #     spatial_size=[args.roi_x, args.roi_y, args.roi_z],
            #     pos=1,                   # 正样本比例 (必须包含 heatmap > 0 的区域)
            #     neg=1,                   # 负样本比例 (背景)
            #     num_samples=2,           # 每次切 2 个块      # 如果 heatmap 尺寸和 image 不一致时需要这个
            #     image_threshold=0,       # 这里的 threshold 是针对 label_key 的
            #     # 注意：因为你的 heatmap 是高斯分布 (0~1)，我们需要告诉它大于多少算"Pos"
            #     # 但 MONAI 这个函数通常针对 0/1 Mask。
            #     # 技巧：CenterNet heatmap 中心是 1.0，所以这能工作。
            #     # 如果不放心，可以增加一个临时的 mask 通道专门用于 Crop。
            # ),
            RandWeightedCropd(
            keys=["ctImage1", "ctImage2", 'heatmap_label', 'offset_label', 'size_label'],
            w_key="heatmap_label",          # 核心：使用 heatmap 作为概率权重图
            spatial_size=[args.roi_x, args.roi_y, args.roi_z],
            num_samples=args.sw_batch_size,
        ),
            ToTensord(keys=["ctImage1", "ctImage2",'text', 'heatmap_label', 'offset_label', 'size_label']),
        ]
    )
    '''
    所以关于CropForegroundd切割前景函数还是需要配准才行，因为CT图像的前景可能不是全体的，所以需要配准到全体上
    '''
    val_transforms = Compose(
        [
            LoadImaged(keys=["ctImage1", "ctImage2", 'heatmap_label', 'offset_label', 'size_label']),
            Lambda(func=lambda d: {
                **d, 
                "offset_label": d["offset_label"].squeeze(-2) if d["offset_label"].shape[-2] == 1 else d["offset_label"]
            }),
            EnsureChannelFirstd(keys=["ctImage1", "ctImage2", 'heatmap_label', 'size_label'],channel_dim="no_channel"),
            EnsureChannelFirstd(keys=['offset_label'], channel_dim=-1),
            Orientationd(keys=["ctImage1", "ctImage2", 'heatmap_label', 'offset_label', 'size_label'], axcodes="RAS"),
            
            ScaleIntensityRanged(
                keys=["ctImage1", "ctImage2"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True 
            ),
            processText,
            SpatialPadd(keys=["ctImage1", "ctImage2", 'heatmap_label', 'offset_label', 'size_label'], spatial_size=[args.roi_x, args.roi_y, args.roi_z]),
            # CropForegroundd(keys=["ctImage1", "ctImage2", 'heatmap_label', 'offset_label', 'size_label'], source_key="ctImage1", k_divisible=[args.roi_x, args.roi_y, args.roi_z]),
            # RandSpatialCropSamplesd(
            #     keys=["ctImage1", "ctImage2", 'heatmap_label', 'offset_label', 'size_label'],
            #     roi_size=[args.roi_x, args.roi_y, args.roi_z],
            #     num_samples=args.sw_batch_size,
            #     random_center=True,
            #     random_size=False,
            # ),
            # RandCropByPosNegLabeld(
            #     keys=["ctImage1", "ctImage2", 'heatmap_label', 'offset_label', 'size_label'],
            #     label_key="heatmap_label",     # 根据 heatmap 来判断哪里是正样本
            #     spatial_size=[args.roi_x, args.roi_y, args.roi_z],
            #     pos=1,                   # 正样本比例 (必须包含 heatmap > 0 的区域)
            #     neg=1,                   # 负样本比例 (背景)
            #     num_samples=2,           # 每次切 2 个块      # 如果 heatmap 尺寸和 image 不一致时需要这个
            #     image_threshold=0,       # 这里的 threshold 是针对 label_key 的
            #     # 注意：因为你的 heatmap 是高斯分布 (0~1)，我们需要告诉它大于多少算"Pos"
            #     # 但 MONAI 这个函数通常针对 0/1 Mask。
            #     # 技巧：CenterNet heatmap 中心是 1.0，所以这能工作。
            #     # 如果不放心，可以增加一个临时的 mask 通道专门用于 Crop。
            # ),
            RandWeightedCropd(
            keys=["ctImage1", "ctImage2", 'heatmap_label', 'offset_label', 'size_label'],
            w_key="heatmap_label",          # 核心：使用 heatmap 作为概率权重图
            spatial_size=[args.roi_x, args.roi_y, args.roi_z],
            num_samples=args.sw_batch_size,
        ),
            ToTensord(keys=["ctImage1", "ctImage2",'text', 'heatmap_label', 'offset_label', 'size_label']),
        ]
    )
    # train_transforms = Compose(
    #     [
    #         LoadImaged(keys=["ctImage1", "ctImage2"]),
    #         EnsureChannelFirstd(keys=["ctImage1", "ctImage2"]),
    #         Orientationd(keys=["ctImage1", "ctImage2"], axcodes="RAS"),
    #         # Spacingd(
    #         # keys=["ctImage1", "ctImage2", 'label_image'],
    #         # pixdim=(1.0, 1.0, 1.0), 
    #         # mode=("bilinear", "bilinear", "bilinear"), # CT用双线性插值，Label(如果是Mask)必须用最近邻
    #         # ),
    #         ScaleIntensityRanged(
    #             keys=["ctImage1", "ctImage2"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
    #         ),
    #         processText,
    #         SpatialPadd(keys=["ctImage1", "ctImage2"], spatial_size=[args.roi_x, args.roi_y, args.roi_z]),

    #         # CropForegroundd(keys=["ctImage1", "ctImage2", 'label_image'], source_key="ctImage1", k_divisible=[args.roi_x, args.roi_y, args.roi_z]),
            
    #         RandSpatialCropSamplesd(
    #             keys=["ctImage1", "ctImage2"],
    #             roi_size=[args.roi_x, args.roi_y, args.roi_z],
    #             num_samples=args.sw_batch_size,
    #             random_center=True,
    #             random_size=False,
    #         ),
    #         ToTensord(keys=["ctImage1", "ctImage2",'text']),
    #     ]
    # )
    # '''
    # 所以关于CropForegroundd切割前景函数还是需要配准才行，因为CT图像的前景可能不是全体的，所以需要配准到全体上
    # '''
    # val_transforms = Compose(
    #     [
    #         LoadImaged(keys=["ctImage1", "ctImage2", 'label_image']),
    #         EnsureChannelFirstd(keys=["ctImage1", "ctImage2", 'label_image']),
    #         Orientationd(keys=["ctImage1", "ctImage2", 'label_image'], axcodes="RAS"),
    #         # Spacingd(
    #         # keys=["ctImage1", "ctImage2", 'label_image'],
    #         # pixdim=(1.0, 1.0, 1.0), 
    #         # mode=("bilinear", "bilinear", "bilinear", "bilinear", "bilinear"), # CT用双线性插值，Label(如果是Mask)必须用最近邻
    #         # ),
    #         ScaleIntensityRanged(
    #             keys=["ctImage1", "ctImage2"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True 
    #         ),
    #         processText,
    #         SpatialPadd(keys=["ctImage1", "ctImage2"], spatial_size=[args.roi_x, args.roi_y, args.roi_z]),
    #         # CropForegroundd(keys=["ctImage1", "ctImage2", 'label_image'], source_key="ctImage1", k_divisible=[args.roi_x, args.roi_y, args.roi_z]),
    #         RandSpatialCropSamplesd(
    #             keys=["ctImage1", "ctImage2"],
    #             roi_size=[args.roi_x, args.roi_y, args.roi_z],
    #             num_samples=args.sw_batch_size,
    #             random_center=True,
    #             random_size=False,
    #         ),
    #         ToTensord(keys=["ctImage1", "ctImage2", 'text']),
    #     ]
    # )

    
    if args.cache_dataset:
        print("Using MONAI Cache Dataset")
        train_ds = CacheDataset(data=datalist, transform=train_transforms, cache_rate=0.5, num_workers=num_workers)
    elif args.smartcache_dataset:
        print("Using MONAI SmartCache Dataset")
        train_ds = SmartCacheDataset(
            data=datalist,
            transform=train_transforms,
            replace_rate=1.0,
            cache_num=2 * args.batch_size * args.sw_batch_size,
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

