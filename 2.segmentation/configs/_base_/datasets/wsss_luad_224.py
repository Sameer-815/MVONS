dataset_type = 'WSSSLUAD'
data_root = 'F:/data/data_all/weak_suprvised_data/LUAD-HistoSeg/LUAD-HistoSeg/'
# img_norm_cfg = dict(
#     mean=[187.5, 129.015, 176.592], std=[47.85, 57.923, 39.454], to_rgb=True) #LUAD
# img_norm_cfg = dict(
#     mean=[180.18, 121.68, 170.13],std=[50.73, 58.032, 44.703],to_rgb=True)    #BCSS
img_norm_cfg = dict(mean=[206.50, 163.79, 206.99],std=[43.28, 52.45, 33.25],to_rgb=True)      #BCSS_10X
crop_size = (224, 224)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(448, 224), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size,cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),             #PhotoMetricDistortion是做数据增强工作
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        # img_scale=(3360,2240),        #WSI测试
        img_scale = (224,224),
        img_ratios=[0.75, 1., 1.25, 1.5, 1.75, 2., 2.5, 3.],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),]

        )
]

data = dict(
    samples_per_gpu=61,     #batch_size
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        pipeline=train_pipeline,
        data_root=data_root,
        img_dir='training/',
        ann_dir='train_PM/PM_multi_scale_campp_bn7/',
        split='train.lst',
        ),
    val=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        data_root=data_root,
        img_dir='val/img/',
        ann_dir='val/mask/',
        split='val.lst',
        ),
    test=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        data_root=data_root,
        img_dir='test/img/',
        ann_dir='test/mask/',
        split='test.lst',
        ))
