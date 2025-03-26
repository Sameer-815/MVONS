_base_ = './pspnet_r50-d8_10k_histo.py'
model = dict(
    pretrained='F:/code/weakly-supervised/OEEM-main/segmentation/weights/res38d.pth',
    # pretrained='G:/train_pth/luad_vote_mask_0.75_1_1.25_1212/iter_20000.pth',
    # pretrained='G:/train_pth/bcss_vote_mask_0.75_1_1.25_0102/best_iter_19000.pth',
    backbone=dict(
        type='WideRes38'),
    decode_head=dict(
        in_channels=4096,
        loss_decode=dict(custom_str='oeem')),
        # loss_decode=dict()),
    auxiliary_head=dict(
        in_channels=1024,
        loss_decode=dict(custom_str='oeem')),
        # loss_decode=dict()),
)
# test_cfg = dict(mode='slide', crop_size=(320, 320), stride=(256, 256), crf=False)
# test_cfg = dict(mode='whole',crf=True)
img_norm_cfg = dict(mean=[187.5, 129.015, 176.592], std=[47.85, 57.923, 39.454], to_rgb=True)      #LUAD
# img_norm_cfg = dict(mean=[180.18, 121.68, 170.13],std=[50.73, 58.032, 44.703],to_rgb=True)      #BCSS
# img_norm_cfg = dict(mean=[206.50, 163.79, 206.99],std=[43.28, 52.45, 33.25],to_rgb=True)      #BCSS_10X
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(224, 224), ratio_range=(0.75, 3.)),
    dict(type='RandomCrop', crop_size=(224, 224)),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=(224, 224), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

# data_root = 'F:/data/data_all/weak_suprvised_data/BCSS-WSSS/BCSS-WSSS/'
data_root = 'F:/data/data_all/weak_suprvised_data/LUAD-HistoSeg/LUAD-HistoSeg/'
# data_root = 'F:/guidian/dataset/BCSS_10x/new/noise_train/'      #BCSS10X
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=1,
    train=dict(
        pipeline=train_pipeline,
        data_root=data_root,
        # img_dir = 'training/',
        # ann_dir = 'amm/bn7_0603',
        img_dir='train/img/',
        # ann_dir='train/noise_mask/',  #BCSS
        ann_dir='train_PM/vote_mask_0.75_1_1.25/',  #LUAD
        split='train.lst',
        ),
    val=dict(
        data_root=data_root,
        img_dir='val/img/',
        ann_dir='val/mask/',
        split='val.lst',
        ),
    test=dict(
        # data_root=data_root,
        # img_dir='test/img_nor_bcss40X/',
        # ann_dir='test/mask/',
        # data_root='F:/guidian/dataset/BCSS_10x/new/',
        img_dir = 'F:/data/data_all/weak_suprvised_data/LUAD-HistoSeg/LUAD-HistoSeg/WSI/wsi_cut/405723/3200_2240/patch/'
        # ann_dir = 'test/mask/',
        # split='test.lst',
        ))
#bash tools/dist_train.sh configs/pspnet_oeem/pspnet_wres38-d8_10k_histo.py 1 G:/train_pth/bcss_vote_mask_0.75_1_1.25_0102