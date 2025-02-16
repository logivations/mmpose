_base_ = ['../_base_/default_runtime.py']

num_keypoints = 4

# runtime
train_cfg = dict(max_epochs=300, val_interval=10)

# optimizer
optim_wrapper = dict(optimizer=dict(
    type='Adam',
    lr=5e-4,
))
# learning policy
param_scheduler = [
    dict(
        type='LinearLR', begin=0, end=500, start_factor=0.001,
        by_epoch=False),  # warm-up
    dict(
        type='MultiStepLR',
        begin=0,
        end=300,
        milestones=[200, 250],
        gamma=0.1,
        by_epoch=True)
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=64)


# hooks
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=1, save_best='coco/AP', rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    cdualization=dict(type='PoseVisualizationHook', enable=True),
    badcase=dict(
        type='BadCaseAnalysisHook',
        enable=False,
        out_dir='badcase',
        metric_type='loss',
        badcase_thr=5
    )
)

# codec settings
codec = dict(
    type='MSRAHeatmap', input_size=(224, 224), heatmap_size=(56, 56), sigma=2)

# model settings
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='ResNet',
        depth=18,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18'),
    ),
    head=dict(
        type='HeatmapHead',
        in_channels=512,
        out_channels=num_keypoints,
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=codec),
    test_cfg=dict(
        flip_test=True,
        flip_mode='heatmap',
        shift_heatmap=True,
    ))

# base dataset settings
dataset_type = 'CocoDataset'
data_mode = 'topdown'
data_root = 'data/new_dataset_14_02/'
work_dir = 'pallet_kp_models/new_dataset_16_02_with_adapted_flip_less_move/'
labels = ["top_left", "top_right", "bottom_left", "bottom_right"]


# pipelines
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale',padding=1.05),
    dict(type='RandomFlip', direction='horizontal'),  # TODO: ASK DOES IT NEEDED
    dict(
        type='RandomBBoxTransform',
        rotate_factor=7,
        rotate_prob=0.5,
        shift_factor=0.03,
        scale_factor=(0.9, 1.1)
    ),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type="RandomBottomHalf", threshold=0.4, p=0.5),
    dict(
        type='Albumentation',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=[0.8, 1.2],
                contrast=[0.8, 1.2],
                saturation=[0.8, 1.2],
                hue=[-0.5, 0.5],
                p=0.4
            ),

            dict(
                type='OneOf',
                transforms=[
                    dict(type='MotionBlur', blur_limit=3, p=0.3),
                    dict(type='MedianBlur', blur_limit=3, p=0.2),
                    dict(type='Blur', blur_limit=3, p=0.2),
                ], p=0.3),

            dict(
                type='OneOf',
                transforms=[
                    dict(type='GaussNoise', var_limit=(10.0, 50.0), p=0.3),
                    dict(type='MultiplicativeNoise', multiplier=(0.9, 1.1), p=0.3),
                ], p=0.4),

            # dict(type='HueSaturationValue', hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.3), # USE WITHOUT TrivialAugmentation
        ]),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs'),
    dict(type='TorchVisionWrapper', transforms=[
        dict(type='TrivialAugmentWide', num_magnitude_bins=31)              
    ], save=True),                                                  # TODO CHECK SAVING will save 100 images in /mmpose/test

]

val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale',padding=1.05),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]

# data loaders
train_dataloader = dict(
    batch_size=64,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            labels=labels,
            data_root=data_root,
            data_mode=data_mode,
            ann_file='coco/train.json',
            data_prefix=dict(img='images/'),
            pipeline=train_pipeline,
        )
    )
)

val_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
	    labels=labels,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='coco/val.json',
        bbox_file='',
        data_prefix=dict(img='images/'),
        test_mode=True,
        pipeline=val_pipeline,
    ))

test_dataloader = val_dataloader

# evaluators
val_evaluator = [
    dict(
        type='CocoMetric',
        ann_file=data_root + 'coco/val.json'
    ),
    dict(
        type='EPE',
    ),
    dict(
        type='PCKAccuracy',
        prefix="5pr_",
    ),
    dict(
        type='PCKAccuracy',
        thr=0.1,
        prefix="10pr_",
    ),
    dict(
        type='AUC',
    ),
]
test_evaluator = val_evaluator
