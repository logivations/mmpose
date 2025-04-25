
## Config for keypoint detection with ResNet-18 and 192x256 input resolution

_base_ = ['../../../_base_/default_runtime.py']

# Model
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
    ),
    backbone=dict(
        type='ResNet',
        depth=18,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18'),
    ),
    head=dict(
        type='HeatmapHead',
        in_channels=512,
        out_channels=7,  # Number of keypoints
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=dict(
            type='MSRAHeatmap',
            input_size=(192, 256),
            heatmap_size=(48, 64),
            sigma=2,
        ),
    ),
    test_cfg=dict(flip_test=True, flip_mode='heatmap', shift_heatmap=True),
)

# Dataset
data_root = '/data/unzipped_export_el/exported_data/'
dataset_type = 'CocoDataset'
data_mode = 'topdown'

train_dataloader = dict(
    batch_size=64,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/forklift_keypoints_train2017.json',
        data_prefix=dict(img='train2017/'),
        data_mode=data_mode,
        pipeline=[
            dict(type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(type='RandomFlip', direction='horizontal'),
            dict(type='RandomBBoxTransform'),
            dict(type='TopdownAffine', input_size=(192, 256)),
            dict(
                type='Albumentation',
                transforms=[
                    dict(type='RandomBrightnessContrast', brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.4),
                    dict(
                        type='OneOf',
                        p=0.3,
                        transforms=[
                            dict(type='MotionBlur', blur_limit=3, p=0.3),
                            dict(type='MedianBlur', blur_limit=3, p=0.2),
                            dict(type='Blur', blur_limit=3, p=0.2),
                        ],
                    ),
                    dict(
                        type='OneOf',
                        p=0.4,
                        transforms=[
                            dict(type='GaussNoise', var_limit=(10.0, 50.0), p=0.3),
                            dict(type='MultiplicativeNoise', multiplier=(0.9, 1.1), p=0.3),
                        ],
                    ),
                    dict(type='HueSaturationValue', hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.3),
                ],
            ),
            dict(
                type='GenerateTarget',
                encoder=dict(
                    type='MSRAHeatmap',
                    input_size=(192, 256),
                    heatmap_size=(48, 64),
                    sigma=2,
                ),
            ),
            dict(type='PackPoseInputs'),
        ],
        labels=['C_Fork', 'L_Fork', 'R_Fork', 'front_left', 'front_right', 'rear_left', 'rear_right'],
    ),
)

val_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/forklift_keypoints_val2017.json',
        data_prefix=dict(img='val2017/'),
        data_mode=data_mode,
        test_mode=True,
        pipeline=[
            dict(type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(type='TopdownAffine', input_size=(192, 256)),
            dict(type='PackPoseInputs'),
        ],
        labels=['C_Fork', 'L_Fork', 'R_Fork', 'front_left', 'front_right', 'rear_left', 'rear_right'],
    ),
)

# Evaluation
val_evaluator = [
    dict(type='CocoMetric', ann_file=f'{data_root}/annotations/forklift_keypoints_val2017.json'),
    dict(type='EPE'),
    dict(type='PCKAccuracy', prefix='5pr_'),
    dict(type='PCKAccuracy', prefix='10pr_', thr=0.1),
    dict(type='AUC'),
]

# Optimization
optim_wrapper = dict(optimizer=dict(type='Adam', lr=5e-4))
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, begin=0, end=500, by_epoch=False),
    dict(type='MultiStepLR', begin=0, end=300, by_epoch=True, milestones=[200, 250], gamma=0.1),
]

# Training settings
train_cfg = dict(by_epoch=True, max_epochs=300, val_interval=10)
default_scope = 'mmpose'
log_level = 'INFO'
visualizer = dict(
    type='PoseLocalVisualizer',
    name='visualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ],
)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    timer=dict(type='IterTimerHook'),
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=1, save_best='coco/AP', rule='greater'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='PoseVisualizationHook', enable=True),
    badcase=dict(type='BadCaseAnalysisHook', enable=False, metric_type='loss', out_dir='badcase', badcase_thr=5),
)

env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
)

work_dir = 'work_dirs/result'
