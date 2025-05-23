log_level = 'INFO'
load_from = None
resume_from = None
dist_params = dict(backend='nccl')
workflow = [('train', 1)]
checkpoint_config = dict(interval=50)
evaluation = dict(interval=10, metric='mAP', key_indicator='AP')

optimizer = dict(
    type='Adam',
    lr=1e-4,
)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[360, 380])
total_epochs = 400
log_config = dict(
    interval=25,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

channel_cfg = dict(
    num_output_channels=6,
    dataset_joints=6,
    dataset_channel=[
        [0, 1, 2, 3, 4, 5],
    ],
    inference_channel=[
        0, 1, 2, 3, 4, 5])

# model settings
model = dict(
    type='TopDown',
    pretrained='torchvision://resnet18',
    backbone=dict(type='ResNet', depth=18),
    keypoint_head=dict(
        type='TopDownSimpleHead',
        in_channels=512,
        out_channels=channel_cfg['num_output_channels'],
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=True,
        post_process='default',
        shift_heatmap=True,
        modulate_kernel=11))

data_cfg = dict(
    image_size=[192, 256],
    heatmap_size=[48, 64],
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
    soft_nms=False,
    nms_thr=1.0,
    oks_thr=0.9,
    vis_thr=0.2,
    use_gt_bbox=True,
    det_bbox_thr=0.0,
    bbox_file='data/coco/person_detection_results/'
    'COCO_val2017_detections_AP_H_56_person.json',
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownRandomFlip', flip_prob=0.5),
    # dict(
    #     type='TopDownHalfBodyTransform',
    #     num_joints_half_body=8,
    #     prob_half_body=0.3),
    dict(
        type='TopDownGetRandomScaleRotation', rot_factor=180, scale_factor=0.5, rot_prob=0.9),
    dict(type='TopDownAffine'),

    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(type='TopDownGenerateTarget', sigma=3),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'bbox_score', 'flip_pairs'
        ]),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'image_file', 'center', 'scale', 'rotation', 'bbox_score',
            'flip_pairs'
        ]),
]

test_pipeline = val_pipeline

data_root = 'data/forklift_coco'
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,
    train=dict(
        type='TopDownForkliftDataset',
        ann_file=f'{data_root}/annotations/forklift_keypoints_train2017.json',
        img_prefix=f'{data_root}/train2017/',
        data_cfg=data_cfg,
        pipeline=train_pipeline),
    val=dict(
        type='TopDownForkliftDataset',
        ann_file=f'{data_root}/annotations/forklift_keypoints_val2017.json',
        img_prefix=f'{data_root}/val2017/',
        data_cfg=data_cfg,
        pipeline=val_pipeline),
    test=dict(
        type='TopDownForkliftDataset',
        ann_file=f'{data_root}/annotations/forklift_keypoints_test2017.json',
        img_prefix=f'{data_root}/val2017/',
        data_cfg=data_cfg,
        pipeline=val_pipeline),
)
