from mmcv import Config


def make_mmpose_cfg(
        work_dir: str,
        labels: list = []
):
    channel_cfg = dict(
        num_output_channels=len(labels),
        dataset_joints=len(labels),
        dataset_channel=list(range(len(labels))),
        inference_channel=list(range(len(labels)))
    )
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
    train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='TopDownRandomFlip', flip_prob=0.5),
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
        bbox_file=f'/dataset/train/coco_train.json',
    )

    cfg = dict(
        work_dir=work_dir,
        total_epochs=100,
        default_scope='mmpose',
        optimizer_config=dict(grad_clip=None),
        log_level='INFO',
        load_from=None,
        resume_from=None,
        dist_params=dict(backend='nccl'),
        optimizer=dict(
            type='Adam',
            lr=1e-4,
        ),
        workflow=[('train', 1)],
        checkpoint_config=dict(interval=100),
        evaluation=dict(interval=10, metric='mAP', key_indicator='AP'),
        lr_config=dict(
            policy='step',
            warmup='linear',
            warmup_iters=5,
            warmup_ratio=0.001,
            step=[360, 380]
        ),
        log_config=dict(
            interval=25,
            hooks=[
                dict(type='TextLoggerHook'),
                dict(type='TensorboardLoggerHook')
            ]
        ),
        default_hooks=dict(
            # record the time of every iteration.
            timer=dict(type='EpochTimerHook'),

            # print log every 10 iterations.
            logger=dict(type='LoggerHook', interval=10),

            # enable the parameter scheduler.
            param_scheduler=dict(type='ParamSchedulerHook'),

            # save checkpoint per epoch.
            checkpoint=dict(type='CheckpointHook', interval=100),

            # set sampler seed in distributed evrionment.
            sampler_seed=dict(type='DistSamplerSeedHook'),

            # validation results visualization, set True to enable it.
            visualization=dict(type='VisualizationHook', enable=True),
        ),
        val_pipeline=val_pipeline,
        test_pipeline=val_pipeline,
        data=dict(
            samples_per_gpu=8,
            workers_per_gpu=2,
            train=dict(
                type='LiftedForkDatasetAnyKP',
                ann_file='/dataset/annotations/coco_train.json',
                img_prefix='/dataset/train/',
                data_cfg=data_cfg,
                pipeline=train_pipeline,
                num_joints=len(labels)
            ),
            val=dict(
                type='LiftedForkDatasetAnyKP',
                ann_file='/dataset/annotations/coco_val.json',
                img_prefix='/dataset/val/',
                data_cfg=data_cfg,
                pipeline=val_pipeline,
                num_joints=len(labels)
            ),
            test=dict(
                type='LiftedForkDatasetAnyKP',
                ann_file='/dataset/annotations/coco_test.json',
                img_prefix='/dataset/test/',
                data_cfg=data_cfg,
                pipeline=val_pipeline,
                num_joints=len(labels)
            ),
        ),
        val_evaluator=[
            dict(prefix='val', topk=(1,), type='Accuracy'),
            dict(prefix='val', type='AveragePrecision'),
            dict(prefix='val', type='SingleLabelMetric'),
        ],
        model=dict(
            type='TopDown',
            pretrained='torchvision://resnet18',
            backbone=dict(type='ResNet', depth=18),
            keypoint_head=dict(
                type='TopDownSimpleHead',
                in_channels=512,
                out_channels=channel_cfg['num_output_channels'],
                loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
            train_cfg=dict(by_epoch=True),
            test_cfg=dict(
                flip_test=True,
                post_process='default',
                shift_heatmap=True,
                modulate_kernel=11
            )
        )

    )
    cfg = Config(cfg)
    return cfg
