# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import torch
import cv2

from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from mmpose.datasets import build_dataset
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Train a pose model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--resume', nargs='?', type=str, const='auto')
    parser.add_argument('--data-root', type=str)
    parser.add_argument('--amp', action='store_true', default=False)
    parser.add_argument('--no-validate', action='store_true')
    parser.add_argument('--auto-scale-lr', action='store_true')
    parser.add_argument('--show-dir')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--interval', type=int, default=1)
    parser.add_argument('--wait-time', type=float, default=1)
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--num-samples', type=int, default=20)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def merge_args(cfg, args):
    if args.no_validate:
        cfg.val_cfg = None
        cfg.val_dataloader = None
        cfg.val_evaluator = None

    cfg.launcher = args.launcher

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])

    if args.data_root is not None:
        cfg.train_dataloader.dataset.data_root = args.data_root
        cfg.val_dataloader.dataset.data_root = args.data_root
        cfg.test_dataloader.dataset.data_root = args.data_root
        for evaluator in cfg.val_evaluator:
            if 'ann_file' in evaluator:
                evaluator['ann_file'] = osp.join(args.data_root, 'annotations/forklift_keypoints_val2017.json')

    if args.amp is True:
        from mmengine.optim import AmpOptimWrapper, OptimWrapper
        optim_wrapper = cfg.optim_wrapper.get('type', OptimWrapper)
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.setdefault('loss_scale', 'dynamic')

    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    if args.auto_scale_lr:
        cfg.auto_scale_lr.enable = True

    if args.show or (args.show_dir is not None):
        assert 'visualization' in cfg.default_hooks
        cfg.default_hooks.visualization.enable = True
        cfg.default_hooks.visualization.show = args.show
        if args.show:
            cfg.default_hooks.visualization.wait_time = args.wait_time
        cfg.default_hooks.visualization.out_dir = args.show_dir
        cfg.default_hooks.visualization.interval = args.interval

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    return cfg


COLORS = [
    (255, 0, 0),      # red
    (0, 255, 0),      # green
    (0, 0, 255),      # blue
    (255, 255, 0),    # cyan
    (255, 0, 255),    # magenta
    (0, 255, 255),    # yellow
    (128, 128, 128),  # gray
]

LABEL_SHORT = {
    "front_right": "FR",
    "rear_right": "RR",
    "front_left": "FL",
    "rear_left": "RL",
    "R_Fork": "RF",
    "L_Fork": "LF",
    "C_Fork": "CF"
}

def plot_keypoints_on_image_cv2(image, heatmap, labels=None):
    num_keypoints, h_heat, w_heat = heatmap.shape
    scale_x = image.shape[1] / w_heat
    scale_y = image.shape[0] / h_heat

    if len(image.shape) == 2 or image.shape[2] == 1:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_bgr = image.copy()

    for i in range(num_keypoints):
        _, _, _, max_loc = cv2.minMaxLoc(heatmap[i])
        x_img = int(max_loc[0] * scale_x)
        y_img = int(max_loc[1] * scale_y)

        color = COLORS[i % len(COLORS)]

        cv2.drawMarker(image_bgr, (x_img, y_img), color=color, markerType=cv2.MARKER_CROSS, markerSize=8, thickness=1)

        label = labels[i] if labels and i < len(labels) else f"KP{i}"
        label_text = LABEL_SHORT.get(label, label[:2])

        cv2.putText(image_bgr, label_text, (x_img + 2, y_img - 2), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.4, color=color, thickness=1, lineType=cv2.LINE_AA)

    return image_bgr


def visualize_samples(cfg, classes, num_samples=5, show_dir=None):
    dataset_cfg = cfg.train_dataloader['dataset']
    dataset_cfg['labels'] = classes  # Ensure labels are passed to dataset constructor
    dataset = build_dataset(dataset_cfg)
   
    print("Labels from config:", classes)
    print("Number of labels:", len(classes))

    for i in range(num_samples):
        data_info = dataset[i]
        data_samples = data_info.get('data_samples', {})
        img = data_info.get('inputs')

        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0).numpy()

        vis_img = img.copy()
        if not hasattr(data_samples.gt_fields, "heatmaps"):
            print(f"[!] Sample {i} does not contain heatmaps!")
            continue

        vis_img = plot_keypoints_on_image_cv2(vis_img, data_samples.gt_fields.heatmaps.numpy(), classes)

        os.makedirs(show_dir, exist_ok=True)
        vis_img = np.clip(vis_img, 0, 255).astype(np.uint8)
        save_path = osp.join(show_dir, f'sample_{i}.jpg')
        if not cv2.imwrite(save_path, vis_img):
            print(f"[!] Failed to save {save_path}")
        else:
            print(f"[✓] Saved: {save_path}")


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg = merge_args(cfg, args)

    if 'preprocess_cfg' in cfg:
        cfg.model.setdefault('data_preprocessor', cfg.get('preprocess_cfg', {}))

    if args.visualize:
        from mmengine.registry import TRANSFORMS
        from mmpose.datasets.transforms import (RandomFlip, LoadImage, TorchVisionWrapper, PackPoseInputs,
                                                RandomBottomHalf, GetBBoxCenterScale, RandomBBoxTransform,
                                                TopdownAffine, Albumentation, GenerateTarget)

        TRANSFORMS.register_module(module=LoadImage)
        TRANSFORMS.register_module(module=RandomFlip, force=True)
        TRANSFORMS.register_module(module=GetBBoxCenterScale)
        TRANSFORMS.register_module(module=TopdownAffine)
        TRANSFORMS.register_module(module=Albumentation)
        TRANSFORMS.register_module(module=GenerateTarget)
        TRANSFORMS.register_module(module=RandomBBoxTransform)
        TRANSFORMS.register_module(module=TorchVisionWrapper)
        TRANSFORMS.register_module(module=RandomBottomHalf)
        TRANSFORMS.register_module(module=PackPoseInputs)

        classes = cfg.train_dataloader.dataset.get('labels', [f"KP{i}" for i in range(cfg.model.head.out_channels)])
        visualize_samples(cfg, classes, num_samples=args.num_samples, show_dir=args.show_dir or "/tmp/viz")
        return

    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == '__main__':
    main()
