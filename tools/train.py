# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(description='Train a pose model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')
    parser.add_argument(
        '--data-root',
        type=str,
        help='Root directory for dataset. This will override data_root in the config file.'
    )
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='whether to auto scale the learning rate according to the '
        'actual batch size and the original batch size.')
    parser.add_argument(
        '--show-dir',
        help='directory where the visualization images will be saved.')
    parser.add_argument(
        '--show',
        action='store_true',
        help='whether to display the prediction results in a window.')
    parser.add_argument(
        '--interval',
        type=int,
        default=1,
        help='visualize per interval samples.')
    parser.add_argument(
        '--wait-time',
        type=float,
        default=1,
        help='display time of every window. (second)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    parser.add_argument('--visualize', action='store_true', help='Visualize augmented dataset samples instead of training')
    parser.add_argument('--num-samples', type=int, default=5, help='Number of samples to visualize')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def merge_args(cfg, args):
    """Merge CLI arguments to config."""
    if args.no_validate:
        cfg.val_cfg = None
        cfg.val_dataloader = None
        cfg.val_evaluator = None

    cfg.launcher = args.launcher

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # Update data_root
    if args.data_root is not None:
        cfg.train_dataloader.dataset.data_root = args.data_root
        cfg.val_dataloader.dataset.data_root = args.data_root
        cfg.test_dataloader.dataset.data_root = args.data_root

        # Update evaluator paths if necessary
        for evaluator in cfg.val_evaluator:
            if 'ann_file' in evaluator:
                evaluator['ann_file'] = osp.join(args.data_root, 'annotations/forklift_keypoints_val2017.json')

    # enable automatic-mixed-precision training
    if args.amp is True:
        from mmengine.optim import AmpOptimWrapper, OptimWrapper
        optim_wrapper = cfg.optim_wrapper.get('type', OptimWrapper)
        assert optim_wrapper in (OptimWrapper, AmpOptimWrapper,
                                 'OptimWrapper', 'AmpOptimWrapper'), \
            '`--amp` is not supported custom optimizer wrapper type ' \
            f'`{optim_wrapper}.'
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.setdefault('loss_scale', 'dynamic')

    # resume training
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    # enable auto scale learning rate
    if args.auto_scale_lr:
        cfg.auto_scale_lr.enable = True

    # visualization
    if args.show or (args.show_dir is not None):
        assert 'visualization' in cfg.default_hooks, \
            'PoseVisualizationHook is not set in the ' \
            '`default_hooks` field of config. Please set ' \
            '`visualization=dict(type="PoseVisualizationHook")`'

        cfg.default_hooks.visualization.enable = True
        cfg.default_hooks.visualization.show = args.show
        if args.show:
            cfg.default_hooks.visualization.wait_time = args.wait_time
        cfg.default_hooks.visualization.out_dir = args.show_dir
        cfg.default_hooks.visualization.interval = args.interval

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    return cfg


def plot_keypoints_on_image_cv2(image, heatmap, labels=None):
    """
    Draws keypoints extracted from a heatmap onto an image using OpenCV.

    Parameters:
        image (np.ndarray): Input image of shape (256,256) or (256,256,3).
        heatmap (np.ndarray): Heatmap of shape (4,64,64); each channel represents one keypoint.
        labels (list of str): Optional list of labels corresponding to each keypoint.

    Returns:
        np.ndarray: The image with keypoints and labels drawn.
    """
    num_keypoints, h_heat, w_heat = heatmap.shape

    # Compute scaling factors from heatmap size to image size.
    scale_x = image.shape[1] / w_heat   # e.g. 256/64 = 4
    scale_y = image.shape[0] / h_heat     # e.g. 256/64 = 4

    # If the image is grayscale, convert it to BGR for colored drawing.
    if len(image.shape) == 2 or image.shape[2] == 1:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_bgr = image.copy()

    # Process each heatmap channel.
    for i in range(num_keypoints):
        # Use cv2.minMaxLoc to find the location of the maximum value.
        # Note: cv2.minMaxLoc returns (x, y) as (col, row)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(heatmap[i])
        x_heat, y_heat = max_loc  # These coordinates are in the 64x64 space

        # Scale coordinates to the image size.
        x_img = int(x_heat * scale_x)
        y_img = int(y_heat * scale_y)

        # Draw a marker (a red cross) at the keypoint location.
        cv2.drawMarker(
            image_bgr, (x_img, y_img), color=(0, 0, 255),
            markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2
        )

        # Determine the label for this keypoint.
        label_text = labels[i] if labels is not None and i < len(labels) else f'KP {i}'

        # Draw the label slightly offset from the keypoint.
        cv2.putText(
            image_bgr, label_text, (x_img + 5, y_img - 5),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
            color=(0, 255, 255), thickness=2, lineType=cv2.LINE_AA
        )

    return image_bgr


def visualize_samples(cfg, classes, num_samples=5, show_dir=None):
    """Visualize augmented dataset samples with keypoint annotations."""
    dataset_cfg = cfg.train_dataloader['dataset']
    dataset = build_dataset(dataset_cfg)

    print(f"Visualizing {num_samples} samples from the dataset...")

    for i in range(num_samples):
        data_info = dataset[i]
        data_samples = data_info.get('data_samples', {})

        img = data_info.get('inputs')

        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0).numpy()

        vis_img = img.copy()
        # Draw the keypoints and labels on the image.
        vis_img = plot_keypoints_on_image_cv2(vis_img, data_samples.gt_fields.heatmaps.numpy(), classes)

        os.makedirs(show_dir, exist_ok=True)
        save_path = osp.join(show_dir, f'sample_{i}.jpg')
        cv2.imwrite(save_path, vis_img)


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)

    # merge CLI arguments to config
    cfg = merge_args(cfg, args)

    # set preprocess configs to model
    if 'preprocess_cfg' in cfg:
        cfg.model.setdefault('data_preprocessor',
                             cfg.get('preprocess_cfg', {}))

    if args.visualize:
        visualize_samples(cfg, args.classes, num_samples=args.num_samples, show_dir="/tmp/viz")
        return

    # build the runner from config
    runner = Runner.from_cfg(cfg)

    # start training
    runner.train()


if __name__ == '__main__':
    main()
