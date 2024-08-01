import json
import mmcv
from mmpose.models import build_posenet
import os

import cv2
import mmcv
import numpy as np
import torch
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

from mmpose.datasets.pipelines import Compose
from mmpose.utils.hooks import OutputHook

DATASET_TYPE = "LiftedForkDatasetAnyKP"


class MMPoseInferencer:
    """MMPose Inferencer. It's a unified inferencer interface for pose
        estimation task, currently including: Pose2D. and it can be used to perform
        2D keypoint detection.

        Args:
            pose2d (str, optional): Pretrained 2D pose estimation algorithm.
                It's the path to the config file or the model name defined in
                metafile. For example, it could be:

                - model alias, e.g. ``'body'``,
                - config name, e.g. ``'simcc_res50_8xb64-210e_coco-256x192'``,
                - config path

                Defaults to ``None``.
            pose2d_weights (str, optional): Path to the custom checkpoint file of
                the selected pose2d model. If it is not specified and "pose2d" is
                a model name of metafile, the weights will be loaded from
                metafile. Defaults to None.
            device (str, optional): Device to run inference. If None, the
                available device will be automatically used. Defaults to None.
            scope (str, optional): The scope of the model. Defaults to "mmpose".
            det_model(str, optional): Config path or alias of detection model.
                Defaults to None.
            det_weights(str, optional): Path to the checkpoints of detection
                model. Defaults to None.
            det_cat_ids(int or list[int], optional): Category id for
                detection model. Defaults to None.
            output_heatmaps (bool, optional): Flag to visualize predicted
                heatmaps. If set to None, the default setting from the model
                config will be used. Default is None.
        """

    def __init__(self,
                 config: str,
                 pretrained: str,
                 bboxes_filepath: str,
                 bbox_thr: float = None,
                 ):
        if isinstance(config, str):
            self.config = mmcv.Config.fromfile(config)
        elif not isinstance(config, mmcv.Config):
            raise TypeError('config must be a filename or Config object, '
                            f'but got {type(config)}')
        else:
            self.config = config

        if not os.path.exists(bboxes_filepath):
            raise TypeError(f'File path does not exists: {bboxes_filepath}')

        with open(bboxes_filepath, "r") as f:
            self.bboxes = json.load(f)
            print(f'bboxes: {self.bboxes}')

        self.devices = self.config.gpu_ids
        self.pretrained = pretrained
        self.dataset_type = DATASET_TYPE
        self.model = self.init_model()

    def init_model(self):
        self.config.model.pretrained = None
        model = build_posenet(self.config.model)
        device = f'cuda:{self.devices[0]}'
        if self.pretrained is not None:
            # load model checkpoint
            load_checkpoint(model, self.pretrained, map_location=device)
        # save the config in the model for convenience
        model.cfg = self.config
        model.to(device)
        model.eval()
        return model

    def __call__(self, image_path):
        image_name = image_path.split("/")[-1]
        bboxes = self.bboxes[image_name]
        self.inference(
            img_or_path=image_path,
            format='xyxy'
        )

    def process_single(
        self,
        model,
        img_or_path,
        bboxes,
        dataset,
        return_heatmap=False
    ):
        """Inference a single bbox.

        num_keypoints: K

        Args:
            model (nn.Module): The loaded pose model.
            img_or_path (str | np.ndarray): Image filename or loaded image.
            bboxes (list | np.ndarray): All bounding boxes (with scores),
                shaped (N, 4) or (N, 5). (left, top, width, height, [score])
                where N is number of bounding boxes.
            dataset (str): Dataset name.
            outputs (list[str] | tuple[str]): Names of layers whose output is
                to be returned, default: None

        Returns:
            ndarray[Kx3]: Predicted pose x, y, score.
            heatmap[N, K, H, W]: Model output heatmap.
        """

        cfg = model.cfg
        device = next(model.parameters()).device

        # build the data pipeline
        channel_order = cfg.test_pipeline[0].get('channel_order', 'rgb')
        test_pipeline = [LoadImage(channel_order=channel_order)
                         ] + cfg.test_pipeline[1:]
        test_pipeline = Compose(test_pipeline)

        assert len(bboxes[0]) in [4, 5]

        flip_pairs = None
        if dataset in ('TopDownCocoDataset', 'TopDownOCHumanDataset',
                       'AnimalMacaqueDataset'):
            flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12],
                          [13, 14], [15, 16]]
        elif dataset == 'TopDownCocoWholeBodyDataset':
            body = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14],
                    [15, 16]]
            foot = [[17, 20], [18, 21], [19, 22]]

            face = [[23, 39], [24, 38], [25, 37], [26, 36], [27, 35], [28, 34],
                    [29, 33], [30, 32], [40, 49], [41, 48], [42, 47], [43, 46],
                    [44, 45], [54, 58], [55, 57], [59, 68], [60, 67], [61, 66],
                    [62, 65], [63, 70], [64, 69], [71, 77], [72, 76], [73, 75],
                    [78, 82], [79, 81], [83, 87], [84, 86], [88, 90]]

            hand = [[91, 112], [92, 113], [93, 114], [94, 115], [95, 116],
                    [96, 117], [97, 118], [98, 119], [99, 120], [100, 121],
                    [101, 122], [102, 123], [103, 124], [104, 125], [105, 126],
                    [106, 127], [107, 128], [108, 129], [109, 130], [110, 131],
                    [111, 132]]
            flip_pairs = body + foot + face + hand
        elif dataset == 'TopDownAicDataset':
            flip_pairs = [[0, 3], [1, 4], [2, 5], [6, 9], [7, 10], [8, 11]]
        elif dataset == 'TopDownMpiiDataset':
            flip_pairs = [[0, 5], [1, 4], [2, 3], [10, 15], [11, 14], [12, 13]]
        elif dataset == 'TopDownMpiiTrbDataset':
            flip_pairs = [[0, 1], [2, 3], [4, 5], [6, 7],
                          [8, 9], [10, 11], [14, 15], [16, 22], [28, 34], [17, 23],
                          [29, 35], [18, 24], [30, 36], [19, 25], [31,
                                                                   37], [20, 26],
                          [32, 38], [21, 27], [33, 39]]
        elif dataset in ('OneHand10KDataset', 'FreiHandDataset', 'PanopticDataset',
                         'InterHand2DDataset'):
            flip_pairs = []
        elif dataset in 'Face300WDataset':
            flip_pairs = [[0, 16], [1, 15], [2, 14], [3, 13], [4, 12], [5, 11],
                          [6, 10], [7, 9], [17, 26], [18, 25], [19, 24], [20, 23],
                          [21, 22], [31, 35], [32, 34], [36, 45], [37,
                                                                   44], [38, 43],
                          [39, 42], [40, 47], [41, 46], [48, 54], [49,
                                                                   53], [50, 52],
                          [61, 63], [60, 64], [67, 65], [58, 56], [59, 55]]

        elif dataset in 'FaceAFLWDataset':
            flip_pairs = [[0, 5], [1, 4], [2, 3], [6, 11], [7, 10], [8, 9],
                          [12, 14], [15, 17]]

        elif dataset in 'FaceCOFWDataset':
            flip_pairs = [[0, 1], [4, 6], [2, 3], [5, 7], [8, 9], [10, 11],
                          [12, 14], [16, 17], [13, 15], [18, 19], [22, 23]]

        elif dataset in 'FaceWFLWDataset':
            flip_pairs = [[0, 32], [1, 31], [2, 30], [3, 29], [4, 28], [5, 27],
                          [6, 26], [7, 25], [8, 24], [9, 23], [10, 22], [11, 21],
                          [12, 20], [13, 19], [14, 18], [15, 17], [33,
                                                                   46], [34, 45],
                          [35, 44], [36, 43], [37, 42], [38, 50], [39,
                                                                   49], [40, 48],
                          [41, 47], [60, 72], [61, 71], [62, 70], [63,
                                                                   69], [64, 68],
                          [65, 75], [66, 74], [67, 73], [55, 59], [56,
                                                                   58], [76, 82],
                          [77, 81], [78, 80], [87, 83], [86, 84], [88, 92],
                          [89, 91], [95, 93], [96, 97]]
        elif dataset in 'LiftedForkDataset3KP':
            flip_pairs = [[0, 1]]
        elif dataset in 'TopDownForkliftDataset':
            flip_pairs = [[0, 1], [2, 3], [4, 5]]
        elif dataset in 'TopDownForkliftDataset4KP':
            flip_pairs = [[0, 1], [2, 3]]
        elif dataset in 'AnimalFlyDataset':
            flip_pairs = [[1, 2], [6, 18], [7, 19], [8, 20], [9, 21], [10, 22],
                          [11, 23], [12, 24], [13, 25], [14, 26], [15, 27],
                          [16, 28], [17, 29], [30, 31]]
        elif dataset in 'AnimalHorse10Dataset':
            flip_pairs = []

        elif dataset in 'AnimalLocustDataset':
            flip_pairs = [[5, 20], [6, 21], [7, 22], [8, 23], [9, 24], [10, 25],
                          [11, 26], [12, 27], [13, 28], [14, 29], [15, 30],
                          [16, 31], [17, 32], [18, 33], [19, 34]]
        elif dataset in 'AnimalZebraDataset':
            flip_pairs = [[3, 4], [5, 6]]
        elif dataset in 'AnimalPoseDataset':
            flip_pairs = [[0, 1], [2, 3], [8, 9], [10, 11], [12, 13], [14, 15],
                          [16, 17], [18, 19]]
        else:
            raise NotImplementedError()

        batch_data = []
        for bbox in bboxes:
            center, scale = _box2cs(cfg, bbox)

            # prepare data
            data = {
                'img_or_path':
                img_or_path,
                'center':
                center,
                'scale':
                scale,
                'bbox_score':
                bbox[4] if len(bbox) == 5 else 1,
                'bbox_id':
                0,  # need to be assigned if batch_size > 1
                'dataset':
                dataset,
                'joints_3d':
                np.zeros((cfg.data_cfg.num_joints, 3), dtype=np.float32),
                'joints_3d_visible':
                np.zeros((cfg.data_cfg.num_joints, 3), dtype=np.float32),
                'rotation':
                0,
                'ann_info': {
                    'image_size': cfg.data_cfg['image_size'],
                    'num_joints': cfg.data_cfg['num_joints'],
                    'flip_pairs': flip_pairs
                }
            }
            data = test_pipeline(data)
            batch_data.append(data)

        batch_data = collate(batch_data, samples_per_gpu=1)

        if next(model.parameters()).is_cuda:
            # scatter not work so just move image to cuda device
            batch_data['img'] = batch_data['img'].to(device)
        # get all img_metas of each bounding box
        batch_data['img_metas'] = [
            img_metas[0] for img_metas in batch_data['img_metas'].data
        ]

        # forward the model
        with torch.no_grad():
            result = model(
                img=batch_data['img'],
                img_metas=batch_data['img_metas'],
                return_loss=False,
                return_heatmap=return_heatmap)

        return result['preds'], result['output_heatmap']

    def inference(
            self,
            img_or_path: str,
            format='xywh',
            dataset='TopDownCocoDataset',
            return_heatmap=False,
            outputs=None
    ):
        # only two kinds of bbox format is supported.
        assert format in ['xyxy', 'xywh']

        pose_results = []
        returned_outputs = []

        if len(person_results) == 0:
            return pose_results, returned_outputs

        # Change for-loop preprocess each bbox to preprocess all bboxes at once.
        bboxes = np.array([box['bbox'] for box in person_results])

        # Select bboxes by score threshold
        if bbox_thr is not None:
            assert bboxes.shape[1] == 5
            valid_idx = np.where(bboxes[:, 4] > bbox_thr)[0]
            bboxes = bboxes[valid_idx]
            person_results = [person_results[i] for i in valid_idx]

        if format == 'xyxy':
            bboxes_xyxy = bboxes
            bboxes_xywh = _xyxy2xywh(bboxes)
        else:
            # format is already 'xywh'
            bboxes_xywh = bboxes
            bboxes_xyxy = _xywh2xyxy(bboxes)

        # if bbox_thr remove all bounding box
        if len(bboxes_xywh) == 0:
            return [], []

        with OutputHook(model, outputs=outputs, as_tensor=False) as h:
            # poses is results['pred'] # N x 17x 3
            poses, heatmap = _inference_single_pose_model(
                model,
                img_or_path,
                bboxes_xywh,
                dataset,
                return_heatmap=return_heatmap)

            if return_heatmap:
                h.layer_outputs['heatmap'] = heatmap

            returned_outputs.append(h.layer_outputs)

        assert len(poses) == len(person_results), print(
            len(poses), len(person_results), len(bboxes_xyxy))
        for pose, person_result, bbox_xyxy in zip(poses, person_results,
                                                  bboxes_xyxy):
            pose_result = person_result.copy()
            pose_result['keypoints'] = pose
            pose_result['bbox'] = bbox_xyxy
            pose_results.append(pose_result)

        return pose_results, returned_outputs

    def prepare_data(self, annotations_json: str):
        """Prepare forklift bboxes"""
        person_results = []
        for bbox in face_det_results:
            person = {}
            # left, top, right, bottom
            person['bbox'] = [bbox[3], bbox[0], bbox[1], bbox[2]]
            person_results.append(person)

        return person_results
