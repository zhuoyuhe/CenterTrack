from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import json
import cv2
import numpy as np
import time
import torch
import copy

from opts import opts
from dataset.dataset_factory import dataset_factory
from forward_detector import ForwardDetector
from utils.utils import AverageMeter

def forward(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    CAMERA_CALIB = np.array([[721.53772, 0, 0, 44.85728],[0, 721.53772, 0, 0.21638], [0, 0, 1, 0.00275]])
    results = {}
    Dataset = dataset_factory[opt.test_dataset]
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    detector = ForwardDetector(opt)
    frame_id = 1
    time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post']
    avg_time_stats = {t: AverageMeter() for t in time_stats}
    for i in range(10):
        img_path = "C:/Users/zhuoyuhe/Desktop/CIS-700/CenterTrack/data/kitti_tracking/data_tracking_image_2/training/image_02/0000/00000{}.png".format(i)
        input_meta = {}
        input_meta['calib'] = CAMERA_CALIB
        if (opt.tracking and frame_id == 1):
            detector.reset_tracking()

        ret = detector.run(img_path, input_meta)
        results[frame_id] = ret['results']

        for t in avg_time_stats:
            avg_time_stats[t].update(ret[t])
        frame_id += 1
    for t in avg_time_stats:
        print(t, avg_time_stats[t].avg)
    a = 1
    b = 2

def _to_list(results):
    for img_id in results:
        for t in range(len(results[img_id])):
            for k in results[img_id][t]:
                if isinstance(results[img_id][t][k], (np.ndarray, np.float32)):
                    results[img_id][t][k] = results[img_id][t][k].tolist()
    return results


if __name__ == '__main__':
    opt = opts().parse()
    forward(opt)
