from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import distutils.util
import os
import sys
import pprint
# import subprocess
# from collections import defaultdict
# from six.moves import xrange

import numpy as np
import cv2

import torch
# import torch.nn as nn
# from torch.autograd import Variable

import _init_paths
import nn as mynn
from core.config import cfg, cfg_from_file, cfg_from_list, assert_and_infer_cfg
from core.test import im_detect_all
from modeling.model_builder import Generalized_RCNN
# import datasets.dummy_datasets as datasets
# import utils.misc as misc_utils
# import utils.net as net_utils
import pycocotools.mask as mask_util

# import utils.vis as vis_utils
# from utils.detectron_weight_helper import load_detectron_weight
# from utils.timer import Timer

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

def parse_args():
    class Args():
        pass

    args = Args()
    args.cfg_file = "configs/baselines/e2e_mask_rcnn_R-50-C4_1x.yaml"
    args.image_dir = "demo/sample_images"
    args.load_ckpt = "Outputs/sample_model.pth"
    args.score_thresh = 0.85

    return args

def load_ckpt(model, ckpt):
    """Load checkpoint"""
    mapping, _ = model.detectron_weight_mapping
    state_dict = {}
    for name in ckpt:
        if mapping[name]:
            state_dict[name] = ckpt[name]
    model.load_state_dict(state_dict, strict=False)

if __name__ == '__main__':
    import glob

    args = parse_args()
    cfg_from_file(args.cfg_file)

    cfg.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS = False  # Don't need to load imagenet pretrained weights
    cfg.MODEL.NUM_CLASSES = 81
    assert_and_infer_cfg()

    from maskrcnn import MaskRCNN
    maskRCNN = MaskRCNN()
    # maskRCNN = Generalized_RCNN()
    model = maskRCNN
    maskRCNN.cuda()

    if args.load_ckpt:
        load_name = args.load_ckpt
        checkpoint = torch.load(load_name, map_location=lambda storage, loc: storage)
        load_ckpt(maskRCNN, checkpoint['model'])

    maskRCNN = mynn.DataParallel(maskRCNN, cpu_keywords=['im_info', 'roidb'],
                             minibatch=True, device_ids=[0])  # only support single GPU
    maskRCNN.eval()

    img_file = "demo/sample_images/img1.jpg"
    img = cv2.imread(img_file)
    cls_boxes, cls_segms, cls_keyps = im_detect_all(maskRCNN, img)

    boxes = [b for b in cls_boxes if len(b) > 0]
    if len(boxes) > 0:
        boxes = np.concatenate(boxes)
    segms = [s for slist in cls_segms for s in slist] if cls_segms is not None else []
    classes = []
    for j in range(len(cls_boxes)):
        classes += [j] * len(cls_boxes[j])

    if segms is not None:
        masks = mask_util.decode(segms)

    # Display in largest to smallest order to reduce occlusion
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    sorted_inds = np.argsort(-areas)

    # colors = np.random.randint(0,255,size=(len(boxes), 3))
    for i in sorted_inds:
        bbox = boxes[i, :4]
        score = boxes[i, -1]

        color = np.random.randint(0,255,size=(3))

        if score < args.score_thresh:
            continue
        
        bbox = np.round(bbox).astype(np.int32)
        m = masks[:,:,i]
        _, contour, hier = cv2.findContours(
                m.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

        img_draw = cv2.rectangle(img, tuple(bbox[:2]), tuple(bbox[2:]), color)
        cv2.drawContours(img_draw, contour, -1, color, 2)
    cv2.imshow("d", img_draw)
    cv2.waitKey(0)

