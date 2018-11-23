from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import distutils.util
import os
import sys
# import pprint

import numpy as np
import cv2

import torch
import torch.nn as nn
# from torch.autograd import Variable

import _init_paths
import nn as mynn
from core.config import cfg, cfg_from_file, assert_and_infer_cfg
import utils.net as net_utils

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

def parse_args():
    # import argparse
    class Args():
        pass

    args = Args()
    args.dataset = 'cocodebug2014'
    args.cfg_file = 'configs/baselines/e2e_mask_rcnn_R-50-C4_1x.yaml'
    args.disp_interval = 20
    
    # args.no_cuda = False

    args.batch_size = 2
    args.num_workers = 2
    # args.iter_size = 1

    # TRAIN VARS
    args.optimizer = 'Adam' # 'SGD'
    args.lr = 0.001
    args.lr_decay_gamma = 0.1
    args.start_step = 0

    # CHECKPOINTS
    args.no_save = False
    args.resume = False
    args.load_ckpt = None
    args.use_tfboard = False

    return args

def set_cfg(args):
    cfg_from_file(args.cfg_file)

    # dataset cfg
    if args.dataset == "coco2017":
        cfg.TRAIN.DATASETS = ('coco_2017_train',)
        cfg.MODEL.NUM_CLASSES = 81
    elif args.dataset == "coco2014":
        cfg.TRAIN.DATASETS = ('coco_2014_train',)
        cfg.MODEL.NUM_CLASSES = 81
    elif args.dataset == "cocoval2014":
        cfg.TRAIN.DATASETS = ('coco_2014_val',)
        cfg.MODEL.NUM_CLASSES = 81
    elif args.dataset == "cocodebug2014":
        cfg.TRAIN.DATASETS = ('coco_2014_debug',)
        cfg.MODEL.NUM_CLASSES = 81
    # elif args.dataset == "keypoints_coco2017":
    #     cfg.TRAIN.DATASETS = ('keypoints_coco_2017_train',)
    #     cfg.MODEL.NUM_CLASSES = 2
    else:
        raise ValueError("Unexpected args.dataset: {}".format(args.dataset))

    cfg.MODEL.NUM_CLASSES = 81

    cfg.NUM_GPUS = 1
    cfg.TRAIN.IMS_PER_BATCH = args.batch_size // cfg.NUM_GPUS
    # effective_batch_size = args.iter_size * args.batch_size

    # optimizer cfg
    cfg.SOLVER.TYPE = args.optimizer
    cfg.SOLVER.BASE_LR = args.lr
    cfg.SOLVER.GAMMA = args.lr_decay_gamma

    cfg.DATA_LOADER.NUM_THREADS = args.num_workers

    assert_and_infer_cfg()

def load_ckpt(args, model):
    if args.load_ckpt:
        load_name = args.load_ckpt
        logging.info("loading checkpoint %s", load_name)
        checkpoint = torch.load(load_name, map_location=lambda storage, loc: storage)
        net_utils.load_ckpt(model, checkpoint['model'])
        if args.resume:
            args.start_step = checkpoint['step'] + 1
            if 'train_size' in checkpoint:  # For backward compatibility
                if checkpoint['train_size'] != train_size:
                    print('train_size value: %d different from the one in checkpoint: %d'
                          % (train_size, checkpoint['train_size']))

            # reorder the params in optimizer checkpoint's params_groups if needed
            # misc_utils.ensure_optimizer_ckpt_params_order(param_names, checkpoint)

            # There is a bug in optimizer.load_state_dict on Pytorch 0.3.1.
            # However it's fixed on master.
            optimizer.load_state_dict(checkpoint['optimizer'])
            # misc_utils.load_optimizer_state_dict(optimizer, checkpoint['optimizer'])
        del checkpoint
        torch.cuda.empty_cache()


def get_trainable_params(model):
    gn_param_nameset = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.GroupNorm):
            gn_param_nameset.add(name+'.weight')
            gn_param_nameset.add(name+'.bias')
    gn_params = []
    gn_param_names = []
    bias_params = []
    bias_param_names = []
    nonbias_params = []
    nonbias_param_names = []
    nograd_param_names = []
    for key, value in model.named_parameters():
        if value.requires_grad:
            if 'bias' in key:
                bias_params.append(value)
                bias_param_names.append(key)
            elif key in gn_param_nameset:
                gn_params.append(value)
                gn_param_names.append(key)
            else:
                nonbias_params.append(value)
                nonbias_param_names.append(key)
        else:
            nograd_param_names.append(key)
    assert (gn_param_nameset - set(nograd_param_names) - set(bias_param_names)) == set(gn_param_names)

    # Learning rate of 0 is a dummy value to be set properly at the start of training
    params = [
        {'params': nonbias_params,
         'lr': 0,
         'weight_decay': cfg.SOLVER.WEIGHT_DECAY},
        {'params': bias_params,
         'lr': 0 * (cfg.SOLVER.BIAS_DOUBLE_LR + 1),
         'weight_decay': cfg.SOLVER.WEIGHT_DECAY if cfg.SOLVER.BIAS_WEIGHT_DECAY else 0},
        {'params': gn_params,
         'lr': 0,
         'weight_decay': cfg.SOLVER.WEIGHT_DECAY_GN}
    ]
    # names of paramerters for each paramter
    # param_names = [nonbias_param_names, bias_param_names, gn_param_names]
    return params

if __name__ == '__main__':
    from datasets.roidb import combined_roidb_for_training
    from roi_data.loader import RoiDataLoader, MinibatchSampler, BatchSampler, collate_minibatch

    from maskrcnn import MaskRCNN
    
    # python tools/train_net_step.py --dataset coco2014 --cfg configs/baselines/e2e_mask_rcnn_R-50-C4.yml --use_tfboard --bs {batch_size} --nw {num_workers}
    args = parse_args()

    set_cfg(args)

    roidb, ratio_list, ratio_index = combined_roidb_for_training(
        cfg.TRAIN.DATASETS, cfg.TRAIN.PROPOSAL_FILES)
    roidb_size = len(roidb)
    train_size = roidb_size // args.batch_size * args.batch_size

    batchSampler = BatchSampler(
        sampler=MinibatchSampler(ratio_list, ratio_index),
        batch_size=args.batch_size,
        drop_last=True
    )
    dataset = RoiDataLoader(
        roidb,
        cfg.MODEL.NUM_CLASSES,
        training=True)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batchSampler,
        num_workers=cfg.DATA_LOADER.NUM_THREADS,
        collate_fn=collate_minibatch)
    dataiterator = iter(dataloader)

    ### Model ###
    maskRCNN = MaskRCNN()
    maskRCNN.cuda()

    ### Optimizer ###
    params = get_trainable_params(maskRCNN)
    if cfg.SOLVER.TYPE == "SGD":
        optimizer = torch.optim.SGD(params, momentum=cfg.SOLVER.MOMENTUM)
    elif cfg.SOLVER.TYPE == "Adam":
        optimizer = torch.optim.Adam(params)

    load_ckpt(args, maskRCNN)

    lr = optimizer.param_groups[0]['lr']  # lr of non-bias parameters, for commmand line outputs.
    maskRCNN = mynn.DataParallel(maskRCNN, cpu_keywords=['im_info', 'roidb'], minibatch=True)

    ### Training Loop ###
    maskRCNN.train()
