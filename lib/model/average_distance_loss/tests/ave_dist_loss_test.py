
import numpy as np
import torch
import torch.nn as nn

from ..modules.average_distance_loss import AverageDistanceLoss

FCT = torch.cuda.FloatTensor

if __name__ == '__main__': # run with ipython -i -m average_distance_loss.tests.ave_dist_loss_test 
    
    root_dir = "/home/vincent/Documents/deep_learning/PoseCNN/"
    pose_p = np.load(root_dir + "poses_pred.npy")
    pose_t = np.load(root_dir + "poses_target.npy")
    pose_w = np.load(root_dir + "poses_weight.npy")
    pts = np.load(root_dir + "points_all.npy")
    symmetry = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])

    num_classes = 22
    margin = 0.01

    ave_dist_loss_op = AverageDistanceLoss(num_classes, margin)
    loss, diff = ave_dist_loss_op(FCT(pose_p), FCT(pose_t), FCT(pose_w), FCT(pts), FCT(symmetry))
    