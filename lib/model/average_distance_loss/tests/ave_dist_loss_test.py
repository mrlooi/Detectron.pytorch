
import numpy as np
import torch
import torch.nn as nn

from ..modules.average_distance_loss import AverageDistanceLoss

def T(x, dtype=torch.float32, requires_grad=False):
    return torch.tensor(x, dtype=dtype, requires_grad=requires_grad, device="cuda")
def FCT(x, requires_grad=False):
    return T(x, requires_grad=requires_grad)
def ICT(x, requires_grad=False):
    return T(x, dtype=torch.int32, requires_grad=requires_grad)

if __name__ == '__main__': # run with ipython -i -m average_distance_loss.tests.ave_dist_loss_test 
    
    root_dir = "/home/vincent/Documents/py/ml/PoseCNN/"
    pose_p = np.load(root_dir + "poses_pred.npy")
    pose_t = np.load(root_dir + "poses_targets.npy")
    pose_w = np.load(root_dir + "poses_weights.npy")
    pts = np.load(root_dir + "points_all.npy")
    symmetry = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])

    num_classes = 22
    margin = 0.01

    fpp = FCT(pose_p, True)
    fpt = FCT(pose_t, False)
    fpw = FCT(pose_w, False)
    fpts = FCT(pts, False)
    symm = FCT(symmetry, False)

    ave_dist_loss_op = AverageDistanceLoss(num_classes, margin)
    loss = ave_dist_loss_op(fpp, fpt, fpw, fpts, symm)
    loss.backward()
    print(loss.grad)