
import numpy as np
import torch
import torch.nn as nn


def T(x, dtype=torch.float32, requires_grad=False):
    return torch.tensor(x, dtype=dtype, requires_grad=requires_grad, device="cuda")
def FCT(x, requires_grad=False):
    return T(x, requires_grad=requires_grad)
def ICT(x, requires_grad=False):
    return T(x, dtype=torch.int32, requires_grad=requires_grad)
def LCT(x, requires_grad=False):
    return T(x, dtype=torch.long, requires_grad=requires_grad)


def get_error(d1, d2):
    x = get_pth_tensor_as_np(d1) if isinstance(d1, torch.Tensor) else d1
    y = get_pth_tensor_as_np(d2) if isinstance(d2, torch.Tensor) else d2
    e = np.abs(x-y)
    return np.sum(e)

def get_pth_tensor_as_np(x):
    return x.data.cpu().numpy()

def convert_poses_NC4_to_N4(pose_p, pose_t, pose_w):
    pose_preds = [] 
    pose_targets = []
    pose_weights = []
    pose_labels = []
    POSE_CHANNELS = 4
    for pp, pt, pw in zip(pose_p, pose_t, pose_w):
        pos_ind = np.where(pw==1)[0]
        if len(pos_ind) == 0:
            continue
        pose_preds.append(pp[pos_ind])
        pose_targets.append(pt[pos_ind])
        pose_weights.append(pw[pos_ind])
        pose_labels.append(pos_ind[0] / POSE_CHANNELS)
    return np.array(pose_preds), np.array(pose_targets), np.array(pose_weights), np.array(pose_labels)

def filter_empty_pose_weights(pose_p, pose_t, pose_w):
    pose_preds = [] 
    pose_targets = []
    pose_weights = []    
    for pp, pt, pw in zip(pose_p, pose_t, pose_w):
        pos_ind = np.where(pw==1)[0]
        if len(pos_ind) == 0:
            continue
        pose_preds.append(pp)
        pose_targets.append(pt)
        pose_weights.append(pw)
    return np.array(pose_preds), np.array(pose_targets), np.array(pose_weights)

if __name__ == '__main__': # run with ipython -i -m average_distance_loss.tests.ave_dist_loss_test 
    
    root_dir = "/home/vincent/Documents/py/ml/PoseCNN/"
    pose_preds = np.load(root_dir + "poses_pred.npy")
    pose_targets = np.load(root_dir + "poses_targets.npy")
    pose_weights = np.load(root_dir + "poses_weights.npy")
    pose_preds, pose_targets, pose_weights = filter_empty_pose_weights(pose_preds, pose_targets, pose_weights)
    pts_all = np.load(root_dir + "points_all.npy")
    symmetry = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
    
    fpts = FCT(pts_all, False)
    symm = FCT(symmetry, False)
    margin = 0.01
    
    # # OLD 
    # num_classes = 22
    # fpp = FCT(pose_preds, True)
    # fpt = FCT(pose_targets, False)
    # fpw = FCT(pose_weights, False)

    # from layer.ave_dist_loss_layer import AverageDistanceLoss
    # ave_dist_loss_op = AverageDistanceLoss(num_classes, margin)
    # loss = ave_dist_loss_op(fpp, fpt, fpw, fpts, symm)
    # print(loss)
    # loss.backward()
    # print(np.sum(fpp.grad.cpu().numpy()))

    # NEW
    # pose_p, pose_t, pose_w, pose_labels = convert_poses_NC4_to_N4(pose_preds, pose_targets, pose_weights)

    root_dir = "/home/bot/Documents/deep_learning/maskrcnn-benchmark/"
    pose_p = np.load(root_dir + "poses_preds.npy")
    pose_t = np.load(root_dir + "poses_targets.npy")
    pose_labels = np.load(root_dir + "poses_labels.npy")
    pts = np.load(root_dir + "points.npy")
    symmetry = np.load(root_dir + "symmetry.npy")

    fpp = FCT(pose_p, True)
    fpt = FCT(pose_t, False)
    fpl = LCT(pose_labels, False)
    # fpts = FCT(pts, False)
    # symm = FCT(symmetry, False)

    from layer.ave_dist_loss_layer import AverageDistanceLoss2
    ave_dist_loss_op2 = AverageDistanceLoss2(margin)
    loss = ave_dist_loss_op2(fpp, fpt, fpl, fpts, symm)
    print(loss)
    # loss.backward()
    # print(np.sum(fpp.grad.cpu().numpy()))

