import numpy as np
import torch
import torch.nn as nn
from torch.autograd.function import once_differentiable


# FT = torch.FloatTensor
def FT(x, req_grad=False):
	return torch.tensor(x, dtype=torch.float, device='cuda', requires_grad=req_grad)
def IT(x, req_grad=False): 
	return torch.tensor(x, dtype=torch.int, device='cuda')#, requires_grad=req_grad)  # INT TENSORS DO NOT ACCEPT GRAD IN TORCH 1.0

def get_data(num_classes):

    root_dir = "/home/vincent/Documents/deep_learning/PoseCNN/data"
    # # extents blob
    # extent_file = root_dir + "/LOV/extents.txt"
    # extents = np.zeros((num_classes, 3), dtype=np.float32)
    # extents[1:, :] = np.loadtxt(extent_file)

    points_file = root_dir + "/LOV/points_all_orig.npy"
    points = np.load(points_file)
    extents = np.max(points, axis=1) - np.min(points,axis=1)

    # meta blob
    im_scale = 1.0
    intrinsics = np.array([[1066.778, 0, 312.9869], [0, 1067.487, 241.3109], [0.0, 0.0, 1.0]]) * im_scale
    # mdata[:,9:18] = Kinv.flatten()

    base_file = root_dir + "/LOV/data/0000/000001-"
    img_file = base_file + "color.png"
    img = cv2.imread(img_file)

    # labels
    labels_file = base_file + "labels_mrcnn.npy"
    labels = np.load(labels_file)

    # masks blob
    masks_file = base_file + "masks_mrcnn.npy"
    masks = np.load(masks_file)

    # vertex pred blob
    vertex_pred_file = base_file + "vert_pred_mrcnn.npy"
    vertex_pred = np.load(vertex_pred_file)
    # vertex_pred = np.transpose(vertex_pred, [0,2,3,1])
    assert masks.shape == vertex_pred.shape[:3]

    # poses = np.zeros((len(masks), 13))
    poses_pred_file = base_file + "poses_mrcnn.npy" 
    poses = np.load(poses_pred_file)

    # return img, labels[:1], masks[:1], vertex_pred[:1], extents, poses[:1], intrinsics, points
    return img, labels, masks, vertex_pred, extents, poses, intrinsics, points


def convert_to_OLD_format(labels, masks, vertex_pred, num_classes):
    N,H,W = masks.shape

    label_2d = np.zeros((H,W), dtype=np.int32)
    vert_p = np.zeros((H,W,num_classes*3), dtype=np.float32)

    for ix, m in enumerate(masks):
        cls = labels[ix]
        label_2d[m==1] = cls
        vert_p[:,:,cls*3:cls*3+3] = vertex_pred[ix]

    return label_2d, vert_p

def vis_rois(im, rois):
    img = im.copy()

    rois = np.round(rois).astype(np.int32)
    RED = (0,0,255)
    GREEN = (0,255,0)
    for roi in rois:
        cx = (roi[2] + roi[4]) // 2
        cy = (roi[3] + roi[5]) // 2
        bbox1 = roi[2:-1]
        img = cv2.rectangle(img, tuple(bbox1[:2]), tuple(bbox1[2:]), RED)
        img = cv2.circle(img, (cx,cy), 3, GREEN, -1)

    cv2.imshow("rois", img)
   
def vis_pose(im, labels, poses, intrinsics, points):
    from transforms3d.quaternions import quat2mat#, mat2quat

    img = im.copy()

    N = poses.shape[0]

    colors = [tuple(np.random.randint(0,255,size=3)) for i in range(N)]
    for i in range(N):
        cls = labels[i]
        if cls > 0:
            # extract 3D points
            x3d = np.ones((4, points.shape[1]), dtype=np.float32)
            x3d[0, :] = points[cls,:,0]
            x3d[1, :] = points[cls,:,1]
            x3d[2, :] = points[cls,:,2]

            # projection
            RT = np.zeros((3, 4), dtype=np.float32)
            RT[:3, :3] = quat2mat(poses[i, :4])
            RT[:, 3] = poses[i, 4:7]
            x2d = np.matmul(intrinsics, np.matmul(RT, x3d))
            x2d[0, :] = np.divide(x2d[0, :], x2d[2, :])
            x2d[1, :] = np.divide(x2d[1, :], x2d[2, :])

            color = colors[i]

            proj_pts = x2d[:2].transpose()
            proj_pts = np.round(proj_pts).astype(np.int32)
            for px in proj_pts:
                img = cv2.circle(img, tuple(px), 1, (int(color[0]),int(color[1]),int(color[2])), -1)

            # plt.plot(x2d[0, :], x2d[1, :], '.', , alpha=0.5)
            # plt.scatter(x2d[0, :], x2d[1, :], marker='o', color=np.divide(colors[cls], 255.0), s=10)
    cv2.imshow("proj points", img)

def run_hough_voting_OLD(T_label_2d, T_vertex_pred, T_extents, T_poses, T_intrinsics):
    from layer_old.hough_voting_layer import HoughVoting

    num_classes = 22
    is_train = False
    vote_threshold = -1.0
    vote_percentage = 0.02
    skip_pixels = 20
    label_threshold = 500
    inlier_threshold = 0.9

    model = HoughVoting(num_classes, threshold_vote=vote_threshold, threshold_percentage=vote_percentage, label_threshold=label_threshold, 
        inlier_threshold=inlier_threshold, skip_pixels=skip_pixels, is_train=is_train)

    model.cuda()
    hough_outputs = model.forward(T_label_2d, T_vertex_pred, T_extents, T_poses, T_intrinsics)
    return hough_outputs

def run_hough_voting(T_labels, T_masks, T_vertex_pred, T_extents, T_poses, T_intrinsics):
    from layer.hough_voting_layer import HoughVoting

    skip_pixels = 100
    inlier_threshold = 0.9

    model = HoughVoting(inlier_threshold=inlier_threshold, skip_pixels=skip_pixels)

    model.cuda()
    hough_outputs = model.forward(T_labels, T_masks, T_vertex_pred, T_extents, T_poses, T_intrinsics)
    return hough_outputs

if __name__ == '__main__': 
    import cv2

    USE_CUDA = 1

    num_classes = 22

    img, labels, masks, vertex_pred, extents, poses, intrinsics, points = get_data(num_classes)
    print(masks.shape, vertex_pred.shape, extents.shape, poses.shape, intrinsics.shape)


    T_labels = IT(labels)
    T_masks = IT(masks)
    T_vertex_pred = FT(vertex_pred)
    T_extents = FT(extents)
    T_poses = FT(poses)
    T_intrinsics = FT(intrinsics.flatten())

    """OLD STUFF"""
    # label_2d_old, vertex_pred_old = convert_to_OLD_format(labels, masks, vertex_pred, num_classes)
    # T_label_2d_old = IT([label_2d_old])#, True)#.type(torch.IntTensor)
    # T_vertex_pred_old = FT([vertex_pred_old], True)
    # hough_outputs = run_hough_voting_OLD(T_label_2d_old, T_vertex_pred_old, T_extents, T_poses, T_intrinsics)

    hough_outputs = run_hough_voting(T_labels, T_masks, T_vertex_pred, T_extents, T_poses, T_intrinsics)


    rois, poses = hough_outputs

    rois = rois.detach().cpu().numpy()
    poses = poses.detach().cpu().numpy()

    # sorted_idx = np.argsort(rois[:,1])  # sort assumption obv breaks if there are multi instances of same class
    # final_poses = poses[sorted_idx]
    # sorted_idx = np.argsort(labels)
    # final_labels = labels[sorted_idx] 

    vis_rois(img, rois)
    vis_pose(img, labels, poses, intrinsics, points)

    cv2.waitKey(0)
