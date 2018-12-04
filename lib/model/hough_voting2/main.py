import numpy as np
import torch
import torch.nn as nn
from torch.autograd.function import once_differentiable

from layer.hough_voting_layer import HoughVoting


# FT = torch.FloatTensor
def FT(x, req_grad=False):
	return torch.tensor(x, dtype=torch.float, device='cuda', requires_grad=req_grad)
def IT(x, req_grad=False): 
	return torch.tensor(x, dtype=torch.int, device='cuda')#, requires_grad=req_grad)  # INT TENSORS DO NOT ACCEPT GRAD IN TORCH 1.0

def get_data(num_classes):
    im_scale = 1.0

    root_dir = "/home/vincent/Documents/deep_learning/PoseCNN/data"
    # extents blob
    extent_file = root_dir + "/LOV/extents.txt"
    extents = np.zeros((num_classes, 3), dtype=np.float32)
    extents[1:, :] = np.loadtxt(extent_file)

    # meta blob
    K = np.array([[1066.778, 0, 312.9869], [0, 1067.487, 241.3109], [0.0, 0.0, 1.0]])
    factor_depth = 10000.0
    meta_data = dict({'intrinsic_matrix': K, 'factor_depth': factor_depth})
    K = np.matrix(meta_data['intrinsic_matrix']) * im_scale
    K[2, 2] = 1.0
    Kinv = np.linalg.pinv(K)
    mdata = np.zeros((1,9), dtype=np.float32)
    mdata[:,0:9] = K.flatten()
    # mdata[:,9:18] = Kinv.flatten()
    # mdata = K.flatten()

    # label_2d blob
    # label_2d_file = root_dir + "/demo_images/000005-label2d.npy"
    label_2d_file = root_dir + "/LOV/data/0002/000001-label2d_mrcnn.npy"
    label_2d = np.load(label_2d_file)

    # vertex pred blob
    # vertex_pred_file = root_dir + "/demo_images/000005-vert_pred.npy"
    vertex_pred_file = root_dir + "/LOV/data/0002/000001-vert_pred_mrcnn.npy"
    vertex_pred = np.load(vertex_pred_file)
    vertex_pred = np.transpose(vertex_pred, [0,2,3,1])
    assert label_2d.shape == vertex_pred.shape[:3]
    assert vertex_pred.shape[-1] == num_classes * 3

    poses = np.zeros((len(label_2d), 13))

    # vertex_pred = np.transpose(vertex_pred, [0,2,3,1])
    # label_2d = np.tile(label_2d, (2,1,1,1))
    # vertex_pred = np.tile(vertex_pred, (2,1,1,1))
    # is_train = False

    return label_2d, vertex_pred, extents, poses, mdata

class HoughNet(nn.Module):
    def __init__(self):
        super(HoughNet, self).__init__()

        self.num_classes = 22
        self.is_train = False
        self.vote_threshold = -1.0
        self.vote_percentage = 0.02
        self.skip_pixels = 20
        self.label_threshold = 500
        self.inlier_threshold = 0.9

        self.hough_voting = HoughVoting(self.num_classes, threshold_vote=self.vote_threshold, threshold_percentage=self.vote_percentage, label_threshold=self.label_threshold, 
            inlier_threshold=self.inlier_threshold, skip_pixels=self.skip_pixels, is_train=self.is_train)


    def forward(self, label_2d, vertex_pred, extents, poses, meta_data):
        return self.hough_voting(label_2d, vertex_pred, extents, poses, meta_data)

    def test(self, is_cuda=True):
        self.cuda()
        self.eval()

        label_2d, vertex_pred, extents, poses, mdata = get_data(self.num_classes)

        label_2d = IT(label_2d, False)#.type(torch.IntTensor)
        # label_2d = FT(label_2d)
        vertex_pred = FT(vertex_pred, True)
        extents = FT(extents)
        poses = FT(poses)
        mdata = FT(mdata)

        print(label_2d.shape, vertex_pred.shape, extents.shape, poses.shape, mdata.shape)

        hough_outputs = self.forward(label_2d, vertex_pred, extents, poses, mdata)
        return hough_outputs[:4]


if __name__ == '__main__': # run with ipython -i -m hough_voting.tests.hough_voting_test 
    import cv2
    import numpy as np

    USE_CUDA = 1

    model = HoughNet()
    model.cuda()
    model.eval()
    label_2d, vertex_pred, extents, poses, mdata = get_data(model.num_classes)

    label_2d = IT(label_2d)#, True)#.type(torch.IntTensor)
    # label_2d = FT(label_2d)
    vertex_pred = FT(vertex_pred, True)
    extents = FT(extents)
    poses = FT(poses)
    mdata = FT(mdata)

    print(label_2d.shape, vertex_pred.shape, extents.shape, poses.shape, mdata.shape)
    hough_outputs = model.forward(label_2d, vertex_pred, extents, poses, mdata)

    rois, poses_init, poses_target, poses_weight, _ = hough_outputs

    # "to verify backward pass, run this"
    # poses_target.sum().backward()
    # assert np.sum(vertex_pred.grad.detach().cpu().numpy()) == 0

    img = cv2.imread("/home/bot/Documents/deep_learning/PoseCNN/data/LOV/data/0002/000001-color.png")

    rois = rois.detach().cpu().numpy()
    RED = (0,0,255)
    GREEN = (0,255,0)
    for roi in rois:
        bbox1 = np.round(roi[2:-1]).astype(np.int32)
        img = cv2.rectangle(img, tuple(bbox1[:2]), tuple(bbox1[2:]), RED)

    cv2.imshow("img", img)
    cv2.waitKey(0)
