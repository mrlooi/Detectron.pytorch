
import sys
# sys.path.append("..")

import numpy as np
import torch
import torch.nn as nn
from modules.hough_voting import HoughVoting

FT = torch.FloatTensor

class HoughNet(nn.Module):
    def __init__(self):
        super(HoughNet, self).__init__()

        self.is_train = False
        self.num_classes = 22
        self.vote_threshold = -1.0
        self.vote_percentage = 0.02
        self.skip_pixels = 20
        self.label_threshold = 500
        self.inlier_threshold = 0.9

        self.hough_voting = HoughVoting(self.num_classes, self.vote_threshold, self.vote_percentage, label_threshold=self.label_threshold, 
        	inlier_threshold=self.inlier_threshold, skip_pixels=self.skip_pixels, is_train=self.is_train)

    def forward(self, label_2d, vertex_pred, extents, poses, meta_data):
        return self.hough_voting(label_2d, vertex_pred, extents, poses, meta_data)

    def test(self, is_cuda=True):
        self.cuda()
        self.eval()

        im_scale = 1

        root_dir = "/home/vincent/Documents/deep_learning/PoseCNN/data"
        # extents blob
        extent_file = root_dir + "/LOV/extents.txt"
        extents = np.zeros((self.num_classes, 3), dtype=np.float32)
        extents[1:, :] = np.loadtxt(extent_file)

        # meta blob
        K = np.array([[1066.778, 0, 312.9869], [0, 1067.487, 241.3109], [0, 0, 1]])
        factor_depth = 10000.0
        meta_data = dict({'intrinsic_matrix': K, 'factor_depth': factor_depth})
        K = np.matrix(meta_data['intrinsic_matrix']) * im_scale
        K[2, 2] = 1
        Kinv = np.linalg.pinv(K)
        mdata = np.zeros(48, dtype=np.float32)
        mdata[0:9] = K.flatten()
        mdata[9:18] = Kinv.flatten()

        # label_2d blob
        label_2d_file = root_dir + "/demo_images/000005-label2d.npy"
        label_2d = np.load(label_2d_file)

        # vertex pred blob
        vertex_pred_file = root_dir + "/demo_images/000005-vert_pred.npy"
        vertex_pred = np.load(vertex_pred_file)
        assert label_2d.shape == vertex_pred.shape[:3]
        assert vertex_pred.shape[-1] == self.num_classes * 3

        # vertex_pred = np.transpose(vertex_pred, [0,2,3,1])

        label_2d = torch.IntTensor(label_2d)
        # label_2d = FT(label_2d)
        vertex_pred = FT(vertex_pred)
        extents = FT(extents)
        poses = torch.ones(len(label_2d), 13)
        mdata = FT(mdata)

        print(label_2d.shape, vertex_pred.shape, extents.shape, poses.shape, mdata.shape)
        if is_cuda:
            label_2d = label_2d.cuda()
            vertex_pred = vertex_pred.cuda()
            extents = extents.cuda()
            poses = poses.cuda()
            mdata = mdata.cuda()

        self.forward(label_2d, vertex_pred, extents, poses, mdata)

USE_CUDA = 1

model = HoughNet()
model.test(is_cuda=USE_CUDA)

