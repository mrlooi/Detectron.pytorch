import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from hough_voting import _C  # TODO: CHANGE

class _HoughVotingFunction(Function):
    @staticmethod
    def forward(ctx, label_2d, vertex_pred, extents, poses, meta_data, num_classes, is_train, 
            inlier_threshold, label_threshold, threshold_vote, threshold_percentage, skip_pixels):
        assert extents.size(0) == num_classes
        ctx.save_for_backward(label_2d, vertex_pred)

        if label_2d.is_cuda:
            label_2d_int = label_2d.type(torch.cuda.IntTensor) if label_2d.type() != 'torch.cuda.IntTensor' else label_2d
            print(meta_data)
            output = _C.hough_voting_forward(label_2d_int, vertex_pred, extents, meta_data, poses, 
                num_classes, is_train, inlier_threshold, label_threshold, threshold_vote, threshold_percentage, skip_pixels)
        else:
            raise NotImplementedError("Hough Voting Forward CPU layer not implemented!")

        top_box, top_pose, top_target, top_weight, top_domain = output  # MUST UNROLL THIS AND RETURN
        return top_box, top_pose, top_target, top_weight, top_domain

    @staticmethod
    @once_differentiable
    def backward(ctx, grad, tmp, tmp1, tmp2, _):
        label_2d, vertex_pred, = ctx.saved_tensors
        g_lab = label_2d.new(*label_2d.size()).zero_()
        g_vp = vertex_pred.new(*vertex_pred.size()).zero_()

        """
        gradients for: label_2d_int, vertex_pred, extents, meta_data, poses, 
            num_classes, is_train, inlier_threshold, label_threshold, threshold_vote, threshold_percentage, skip_pixels
        """
        return g_lab, g_vp, None, None, None, None, None, None, None, None, None, None 


class HoughVoting(nn.Module):
    def __init__(self, num_classes, threshold_vote, threshold_percentage, label_threshold=500, inlier_threshold=0.9, skip_pixels=1, is_train=False):
        super(HoughVoting, self).__init__()

        self.num_classes = int(num_classes)
        self.label_threshold = int(label_threshold)
        self.inlier_threshold = float(inlier_threshold)
        self.threshold_vote = float(threshold_vote)
        self.threshold_percentage = float(threshold_percentage)
        self.skip_pixels = int(skip_pixels)

        self.is_train = int(is_train)

    def forward(self, label_2d, vertex_pred, extents, poses, meta_data):
        return _HoughVotingFunction.apply(label_2d, vertex_pred, extents, poses, meta_data, self.num_classes, self.is_train, 
            self.inlier_threshold, self.label_threshold, self.threshold_vote, self.threshold_percentage, self.skip_pixels)

    def __repr__(self):
        tmpstr = "%s (num_classes=%d, threshold_vote=%.2f, threshold_percentage=%.2f, label_threshold=%d, \
                        inlier_threshold=%.2f, skip_pixels=%d, is_train=%s)"%(self.__class__.__name__,  self.num_classes,
                self.threshold_vote, self.threshold_percentage, self.label_threshold, self.inlier_threshold, self.skip_pixels, self.is_train)
        return tmpstr
