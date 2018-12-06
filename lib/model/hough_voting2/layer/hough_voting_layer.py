import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable

import _C

class _HoughVotingFunction(Function):
    @staticmethod
    def forward(ctx, labels, masks, vertex_pred, extents, poses, meta_data, num_classes, is_train, 
            inlier_threshold, label_threshold, threshold_vote, threshold_percentage, skip_pixels):
        assert extents.size(0) == num_classes
        # ctx.save_for_backward(masks, vertex_pred)

        if masks.is_cuda:
            masks_int = masks.type(torch.cuda.IntTensor) if masks.type() != 'torch.cuda.IntTensor' else masks
            labels_int = labels.type(torch.cuda.IntTensor) if labels.type() != 'torch.cuda.IntTensor' else labels
            output = _C.hough_voting_forward(labels_int, masks_int, vertex_pred, extents, meta_data, poses, 
                num_classes, is_train, inlier_threshold, label_threshold, threshold_vote, threshold_percentage, skip_pixels)
        else:
            raise NotImplementedError("Hough Voting Forward CPU layer not implemented!")

        top_box, top_pose, top_target, top_weight, top_domain = output  # MUST UNROLL THIS AND RETURN
        return top_box, top_pose, top_target, top_weight, top_domain

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

    def forward(self, labels, masks, vertex_pred, extents, poses, meta_data):
        return _HoughVotingFunction.apply(labels, masks, vertex_pred, extents, poses, meta_data, self.num_classes, self.is_train, 
            self.inlier_threshold, self.label_threshold, self.threshold_vote, self.threshold_percentage, self.skip_pixels)

    def __repr__(self):
        tmpstr = "%s (num_classes=%d, threshold_vote=%.2f, threshold_percentage=%.2f, label_threshold=%d, \
                        inlier_threshold=%.2f, skip_pixels=%d, is_train=%s)"%(self.__class__.__name__,  self.num_classes,
                self.threshold_vote, self.threshold_percentage, self.label_threshold, self.inlier_threshold, self.skip_pixels, self.is_train)
        return tmpstr
