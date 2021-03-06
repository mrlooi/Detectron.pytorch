import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from ave_dist_loss import _C  # TODO: CHANGE

class _AverageDistanceLossFunction(Function):
    @staticmethod
    def forward(ctx, poses_pred, poses_target, poses_weight, points, symmetry, num_classes, margin):
        assert points.size(0) == symmetry.size(0) == num_classes
        assert poses_pred.size(-1) == num_classes * 4
        assert poses_pred.size() == poses_target.size()

        outputs = _C.average_distance_loss_forward(poses_pred, poses_target, poses_weight, points, symmetry, num_classes, margin)
        
        loss, bottom_diff = outputs
        ctx.save_for_backward(bottom_diff)

        return loss

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        bottom_diff, = ctx.saved_tensors
        # grad_input = bottom_diff.new(*bottom_diff.size()).zero_()
        grad_input = _C.average_distance_loss_backward(grad, bottom_diff);

        """
        gradients for: poses_pred, poses_target, poses_weight, points, symmetry, num_classes, margin
        """
        return grad_input, None, None, None, None, None, None


class AverageDistanceLoss(nn.Module):
    def __init__(self, num_classes, margin):
        super(AverageDistanceLoss, self).__init__()

        self.num_classes = int(num_classes)
        self.margin = float(margin)

    def forward(self, poses_pred, poses_target, poses_weight, points, symmetry):
        return _AverageDistanceLossFunction.apply(poses_pred, poses_target, poses_weight, points, symmetry, self.num_classes, self.margin)

    def __repr__(self):
        tmpstr = "%s (num_classes=%d, margin=%.3f)"%(self.__class__.__name__,  self.num_classes, self.margin)
        return tmpstr


class _AverageDistanceLossFunction2(Function):
    @staticmethod
    def forward(ctx, poses_pred, poses_target, poses_labels, points, symmetry, margin):
        assert points.size(0) == symmetry.size(0)
        assert poses_pred.size(-1) == 4
        assert poses_pred.size() == poses_target.size()
        assert poses_pred.size(0) == poses_labels.size(0)

        if poses_labels.is_cuda:
            poses_labels_int = poses_labels.type(torch.cuda.IntTensor) if poses_labels.type() != 'torch.cuda.IntTensor' else poses_labels
            print(poses_labels_int)
            outputs = _C.average_distance_loss_forward2(poses_pred, poses_target, poses_labels_int, points, symmetry, margin)
        else:
            raise NotImplementedError("Average Distance Loss Forward CPU layer not implemented!")
        
        loss, bottom_diff = outputs
        ctx.save_for_backward(bottom_diff)

        return loss

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        bottom_diff, = ctx.saved_tensors
        # grad_input = bottom_diff.new(*bottom_diff.size()).zero_()
        grad_input = _C.average_distance_loss_backward(grad, bottom_diff);

        """
        gradients for: poses_pred, poses_target, poses_labels, points, symmetry, margin
        """
        return grad_input, None, None, None, None, None


class AverageDistanceLoss2(nn.Module):
    def __init__(self, margin):
        super(AverageDistanceLoss2, self).__init__()

        self.margin = float(margin)

    def forward(self, poses_pred, poses_target, poses_labels, points, symmetry):
        return _AverageDistanceLossFunction2.apply(poses_pred, poses_target, poses_labels, points, symmetry, self.margin)

    def __repr__(self):
        tmpstr = "%s (margin=%.3f)"%(self.__class__.__name__, self.margin)
        return tmpstr
