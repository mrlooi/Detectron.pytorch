import torch
from torch.autograd import Function
from .._ext import average_distance_loss

class AverageDistanceLossFunction(Function):
    def __init__(ctx, num_classes, margin):

        ctx.num_classes = int(num_classes)
        ctx.margin = float(margin)

    def forward(ctx, poses_pred, poses_target, poses_weight, points, symmetry): 
        assert(points.size()[0] == symmetry.size()[0] == ctx.num_classes)

        ctx.poses_pred_size = poses_pred.size()           
        num_rois, cn = ctx.poses_pred_size
        num_pts = points.size()[1]

        # assert(cn == ctx.num_classes * 4)

        loss = poses_pred.new(1).zero_()

        if poses_pred.is_cuda:
            average_distance_loss.average_distance_loss_forward_cuda(poses_pred, poses_target, poses_weight, points, symmetry, 
                ctx.num_classes, num_pts, ctx.margin, loss)
        else:
            raise NotImplementedError("Average Distance Loss Forward CPU layer not implemented!")

        return loss 

    def backward(ctx, grad_output):
        raise NotImplementedError("Average Distance Loss Backward layer not implemented!")        

        return None, None
