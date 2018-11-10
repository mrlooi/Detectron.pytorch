import torch
from torch.autograd import Function
from .._ext import hard_label
# from _ext import hard_label

class HardLabelFunction(Function):
    def __init__(ctx, threshold):

        ctx.threshold = float(threshold)

    def forward(ctx, label_gt, prob): 
        ctx.label_gt = label_gt
        ctx.prob = prob

        out = prob.new(*prob.size()).zero_()

        if label_gt.is_cuda:
            label_gt_int = label_gt.type(torch.cuda.IntTensor) if label_gt.type() != 'torch.cuda.IntTensor' else label_gt
            # hard_label.allocate_outputs(top_box, top_pose, top_target, top_weight, top_domain, num_rois, ctx.num_classes);
            hard_label.hard_label_forward_cuda(label_gt_int, prob, out, ctx.threshold)
        else:
            raise NotImplementedError("Hard Label Forward CPU layer not implemented!")

        return out

    def backward(ctx, grad): 
        """
        No gradients in hard label layer
        """
        g_lab = ctx.label_gt.new(*ctx.label_gt.size()).zero_()
        g_prob = ctx.prob.new(*ctx.prob.size()).zero_()

        return g_lab, g_prob
