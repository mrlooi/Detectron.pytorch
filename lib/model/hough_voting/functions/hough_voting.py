import torch
from torch.autograd import Function
from .._ext import hough_voting
# from _ext import hough_voting

class HoughVotingFunction(Function):
    def __init__(ctx, num_classes, threshold_vote, threshold_percentage, label_threshold, inlier_threshold, skip_pixels=1, is_train=False):

        ctx.num_classes = num_classes
        ctx.threshold_vote = float(threshold_vote)
        ctx.threshold_percentage = float(threshold_percentage)
        ctx.label_threshold = int(label_threshold)
        ctx.inlier_threshold = float(inlier_threshold)
        ctx.skip_pixels = int(skip_pixels)

        ctx.is_train = is_train

    def forward(ctx, label_2d, vertex_pred, extents, poses, meta_data): 
        ctx.label_size = label_2d.size()           
        batch_size, data_height, data_width = ctx.label_size
        assert(extents.size()[0] == ctx.num_classes)
        # print(batch_size, data_height, data_width)

        # float tensors
        top_box = vertex_pred.new()
        top_pose = vertex_pred.new()
        top_target = vertex_pred.new()
        top_weight = vertex_pred.new()

        # int tensors

        if label_2d.is_cuda:
            label_2d_int = label_2d.type(torch.cuda.IntTensor) if label_2d.type() != 'torch.cuda.IntTensor' else label_2d
            top_domain = label_2d_int.new()  
            # hough_voting.allocate_outputs(top_box, top_pose, top_target, top_weight, top_domain, num_rois, ctx.num_classes);
            hough_voting.hough_voting_forward_cuda(label_2d_int, vertex_pred, extents, meta_data, poses, 
                ctx.num_classes,
                ctx.is_train, ctx.inlier_threshold, ctx.label_threshold, ctx.threshold_vote, ctx.threshold_percentage, 
                ctx.skip_pixels, 
                top_box, top_pose, top_target, top_weight, top_domain
                )
        else:
            raise NotImplementedError("Hough Voting Forward CPU layer not implemented!")

        # top_box is rois, which is shape (N, 7): batch_index, cls, x1, y1, x2, y2, max_hough_idx
        return top_box, top_pose, top_target, top_weight, top_domain

    def backward(ctx, grad_output):
        raise NotImplementedError("Hough Voting Backward layer not implemented!")        
        # hough_voting.hough_voting_backward_cuda(ctx.pooled_height, ctx.pooled_width, ctx.spatial_scale,
        #                                       grad_output, ctx.rois, grad_input, ctx.argmax)

        return None, None
