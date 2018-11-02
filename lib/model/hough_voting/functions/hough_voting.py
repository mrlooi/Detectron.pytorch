import torch
from torch.autograd import Function
# from .._ext import hough_voting
from _ext import hough_voting

# import pdb

# int hough_voting_forward_cuda(THCudaIntTensor* labelmap, THCudaTensor* vertmap, THCudaTensor* extents, THCudaTensor* meta_data, THCudaTensor* gt,
#     const int num_classes, 
#     const int is_train, const float inlierThreshold, const int labelThreshold, const float votingThreshold, const float perThreshold, 
#     const int skip_pixels
#     )

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

        # output = features.new(num_rois, num_channels, ctx.pooled_height, ctx.pooled_width).zero_()

        if label_2d.is_cuda:
            hough_voting.hough_voting_forward_cuda(label_2d, vertex_pred, extents, meta_data, poses, 
                ctx.num_classes,
                ctx.is_train, ctx.inlier_threshold, ctx.label_threshold, ctx.threshold_vote, ctx.threshold_percentage, 
                ctx.skip_pixels)
        else:
            raise NotImplementedError("Hough Voting Forward CPU layer not implemented!")

        return label_2d

    def backward(ctx, grad_output):
        raise NotImplementedError("Hough Voting Backward layer not implemented!")        
        # hough_voting.hough_voting_backward_cuda(ctx.pooled_height, ctx.pooled_width, ctx.spatial_scale,
        #                                       grad_output, ctx.rois, grad_input, ctx.argmax)

        return None, None
