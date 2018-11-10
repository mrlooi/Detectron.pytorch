#include <THC/THC.h>
#include <math.h>

#include <stdio.h>

#include "hough_voting_kernel.h"

extern THCState *state;

int resize_outputs(THCudaTensor* top_box, THCudaTensor* top_pose, THCudaTensor* top_target, THCudaTensor* top_weight, 
    THCudaIntTensor* top_domain, const int num_rois, const int num_classes)
{
    THCudaTensor_resize2d(state, top_box, num_rois, 7);  // batch_index, cls, x1, y1, x2, y2, max_hough_idx
    THCudaTensor_resize2d(state, top_pose, num_rois, 7);
    THCudaTensor_resize2d(state, top_target, num_rois, 4 * num_classes);
    THCudaTensor_resize2d(state, top_weight, num_rois, 4 * num_classes);
    
    THCudaIntTensor_resize1d(state, top_domain, num_rois);
    // THCudaIntTensor_resize1d(state, num_rois, 1);
    // THCudaIntTensor_fill(state, num_rois, 0);

    return 1;
}

int reset_outputs(THCudaTensor* top_box, THCudaTensor* top_pose, THCudaTensor* top_target, THCudaTensor* top_weight, 
    THCudaIntTensor* top_domain)
{
    THCudaTensor_fill(state, top_box, 0);
    THCudaTensor_fill(state, top_pose, 0);
    THCudaTensor_fill(state, top_target, 0);
    THCudaTensor_fill(state, top_weight, 0);
    THCudaIntTensor_fill(state, top_domain, 0);
    return 1;
}

int hough_voting_forward_cuda(THCudaIntTensor* labelmap, THCudaTensor* vertmap, THCudaTensor* extents, THCudaTensor* meta_data, THCudaTensor* gt,
    const int num_classes, 
    const int is_train, const float inlierThreshold, const int labelThreshold, const float votingThreshold, const float perThreshold, 
    const int skip_pixels,
    THCudaTensor* top_box, THCudaTensor* top_pose, THCudaTensor* top_target, THCudaTensor* top_weight, 
    THCudaIntTensor* top_domain
    )
{
    // Grab the input tensor
    const int* labelmap_flat = THCudaIntTensor_data(state, labelmap);
    const float* vertmap_flat = THCudaTensor_data(state, vertmap);

    const float* extents_flat = THCudaTensor_data(state, extents);
    const float* meta_data_flat = THCudaTensor_data(state, meta_data);
    const float* gt_flat = THCudaTensor_data(state, gt);

    // Number of ROIs
    int batch_size = THCudaIntTensor_size(state, labelmap, 0);
    int height = THCudaIntTensor_size(state, labelmap, 1);
    int width = THCudaIntTensor_size(state, labelmap, 2);
    int num_gt = THCudaTensor_size(state, gt, 0);

    cudaStream_t stream = THCState_getCurrentStream(state);

    int max_allowed_rois = is_train ? MAX_ROI * 9 : MAX_ROI; // maximum allowed per forward pass
    resize_outputs(top_box, top_pose, top_target, top_weight, top_domain, max_allowed_rois, num_classes);
    reset_outputs(top_box, top_pose, top_target, top_weight, top_domain);

    float* top_box_data = THCudaTensor_data(state, top_box); 
    float* top_pose_data = THCudaTensor_data(state, top_pose); 
    float* top_target_data = THCudaTensor_data(state, top_target); 
    float* top_weight_data = THCudaTensor_data(state, top_weight);
    int* top_domain_data = THCudaIntTensor_data(state, top_domain);

    int num_rois = 0;
    // THCudaIntTensor* num_rois = THCudaIntTensor_new(state);
    // int* num_rois_data = THCudaIntTensor_data(state, num_rois);

    for (int n = 0; n < batch_size; n++)
    {
        HoughVotingForwardLaucher(
            labelmap_flat, vertmap_flat, extents_flat, meta_data_flat, gt_flat,
            n, batch_size, height, width, num_classes, num_gt,
            is_train, inlierThreshold, labelThreshold, votingThreshold, perThreshold, 
            skip_pixels,
            top_box_data, top_pose_data, top_target_data, top_weight_data, top_domain_data, &num_rois, stream);
    }

    // then resize outputs based on num_rois
    // printf("num_rois: %d\n", num_rois);
    resize_outputs(top_box, top_pose, top_target, top_weight, top_domain, num_rois, num_classes);

    // THCudaTensor_free(state, num_rois);

    return 1;
}

// int hough_voting_backward_cuda(int pooled_height, int pooled_width, float spatial_scale,
//                         THCudaTensor * top_grad, THCudaTensor * rois, THCudaTensor * bottom_grad, THCudaIntTensor * argmax)
// {

//     return 1;
// }
