#include <THC/THC.h>
#include <math.h>

#include <stdio.h>

#include "hough_voting_kernel.h"

extern THCState *state;

// int HoughVotingForwardLaucher(
//     const int* labelmap, const float* vertmap, const float* extents, const float* meta_data, const float* gt,
//     const int batch_index, const int batch_size, const int height, const int width, const int num_classes, const int num_gt, 
//     const int is_train, const float inlierThreshold, const int labelThreshold, const float votingThreshold, const float perThreshold, 
//     const int skip_pixels, 
//     float* top_box, float* top_pose, float* top_target, float* top_weight, int* top_domain, int* num_rois, cudaStream_t stream);

int hough_voting_forward_cuda(THCudaIntTensor* labelmap, THCudaTensor* vertmap, THCudaTensor* extents, THCudaTensor* meta_data, THCudaTensor* gt,
    const int num_classes, 
    const int is_train, const float inlierThreshold, const int labelThreshold, const float votingThreshold, const float perThreshold, 
    const int skip_pixels
    )
{
    // Grab the input tensor
    int* labelmap_flat = THCudaIntTensor_data(state, labelmap);
    float* vertmap_flat = THCudaTensor_data(state, vertmap);

    float* extents_flat = THCudaTensor_data(state, extents);
    float* meta_data_flat = THCudaTensor_data(state, meta_data);
    float* gt_flat = THCudaTensor_data(state, gt);

    // Number of ROIs
    int batch_size = THCudaIntTensor_size(state, labelmap, 0);
    int height = THCudaIntTensor_size(state, labelmap, 1);
    int width = THCudaIntTensor_size(state, labelmap, 2);
    int num_gt = THCudaTensor_size(state, gt, 0);

    cudaStream_t stream = THCState_getCurrentStream(state);

    float* top_box; 
    float* top_pose; 
    float* top_target; 
    float* top_weight;
    int* top_domain;
    int* num_rois;

    printf("batch_size: %d\n", batch_size);

    for (int n = 0; n < batch_size; n++)
    {
        printf("n: %d\n", n);
        HoughVotingForwardLaucher(
            labelmap_flat, vertmap_flat, extents_flat, meta_data_flat, gt_flat,
            n, batch_size, height, width, num_classes, num_gt,
            is_train, inlierThreshold, labelThreshold, votingThreshold, perThreshold, 
            skip_pixels,
            top_box, top_pose, top_target, top_weight, top_domain, num_rois, stream);
    }

    return 1;
}

// int hough_voting_backward_cuda(int pooled_height, int pooled_width, float spatial_scale,
//                         THCudaTensor * top_grad, THCudaTensor * rois, THCudaTensor * bottom_grad, THCudaIntTensor * argmax)
// {

//     return 1;
// }
