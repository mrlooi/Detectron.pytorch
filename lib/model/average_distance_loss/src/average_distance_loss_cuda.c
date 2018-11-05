#include <THC/THC.h>
#include <math.h>

#include <stdio.h>

#include "average_distance_loss_kernel.h"

extern THCState *state;

int average_distance_loss_forward_cuda(THCudaTensor* poses_pred, THCudaTensor* poses_target, THCudaTensor* poses_weight, 
    THCudaTensor* points, THCudaTensor* symmetry, const int num_classes, const int num_points, const float margin,
    THCudaTensor* loss, THCudaTensor* bottom_diff
)
{
    // Grab the input tensor
    const float* poses_pred_flat = THCudaTensor_data(state, poses_pred);
    const float* poses_target_flat = THCudaTensor_data(state, poses_target);
    const float* poses_weight_flat = THCudaTensor_data(state, poses_weight);
    const float* points_flat = THCudaTensor_data(state, points);
    const float* symmetry_flat = THCudaTensor_data(state, symmetry);

    float* loss_flat = THCudaTensor_data(state, loss); 
    float* bottom_diff_flat = THCudaTensor_data(state, bottom_diff); ;

    int batch_size = THCudaTensor_size(state, poses_pred, 0);

    cudaStream_t stream = THCState_getCurrentStream(state);

    AveragedistanceForwardLaucher(    
        poses_pred_flat, poses_target_flat, poses_weight_flat, points_flat, symmetry_flat, 
        batch_size, num_classes, num_points, margin,
        loss_flat, bottom_diff_flat, stream
    );

    return 1;
}

// int hough_voting_backward_cuda(int pooled_height, int pooled_width, float spatial_scale,
//                         THCudaTensor * top_grad, THCudaTensor * rois, THCudaTensor * bottom_grad, THCudaIntTensor * argmax)
// {

//     return 1;
// }
