#include <THC/THC.h>
#include <math.h>

#include <stdio.h>

#include "hard_label_kernel.h"

extern THCState *state;

int hard_label_forward_cuda(THCudaIntTensor* labelmap, THCudaTensor* prob, THCudaTensor* out, const float threshold)
{

    // Grab the input tensor
    const int* labelmap_flat = THCudaIntTensor_data(state, labelmap);
    const float* prob_flat = THCudaTensor_data(state, prob);
    float* out_flat = THCudaTensor_data(state, out);

    int batch_size = THCudaTensor_size(state, prob, 0);
    int height = THCudaTensor_size(state, prob, 1);
    int width = THCudaTensor_size(state, prob, 2);
    int num_classes = THCudaTensor_size(state, prob, 3);

    cudaStream_t stream = THCState_getCurrentStream(state);

    int rt = HardlabelForwardLaucher(prob_flat, labelmap_flat, batch_size, height, width, num_classes, threshold, out_flat, stream);

    return rt;
}

