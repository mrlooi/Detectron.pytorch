#pragma once

#include <vector>
#include <torch/extension.h>

#ifdef WITH_CUDA
#include "hough_voting_cuda.h"
#endif

// Interface for Python
std::vector<at::Tensor> hough_voting_forward(
    const at::Tensor& labelmap, const at::Tensor& vertmap, const at::Tensor& extents, const at::Tensor& meta_data, const at::Tensor& gt,
    const int num_classes, 
    const int is_train, const float inlierThreshold, const int labelThreshold, const float votingThreshold, const float perThreshold, 
    const int skip_pixels
) 
{
  if (labelmap.type().is_cuda()) 
  {
#ifdef WITH_CUDA
    return hough_voting_forward_cuda(labelmap, vertmap, extents, meta_data, gt, num_classes, is_train, 
        inlierThreshold, labelThreshold, votingThreshold, perThreshold, skip_pixels);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
  // return hough_voting_forward_cpu(input1, input2);
}

// std::vector<at::Tensor> hough_voting_backward(const at::Tensor& grad) 
// {
//   if (grad.type().is_cuda()) {
// #ifdef WITH_CUDA
//     return hough_voting_backward_cuda(grad);
// #else
//     AT_ERROR("Not compiled with GPU support");
// #endif
//   }
//   AT_ERROR("Not implemented on the CPU");
//   // return hough_voting_backward_cpu(grad);
// }
