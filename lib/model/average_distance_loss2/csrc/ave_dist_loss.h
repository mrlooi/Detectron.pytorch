#pragma once

#include <vector>
#include <torch/extension.h>

#ifdef WITH_CUDA
#include "ave_dist_loss_cuda.h"
#endif

// Interface for Python
std::vector<at::Tensor> ave_dist_loss_forward(
    const at::Tensor& poses_pred, const at::Tensor& poses_target, const at::Tensor& poses_weight, const at::Tensor& points, const at::Tensor& symmetry,
    const int num_classes, const float margin
) 
{
  if (poses_pred.type().is_cuda()) 
  {
#ifdef WITH_CUDA
    return ave_dist_loss_forward_cuda(poses_pred, poses_target, poses_weight, points, symmetry, num_classes, margin);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
  // return ave_dist_loss_forward_cpu(input1, input2);
}

at::Tensor ave_dist_loss_backward(const at::Tensor& grad, const at::Tensor& bottom_diff) 
{
  if (grad.type().is_cuda()) 
  {
#ifdef WITH_CUDA
    return ave_dist_loss_backward_cuda(grad, bottom_diff);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
  // return ave_dist_loss_backward_cpu(grad);
}
