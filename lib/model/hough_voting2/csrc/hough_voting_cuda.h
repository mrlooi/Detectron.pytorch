#pragma once
#include <torch/extension.h>


std::vector<at::Tensor> hough_voting_forward_cuda(
    const at::Tensor& labels, const at::Tensor& masks, const at::Tensor& vertmap, const at::Tensor& extents, 
    const at::Tensor& meta_data, const at::Tensor& poses,
    const int num_classes, 
    const int is_train, const float inlierThreshold, const int labelThreshold, const float votingThreshold, const float perThreshold, 
    const int skip_pixels);

// std::vector<at::Tensor> hough_voting_backward_cuda(const at::Tensor& grad);