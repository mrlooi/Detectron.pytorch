
int average_distance_loss_forward_cuda(THCudaTensor* poses_pred, THCudaTensor* poses_target, THCudaTensor* poses_weight, 
	THCudaTensor* points, THCudaTensor* symmetry, const int num_classes, const int num_points, const float margin,
    THCudaTensor* top_data
    );

// int hough_voting_backward_cuda(int pooled_height, int pooled_width, float spatial_scale,
//                         THCudaTensor * top_grad, THCudaTensor * rois, THCudaTensor * bottom_grad, THCudaIntTensor * argmax);
