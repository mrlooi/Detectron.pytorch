
int resize_outputs(THCudaTensor* top_box, THCudaTensor* top_pose, THCudaTensor* top_target, THCudaTensor* top_weight, 
    THCudaIntTensor* top_domain, const int num_rois, const int num_classes);

int reset_outputs(THCudaTensor* top_box, THCudaTensor* top_pose, THCudaTensor* top_target, THCudaTensor* top_weight, 
    THCudaIntTensor* top_domain);

int hough_voting_forward_cuda(THCudaIntTensor* labelmap, THCudaTensor* vertmap, THCudaTensor* extents, THCudaTensor* meta_data, THCudaTensor* gt,
    const int num_classes, 
    const int is_train, const float inlierThreshold, const int labelThreshold, const float votingThreshold, const float perThreshold, 
    const int skip_pixels,
    THCudaTensor* top_box, THCudaTensor* top_pose, THCudaTensor* top_target, THCudaTensor* top_weight, 
    THCudaIntTensor* top_domain
    );

// int hough_voting_backward_cuda(int pooled_height, int pooled_width, float spatial_scale,
//                         THCudaTensor * top_grad, THCudaTensor * rois, THCudaTensor * bottom_grad, THCudaIntTensor * argmax);
