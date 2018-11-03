#ifndef _HOUGH_VOTING_KERNEL
#define _HOUGH_VOTING_KERNEL

#define VERTEX_CHANNELS 3
#define MAX_ROI 128

#ifdef __cplusplus
extern "C" {
#endif

int HoughVotingForwardLaucher(
    const int* labelmap, const float* vertmap, const float* extents, const float* meta_data, const float* gt,
    const int batch_index, const int batch_size, const int height, const int width, const int num_classes, const int num_gt, 
    const int is_train, const float inlierThreshold, const int labelThreshold, const float votingThreshold, const float perThreshold, 
    const int skip_pixels, 
    float* top_box, float* top_pose, float* top_target, float* top_weight, int* top_domain, int* num_rois, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif

