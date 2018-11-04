#ifndef _AVERAGE_DISTANCE_LOSS_KERNEL
#define _AVERAGE_DISTANCE_LOSS_KERNEL

#define POSE_CHANNELS 4

#ifdef __cplusplus
extern "C" {
#endif

int AveragedistanceForwardLaucher(
    const float* bottom_prediction, const float* bottom_target, const float* bottom_weight, const float* bottom_point,
    const float* bottom_symmetry, const int batch_size, const int num_classes, const int num_points, const float margin,
    float* top_data, float* bottom_diff, cudaStream_t stream);

int AveragedistanceBackwardLaucher(const float* top_diff, const float* bottom_diff, const int batch_size,
    const int channels, float* output, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif

