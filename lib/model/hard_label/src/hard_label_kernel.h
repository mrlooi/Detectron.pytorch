#ifndef _HOUGH_VOTING_KERNEL
#define _HOUGH_VOTING_KERNEL

#ifdef __cplusplus
extern "C" {
#endif

int HardlabelForwardLaucher(const float* bottom_prob, const int* bottom_gt,
  const int batch_size, const int height, const int width, const int num_classes,
  const float threshold, float* top_data, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif

