// #ifdef __cplusplus
// extern "C" {
// #endif

#include <stdio.h>
#include <vector>
#include <math.h>
#include <float.h>

// #include <thrust/device_vector.h>
// #include <thrust/copy.h>
// #include <thrust/extrema.h>
#include <cuda_runtime.h>

#include "average_distance_loss_kernel.h"


#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

 // CUDA: various checks for different function calls.
#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
  } while (0)

int AveragedistanceForwardLaucher(
    const float* bottom_prediction, const float* bottom_target, const float* bottom_weight, const float* bottom_point,
    const float* bottom_symmetry, const int batch_size, const int num_classes, const int num_points, const float margin,
    float* top_data, float* bottom_diff, cudaStream_t stream)
{
  
  return 1;
}