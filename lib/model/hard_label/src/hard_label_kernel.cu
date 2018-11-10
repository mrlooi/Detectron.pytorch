// #ifdef __cplusplus
// extern "C" {
// #endif

#include <stdio.h>
#include <vector>
#include <math.h>
#include <float.h>

#include "hard_label_kernel.h"


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



template <typename Dtype>
__global__ void HardlabelForward(const int nthreads, const float* bottom_prob, const int* bottom_gt, 
  const int num_classes, const float threshold, Dtype* top_data) 
{
  CUDA_1D_KERNEL_LOOP(index, nthreads) 
  {
    // for (int c = 0; c < num_classes; c++)
    //   top_data[index * num_classes + c] = 0.0;

    int gt_label = bottom_gt[index];
    if (gt_label != -1 && (gt_label > 0 || bottom_prob[index * num_classes + gt_label] < threshold))
      top_data[index * num_classes + gt_label] = 1.0;
  }
}


int HardlabelForwardLaucher(const float* bottom_prob, const int* bottom_gt,
  const int batch_size, const int height, const int width, const int num_classes,
  const float threshold, float* top_data, cudaStream_t stream)
{
  const int kThreadsPerBlock = 1024;
  const int output_size = batch_size * height * width;
  cudaError_t err;

  HardlabelForward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock,
                       kThreadsPerBlock, 0, stream>>>(output_size, bottom_prob, bottom_gt, num_classes, threshold, top_data);

  cudaThreadSynchronize();
  err = cudaGetLastError();
  if(cudaSuccess != err)
  {
    fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
    exit( -1 );
  }

  return 1;
}
