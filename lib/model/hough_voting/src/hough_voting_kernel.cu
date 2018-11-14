// #ifdef __cplusplus
// extern "C" {
// #endif

#include <stdio.h>
#include <vector>
#include <math.h>
#include <float.h>

#include <thrust/execution_policy.h>  // for certain cuda versions, this is where 'thrust::device' is

// #include <thrust/device_vector.h>
// #include <thrust/copy.h>
#include <thrust/extrema.h>
#include <cuda_runtime.h>

#include "hough_voting_kernel.h"
#include "hough_voting_cuda_utils.h"


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

__global__ void compute_arrays_kernel(const int nthreads, const int* labelmap,
    int* arrays, int* array_size, const int height, const int width) 
{
  CUDA_1D_KERNEL_LOOP(index, nthreads) 
  {
    int cls = labelmap[index];
    if (cls > 0)
    {
      int size = atomicAdd(array_size + cls, 1);
      int offset = cls * height * width + size;
      arrays[offset] = index;
    }
  }
}

__global__ void compute_hough_kernel(const int nthreads, float* hough_space, float* hough_data, const int* labelmap, 
    const float* vertmap, const float* extents, const float* meta_data, const int* arrays, const int* array_size, 
    const int* class_indexes, const int height, const int width, const int num_classes, const int count, const float inlierThreshold, const int skip_pixels) 
{
  CUDA_1D_KERNEL_LOOP(index, nthreads) 
  {
    // (cls, cx, cy) is an element in the hough space
    int ind = index / (height * width);
    int cls = class_indexes[ind];
    int n = index % (height * width);
    int cx = n % width;
    int cy = n / width;
    int size = array_size[cls];
    float distance = 0;
    float threshold;

    for (int i = 0; i < size; i += skip_pixels)
    {
      int offset = cls * height * width + i;
      int location = arrays[offset];
      int x = location % width;
      int y = location / width;

      // read the direction
      offset = VERTEX_CHANNELS * cls + VERTEX_CHANNELS * num_classes * (y * width + x);
      float u = vertmap[offset];
      float v = vertmap[offset + 1];
      float d = exp(vertmap[offset + 2]);

      // vote
      if (angle_distance(cx, cy, x, y, u, v) > inlierThreshold)
      {
        project_box(cls, extents, meta_data, d, 0.6, &threshold);
        float dx = fabsf(x - cx);
        float dy = fabsf(y - cy);
        if (dx < threshold && dy < threshold)
        {
          hough_space[index]++;
          distance += d;
        }
      }
    }

    if (hough_space[index] > 0)
    {
      distance /= hough_space[index];

      float bb_width = -1;
      float bb_height = -1;
      for (int i = 0; i < size; i += skip_pixels)
      {
        int offset = cls * height * width + i;
        int location = arrays[offset];
        int x = location % width;
        int y = location / width;

        // read the direction
        offset = VERTEX_CHANNELS * cls + VERTEX_CHANNELS * num_classes * (y * width + x);
        float u = vertmap[offset];
        float v = vertmap[offset + 1];

        // vote
        if (angle_distance(cx, cy, x, y, u, v) > inlierThreshold)
        {
          project_box(cls, extents, meta_data, distance, 0.6, &threshold);
          float dx = fabsf(x - cx);
          float dy = fabsf(y - cy);
          if (dx > bb_width && dx < threshold && dy < threshold)
            bb_width = dx;
          if (dy > bb_height && dx < threshold && dy < threshold)
            bb_height = dy;
        }
      }

      int offset = ind * height * width * 3 + 3 * (cy * width + cx);
      hough_data[offset] = distance;
      hough_data[offset + 1] = 2 * bb_height;
      hough_data[offset + 2] = 2 * bb_width;
    }
  }
}

__global__ void compute_max_indexes_kernel(const int nthreads, int* max_indexes, int index_size, int* num_max, float* hough_space, 
  float* hough_data, int height, int width, float threshold, float perThreshold)
{
  CUDA_1D_KERNEL_LOOP(index, nthreads) 
  {
    // (ind, cx, cy) is an element in the hough space
    int ind = index / (height * width);
    int n = index % (height * width);
    int cx = n % width;
    int cy = n / width;
    int kernel_size = 3;

    int offset = ind * height * width * 3 + 3 * (cy * width + cx);
    float bb_height = hough_data[offset + 1];
    float bb_width = hough_data[offset + 2];

    if (hough_space[index] > threshold && bb_height > 0 && bb_width > 0)
    {
      // check if the location is local maximum
      int flag = 0;
      for (int x = cx - kernel_size; x <= cx + kernel_size; x++)
      {
        for (int y = cy - kernel_size; y <= cy + kernel_size; y++)
        {
          if (x >= 0 && x < width && y >= 0 && y < height)
          {
            if (hough_space[ind * height * width + y * width + x] > hough_space[index])
            {
              flag = 1;
              break;
            }
          }
        }

        // check the percentage of voting
        if (hough_space[index] / (bb_height * bb_width) < perThreshold)
          flag = 1;
      }

      if (flag == 0)
      {
        // add the location to max_indexes
        int max_index = atomicAdd(num_max, 1);
        if (max_index < index_size)
          max_indexes[max_index] = index;
      }
    }
  }
}


__global__ void compute_rois_kernel(const int nthreads, float* top_box, float* top_pose, float* top_target, float* top_weight, int* top_domain,
    const float* extents, const float* meta_data, const float* gt, const float* hough_space, const float* hough_data, const int* max_indexes, const int* class_indexes,
    int is_train, int batch_index, const int height, const int width, const int num_classes, const int num_gt, int* num_rois) 
{
  __shared__ float s_f[4];

  CUDA_1D_KERNEL_LOOP(index, nthreads) 
  {
    if (threadIdx.x == 0)
    {
      s_f[0] = meta_data[0]; // fx
      s_f[1] = meta_data[4]; // fy
      s_f[2] = meta_data[2]; // px
      s_f[3] = meta_data[5]; // py
    }
    __syncthreads();

    float fx = s_f[0];
    float fy = s_f[1];
    float px = s_f[2];
    float py = s_f[3];

    float scale = 0.05;
    int max_index = max_indexes[index];
    float max_hs_idx = hough_space[max_index];
    int ind = max_index / (height * width);
    int cls = class_indexes[ind];
    int n = max_index % (height * width);
    int x = n % width;
    int y = n / width;

    float rx = (x - px) / fx;
    float ry = (y - py) / fy;

    int offset = ind * height * width * 3 + 3 * (y * width + x);
    float bb_distance = hough_data[offset];
    float bb_height = hough_data[offset + 1];
    float bb_width = hough_data[offset + 2];

    if (!is_train)
    {
      int roi_index = atomicAdd(num_rois, 1);
      top_box[roi_index * 7 + 0] = batch_index;
      top_box[roi_index * 7 + 1] = cls;
      top_box[roi_index * 7 + 2] = x - bb_width * (0.5 + scale);
      top_box[roi_index * 7 + 3] = y - bb_height * (0.5 + scale);
      top_box[roi_index * 7 + 4] = x + bb_width * (0.5 + scale);
      top_box[roi_index * 7 + 5] = y + bb_height * (0.5 + scale);
      top_box[roi_index * 7 + 6] = max_hs_idx;
      
      top_pose[roi_index * 7 + 0] = 1;
      top_pose[roi_index * 7 + 1] = 0;
      top_pose[roi_index * 7 + 2] = 0;
      top_pose[roi_index * 7 + 3] = 0;
      top_pose[roi_index * 7 + 4] = rx * bb_distance;
      top_pose[roi_index * 7 + 5] = ry * bb_distance;
      top_pose[roi_index * 7 + 6] = bb_distance;
    }
    else
    {
      int roi_index = atomicAdd(num_rois, 9);
      top_box[roi_index * 7 + 0] = batch_index;
      top_box[roi_index * 7 + 1] = cls;
      top_box[roi_index * 7 + 2] = x - bb_width * (0.5 + scale);
      top_box[roi_index * 7 + 3] = y - bb_height * (0.5 + scale);
      top_box[roi_index * 7 + 4] = x + bb_width * (0.5 + scale);
      top_box[roi_index * 7 + 5] = y + bb_height * (0.5 + scale);
      top_box[roi_index * 7 + 6] = max_hs_idx;

      for (int i = 0; i < 9; i++)
      {
        top_pose[(roi_index + i) * 7 + 0] = 1;
        top_pose[(roi_index + i) * 7 + 1] = 0;
        top_pose[(roi_index + i) * 7 + 2] = 0;
        top_pose[(roi_index + i) * 7 + 3] = 0;
        top_pose[(roi_index + i) * 7 + 4] = rx * bb_distance;
        top_pose[(roi_index + i) * 7 + 5] = ry * bb_distance;
        top_pose[(roi_index + i) * 7 + 6] = bb_distance;

        if (num_gt == 0)
          top_domain[roi_index + i] = 1;
        else
          top_domain[roi_index + i] = 0;
      }

      // compute pose target
      for (int i = 0; i < num_gt; i++)
      {
        int gt_batch = int(gt[i * 13 + 0]);
        int gt_id = int(gt[i * 13 + 1]);
        if(cls == gt_id && batch_index == gt_batch)
        {
          int gt_ind = i;

          float overlap = compute_box_overlap(cls, extents, meta_data, gt + gt_ind * 13, top_box + roi_index * 7 + 2);
          if (overlap > 0.2)
          {
            for (int j = 0; j < 9; j++)
            {
              top_target[(roi_index + j) * 4 * num_classes + 4 * cls + 0] = gt[gt_ind * 13 + 6];
              top_target[(roi_index + j) * 4 * num_classes + 4 * cls + 1] = gt[gt_ind * 13 + 7];
              top_target[(roi_index + j) * 4 * num_classes + 4 * cls + 2] = gt[gt_ind * 13 + 8];
              top_target[(roi_index + j) * 4 * num_classes + 4 * cls + 3] = gt[gt_ind * 13 + 9];

              top_weight[(roi_index + j) * 4 * num_classes + 4 * cls + 0] = 1;
              top_weight[(roi_index + j) * 4 * num_classes + 4 * cls + 1] = 1;
              top_weight[(roi_index + j) * 4 * num_classes + 4 * cls + 2] = 1;
              top_weight[(roi_index + j) * 4 * num_classes + 4 * cls + 3] = 1;
            }
            break;
          }
        }
      }

      // add jittering boxes
      float x1 = top_box[roi_index * 7 + 2];
      float y1 = top_box[roi_index * 7 + 3];
      float x2 = top_box[roi_index * 7 + 4];
      float y2 = top_box[roi_index * 7 + 5];
      float ww = x2 - x1;
      float hh = y2 - y1;

      // (-1, -1)
      roi_index++;
      top_box[roi_index * 7 + 0] = batch_index;
      top_box[roi_index * 7 + 1] = cls;
      top_box[roi_index * 7 + 2] = x1 - 0.05 * ww;
      top_box[roi_index * 7 + 3] = y1 - 0.05 * hh;
      top_box[roi_index * 7 + 4] = top_box[roi_index * 7 + 2] + ww;
      top_box[roi_index * 7 + 5] = top_box[roi_index * 7 + 3] + hh;
      top_box[roi_index * 7 + 6] = max_hs_idx;

      // (+1, -1)
      roi_index++;
      top_box[roi_index * 7 + 0] = batch_index;
      top_box[roi_index * 7 + 1] = cls;
      top_box[roi_index * 7 + 2] = x1 + 0.05 * ww;
      top_box[roi_index * 7 + 3] = y1 - 0.05 * hh;
      top_box[roi_index * 7 + 4] = top_box[roi_index * 7 + 2] + ww;
      top_box[roi_index * 7 + 5] = top_box[roi_index * 7 + 3] + hh;
      top_box[roi_index * 7 + 6] = max_hs_idx;

      // (-1, +1)
      roi_index++;
      top_box[roi_index * 7 + 0] = batch_index;
      top_box[roi_index * 7 + 1] = cls;
      top_box[roi_index * 7 + 2] = x1 - 0.05 * ww;
      top_box[roi_index * 7 + 3] = y1 + 0.05 * hh;
      top_box[roi_index * 7 + 4] = top_box[roi_index * 7 + 2] + ww;
      top_box[roi_index * 7 + 5] = top_box[roi_index * 7 + 3] + hh;
      top_box[roi_index * 7 + 6] = max_hs_idx;

      // (+1, +1)
      roi_index++;
      top_box[roi_index * 7 + 0] = batch_index;
      top_box[roi_index * 7 + 1] = cls;
      top_box[roi_index * 7 + 2] = x1 + 0.05 * ww;
      top_box[roi_index * 7 + 3] = y1 + 0.05 * hh;
      top_box[roi_index * 7 + 4] = top_box[roi_index * 7 + 2] + ww;
      top_box[roi_index * 7 + 5] = top_box[roi_index * 7 + 3] + hh;
      top_box[roi_index * 7 + 6] = max_hs_idx;

      // (0, -1)
      roi_index++;
      top_box[roi_index * 7 + 0] = batch_index;
      top_box[roi_index * 7 + 1] = cls;
      top_box[roi_index * 7 + 2] = x1;
      top_box[roi_index * 7 + 3] = y1 - 0.05 * hh;
      top_box[roi_index * 7 + 4] = top_box[roi_index * 7 + 2] + ww;
      top_box[roi_index * 7 + 5] = top_box[roi_index * 7 + 3] + hh;
      top_box[roi_index * 7 + 6] = max_hs_idx;

      // (-1, 0)
      roi_index++;
      top_box[roi_index * 7 + 0] = batch_index;
      top_box[roi_index * 7 + 1] = cls;
      top_box[roi_index * 7 + 2] = x1 - 0.05 * ww;
      top_box[roi_index * 7 + 3] = y1;
      top_box[roi_index * 7 + 4] = top_box[roi_index * 7 + 2] + ww;
      top_box[roi_index * 7 + 5] = top_box[roi_index * 7 + 3] + hh;
      top_box[roi_index * 7 + 6] = max_hs_idx;

      // (0, +1)
      roi_index++;
      top_box[roi_index * 7 + 0] = batch_index;
      top_box[roi_index * 7 + 1] = cls;
      top_box[roi_index * 7 + 2] = x1;
      top_box[roi_index * 7 + 3] = y1 + 0.05 * hh;
      top_box[roi_index * 7 + 4] = top_box[roi_index * 7 + 2] + ww;
      top_box[roi_index * 7 + 5] = top_box[roi_index * 7 + 3] + hh;
      top_box[roi_index * 7 + 6] = max_hs_idx;

      // (+1, 0)
      roi_index++;
      top_box[roi_index * 7 + 0] = batch_index;
      top_box[roi_index * 7 + 1] = cls;
      top_box[roi_index * 7 + 2] = x1 + 0.05 * ww;
      top_box[roi_index * 7 + 3] = y1;
      top_box[roi_index * 7 + 4] = top_box[roi_index * 7 + 2] + ww;
      top_box[roi_index * 7 + 5] = top_box[roi_index * 7 + 3] + hh;
      top_box[roi_index * 7 + 6] = max_hs_idx;
    }
  }
}

int HoughVotingForwardLaucher(
    const int* labelmap, const float* vertmap, const float* extents, const float* meta_data, const float* gt,
    const int batch_index, const int batch_size, const int height, const int width, const int num_classes, const int num_gt, 
    const int is_train, const float inlierThreshold, const int labelThreshold, const float votingThreshold, const float perThreshold, 
    const int skip_pixels, 
    float* top_box, float* top_pose, float* top_target, float* top_weight, int* top_domain, int* num_rois, cudaStream_t stream)
{
  const int kThreadsPerBlock = 1024;
  cudaError_t err;

  int output_size = height * width;

  // step 1: compute a label index array for each class
  int dims = num_classes * height * width;
  // thrust::device_vector<int> arrays_vec(dims);
  // int* arrays = thrust::raw_pointer_cast(arrays_vec.data());

  // unique_ptr_device<int> arrays_vec;
  // cudaMalloc(arrays_vec, dims * sizeof(int));
  int* arrays;// = arrays_vec.get();
  cudaMalloc((void **)&arrays, dims * sizeof(int));

  // thrust::device_vector<int> array_sizes_vec(num_classes);
  // int* array_sizes = thrust::raw_pointer_cast(array_sizes_vec.data());

  // unique_ptr_device<int> array_sizes_vec;
  // cudaMalloc(array_sizes_vec, num_classes * sizeof(int));
  int* array_sizes;// = array_sizes_vec.get();  
  cudaMalloc((void **)&array_sizes, num_classes * sizeof(int));

  cudaMemset(array_sizes, 0, num_classes * sizeof(int));

  compute_arrays_kernel<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock,
                       kThreadsPerBlock, 0, stream>>>(
      output_size, labelmap, arrays, array_sizes, height, width);
  cudaThreadSynchronize();

  // compute class indexes
  std::vector<int> array_sizes_host(num_classes);
  // thrust::copy(array_sizes_vec.begin(), array_sizes_vec.end(), array_sizes_host.begin());
  cudaMemcpy(&array_sizes_host[0], array_sizes, num_classes * sizeof(int), cudaMemcpyDeviceToHost);

  std::vector<int> class_indexes_host(num_classes);
  int count = 0;
  for (int c = 1; c < num_classes; c++)
  {
    if (array_sizes_host[c] > labelThreshold)
    {
      class_indexes_host[count] = c;
      // printf("Class %d) %d (labels count: %d)\n", c, count, array_sizes_host[c]);
      count++;
    }
  }

  if (count == 0)
  {
    // printf("RETURN\n");
    cudaFree(arrays);
    cudaFree(array_sizes);
    return 1;
  }

  // thrust::device_vector<int> class_indexes_vec(count);
  int* class_indexes;// = thrust::raw_pointer_cast(class_indexes_vec.data());
  cudaMalloc((void **)&class_indexes, count * sizeof(int));
  cudaMemcpy(class_indexes, &class_indexes_host[0], count * sizeof(int), cudaMemcpyHostToDevice);

  err = cudaGetLastError();
  if(cudaSuccess != err)
  {
    fprintf( stderr, "cudaCheckError() failed compute label index: %s\n", cudaGetErrorString( err ) );
    exit( -1 );
  }

  // step 2: compute the hough space
  // thrust::device_vector<float> hough_space_vec(count * height * width);
  float* hough_space; // = thrust::raw_pointer_cast(hough_space_vec.data());
  cudaMalloc((void **)&hough_space, count * height * width * sizeof(float));
  if (cudaMemset(hough_space, 0, count * height * width * sizeof(float)) != cudaSuccess)
    fprintf(stderr, "reset error\n");

  // thrust::device_vector<float> hough_data_vec(count * height * width * 3);
  float* hough_data; // = thrust::raw_pointer_cast(hough_data_vec.data());
  cudaMalloc((void **)&hough_data, count * height * width * 3 * sizeof(float));
  if (cudaMemset(hough_data, 0, count * height * width * 3 * sizeof(float)) != cudaSuccess)
    fprintf(stderr, "reset error\n");

  output_size = count * height * width;
  compute_hough_kernel<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock,
                       kThreadsPerBlock, 0, stream>>>(
      output_size, hough_space, hough_data, labelmap, vertmap, extents, meta_data,
      arrays, array_sizes, class_indexes, height, width, num_classes, count, inlierThreshold, skip_pixels);
  cudaThreadSynchronize();

  err = cudaGetLastError();
  if(cudaSuccess != err)
  {
    fprintf( stderr, "cudaCheckError() failed compute hough space: %s\n", cudaGetErrorString( err ) );
    exit( -1 );
  }

  // step 3: find the maximum in hough space
  // thrust::device_vector<int> num_max_vec(1);
  int* num_max; // = thrust::raw_pointer_cast(num_max_vec.data());
  cudaMalloc((void **)&num_max, sizeof(int));
  if (cudaMemset(num_max, 0, sizeof(int)) != cudaSuccess)
    fprintf(stderr, "reset error\n");

  // printf("num_rois_host %d\n", num_rois_host);
  int index_size = (MAX_ROI - *num_rois) / (batch_size - batch_index);
  // thrust::device_vector<int> max_indexes_vec(index_size);
  int* max_indexes; // = thrust::raw_pointer_cast(max_indexes_vec.data());
  cudaMalloc((void **)&max_indexes, index_size * sizeof(int));
  if (cudaMemset(max_indexes, 0, index_size * sizeof(int)) != cudaSuccess)
    fprintf(stderr, "reset error\n");

  if (votingThreshold > 0)
  {
    output_size = count * height * width;
    compute_max_indexes_kernel<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock,
                       kThreadsPerBlock, 0, stream>>>(
      output_size, max_indexes, index_size, num_max, hough_space, hough_data, height, width, votingThreshold, perThreshold);
    cudaThreadSynchronize();
  }
  else
  {
    std::vector<int> max_indexes_host(count);
    // memset(&max_indexes_host[0], 0, count * sizeof(int));
    for (int i = 0; i < count; i++)
    {
      float *hmax = thrust::max_element(thrust::device, hough_space + i * height * width, hough_space + (i+1) * height * width);
      max_indexes_host[i] = hmax - hough_space;
      // printf("Max indexes %d) %d\n", i, max_indexes_host[i]);
    }
    cudaMemcpy(num_max, &count, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(max_indexes, &max_indexes_host[0], count * sizeof(int), cudaMemcpyHostToDevice);
  }
  err = cudaGetLastError();
  if(cudaSuccess != err)
  {
    fprintf( stderr, "cudaCheckError() failed compute maximum: %s\n", cudaGetErrorString( err ) );
    exit( -1 );
  }

  // step 4: compute outputs
  int num_max_host;
  cudaMemcpy(&num_max_host, num_max, sizeof(int), cudaMemcpyDeviceToHost);
  num_max_host = std::min(num_max_host, index_size);

  // printf("num_max: %d\n", num_max_host);
  if (num_max_host > 0)
  {
    output_size = num_max_host;
    if (cudaMemset(num_max, 0, sizeof(int)) != cudaSuccess)
      fprintf(stderr, "reset error\n");
    compute_rois_kernel<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock,
                         kThreadsPerBlock, 0, stream>>>(
        output_size, top_box, top_pose, top_target, top_weight, top_domain,
        extents, meta_data, gt, hough_space, hough_data, max_indexes, class_indexes,
        is_train, batch_index, height, width, num_classes, num_gt, num_max);
    cudaThreadSynchronize();
    cudaMemcpy(&num_max_host, num_max, sizeof(int), cudaMemcpyDeviceToHost);
    *num_rois += num_max_host;
  }
  
  // err checking
  err = cudaGetLastError();
  if(cudaSuccess != err)
  {
    fprintf( stderr, "cudaCheckError() failed compute outputs: %s\n", cudaGetErrorString( err ) );
    exit( -1 );
  }

  cudaFree(arrays);
  cudaFree(array_sizes);
  cudaFree(class_indexes);
  cudaFree(hough_space);
  cudaFree(hough_data);
  cudaFree(num_max);
  cudaFree(max_indexes);

  return 1;
}