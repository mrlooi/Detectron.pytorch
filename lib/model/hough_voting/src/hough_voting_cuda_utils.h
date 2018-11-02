#ifndef _HOUGH_VOTING_CUDA_UTILS
#define _HOUGH_VOTING_CUDA_UTILS

// TODO: remove eigen 
#include <Eigen/Geometry> 

__device__ inline float point2line(int cx, int cy, int x, int y, float u, float v)
{
  float n1 = -v;
  float n2 = u;

  return fabs(n1 * (cx - x) + n2 * (cy - y)) / sqrt(n1 * n1 + n2 * n2);
}


__device__ inline float angle_distance(int cx, int cy, int x, int y, float u, float v)
{
  float dx = cx - x;
  float dy = cy - y;
  float n1 = sqrt(u * u + v * v);
  float n2 = sqrt(dx * dx + dy * dy);
  float dot = u * dx + v * dy;
  float distance = dot / (n1 * n2);

  return distance;
}

__device__ inline float angle_distance_label(int cx, int cy, int x, int y, float u, float v, 
  int cls, const int height, const int width, const int* labelmap)
{
  float dx = cx - x;
  float dy = cy - y;
  float n1 = sqrt(u * u + v * v);
  float n2 = sqrt(dx * dx + dy * dy);
  float dot = u * dx + v * dy;
  float distance = dot / (n1 * n2);

  int num = 20;
  int count = 0;
  for (int i = 1; i <= num; i++)
  {
    float step = float(i) / float(num);
    int px = int(x + step * dx);
    int py = int(y + step * dy);
    if (px >= 0 && px < width && py >= 0 && py < height)
    {
      if (labelmap[py * width + px] == cls)
        count++;
    }
  }
  if ((float)count / float(num) < 0.8)
    distance = 0;

  return distance;
}

__device__ inline float IoU(float* a, float* b) 
{
  float left = fmax(a[0], b[0]), right = fmin(a[2], b[2]);
  float top = fmax(a[1], b[1]), bottom = fmin(a[3], b[3]);
  float width = fmax(right - left + 1, 0.f), height = fmax(bottom - top + 1, 0.f);
  float interS = width * height;
  float Sa = (a[2] - a[0] + 1) * (a[3] - a[1] + 1);
  float Sb = (b[2] - b[0] + 1) * (b[3] - b[1] + 1);
  return interS / (Sa + Sb - interS);
}

__device__ inline void project_box(int cls, const float* extents, const float* meta_data, float distance, float factor, float* threshold)
{
  float xHalf = extents[cls * 3 + 0] * 0.5;
  float yHalf = extents[cls * 3 + 1] * 0.5;
  float zHalf = extents[cls * 3 + 2] * 0.5;
  float bb3D[24];

  bb3D[0] = xHalf; bb3D[1] = yHalf; bb3D[2] = zHalf + distance;
  bb3D[3] = -xHalf; bb3D[4] = yHalf; bb3D[5] = zHalf + distance;
  bb3D[6] = xHalf; bb3D[7] = -yHalf; bb3D[8] = zHalf + distance;
  bb3D[9] = -xHalf; bb3D[10] = -yHalf; bb3D[11] = zHalf + distance;
  bb3D[12] = xHalf; bb3D[13] = yHalf; bb3D[14] = -zHalf + distance;
  bb3D[15] = -xHalf; bb3D[16] = yHalf; bb3D[17] = -zHalf + distance;
  bb3D[18] = xHalf; bb3D[19] = -yHalf; bb3D[20] = -zHalf + distance;
  bb3D[21] = -xHalf; bb3D[22] = -yHalf; bb3D[23] = -zHalf + distance;

  float fx = meta_data[0];
  float fy = meta_data[4];
  float px = meta_data[2];
  float py = meta_data[5];
  float minX = 1e8;
  float maxX = -1e8;
  float minY = 1e8;
  float maxY = -1e8;
  for (int i = 0; i < 8; i++)
  {
    float x = fx * (bb3D[i * 3] / bb3D[i * 3 + 2])  + px;
    float y = fy * (bb3D[i * 3 + 1] / bb3D[i * 3 + 2])  + py;
    minX = fmin(minX, x);
    minY = fmin(minY, y);
    maxX = fmax(maxX, x);
    maxY = fmax(maxY, y);
  }
  float width = maxX - minX + 1;
  float height = maxY - minY + 1;
  *threshold = fmax(width, height) * factor;
}


__device__ inline float compute_box_overlap(int cls, const float* extents, const float* meta_data, const float* pose, float* box)
{
  float xHalf = extents[cls * 3 + 0] * 0.5;
  float yHalf = extents[cls * 3 + 1] * 0.5;
  float zHalf = extents[cls * 3 + 2] * 0.5;

  Eigen::Matrix<float,8,3,Eigen::DontAlign> bb3D;
  bb3D(0, 0) = xHalf; bb3D(0, 1) = yHalf; bb3D(0, 2) = zHalf;
  bb3D(1, 0) = -xHalf; bb3D(1, 1) = yHalf; bb3D(1, 2) = zHalf;
  bb3D(2, 0) = xHalf; bb3D(2, 1) = -yHalf; bb3D(2, 2) = zHalf;
  bb3D(3, 0) = -xHalf; bb3D(3, 1) = -yHalf; bb3D(3, 2) = zHalf;
  bb3D(4, 0) = xHalf; bb3D(4, 1) = yHalf; bb3D(4, 2) = -zHalf;
  bb3D(5, 0) = -xHalf; bb3D(5, 1) = yHalf; bb3D(5, 2) = -zHalf;
  bb3D(6, 0) = xHalf; bb3D(6, 1)= -yHalf; bb3D(6, 2) = -zHalf;
  bb3D(7, 0) = -xHalf; bb3D(7, 1) = -yHalf; bb3D(7, 2) = -zHalf;

  // rotation
  Eigen::Quaternionf quaternion(pose[6], pose[7], pose[8], pose[9]);
  Eigen::Matrix3f rmatrix = quaternion.toRotationMatrix();
  Eigen::Matrix<float,3,8,Eigen::DontAlign> bb3D_new = rmatrix * bb3D.transpose();

  // projection
  float fx = meta_data[0];
  float fy = meta_data[4];
  float px = meta_data[2];
  float py = meta_data[5];
  float x1 = 1e8;
  float x2 = -1e8;
  float y1 = 1e8;
  float y2 = -1e8;
  for (int i = 0; i < 8; i++)
  {
    float X = bb3D_new(0, i) + pose[10];
    float Y = bb3D_new(1, i) + pose[11];
    float Z = bb3D_new(2, i) + pose[12];
    float x = fx * (X / Z)  + px;
    float y = fy * (Y / Z)  + py;
    x1 = fmin(x1, x);
    y1 = fmin(y1, y);
    x2 = fmax(x2, x);
    y2 = fmax(y2, y);
  }

  float box_gt[4];
  box_gt[0] = x1;
  box_gt[1] = y1;
  box_gt[2] = x2;
  box_gt[3] = y2;
  return IoU(box, box_gt);
}

#endif