#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <cuda.h>
#include <cuda_runtime.h>

const int kSize = 2000;
const int kKernelSize = 13; // odd

void Generate(float *const a, float *const w)
{
  std::random_device r;
  std::default_random_engine generator(r());
  // std::default_random_engine generator;
  std::uniform_real_distribution<float> distribution(-10, 10);
  //std::uniform_int_distribution<int> distribution(0, 2);
  for (int i = 0; i < kSize * kSize; ++i)
    a[i] = distribution(generator);
  for (int i = 0; i < kKernelSize * kKernelSize; ++i)
    w[i] = distribution(generator);
}

void Conv(const float *const a, const float *const w, float *const b)
{
#pragma omp parallel for
  for (int i = 0; i < kSize; ++i)
  {
    for (int j = 0; j < kSize; ++j)
    {
      float conv = 0;
      int x = i - kKernelSize / 2, y = j - kKernelSize / 2;
      for (int k = 0; k < kKernelSize; ++k)
      {
        for (int l = 0; l < kKernelSize; ++l)
        {
          if (!(x < 0 || x >= kSize || y < 0 || y >= kSize))
            conv += a[x * kSize + y] * w[k * kKernelSize + l];
          y++;
        }
        x++;
        y -= kKernelSize;
      }
      b[i * kSize + j] = conv;
    }
  }
}

void Check(const float *const a, const float *const w, float *const b)
{
  auto b_std = new float[kSize * kSize];
  Conv(a, w, b_std);
  for (int i = 0; i < kSize; ++i)
  {
    //if (b[i] != b_std[i])
    if (abs(b[i] / b_std[i] - 1) > 1e-3)
    {
      std::cout << "\x1b[31m"
                   "Wrong Answer"
                   "\x1b[0m"
                   " at "
                << i << std::endl;
      std::cout << "expected " << b_std[i] << " but found " << b[i]
                << std::endl;
      delete[] b_std;
      return;
    }
  }
  std::cout << "\x1b[32m"
               "Correct"
               "\x1b[0m"
            << std::endl;

  delete[] b_std;
}

void Output(const float *const a, const float *const w, const float *const b)
{
  for (int i = 0; i < kSize; ++i)
  {
    for (int j = 0; j < kSize; ++j)
      std::cout << std::setw(2) << a[i * kSize + j] << ' ';
    std::cout << std::endl;
  }

  for (int i = 0; i < kKernelSize; ++i)
  {
    for (int j = 0; j < kKernelSize; ++j)
      std::cout << std::setw(2) << w[i * kKernelSize + j] << ' ';
    std::cout << std::endl;
  }

  for (int i = 0; i < kSize; ++i)
  {
    for (int j = 0; j < kSize; ++j)
      std::cout << std::setw(2) << b[i * kSize + j] << ' ';
    std::cout << std::endl;
  }
}

__global__ void conv_cuda_kernel(float *a, float *w, float *b, size_t anchor)
{
  // int row = (int)blockIdx.x + ((int)threadIdx.x - anchor);
  // int col = (int)blockIdx.y + ((int)threadIdx.y - anchor);
  size_t col = (blockIdx.x + threadIdx.x) - anchor;
  size_t row = (blockIdx.y + threadIdx.y) - anchor;
  // if(row>=0&&row<blockDim.x&&col>=0&&col<blockDim.y)
  if (row < gridDim.y && col < gridDim.x)
  {
    size_t a_pos = row * gridDim.x + col;
    size_t w_pos = threadIdx.x + threadIdx.y * blockDim.x;
    size_t b_pos = blockIdx.x + blockIdx.y * gridDim.x;
    atomicAdd(b + b_pos, a[a_pos] * w[w_pos]);
  }
}
// naive and shit
// only for testing correctness and precision
void conv_cuda(const float *const a, const float *const w, float *const b)
{
  float *a_kernel, *w_kernel, *b_kernel;
  cudaMalloc(&a_kernel, kSize * kSize * sizeof(float));
  cudaMemcpy(a_kernel, a, kSize * kSize * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc(&w_kernel, kKernelSize * kKernelSize * sizeof(float));
  cudaMemcpy(w_kernel, w, kKernelSize * kKernelSize * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc(&b_kernel, kSize * kSize * sizeof(float));
  cudaMemset(b_kernel, 0, kSize * kSize * sizeof(float));
  dim3 grid(kSize, kSize);
  dim3 block(kKernelSize, kKernelSize);
  conv_cuda_kernel<<<grid, block>>>(a_kernel, w_kernel, b_kernel, kKernelSize / 2);
  cudaDeviceSynchronize();
  cudaMemcpy(b, b_kernel, kSize * kSize * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(a_kernel);
  cudaFree(w_kernel);
  cudaFree(b_kernel);
}

int main()
{
  auto a = new float[kSize * kSize];
  auto w = new float[kKernelSize * kKernelSize];
  auto b = new float[kSize * kSize];
  Generate(a, w);

  auto start = std::chrono::high_resolution_clock::now();

  // Conv(a, w, b);
  conv_cuda(a, w, b);

  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();

  Check(a, w, b);
  std::chrono::nanoseconds diff =
      std::chrono::duration_cast<decltype(diff)>(end - start);
  std::cout << diff.count() << " nanoseconds" << std::endl;

  //Output(a, w, b);

  delete[] a;
  delete[] w;
  delete[] b;
  return 0;
}
