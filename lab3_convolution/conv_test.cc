#include <cuda.h>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>

const int kSize = 5000;
const int kKernelSize = 13;  // odd

#define InitRandom()                         \
  std::random_device r;                      \
  std::default_random_engine generator(r()); \
  std::uniform_int_distribution<int> distribution(-5, 5);

void Generate(float *const a, float *const w) {
#pragma omp parallel for
  for (int i = 0; i < kSize; ++i) {
    InitRandom();
    const int j_upperbound = (i + 1) * kSize;
    for (int j = i * kSize; j < j_upperbound; ++j)
      a[j] = distribution(generator);
  }
  {
    InitRandom();
    for (int i = 0; i < kKernelSize * kKernelSize; ++i)
      w[i] = distribution(generator);
  }
}

void Conv(const float *const a, const float *const w, float *const b) {
#pragma omp parallel for
  for (int i = 0; i < kSize; ++i) {
    for (int j = 0; j < kSize; ++j) {
      float conv = 0;
      int x = i - kKernelSize / 2, y = j - kKernelSize / 2;
      for (int k = 0; k < kKernelSize; ++k) {
        for (int l = 0; l < kKernelSize; ++l) {
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

void Check(const float *const a, const float *const w, float *const b) {
  auto b_std = new float[kSize * kSize];
  Conv(a, w, b_std);
  for (int i = 0; i < kSize * kSize; ++i) {
    if (b[i] != b_std[i]) {
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

void Output(const float *const a, const float *const w, const float *const b) {
  for (int i = 0; i < kSize; ++i) {
    for (int j = 0; j < kSize; ++j)
      std::cout << std::setw(2) << a[i * kSize + j] << ' ';
    std::cout << std::endl;
  }

  for (int i = 0; i < kKernelSize; ++i) {
    for (int j = 0; j < kKernelSize; ++j)
      std::cout << std::setw(2) << w[i * kKernelSize + j] << ' ';
    std::cout << std::endl;
  }

  for (int i = 0; i < kSize; ++i) {
    for (int j = 0; j < kSize; ++j)
      std::cout << std::setw(2) << b[i * kSize + j] << ' ';
    std::cout << std::endl;
  }
}

const int kBlockFactor = 16;
__global__ void conv_cuda_kernel(float *a, float *w, float *b) {
  const int i = blockIdx.x * kBlockFactor + threadIdx.x;
  const int j = blockIdx.y * kBlockFactor + threadIdx.y;
  if (i < kSize && j < kSize) {
    float conv = 0;
    int x = i - kKernelSize / 2, y = j - kKernelSize / 2;
    for (int k = 0; k < kKernelSize; ++k) {
      for (int l = 0; l < kKernelSize; ++l) {
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
// naive and shit
// only for testing correctness and precision
void conv_cuda(const float *const a, const float *const w, float *const b) {
  float *a_kernel, *w_kernel, *b_kernel;
  cudaMalloc(&a_kernel, kSize * kSize * sizeof(float));
  cudaMemcpy(a_kernel, a, kSize * kSize * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMalloc(&w_kernel, kKernelSize * kKernelSize * sizeof(float));
  cudaMemcpy(w_kernel, w, kKernelSize * kKernelSize * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMalloc(&b_kernel, kSize * kSize * sizeof(float));
  dim3 grid((kSize + kBlockFactor - 1) / kBlockFactor,
            (kSize + kBlockFactor - 1) / kBlockFactor);
  dim3 block(kBlockFactor, kBlockFactor);
  conv_cuda_kernel<<<grid, block>>>(a_kernel, w_kernel, b_kernel);
  cudaDeviceSynchronize();
  cudaMemcpy(b, b_kernel, kSize * kSize * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaFree(a_kernel);
  cudaFree(w_kernel);
  cudaFree(b_kernel);
}

int main() {
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

  // Output(a, w, b);

  delete[] a;
  delete[] w;
  delete[] b;
  return 0;
}
