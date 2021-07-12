#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>

const int kSize = 2000;
const int kKernelSize = 13;  // odd

void Generate(float *const a, float *const w) {
  std::random_device r;
  std::default_random_engine generator(r());
  // std::default_random_engine generator;
  std::uniform_real_distribution<float> distribution(-1e3, 1e3);
  // std::uniform_int_distribution<int> distribution(-1, 1);
  for (int i = 0; i < kSize * kSize; ++i) a[i] = distribution(generator);
  for (int i = 0; i < kKernelSize * kKernelSize; ++i)
    w[i] = distribution(generator);
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
  for (int i = 0; i < kSize; ++i) {
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

int main() {
  auto a = new float[kSize * kSize];
  auto w = new float[kKernelSize * kKernelSize];
  auto b = new float[kSize * kSize];
  Generate(a, w);

  auto start = std::chrono::high_resolution_clock::now();

  Conv(a, w, b);

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
