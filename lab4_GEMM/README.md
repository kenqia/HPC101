<font size=10>实验四：通用矩阵乘法</font>

---

# 1 实验简介

**BLAS**（英语：**Basic Linear Algebra Subprograms**，基础线性代数程序集）是一个[应用程序接口](https://zh.wikipedia.org/wiki/应用程序接口)（API）标准，用以规范发布基础线性代数操作的数值库（如矢量或矩阵乘法）。该程序集最初发布于1979年，并用于创建更大的数值程序包（如 [LAPACK](https://zh.wikipedia.org/wiki/LAPACK)）。[1] 在高性能计算领域，BLAS 被广泛使用，numpy 的底层也依赖于 BLAS。

通用矩阵乘法（[General matrix multiply](https://en.wikipedia.org/wiki/General_matrix_multiply), GEMM）是 BLAS 中经典的子程序之一。[2] 作为当今科学计算最常见的计算任务之一，GEMM 需要实现一个非常高效的矩阵乘法。优化 GEMM 也是 HPC 界非常基础的任务。

本次实验需要你使用 OpenMP、MPI 等 API 手工完成一个支持分布式计算高性能 GEMM 实现。

# 2 实验环境

请大家在我们提供的集群上创建四个容器，每台容器包含四个 CPU 核心。大家需要配置好容器之间的网络通信以及 MPI 环境，方便起见你可以直接使用包含 MPI 的 Docker 镜像。

# 3 实验基础知识介绍

## 3.1 程序局部性与计算机层次存储结构



## 3.2 并行计算

### 3.2.1 SIMD 子字并行



### 3.2.2 指令级并行



### 3.2.3 线程级并行



### 3.2.4 进程级并行



## 3.3 IO 与通信开销



# 4 实验步骤

接下来我们讨论的优化技巧全部是针对两个稠密矩阵的乘法。我们给出以下形式化定义：

给定矩阵 $$A, B, C$$：
$$
{\displaystyle \mathbf {A} ={\begin{pmatrix}a_{11}&a_{12}&\cdots &a_{1n}\\a_{21}&a_{22}&\cdots &a_{2n}\\\vdots &\vdots &\ddots &\vdots \\a_{m1}&a_{m2}&\cdots &a_{mn}\\\end{pmatrix}},\quad \mathbf {B} ={\begin{pmatrix}b_{11}&b_{12}&\cdots &b_{1p}\\b_{21}&b_{22}&\cdots &b_{2p}\\\vdots &\vdots &\ddots &\vdots \\b_{n1}&b_{n2}&\cdots &b_{np}\\\end{pmatrix}}},\quad{\displaystyle \mathbf {C} ={\begin{pmatrix}c_{11}&c_{12}&\cdots &c_{1p}\\c_{21}&c_{22}&\cdots &c_{2p}\\\vdots &\vdots &\ddots &\vdots \\c_{m1}&c_{m2}&\cdots &c_{mp}\\\end{pmatrix}}}
$$
矩阵乘法 $$C = AB$$ 定义为对任意 $$c_{ij}$$ 有：
$$
{\displaystyle c_{ij}=a_{i1}b_{1j}+a_{i2}b_{2j}+\cdots +a_{in}b_{nj}=\sum _{k=1}^{n}a_{ik}b_{kj},}
$$
即：
$$
{\displaystyle \mathbf {C} ={\begin{pmatrix}a_{11}b_{11}+\cdots +a_{1n}b_{n1}&a_{11}b_{12}+\cdots +a_{1n}b_{n2}&\cdots &a_{11}b_{1p}+\cdots +a_{1n}b_{np}\\a_{21}b_{11}+\cdots +a_{2n}b_{n1}&a_{21}b_{12}+\cdots +a_{2n}b_{n2}&\cdots &a_{21}b_{1p}+\cdots +a_{2n}b_{np}\\\vdots &\vdots &\ddots &\vdots \\a_{m1}b_{11}+\cdots +a_{mn}b_{n1}&a_{m1}b_{12}+\cdots +a_{mn}b_{n2}&\cdots &a_{m1}b_{1p}+\cdots +a_{mn}b_{np}\\\end{pmatrix}}}
$$
为了简化问题，我们假设所有的矩阵都是 $$N \times N$$ 的方阵。 

## 4.1 单机优化

### 4.1.1 基准

最基础的矩阵乘法自然是三层循环，即对二维矩阵 $$C$$ 的每一项通过单层循环计算其结果。

```c++
for (int x = 0; x < N; x++) {
  for (int y = 0; y < N; y++) {
    C[(x * N) + y] = 0
    for (int k = 0; k < N; k++) {
      C[(x * N) + y] += A[(x * N) + k] * B[(k * N) + y]
    }
  }
}
```

### 4.1.2 分块

基准代码尽可能多的使用了行遍历来提高内存的访问效率，但是即便如此，由于矩阵本身大小过大，导致部分按列访问的情况下，整体局部性还是不高。我们引入分块技术来提高程序的局部性，降低 cache miss 的概率。
$$
A=\left(\begin{array}{ccc}
A_{0,0} & \cdots & A_{0, K-1} \\
\vdots & & \vdots \\
A_{M-1,0} & \cdots & A_{M-1, K-1}
\end{array}\right), B=\left(\begin{array}{ccc}
B_{0,0} & \cdots & B_{0, N-1} \\
\vdots & & \vdots \\
B_{K-1,0} & \cdots & B_{K-1, N-1}
\end{array}\right), C=\left(\begin{array}{ccc}
C_{0,0} & \cdots & C_{0, N-1} \\
\vdots & & \vdots \\
C_{M-1,0} & \cdots & C_{M-1, N-1}
\end{array}\right)
$$

此时对于每一个 $$C$$，我们都有：
$$
{\displaystyle C_{ij}=\sum _{k=0}^{N-1}A_{ik}B_{kj},}
$$
大致代码实现如下：

```c++
for (int outerX = 0; outerX < N; outerX++) {
  for (int outerY = 0; outerY < N; outerY++) {
    // Clear C_ij matrix
    for (int innerX = 0; innerX < blockSize; innerX++) {
      for (int innerY = 0; innerY < blockSize; innerY++) {
        // TODO: Clear C_ij matrix
      }
    }
    for (int K = 0; K < N; K++) {
      // TODO: calculate C_ij = sum(A_ik * B_kj)
    }
  }
}
```

为了保证计算的正确性，你需要非常谨慎的处理循环层数、下标。

### 4.1.3 向量化

对于循环的最内层，如果我们将其展开，往往会发现类似下面的模式：

```c++
c[i+0] = a[j+0] * b[someIdx0]
c[i+1] = a[j+1] * b[someIdx1]
c[i+2] = a[j+2] * b[someIdx2]
c[i+3] = a[j+3] * b[someIdx3]
...
```

这种规律、统一的计算非常适合使用 SIMD 指令进行计算。以计算 $$C_i = A_i * B_i$$ 为例，我们可以写出以下利用 AVX 指令的向量化代码：

```c++
__m256d a[N], b[N], c[N];
for (int i = 0; i < N; i++)
    c[i] = __mm256_mul_pd(a[i], b[i]);
```

这段代码能够在一条 CPU 指令内完成 4 个 double 的乘法，从而大幅提高系统的计算性能。

同样，为了保证计算的正确性，你需要非常谨慎的处理循环层数、地址。

由于向量化指令集在不同厂商、不同型号的 CPU 上可能都是不同的，手写向量化的成本非常的高。因此我们往往直接使用编译器的自动向量化。具体开启方法请查阅不同平台上相关编译器的编译选项。

### 4.1.4 循环重排



### 4.1.5 数组封装



### 4.1.6 写缓存



### 4.1.7 线程级并行



## 4.3 多机分布式



# 5 实验任务与要求

利用以上技术完成 GEMM 分布式实现，

# 参考资料

1. [https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms)
2. [https://en.wikipedia.org/wiki/General_matrix_multiply](https://en.wikipedia.org/wiki/General_matrix_multiply)
3. [https://github.com/flame/how-to-optimize-gemm/wiki](https://github.com/flame/how-to-optimize-gemm/wiki)
4. [https://tvm.apache.org/docs/tutorials/optimize/opt_gemm.html](https://tvm.apache.org/docs/tutorials/optimize/opt_gemm.html)
5. Huang J, van. BLISlab: A Sandbox for Optimizing GEMM. arXiv.org. Published 2016. Accessed July 10, 2021. https://arxiv.org/abs/1609.00076

‌

