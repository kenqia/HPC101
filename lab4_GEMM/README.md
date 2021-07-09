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

### 3.2.1 指令级并行



### 3.2.2 线程级并行



### 3.2.3 进程级并行



## 3.3 通信开销



# 4 实验步骤

## 4.1 单机优化

### 4.1.1 基准



### 4.1.2 分块



### 4.1.3 向量化



### 4.1.4 循环排列



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

