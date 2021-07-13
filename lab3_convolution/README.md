<font size=10>实验三：CUDA 使用基础</font>

---

# 1 实验简介

卷积（[Convolution](https://en.wikipedia.org/wiki/Convolution)）是一种基本的数学运算，想必大家在微积分、概率论与数理统计等数学基础课程中都一定程度上接触过。作为一种基本的数学计算，其在图像处理、机器学习等领域都有重要应用。

本次实验需要你使用 Cuda 完成一个 GPU 上的二维离散卷积。

# 2 实验环境

请大家在我们提供的集群上创建一个开发环境为 TensorFlow 的容器（要求最后在实验报告中展示环境基本信息），容器中含有 Nvidia GeForce RTX 2080 Ti 及 nvcc v10.1，无需自行配置。

下图为某个可能的环境基本信息：

![env_info](./img/env_info.png)

# 3 实验基础知识介绍
该部分简要介绍和实验相关的基础知识  
为方便理解，不保证数学上的严谨性  

## 张量(tensor)
> 张量概念是矢量概念的推广，矢量是一阶张量。张量是一个可用来表示在一些矢量、标量和其他张量之间的线性关系的多线性函数。  
> 同构意义下，第零阶张量(r = 0)为标量(Scalar)，第一阶张量(r = 1)为向量 (Vector)，第二阶张量(r = 2)则为矩阵(Matrix)。  

本实验中，张量概念作了解即可。实验中的卷积运算特指2个矩阵之间的卷积运算。  
## 卷积(convolution)
本实验只涉及离散运算，连续形式的卷积不做介绍，感兴趣的同学可以自行了解。  
### 一维离散卷积
定义$\left(f*g\right)\left(n\right)$为函数$f$与$g$的卷积  
$$\left(f*g\right)\left(n\right)=\Sigma_{t=-\infty}^{+\infty}f\left(t\right)g\left(n-t\right)$$  
函数$f$和$g$定义域可以不是所有整数，修改上式中t的遍历范围可得到新的定义；另一种方式是定义超出定义域的函数值视为0，可得到相同的结果。  
需要注意的是，两个函数的卷积结果仍是函数。  
可以形象地理解为沿着不断移动的 $x+y=n$ 直线，将两个函数卷成一个新的函数，每条直线对应新函数的一组对应关系。
### 二维离散卷积
二维离散卷积可以视为一维离散卷积的推广。  
$$\left(f*g\right)\left(n,m\right)=\Sigma_{i=-\infty}^{+\infty}\Sigma_{j=-\infty}^{+\infty}f\left(i,j\right)g\left(n-i,m-j\right)$$  
我们在实验中的定义卷积与数学上的定义存在差别，我们认为其在广义上属于二维离散卷积。  
简化起见，考虑两个方阵$f$和$g$，$f$的大小为 $a*a$，$g$的大小为 $b*b$，我们将$g$称为核(kernel)函数，且要求$b$为奇数。$f$行列下标均从0开始，
$g$的行列下标则从$-\lfloor b/2\rfloor$到$+\lfloor b/2\rfloor$(包括0)  
则卷积的结果可以定义为:  
$$\left(f*g\right)\left(n,m\right)=\Sigma_{i=-\lfloor b/2\rfloor}^{+\lfloor b/2\rfloor}\Sigma_{j=-\lfloor b/2\rfloor}^{+\lfloor b/2\rfloor}f\left(n+i,m+j\right)g\left(i,j\right)$$  
若$f$的下标范围超出定义范围，本实验的方式是填充一个默认值(0)以解决问题，卷积结果与$f$大小相同

# 4 实验步骤

## 4.1 基准

最基础的 CPU 版本已在 `conv.cu` 中给出，即通过四层循环轮流计算结果矩阵中每个位置的值。

```c++
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
```

# 5 实验任务与要求

利用以上技术，在基准程序的基础上实现卷积计算的 GPU 实现并优化之。

**只允许修改两个计时点之间的代码及 Makefile 文件**

Note: 调试时为使错误可复现，可以将代码中的 `std::default_random_engine generator(r());` 改为 `std::default_random_engine generator();`，这样每次生成的随机矩阵都会是一致的。
