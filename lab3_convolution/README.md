<font size=10>实验三：CUDA 使用基础</font>

---

# 1 实验简介



# 2 实验环境



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



# 5 实验任务与要求

