<font size=10>实验五：简单 CNN 网络训练</font>

---

# 1 实验简介
## LeNet-5

LeNet-5是一个较简单的卷积神经网络。下图显示了其结构：输入的二维图像，先经过两次卷积层到池化层，再经过全连接层，最后输出每种分类预测得到的概率。
有关于其更详细的结构可以在[原论文](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)中找到
![](https://pic4.zhimg.com/v2-308e7517e3f6482a0c376a0d1e90d846_1440w.jpg?source=172ae18b)


## MNIST 手写数字数据集
MNIST数据集(Mixed National Institute of Standards and Technology database)是美国国家标准与技术研究院收集整理的大型手写数字数据库,包含60,000个示例的训练集以及10,000个示例的测试集.


## CIFAR-10 / CIFAR-100 数据集

[CIFAR-10和CIFAR-100](http://www.cs.toronto.edu/~kriz/cifar.html) 是[8000 万个微型图像数据集](http://groups.csail.mit.edu/vision/TinyImages/)的标记子集。它们由 Alex Krizhevsky、Vinod Nair 和 Geoffrey Hinton 收集。

**CIFAR-10** 数据集包含 10 个类别的 60000 张 32x32 彩色图像，每类 6000 张图像。整个数据集由 50000 张训练图像和 10000 张测试图像组成。

官网下载到的数据集中包含5个batch，每个都由1000张各自类别的图像组成。

以下是数据集中的不同种类，以及每个种类的10张随机图像：

![](http://8.136.41.73:5212/api/v3/file/get/92/1.png?sign=AKEhR2PuaP2zpTAJ8NbHsRVRmSKZ9X4XLlQ8CdLSJLs%3D%3A0)

**CIFAR-100**数据集和CIFAR-10数据集类似，不同之处在于它有100种类别，每种类别包含600张图像，分别为500张训练图像和100张测试图像。CIFAR-100 中的 100 个类别分为 20 个superclass。每个图像都带有一个“精细”标签（它所属的类）和一个“粗”标签（它所属的超类）。

与MNIST数据集相比，CIFAR-10 是3 通道的彩色RGB 图像，而MNIST 是灰度图像。CIFAR-10 的图片尺寸为32 × 32 ， 而MNIST 的图片尺寸为28 × 28 ，比MNIST 稍大。

# 2 实验环境



# 3 实验基础知识介绍



# 4 实验步骤
见`MNIST.ipynb`和`CIFAR.ipynb`


# 5 实验任务与要求

1.实现最基本的卷积神经网络 (CNN) LeNet-5 以及一个物体分类的 CNN ，可直接调用 `TensorFlow` 或` PyTorch` 这 2 个常用的深度 学习开发工具的各种构建函数。可直接调用开发工具的训练相关的接口，但不能直接读取各种深度学习开发工具已训练好的 CNN 网络结构与参数。

2.自己用 MNIST 手写数字数据集（ 0 9 一共十个数字） 6 万样本实现对 LeNet-5 的训练，数据集下载： http://yann.lecun.com/exdb/mnist/index.html 。 对MNIST 的 1 万测试样本进行测试，获得识别率是多少。

3.自己用 CIFAR 10 数据库 http://www.cs.utoronto.ca/~kriz/cifar.html 实现CNN 物体分类功能的训练与测试。





# 参考资料

`TensorFlow `框架： https://github.com/tensorflow/tensorflow   https://www.tensorflow.org/

`PyTorch`框架 https://pytorch.org/

MNIST 数据集 http://yann.lecun.com/exdb/mnist/index.html
CIFAR 10/100 数据集 http://www.cs.utoronto.ca/~kriz/cifar.html

LeNet-5网络结构 http://yann.lecun.com/exdb/lenet/

