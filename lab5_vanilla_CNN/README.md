<font size=10>实验五：简单 CNN 网络训练</font>

---

# 1 实验简介
**深度学习**（Deep Learning）是[机器学习](https://zh.wikipedia.org/wiki/机器学习)的分支，是一种以[人工神经网络](https://zh.wikipedia.org/wiki/人工神经网络)为架构，对资料进行表征学习的[算法](https://zh.wikipedia.org/wiki/算法)。深度学习能够取得如此卓越的成就，除了优越的算法、充足的数据，更离不开强劲的算力。近年来，深度学习相关的基础设施逐渐成熟，从网络设计时的训练、优化，到落地的推理加速，都有非常优秀的解决方案。其中，对于算力的需求最大的部分之一是网络的训练，它也因此成为 HPC 领域经常研究的话题。

**卷积神经网络**（Convolutional Neural Network, **CNN**）是一种[前馈神经网络](https://zh.wikipedia.org/wiki/前馈神经网络)，对于大型图像处理有出色表现。

本次实验我们将完成两个简单的 CNN 网络，并在 GPU 上加速它的训练，体会基本的网络设计、训练流程。

# 2 实验环境

TBD

# 3 实验基础知识介绍

## 3.1 网络模型

### 3.1.1 CNN 卷积神经网络

卷积神经网络由一个或多个卷积层和顶端的全连通层（对应经典的神经网络）组成，同时也包括关联权重和池化层（pooling layer）。这一结构使得卷积神经网络能够利用输入数据的二维结构。与其他深度学习结构相比，卷积神经网络在图像和[语音识别](https://zh.wikipedia.org/wiki/语音识别)方面能够给出更好的结果。这一模型也可以使用[反向传播算法](https://zh.wikipedia.org/wiki/反向传播算法)进行训练。相比较其他深度、前馈神经网络，卷积神经网络需要考量的参数更少，使之成为一种颇具吸引力的深度学习结构。

### 3.1.2 LeNet-5

LeNet-5是一个较简单的卷积神经网络。下图显示了其结构：输入的二维图像，先经过两次卷积层到池化层，再经过全连接层，最后输出每种分类预测得到的概率。

![](README.assets/LeNet.jpg)

有关于其更详细的结构可以在[原论文](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)中找到。

### 3.1.3 其他 CNN 模型

LeNet-5 虽然在简单的分类任务上效果不错，但是并不能胜任更复杂的任务。因此你可能需要参考其他的 CNN 模型来设计一个更复杂的 CNN，这样才能完成第二个任务，即 CIFAR-10 上的图像分类任务。

你可以参考 VGG 等网络，也可以直接在 paper with code 上查找在 CIFAR-10 上表现优异的网络。

## 3.2 数据集

### 3.2.1 MNIST 手写数字数据集

MNIST 数据集 (Mixed National Institute of Standards and Technology database) 是美国国家标准与技术研究院收集整理的大型手写数字数据库，包含 60,000 个示例的训练集以及 10,000 个示例的测试集。

<img src="README.assets/MNIST.jpeg" alt="How to Train a Model with MNIST dataset | by Abdullah Furkan Özbek | Medium" style="zoom:50%;" />

MNIST 数据集下载： http://yann.lecun.com/exdb/mnist/index.html 

### 3.2.2 CIFAR-10 / CIFAR-100 数据集

[CIFAR-10 和 CIFAR-100](http://www.cs.toronto.edu/~kriz/cifar.html) 是 [8000 万个微型图像数据集](http://groups.csail.mit.edu/vision/TinyImages/) 的标记子集。它们由 Alex Krizhevsky、Vinod Nair 和 Geoffrey Hinton 收集。

**CIFAR-10** 数据集包含 10 个类别的 60000 张 32x32 彩色图像，每类 6000 张图像。整个数据集由 50000 张训练图像和 10000 张测试图像组成。

官网下载到的数据集中包含5个batch，每个都由1000张各自类别的图像组成。

以下是数据集中的不同种类，以及每个种类的10张随机图像：

<img src="README.assets/CIFAR.png" style="zoom:50%;" />

**CIFAR-100** 数据集和 CIFAR-10 数据集类似，不同之处在于它有 100 种类别，每种类别包含 600 张图像，分别为 500 张训练图像和 100 张测试图像。CIFAR-100 中的 100 个类别分为 20 个 superclass。每个图像都带有一个”精细”标签（它所属的类）和一个“粗”标签（它所属的超类）。

与 MNIST 数据集相比，CIFAR-10 是 3 通道的彩色 RGB 图像，而 MNIST 是灰度图像。CIFAR-10 的图片尺寸为 32 × 32 ，而 MNIST 的图片尺寸为28 × 28 ，略小于 CIFAR-10 的图像尺寸。



# 4 实验步骤
见 `MNIST.ipynb` 和 `CIFAR.ipynb`

## 4.1 环境配置



## 4.2 数据准备



## 4.3 模型编写

### 4.3.1 网络结构



### 4.3.2 正向传播



### 4.3.3 反向传播



### 4.3.4 优化器



## 4.4 训练过程

多卡的训练需要配置 DDP。作为加分项。




# 5 实验任务与要求

1. 使用 `PyTorch` 实现最基本的卷积神经网络 LeNet-5，并在 MNIST 数据集上使用 GPU 进行训练，并对测试集进行测试。

2. 使用 `PyTorch` 实现更复杂的卷积神经网络，结构可以自行设计，并在CIFAR 10 数据集上使用 GPU 进行训练，并对测试集进行测试。
3. 你需要提交：
   1. 全部代码
   2. 实验报告，其中需要包含：
      1. 简要实验过程
      2. 贴上两个 CNN 模型训练过程的 **GPU 占用率截图**（使用 `nvidia-smi` 查看）
      3. Tensorboard **损失曲线、准确率曲线等截图**
      4. 写明测试集上的**识别正确率**。
4. ***不允许直接使用各种深度学习开发工具已训练好的 CNN 网络结构与参数。***



# 参考资料

- `PyTorch` 框架 https://pytorch.org/
- `PyTorch Lightning` 框架 https://www.pytorchlightning.ai/

- MNIST 数据集 http://yann.lecun.com/exdb/mnist/index.html

- CIFAR 10/100 数据集 http://www.cs.utoronto.ca/~kriz/cifar.html

- LeNet-5 网络结构 http://yann.lecun.com/exdb/lenet/

