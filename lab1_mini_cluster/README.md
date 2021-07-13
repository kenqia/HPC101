<font size=10>实验一：简单集群搭建</font>

---

# 1 实验简介

# 2 实验环境


# 3 实验基础知识介绍

# 4 实验步骤
## 下载 Hypervisor 和 Linux 光盘映像文件
### Hypervisor
在创建虚拟机之前，你需要先准备好 Hypervisor，在本手册中，我们以 Virtual Box 为例，其他的 Hypervisor 请自行参阅相关材料：
- [Virtual Box 官网下载](https://www.virtualbox.org/wiki/Downloads)：
![Virtual Box 官网](pics/01.png)
> 注意**不要选错宿主机平台**！
#### Docker
不推荐也不适合使用，请同学们不要钻牛角尖。
### Linux 光盘映像文件
本手册所使用的范例发行版是 Debian 10，同学可根据自己的喜好和经验挑选适合的发行版。
Debian 下载点（如果网速问题可访问国内镜像）：

- [Official Mirror](https://cdimage.debian.org/debian-cd/current-live/amd64/iso-hybrid/)
- [Tuna Mirror](https://mirrors.tuna.tsinghua.edu.cn/debian-cd/current-live/amd64/iso-hybrid/)
![Debian 下载](pics/02.png)
> 花花绿绿的映像，该怎么选择好呢？图中带有 mate 字眼的映像代表附有 mate 桌面，为了最精简安装，本手册采 standard 版本（不含桌面）。
#### 不推荐使用的光盘映像
##### 从互联网下载 (关键字：Install via Internet / netinstall)
此类镜像是在安装过程中访问互联网下载需要的包，由于虚拟机可能无法访问互联网，部分镜像虽然体积小但缺少很多重要的命令，因此不要挑选此类镜像。
##### 带有桌面环境的光盘映像 (关键字：Desktop / 以及后面带有 gnome, kde, lxde 等词的映像)
如果计算机上的磁盘空间较不充足，建议不要下载带有桌面环境的光盘映像，如果你偏好桌面且空间足够可以忽略。
##### 你不熟悉的光盘映像 / 没有图形界面(GUI)安装程序的光盘映像
Linux 发行版相当多，不熟悉或没使用过 Linux 的同学建议参考本手册，相信自己已经有一定基础的同学可以忽略。
## 搭建集群并安装相关程序
### 创建虚拟机
准备好 Hypervisor 跟 光盘映像后，就可以着手安装一个虚拟机了，请参考 Virtual Box 手册和相关教程。  
#### 选择发行版、内存、磁盘空间
![Step1-1](pics/03.jpg)
![Step1-2](pics/04.jpg)
选择自己要安装的发行版（如果没有直接选 Linux 即可），内存和磁盘空间根据实际情况分配。  
#### 插入发行版映像文件和配置网络
![Step2-1](pics/05.jpg)
选取刚下载的映像文件。  
![Step2-2](pics/06.jpg)
网络对虚拟机来说十分复杂，如果你不熟悉相关的名词，在一台虚拟机的情况下，直接默认的 NAT 即可，但因为手册需要节点间彼此互连，
因此使用 NAT Network，这样同时也可访问互联网。  
不需要互联网或者使用有线网的同学也可以考虑使用 Bridged 或 Internal Network，具体请参考手册上的说明：[Virtual Networking](https://www.virtualbox.org/manual/ch06.html)。
> 小贴士
> - 如果你的虚拟机无法连上网，可能是宿主机的问题，请**排查宿主机的网络状况**。
> - 由于学校的网络情况，在学校中使用虚拟机的同学还是**以 NAT Network 为主**较方便。
#### 进入图形安装程序
在启动虚拟机后，不要直接进入 Live CD，直接进入安装程序，照着引导走即可。  
![Step3-1](pics/07.jpg)

#### 安装完成并重启
在刚刚插入映像文件的地方取消选取映像文件，否则下次重启时还会进入 Live CD。  
### 下载并安装 OpenMPI
由于系统中的包通常比较旧，因此我们从 OpenMPI 的官网中下载最新版本源码自行编译安装：[OpenMPI 下载](https://www.open-mpi.org/software/ompi/v4.1/)。  
> 小贴士
> - 在虚拟机**可联网**的环境下，可直接使用 `wget` 或 `curl` 来下载，会方便许多。
> - 在编译前，先了解 `make` 和 `autoconf`，可以减少不少除错时间。
#### 修改 `PATH` 和 `LD_LIBRARY_PATH`
请将找到 OpenMPI 二进制文件的目录加入 `PATH` 环境变量，OpenMPI 库的目录加入 `LD_LIBRARY_PATH` 环境变量。 
### 下载并安装 HPL
下载点：https://netlib.org/benchmark/hpl/software.html
#### BLAS
HPL 的依赖除了一个 MPI 实现（在手册中是 OpenMPI）外，还需要一个 BLAS 实现，我们可以从 netlib 下载[其中一个实现](https://www.netlib.org/blas/#_software)，
虽然没有优化过，但拿来测试已经足够了。

##### 检查 `gcc` /`gfortran` 环境
BLAS 需要 gcc/gfortran 来编译，请务必检查自己虚拟机中编译器是否存在及其版本。
##### 编译 BLAS/CBLAS
先编译 BLAS，再参考 `README` 和 `INSTALL` 修改 CBLAS 的 Makefile 并编译 （需要 BLAS 的链接文件）。
#### 修改 HPL Makefile
解压 HPL 压缩包后，在根目录的 `setup/` 文件夹下有 Makefile 相关文件的模板(Make.xxx，后缀代表架构)，复制到根目录并保存。  
参考 `README` ，根据自己情况修改以下参数：  

```makefile
# Make.test
# arch
ARCH = test
...
# MPI
MPdir = /path/to/your/mpi/
MPinc = -I$(MPdir)/include64
MPlib = $(MPdir)/libmpi.so
# BLAS
LAdir = /path/to/your/blas
LAinc =
LAlib = $(LAdir)/lib
...
# compiler
CC = /path/to/your/mpicc
LINKER = $(CC)
```
修改当前目录下 `Make.top` 中的 `arch` 参数，需与 `Make.test` 中的一致。
#### 编译
```shell
# 替换成你自己的后缀
make arch=test
```
### 克隆节点
在 Virtual Box 中，克隆已经配置完成的节点成为集群中的其他节点，本手册范例中仅克隆一个（集群中两个节点），可克隆更多。  
## 测试集群
### `ping`
`ping` 是测试节点网络连通性最为简单的方式，在进行其他测试前，请先确认能 `ping` 通所有节点。
### 配置 ssh
`ssh` 是相当常用的，实现安全远程连接的方式，其原理和使用、配置方法请查阅相关参考资料：[Open SSH 网站](https://www.openssh.com/manual.html)
如果你没有 ssh 密钥，可以在其中一个节点创建一个：  

```shell
ssh key-gen
```
我们需要将自己的 ssh 公钥复制一份到另一个节点上的 `.ssh/authorized_keys` 中（可以利用 `nc` 命令来传输文件）。复制完成后，注意检查
`authorized_keys`(600) 和 `.ssh/`(700) 目录的权限，否则无法顺利 `ssh`。

#### ssh passphrase
如果自己的密钥有 passphrase，那么请使用 `ssh-agent` 确保能暂时不用输入 passphrase，以免之后影响 `mpirun` 正确运行。
### `mpirun`
尽管对 OpenMPI 十分陌生，我们还是能利用它来跑非 MPI 的程序，同学可以编写简单的 Helloworld 程序来测试 OpenMPI，或者直接使用 Unix 命令。
#### OpenMPI hostfile
两个节点都准备好后，我们可以试试 `mpirun` 了！按照如下格式，编写 MPI 的 hostfile 并保存：
```
localhost slots=1
10.0.2.5  slots=1
```
其中 slots 代表的意思是一个节点的 CPU 有多少核，由于我们创建虚拟机时仅分配一个核，因此这里的 slots 上限为 1 ，同学根据自己虚拟机的
情况，修改 slots 的值。

#### 测试程序与 HPL
OpenMPI 需要测试程序为节点所共有或在节点上有相同路径，因为我们的第二个节点是克隆出来的，因此两个节点上的命令和 HPL 程序都会是相同路径，
此时运行以下命令：

```
mpirun --hostfile myhostfile uptime
```
就能看到每个节点上线了多久。  
HPL：

```
mpirun --hostfile myhostfile ./xhpl
```
# 5 实验任务与要求
1. 搭建四个节点的虚拟机并记录过程
2. 使用 OpenMPI 和 HPL 测试集群表现并记录结果
