---
layout:     post
title:      Pytorch GPU Win
subtitle:   Pytorch GPU Windows 配置
date:       2022-05-10
author:     py
header-img: img/post-bg-ios9-web.jpg
catalog: true
tags:
    - 学习
    - Pytorch
    - Cuda
    - Python
---

**目录**

[1 准备阶段](#t0)

[1.1 安装Anaconda](#t1)

[1.2 查看显卡型号](#t2)

[1.3 更新最新版本显卡驱动](#t3)

[1.4 查看显卡驱动版本信息](#t4)

[2 安装PyTorch（GPU版）](#t5)

[2.1 通过conda创建一个虚拟环境](#t6)

[2.2 方案一：通过官网命令行安装Pytorch](#t7)

[2.2.1 配置官网命令（推荐pip）](#2.2.1%20%E9%85%8D%E7%BD%AE%E5%AE%98%E7%BD%91%E5%91%BD%E4%BB%A4)

[2.2.2 在创建的虚拟环境中通过命令行自动下载安装](#2.2.2%20%E5%9C%A8%E5%88%9B%E5%BB%BA%E7%9A%84%E8%99%9A%E6%8B%9F%E7%8E%AF%E5%A2%83%E4%B8%AD%E9%80%9A%E8%BF%87%E5%91%BD%E4%BB%A4%E8%A1%8C%E8%87%AA%E5%8A%A8%E4%B8%8B%E8%BD%BD%E5%AE%89%E8%A3%85)

[2.2.3 检查是否安装成功](#2.2.3%20%E6%A3%80%E6%9F%A5%E6%98%AF%E5%90%A6%E5%AE%89%E8%A3%85%E6%88%90%E5%8A%9F)

[2.3 方案二：通过本地whl文件的方式安装](#t8)

[2.3.1 下载镜像文件](#2.3.1%20%E4%B8%8B%E8%BD%BD%E9%95%9C%E5%83%8F%E6%96%87%E4%BB%B6)

[2.3.2 安装](#2.3.2%20%E5%AE%89%E8%A3%85)

[2.3.3 检验是否安装成功](#2.3.3%20%E6%A3%80%E9%AA%8C%E6%98%AF%E5%90%A6%E5%AE%89%E8%A3%85%E6%88%90%E5%8A%9F)

* * *

### 1 准备阶段

#### 1.1 安装Anaconda

    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.shchmod +x Miniconda3-latest-Linux-x86_64.sh./Miniconda3-latest-Linux-x86_64.sh

方法一：[官网](https://www.anaconda.com/download/ "官网")下载Anaconda安装。

方法二： 官网下载太慢，推荐国内镜像源下载最新版：

*   [anaconda清华大学开源软件镜像站](https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/ "anaconda清华大学开源软件镜像站")
*   [Anaconda3](https://repo.anaconda.com/archive/ "Anaconda3")
    

安装过程可以参考：[Anaconda安装-超详细版(2023)](https://blog.csdn.net/weixin_43412762/article/details/129599741?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522172260622316800207062977%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=172260622316800207062977&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-129599741-null-null.142%5Ev100%5Epc_search_result_base3&utm_term=anaconda%E5%AE%89%E8%A3%85&spm=1018.2226.3001.4187 "Anaconda安装-超详细版(2023)")

#### 1.2 查看显卡型号

右击开始菜单→任务管理器→性能→GPU 1，其中 NVIDI GeForce GTX 1050表示GPU型号。 

![](https://i-blog.csdnimg.cn/direct/05fd173c886144b4869858252351383d.png)

注：若只显示：GPU 0   Intel(R) HD Graphics 630，是CPU集成显卡，表示没有GPU，不能安装GPU版本的PyTorch。

![](https://i-blog.csdnimg.cn/direct/f6064ccf9d36433e9230246851d943e1.png)

#### 1.3 更新最新版本显卡驱动

在官网下载对应型号的最新显卡驱动，并安装(全部下一步)，地址：[NVIDIA GeForce 驱动程序](https://www.nvidia.cn/geforce/drivers/ "NVIDIA GeForce 驱动程序")

![](https://i-blog.csdnimg.cn/direct/36ed3fdd039540cbb7b3ab2ff30a48b3.png)

安装结束后，重启电脑。

#### 1.4 [查看显卡驱动版本](https://so.csdn.net/so/search?q=%E6%9F%A5%E7%9C%8B%E6%98%BE%E5%8D%A1%E9%A9%B1%E5%8A%A8%E7%89%88%E6%9C%AC&spm=1001.2101.3001.7020)信息

双击NVIDIA设置小图标打开NVIDIA控制面板（或桌面空白处鼠标右击→NVIDA控制面板）→帮助→系统信息→组件，查看电脑驱动的版本：NVIDIA CUDA 12.6.32（12.6版）。

注：后期安装PyTorch时，选取的CUDA不能高于12.6。

![](https://i-blog.csdnimg.cn/direct/5e54f1e51dfe457485185dde0cc80a46.png)

### 2 [安装PyTorch](https://so.csdn.net/so/search?q=%E5%AE%89%E8%A3%85PyTorch&spm=1001.2101.3001.7020)（GPU版）

#### 2.1 通过conda创建一个虚拟环境

按住win＋R ，输入cmd，在命令行输入：conda create –n 虚拟环境名字 python=版本，即可创建成功。本部分创建的虚拟环境名称为：myPytorch，python版本为3.8。

    conda create –n myPytorch python=3.8
    

查看已创建的虚拟环境：

    conda env list

![](https://i-blog.csdnimg.cn/direct/54e9c8a5300a4b8cad064db14c44b333.png)

#### 2.2 方案一：通过官网命令行安装Pytorch

##### 2.2.1 配置官网命令（推荐pip）

官网地址： [PyTorch](https://pytorch.org/ "PyTorch")  或者 [Start Locally | PyTorch](https://pytorch.org/get-started/locally/ "Start Locally | PyTorch")

注：此处**CUDA的版本** ≤ **电脑显卡驱动**版本（最好保持一致），可以通过更新电脑显卡驱动（前面提过）或者在Previous versions of PyTorch 中寻找低版本的CUDA。

![](https://i-blog.csdnimg.cn/direct/8d25dfb0de75495298a2cf95fa725e9e.png)

##### 2.2.2 在创建的虚拟环境中通过命令行自动下载安装

打开Anaconda Prompt命令行窗口：开始→Anaconda Prompt （或者win+R，输入cmd）。

![](https://i-blog.csdnimg.cn/direct/49e2b075cdd0463697ec8082104ea786.png)

进入虚拟环境：输入conda activate 虚拟环境

    conda activate myPytorch
    

在虚拟环境下安装，输入官网配置的命令：

    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

![](https://i-blog.csdnimg.cn/direct/3b4cc509be70413aa5fb72824ad17bbd.png)

##### 2.2.3 检查是否安装成功

方案一：输入`pip list`或者`conda list`，看有没有pytorch或者torch，有表示成功。![](https://i-blog.csdnimg.cn/direct/981bce14edac4b3c92f99c56eb85d62c.png)

方案二：通过python验证

在命令行以此输入：  
python  
import torch  
torch.cuda.is\_available()，这个命令是检查我们pytorch的GPU能否用。

    pythonimport torchtorch.cuda.is_available()

![](https://i-blog.csdnimg.cn/direct/00653efcb95f400097825023c4c68e06.png)

如果显示True，就说明PyTorch安装成功了。

#### 2.3 方案二：通过本地镜像文件离线方式安装（推荐清华镜像源）

##### 2.3.1 下载镜像文件

*   [https://download.pytorch.org/whl/torch/](https://download.pytorch.org/whl/torch/ "https://download.pytorch.org/whl/torch/")  
    ![](https://i-blog.csdnimg.cn/direct/996986f10f484a24af53be39bc388d72.png)

*   [清华大学开源软件镜像站](https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/win-64/ "清华大学开源软件镜像站")  
    ![](https://i-blog.csdnimg.cn/direct/87f2336f14fb4323acc9f860dd1a5085.png)  
    ![](https://i-blog.csdnimg.cn/direct/cc36a27e92d7425ca0ba16ea014df5a7.png)

##### 2.3.2 安装

打开Anaconda Prompt命令行窗口：开始→Anaconda Prompt （或者win+R，输入cmd）。

切换到虚拟环境：conda activate 虚拟环境     

方案一：找到下载好的.whl文件，在命令行运行：

    pip3 install C:\Users\Lenovo\Desktop\torch-2.4.0+cu124-cp38-cp38-win_amd64.whl

注：换成 pip3 install 你的路径

方案二：清华镜像源，下载pytorch和torchvision，在命令行运行：

    conda install --offline C:\Users\Lenovo\Desktop\pytorch-2.4.0-py3.8_cuda12.4_cudnn9_0.tar.bz2conda install --offline C:\Users\Lenovo\Desktop\torchvision-0.19.0-py38_cu124.tar.bz2

注：换成  
                conda install --offline 路径+pytorch压缩包的全称  
                conda install --offline 路径+torchvision压缩包的全称

##### 2.3.3 检验是否安装成功

同方案一验证