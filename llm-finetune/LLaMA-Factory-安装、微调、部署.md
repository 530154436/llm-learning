
## 一、环境配置
当前使用的环境是基于`cuda:12.2.2-cudnn8-devel-ubuntu22.04`镜像构建的，基本能满足条件。

### 1.1 CUDA安装
1、保证当前 Linux 版本支持CUDA
```shell
uname -m && cat /etc/*release
# x86_64
# DISTRIB_ID=Ubuntu
# DISTRIB_RELEASE=22.04
# DISTRIB_DESCRIPTION="Ubuntu 22.04.3 LTS"
```
2、检查是否安装了 gcc
```shell
gcc --version
# gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
```
3、检查cuda版本，官方推荐12.2。
```shell
nvcc -V
# Cuda compilation tools, release 12.2, V12.2.140
# Build cuda_12.2.r12.2/compiler.33191640_0
```
4、查看显卡
```shell
nvidia-smi
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 470.57.02    Driver Version: 470.57.02    CUDA Version: 12.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
```
### 1.2 LLaMA-Factory 安装
通过pypi进行安装 LLaMA-Factory 
```shell
pip install llamafactory==0.9.2
```
校验安装是否成功
```shell
llamafactory-cli version
----------------------------------------------------------
| Welcome to LLaMA Factory, version 0.9.2                |
| Project page: https://github.com/hiyouga/LLaMA-Factory |
----------------------------------------------------------
```


## 二、数据集格式
## 三、



## 参考引用
[1] [LLaMA-Factory-官方文档](https://llamafactory.readthedocs.io/zh-cn/latest/getting_started/installation.html)<br>
[2] [LLaMA-Factory-官方Github](https://github.com/hiyouga/LLaMA-Factory/blob/main/README_zh.md)<br>
