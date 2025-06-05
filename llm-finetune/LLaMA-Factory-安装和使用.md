
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
> 官方文档是 git clone 整个项目以便支持最新的模型。

## 二、LLaMA-Factory 微调
### 2.1 构建数据集
Alpaca格式：alpaca_zh_demo.json
```json lines
[
  {
    "instruction": "识别并解释给定列表中的科学理论：细胞理论。",
    "input": "",
    "output": "细胞理论是生物科学的一个理论，它认为所有生命体都是由微小的基本单元——细胞所构成。"
  }
]
```
注册自定义数据集，将数据集添加到全局配置：dataset_info.json
```
{
  "alpaca_zh_demo": {
    "file_name": "alpaca_zh_demo.json"
  }
}
```
### 2.2 配置文件

`max_samples`决定了模型训练时从数据集中采样的最大样本数量，当设置为1000时，意味着训练过程中最多使用1000条数据进行模型训练。

在实际项目应用中，需要注意以下几点： 
+ 完整训练需求：对于生产环境或正式研究，通常建议移除该参数或设置为足够大的值，以确保模型能够学习到数据集的完整特征 
+ 数据采样影响：当数据集规模超过max_samples设定值时，框架会自动进行采样，可能导致模型无法充分学习数据分布 
+ 性能权衡：较小的max_samples值会牺牲模型性能换取训练速度，需要根据具体场景进行权衡

针对不同使用场景，建议采取以下策略： 
+ 开发调试阶段：可设置为500-1000，快速验证训练流程
+ 小规模实验：建议设置为5000-10000，平衡训练速度与模型性能 
+ 正式训练：应注释掉该参数或设置为None，使用完整数据集

### 2.3 训练
```shell
llamafactory-cli train conf/Qwen2.5-7B-Instruct-lora-sft.yaml
```





## 四、部署
```shell
API_PORT=8000 llamafactory-cli api examples/inference/llama3.yaml infer_backend=vllm vllm_enforce_eager=true
```


```shell
nohup python -m vllm.entrypoints.openai.api_server \
--host 0.0.0.0 \
--port 8000 \
--model ../model_hub/Qwen2.5-7B-Instruct \
--served-model-name Qwen2.5-7B-Instruct \
--enable-lora \
--gpu-memory-utilization 0.8 \
--max-model-len 1024 \
--disable-log-requests \
--lora-modules clue-ner-lora-sft=data/outputs/Qwen2.5-7B-Instruct-clue-ner-lora-sft \
> server.log &
```

## 五、调用

```shell
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "clue-ner-lora-sft",
  "messages": [
    {"role": "system", "content": "你是Qwen，由阿里云创建。你是一个乐于助人的助手。"},
    {"role": "user", "content": "你是谁？"}
  ],
  "temperature": 0.7,
  "top_p": 0.8,
  "repetition_penalty": 1.05,
  "max_tokens": 512
}'
```


## 参考引用
[1] [LLaMA-Factory-官方文档](https://llamafactory.readthedocs.io/zh-cn/latest/getting_started/installation.html)<br>
[2] [LLaMA-Factory-官方Github](https://github.com/hiyouga/LLaMA-Factory/blob/main/README_zh.md)<br>
