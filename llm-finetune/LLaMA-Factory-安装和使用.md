<nav>
<a href="#一环境配置">一、环境配置</a><br/>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#11-cuda安装">1.1 CUDA安装</a><br/>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#12-llama-factory-安装">1.2 LLaMA-Factory 安装</a><br/>
<a href="#二llama-factory-微调">二、LLaMA-Factory 微调</a><br/>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#21-构建数据集">2.1 构建数据集</a><br/>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#22-配置文件">2.2 配置文件</a><br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#221-template参数">2.2.1 template参数</a><br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#222-max_samples参数">2.2.2 max_samples参数</a><br/>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#23-llamafactory-cli">2.3 llamafactory-cli</a><br/>
<a href="#参考引用">参考引用</a><br/>
</nav>


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
注册自定义数据集，将数据集添加到全局配置：dataset_info.json；**columns字段可以不加**。
```
{
  "alpaca_zh_demo": {
    "file_name": "alpaca_zh_demo.json",
    "columns": {
        "prompt": "instruction",
        "query": "input",
        "response": "output",
        "system": "system",
        "history": "history"
    }
  }
}
```
### 2.2 配置文件
```
### model
model_name_or_path: ../model_hub/Qwen2.5-7B-Instruct
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
flash_attn: auto
lora_rank: 8
lora_target: all

### dataset
dataset_dir: data/dataset/alpaca  # 存储数据集的文件夹路径。
dataset: alpaca_clue_train
template: qwen  # Qwen (1-2.5)
cutoff_len: 1024  # 输入的最大 token 数，超过该长度会被截断。
max_samples: 15000  # 每个数据集的最大样本数：设置后，每个数据集的样本数将被截断至指定的 max_samples。
overwrite_cache: true  # 是否覆盖缓存的训练和评估数据集。
preprocessing_num_workers: 16
dataloader_num_workers: 8

### output
output_dir: data/outputs/Qwen2.5-7B-Instruct-clue-ner-lora-sft
logging_steps: 10
save_steps: 500
plot_loss: true
overwrite_output_dir: true
save_only_model: false

### train
per_device_train_batch_size: 4  # 每设备训练批次大小, 默认 8
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
resume_from_checkpoint: null

### eval
#eval_dataset:
val_size: 0.1
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 500
```

> [LLaMA-Factory 完整参数列表](https://llamafactory.readthedocs.io/zh-cn/latest/advanced/arguments.html)

#### 2.2.1 template参数

对于所有“基座”（Base）模型，`template` 参数可以是 `default`, `alpaca`, `vicuna` 等任意值。但“对话”（Instruct/Chat）模型请务必使用**对应的模板**。

| 模型名                                                             | 参数量                              | Template    |
|-----------------------------------------------------------------|----------------------------------|-------------|
| [ChatGLM3](https://huggingface.co/THUDM)                        | 6B                               | chatglm3    |
| [DeepSeek (Code/MoE)](https://huggingface.co/deepseek-ai)       | 7B/16B/67B/236B                  | deepseek    |
| [DeepSeek 2.5/3](https://huggingface.co/deepseek-ai)            | 236B/671B                        | deepseek3   |
| [DeepSeek R1 (Distill)](https://huggingface.co/deepseek-ai)     | 1.5B/7B/8B/14B/32B/70B/671B      | deepseekr1  |
| [Llama](https://github.com/facebookresearch/llama)              | 7B/13B/33B/65B                   | -           |
| [Llama 2](https://huggingface.co/meta-llama)                    | 7B/13B/70B                       | llama2      |
| [Llama 3-3.3](https://huggingface.co/meta-llama)                | 1B/3B/8B/70B                     | llama3      |
| [Llama 4](https://huggingface.co/meta-llama)                    | 109B/402B                        | llama4      |
| [Llama 3.2 Vision](https://huggingface.co/meta-llama)           | 11B/90B                          | mllama      |
| [Qwen (1-2.5) (Code/Math/MoE/QwQ)](https://huggingface.co/Qwen) | 0.5B/1.5B/3B/7B/14B/32B/72B/110B | qwen        |
| [Qwen3 (MoE)](https://huggingface.co/Qwen)                      | 0.6B/1.7B/4B/8B/14B/32B/235B     | qwen3       |
| [Qwen2-Audio](https://huggingface.co/Qwen)                      | 7B                               | qwen2_audio |
| [Qwen2.5-Omni](https://huggingface.co/Qwen)                     | 3B/7B                            | qwen2_omni  |
| [Qwen2-VL/Qwen2.5-VL/QVQ](https://huggingface.co/Qwen)          | 2B/3B/7B/32B/72B                 | qwen2_vl    |

> 详见：[LLaMa-Factory README_zh.md](https://github.com/hiyouga/LLaMA-Factory/blob/main/README_zh.md)

#### 2.2.2 max_samples参数

`max_samples`决定了模型训练时从数据集中采样的最大样本数量，当设置为1000时，意味着训练过程中最多使用1000条数据进行模型训练。
在实际项目应用中，需要注意以下几点： 
+ 完整训练需求：对于生产环境或正式研究，通常建议移除该参数或设置为足够大的值，以确保模型能够学习到数据集的完整特征 
+ 数据采样影响：当数据集规模超过max_samples设定值时，框架会自动进行采样，可能导致模型无法充分学习数据分布 
+ 性能权衡：较小的max_samples值会牺牲模型性能换取训练速度，需要根据具体场景进行权衡

针对不同使用场景，建议采取以下策略： 
+ 开发调试阶段：可设置为500-1000，快速验证训练流程
+ 小规模实验：建议设置为5000-10000，平衡训练速度与模型性能 
+ 正式训练：应注释掉该参数或设置为None，使用完整数据集

### 2.3 llamafactory-cli
+ 训练
```shell
llamafactory-cli train conf/Qwen2.5-7B-Instruct-lora-sft.yaml
```
+ 推理 api
```shell
llamafactory-cli api \
  --model_name_or_path ../model_hub/Qwen2.5-7B-Instruct \
  --adapter_name_or_path data/outputs/Qwen2.5-7B-Instruct-clue-ner-lora-sft \
  --infer_backend vllm \
  --template qwen \
  --finetuning_type lora \
  --vllm_enforce_eager=true
```
+ 调用 api
```shell
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "../model_hub/Qwen2.5-7B-Instruct",
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
[3] [llamafactory-cli 推理](https://llamafactory.readthedocs.io/zh-cn/latest/getting_started/inference.html)<br>