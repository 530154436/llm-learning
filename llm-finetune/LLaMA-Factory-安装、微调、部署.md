
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


inputs:
```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
你是一个文本实体识别领域的专家，请从给定的句子中识别并提取出以下指定类别的实体。

<实体类别集合>
name, organization, scene, company, movie, book, government, position, address, game

<任务说明>
1. 仅提取属于上述类别的实体，忽略其他类型的实体。
2. 以json格式输出，对于每个识别出的实体，请提供：
   - label: 实体类型，必须严格使用原始类型标识（不可更改）
   - text: 实体在原文中的中文内容

<输出格式要求>
``json
[{{"label": "实体类别", "text": "实体名称"}}]
``

浙商银行企业信贷部叶老桂博士则从另一个角度对五道门槛进行了解读。叶老桂认为，对目前国内商业银行而言，<|im_end|>
<|im_start|>assistant
[{"label": "name", "text": "叶老桂"}, {"label": "company", "text": "浙商银行"}]<|im_end|>
```

labels:
```
[{"label": "name", "text": "叶老桂"}, {"label": "company", "text": "浙商银行"}]<|im_end|>
```

number of train:  10748


## 三、微调
配置信息（Qwen2.5-7B-Instruct-lora-sft.yaml）
```
### model
model_name_or_path: /data/dev-nfs/zhengchubin/models/Qwen2.5-7B-Instruct
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
flash_attn: auto
lora_rank: 8
lora_target: all

### dataset
dataset_dir: data/dataset  # 存储数据集的文件夹路径。
dataset: alpaca_clue_train
template: qwen  # Qwen (1-2.5)
cutoff_len: 1024  # 输入的最大 token 数，超过该长度会被截断。
max_samples: 15000  # 每个数据集的最大样本数：设置后，每个数据集的样本数将被截断至指定的 max_samples。
overwrite_cache: true  # 是否覆盖缓存的训练和评估数据集。
preprocessing_num_workers: 16
dataloader_num_workers: 8

### output
output_dir: data/experiment/Qwen2.5-7B-Instruct-lora-sft
logging_steps: 10
save_steps: 500
plot_loss: true
overwrite_output_dir: true
save_only_model: false

### train
per_device_train_batch_size: 4  # 每设备训练批次大小
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
开始训练
```shell
llamafactory-cli train conf/Qwen2.5-7B-Instruct-lora-sft.yaml
```

`max_samples`决定了模型训练时从数据集中采样的最大样本数量，当设置为1000时，意味着训练过程中最多使用1000条数据进行模型训练。

在实际项目应用中，需要注意以下几点： 
+ 完整训练需求：对于生产环境或正式研究，通常建议移除该参数或设置为足够大的值，以确保模型能够学习到数据集的完整特征 
+ 数据采样影响：当数据集规模超过max_samples设定值时，框架会自动进行采样，可能导致模型无法充分学习数据分布 
+ 性能权衡：较小的max_samples值会牺牲模型性能换取训练速度，需要根据具体场景进行权衡

针对不同使用场景，建议采取以下策略： 
+ 开发调试阶段：可设置为500-1000，快速验证训练流程
+ 小规模实验：建议设置为5000-10000，平衡训练速度与模型性能 
+ 正式训练：应注释掉该参数或设置为None，使用完整数据集

per_device_train_batch_size (int, optional, defaults to 8):
每一个GPU/TPU 或者CPU核心训练的批次大小


|===============================+======================+======================|
|   0  NVIDIA A100-PCI...  Off  | 00000000:65:00.0 Off |                    0 |
| N/A   67C    P0   193W / 250W |  30756MiB / 40536MiB |     48%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+

[INFO|trainer.py:2405] 2025-05-30 02:56:21,267 >> ***** Running training *****
[INFO|trainer.py:2406] 2025-05-30 02:56:21,268 >>   Num examples = 9,673
[INFO|trainer.py:2407] 2025-05-30 02:56:21,268 >>   Num Epochs = 3
[INFO|trainer.py:2408] 2025-05-30 02:56:21,268 >>   Instantaneous batch size per device = 4
[INFO|trainer.py:2411] 2025-05-30 02:56:21,268 >>   Total train batch size (w. parallel, distributed & accumulation) = 32
[INFO|trainer.py:2412] 2025-05-30 02:56:21,268 >>   Gradient Accumulation steps = 8
[INFO|trainer.py:2413] 2025-05-30 02:56:21,268 >>   Total optimization steps = 906
[INFO|trainer.py:2414] 2025-05-30 02:56:21,274 >>   Number of trainable parameters = 20,185,088


## 四、部署
```shell
API_PORT=8000 llamafactory-cli api examples/inference/llama3.yaml infer_backend=vllm vllm_enforce_eager=true
```


```shell
nohup python -m vllm.entrypoints.openai.api_server \
--host 0.0.0.0 \
--port 8000 \
--model /data/dev-nfs/zhengchubin/models/Qwen2.5-7B-Instruct \
--served-model-name Qwen2.5-7B-Instruct \
--enable-lora \
--gpu-memory-utilization 0.8 \
--max-model-len 1024 \
--disable-log-requests \
--lora-modules clue-ner-lora-sft=data/experiment/Qwen2.5-7B-Instruct-lora-sft \
> server.log &
```

## 五、调用

```shell
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "clue-ner-lora",
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
