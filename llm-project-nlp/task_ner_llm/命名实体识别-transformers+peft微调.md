<nav>
<a href="#一环境配置">一、环境配置</a><br/>
<a href="#二准备工作">二、准备工作</a><br/>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#21-基座模型下载">2.1 基座模型下载</a><br/>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#22-数据集构建">2.2 数据集构建</a><br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#原始数据">原始数据</a><br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#提示词模板">提示词模板</a><br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#alpaca-格式转换">Alpaca 格式转换</a><br/>
<a href="#三整体流程">三、整体流程</a><br/>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#31-微调阶段">3.1 微调阶段</a><br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#311-对话模板适配">3.1.1 对话模板适配</a><br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#312-tokenization分词与编码">3.1.2 Tokenization（分词与编码）</a><br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#313-主要参数配置">3.1.3 主要参数配置</a><br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#314-训练过程摘要">3.1.4 训练过程摘要</a><br/>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#32-推理阶段">3.2 推理阶段</a><br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#321-vllm-openai-api的部署方式">3.2.1 vLLM + OpenAI API的部署方式</a><br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#启动服务端命令">启动服务端命令</a><br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#查询模型列表接口">查询模型列表接口</a><br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#示例调用ner-实体识别任务">示例调用：NER 实体识别任务</a><br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#322-基于-transformers-peft-的本地推理部署">3.2.2 基于 transformers + PEFT 的本地推理部署</a><br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#加载模型和分词器">加载模型和分词器</a><br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#输出示例">输出示例</a><br/>
<a href="#四模型评估结果">四、模型评估结果</a><br/>
<a href="#参考引用">参考引用</a><br/>
</nav>



## 一、环境配置
docker镜像
```
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04
```
CUDA版本
```shell
nvcc -V
# Cuda compilation tools, release 12.2, V12.2.140
# Build cuda_12.2.r12.2/compiler.33191640_0
```
PYTHON版本和主要依赖库
```
Python 3.10.12 (main, Sep 11 2024, 15:47:36) [GCC 11.4.0] on linux
--------------------------------- -------------- -----
accelerate                        1.7.0
datasets                          3.6.0
torch                             2.4.0
tqdm                              4.66.5
traitlets                         5.14.3
transformers                      4.51.0
vllm                              0.6.3.post1
```

## 二、准备工作

### 2.1 基座模型下载
```python
from modelscope import snapshot_download
snapshot_download('Qwen/Qwen2.5-7B-Instruct', cache_dir='models', revision='master')
```

### 2.2 数据集构建
#### 原始数据
文件名：[clue.train.jsonl](data/dataset/clue.train.jsonl)，该文件采用 JSON Lines (jsonl) 格式存储，每行是一个独立的 JSON 对象。
示例格式如下：
```json lines
{
	"text": "凯尔特人在苏格兰赛场连胜7场，不过连胜含金量要打折扣。双方首回合交锋奥尔堡客场逼平凯尔特人。",
	"label": {  
        "organization": {"苏格兰": [[5, 7]], "奥尔堡": [[34, 36]], "凯尔特人": [[41, 44], [0, 3]]}
    }
}
```
说明：
+ "text" 表示原始文本内容。
+ "label" 包含实体类型（如 organization）及其对应的实体名称与出现位置索引。

#### 提示词模板
文件名：[clue_prompt.txt](data/prompts/clue_prompt.txt)，用于指导模型从输入文本中识别并提取指定类别的实体信息。其内容如下：
```
请从给定的句子中识别并提取出以下指定类别的实体。

<实体类别集合>
name, organization, scene, company, movie, book, government, position, address, game

<任务说明>
1. 仅提取属于上述类别的实体，忽略其他类型的实体。
2. 以json格式输出，对于每个识别出的实体，请提供：
   - label: 实体类型，必须严格使用原始类型标识（不可更改）
   - text: 实体在原文中的中文内容

<输出格式要求>
\```json
[{{"label": "实体类别", "text": "实体名称"}}]
\```

<输入文本>
{text}
```
#### Alpaca 格式转换
为了适配基于指令微调（Instruction Tuning）的训练方式，将原始 CLUE 数据集按照 Alpaca 格式 进行转换。
该格式通常包含三个字段：`instruction`、`input` 和 `output`，分别表示任务描述、输入文本及模型期望输出的结果。
转换后的数据文件：[alpaca_clue_train.json](data/dataset/alpaca/alpaca_clue_train.json)，示例内容如下：
```
[
  {
    "instruction": "你是一个文本实体识别领域的专家，擅长从自然语言中提取不同类别的实体名称。",
    "input": "请从给定的句子中识别并提取出以下指定类别的实体。\n\n<实体类别集合>\nname, organization, scene, company, movie, book, government, position, address, game\n\n<任务说明>\n1. 仅提取属于上述类别的实体，忽略其他类型的实体。\n2. 以json格式输出，对于每个识别出的实体，请提供：\n   - label: 实体类型，必须严格使用原始类型标识（不可更改）\n   - text: 实体在原文中的中文内容\n\n<输出格式要求>\n```json\n[{\"label\": \"实体类别\", \"text\": \"实体名称\"}]\n```\n\n<输入文本>\n凯尔特人在苏格兰赛场连胜7场，不过连胜含金量要打折扣。双方首回合交锋奥尔堡客场逼平凯尔特人。",
    "output": "[{\"label\": \"organization\", \"text\": \"苏格兰\"}, {\"label\": \"organization\", \"text\": \"奥尔堡\"}, {\"label\": \"organization\", \"text\": \"凯尔特人\"}]"
  },
  ...
]
```

## 三、整体流程
### 3.1 微调阶段
#### 3.1.1 对话模板适配
针对 Qwen 模型的对话格式要求，采用如下对话模板对输入数据进行格式化（[通义千问 (Qwen-核心概念-对话模板)](https://qwen.readthedocs.io/zh-cn/latest/getting_started/concepts.html)）：
```
<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{query1}<|im_end|>
<|im_start|>assistant
{response1}<|im_end|>
<|im_start|>user
{query2}<|im_end|>
<|im_start|>assistant
{response2}<|im_end|><|endoftext|>
```

在本任务中，采用 **单轮对话**（Single-turn Dialogue） 的形式对模型进行微调。
具体地，将 Alpaca 格式中的 `instruction` 与 `input` 合并作为模型的输入提示(prompt)，`output` 则作为预期输出标签(label)。
通过这种方式，将原始结构化任务转换为适用于 Qwen 系列模型的对话格式。

```
<|im_start|>system\n你是一个文本实体识别领域的专家，擅长从自然语言中提取不同类别的实体名称。<|im_end|>\n<|im_start|>user\n请从给定的句子中识别并提取出以下指定类别的实体。\n\n<实体类别集合>\nname, organization, scene, company, movie, book, government, position, address, game\n\n<任务说明>\n1. 仅提取属于上述类别的实体，忽略其他类型的实体。\n2. 以json格式输出，对于每个识别出的实体，请提供：\n   - label: 实体类型，必须严格使用原始类型标识（不可更改）\n   - text: 实体在原文中的中文内容\n\n<输出格式要求>\n```json\n[{\"label\": \"实体类别\", \"text\": \"实体名称\"}]\n```\n\n<输入文本>\n凯尔特人在苏格兰赛场连胜7场，不过连胜含金量要打折扣。双方首回合交锋奥尔堡客场逼平凯尔特人。<|im_end|>\n<|im_start|>assistant\n
```
```
[{\"label\": \"organization\", \"text\": \"苏格兰\"}, {\"label\": \"organization\", \"text\": \"奥尔堡\"}, {\"label\": \"organization\", \"text\": \"凯尔特人\"}]<|im_end|><|endoftext|>
```

#### 3.1.2 Tokenization（分词与编码）

在模型训练前，需将输入文本按照 Qwen 的 tokenizer 进行分词和编码处理，转换为 token ID 序列，并构造相应的 `input_ids`、`attention_mask` 和 `labels`。
```shell
input_text    : <|im_start|>system\n你是一个文本实体识别领域的专家，擅长从自然语言中提取不同类别的实体名称。<|im_end|>\n<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n
input_token   : ['<|im_start|>', 'system', 'Ċ', ..., 'éĵ¶è¡Į', '<|im_end|>', '<|endoftext|>']
input_ids     : [151644, 8948, 198, ..., 100358, 151645, 151643]
attention_mask: [1, 1, 1, ..., 1, 1, 1]
labels:       : [-100, -100, -100, ..., 100358, 151645, 151643]
```

+ input_ids: 表示输入文本经过 tokenizer 编码后的 token ID 序列；
+ attention_mask: 指明哪些位置是有效输入（1），哪些是 padding（0）；
+ labels: 在监督学习中用于计算损失函数，仅对模型输出部分（即 assistant 的回答内容）保留真实 token ID，其余部分设为 -100，表示不参与损失计算。

> 1、在构造模型输入 input_ids 时，不仅包含了用户输入（query），还把标签（response）也拼接进去了。<br>
> 因为语言模型是通过上下文预测下一个 token，所以必须把完整的历史上下文（包括 input 和 response）都传给模型。
> 
> 2、为什么 labels 要用 -100 屏蔽 input 部分？<br>
> 目的： 只让模型对 response（输出回答）部分 进行预测和计算损失，忽略 input（用户输入）部分。<br>
> 原因： -100 是 PyTorch 中 `torch.nn.CrossEntropyLoss` 的默认 “忽略索引”（ignore index）。 当标签中出现 -100 时，PyTorch 会自动跳过该位置的损失计算。<br>
> `torch.nn.CrossEntropyLoss 的构造函数 def __init__(self, ..., ignore_index: int = -100, ...)`
>
>3、这和推理阶段有什么不同？
>推理时不需要 labels，只需要 input，模型会自回归地生成 response。<br>


#### 3.1.3 主要参数配置
```
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,   # 因果语言建模任务
    inference_mode=False,           # 训练模式
    r=64,                           # Lora 权重矩阵的秩
    lora_alpha=16,                  # LoRA 的缩放因子
    lora_dropout=0.1,               # 在LoRA层施加的dropout比例
    target_modules=[                # 应用LoRA的子模块
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

# 训练参数配置
args = TrainingArguments(
    output_dir=f"{model_path}/checkpoints",
    per_device_train_batch_size=2,      # 每块GPU上的批次大小
    gradient_accumulation_steps=2,      # 累积梯度步数
    num_train_epochs=2,                 # 训练轮数
    logging_steps=10,                   # 日志记录间隔
    learning_rate=1e-4,                 # 微调学习率，一般比预训练时小
    gradient_checkpointing=True,        # 开启梯度检查点，节省显存
    report_to="none",                   # 不启用默认的日志报告（如TensorBoard）
    save_on_each_node=True,
    save_steps=200,                     # 模型保存间隔
    save_total_limit=2,                 # 最多保留的检查点数
    eval_strategy="steps",              # 评估触发方式（steps或epoch）
    eval_steps=200,                     # 评估步数
    metric_for_best_model="eval_loss",  # 使用验证 loss 判断最佳模型
    load_best_model_at_end=True
)
```
#### 3.1.4 训练过程摘要
以下是微调过程中部分关键训练日志信息：
```
{'loss': 0.0429, 'grad_norm': 0.2003297060728073, 'learning_rate': 5.583126550868486e-07, 'epoch': 1.99}                                                          
{'loss': 0.0275, 'grad_norm': 0.2421131432056427, 'learning_rate': 3.515301902398677e-07, 'epoch': 1.99}
{'loss': 0.0352, 'grad_norm': 0.09518225491046906, 'learning_rate': 1.4474772539288668e-07, 'epoch': 2.0}
{'train_runtime': 8418.1136, 'train_samples_per_second': 2.298, 'train_steps_per_second': 0.574, 'train_loss': 0.05433076250654198, 'epoch': 2.0}                              
100%|██████████████████████| 4836/4836 [2:20:18<00:00,  1.74s/it]
Best model saved at: ./data/outputs/Qwen2.5-7B-Instruct-clue-ner-lora-sft-peft/checkpoints/checkpoint-4200
```
+ Loss: 模型在训练阶段的损失值逐步下降，表明模型正在有效学习；
+ Grad Norm: 梯度范数稳定在合理范围内，未出现梯度爆炸；
+ Learning Rate: 学习率按调度策略递减；
+ Runtime & Throughput: 总训练耗时约 2 小时 20 分钟，平均每秒处理约 2.3 个样本，0.57 步迭代。


GPU 使用情况监控（NVIDIA A100）
```
|===============================+======================+======================|
|   0  NVIDIA A100-PCI...  Off  | 00000000:65:00.0 Off |                    0 |
| N/A   66C    P0   168W / 250W |  35250MiB / 40536MiB |     61%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
```

### 3.2 推理阶段
#### 3.2.1 vLLM + OpenAI API的部署方式

为了高效地部署微调后的 LoRA 模型并提供标准化接口服务，我们采用 vLLM 框架 提供的 OpenAI 兼容接口进行服务部署。该方式支持 LoRA 动态加载，并允许通过统一接口灵活切换不同模型或适配器。

##### 启动服务端命令
使用如下命令启动 vLLM 的 OpenAI API Server，其中包含了对 LoRA 参数的配置支持（如 max_loras、`max_lora_rank`、max_cpu_loras 等）：

```shell
python -m vllm.entrypoints.openai.api_server \
--host 0.0.0.0 \
--port 8000 \
--model ../model_hub/Qwen2.5-7B-Instruct \
--served-model-name Qwen2.5-7B-Instruct \
--gpu-memory-utilization 0.8 \
--max-model-len 1024 \
--disable-log-requests \
--enable-lora \
--max-lora-rank 64 \
--lora-modules clue-ner-lora-sft-peft=data/outputs/Qwen2.5-7B-Instruct-clue-ner-lora-sft-peft
```
+ model: 指定基础模型路径；
+ served-model-name: 对外暴露的模型名称；
+ enable-lora: 启用 LoRA 支持；
+ max-lora-rank: 设置最大 LoRA 秩（rank）；
+ lora-modules: 定义 LoRA 模块名称及其对应的模型路径；

GPU 资源监控信息：
```
|===============================+======================+======================|
|   0  NVIDIA A100-PCI...  Off  | 00000000:65:00.0 Off |                    0 |
| N/A   32C    P0    35W / 250W |  31312MiB / 40536MiB |      0%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
```

报错：
```
Error in model execution: LoRA rank 64 is greater than max_lora_rank 16.
```
在部署时，设置允许更高的 max_lora_rank（如 64）。
```shell
python -m vllm.entrypoints.openai.api_server --max-lora-rank 64 ...
```

##### 查询模型列表接口
服务启动后，可通过 `/v1/models` 接口查询当前加载的所有模型信息（LoRA 模型及其对应的基础模型）：
```shell
curl localhost:8000/v1/models

# 模型列表
{
    "object": "list",
    "data": [
        {
            "id": "Qwen2.5-7B-Instruct",
            "object": "model",
            ...
        },
        {
            "id": "clue-ner-lora-sft-peft",
            "object": "model",
            "root": "data/outputs/Qwen2.5-7B-Instruct-clue-ner-lora-sft-peft",
            "parent": "Qwen2.5-7B-Instruct",
            ...
        }
    ]
}
```
##### 示例调用：NER 实体识别任务

使用标准的 OpenAI 格式发送请求，调用 LoRA 微调后的模型进行实体识别任务：

```shell
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "clue-ner-lora-sft-peft",
  "messages": [
    {"role": "system", "content": "你是一个文本实体识别领域的专家，擅长从自然语言中提取不同类别的实体名称。"},
    {"role": "user", "content": "请从给定的句子中识别并提取出以下指定类别的实体。\n\n<实体类别集合>\nname, organization, scene, company, movie, book, government, position, address, game\n\n<任务说明>\n1. 仅提取属于上述类别的实体，忽略其他类型的实体。\n2. 以json格式输出，对于每个识别出的实体，请提供：\n   - label: 实体类型，必须严格使用原始类型标识（不可更改）\n   - text: 实体在原文中的中文内容\n\n<输出格式要求>\n```json\n[{"label": "实体类别", "text": "实体名称"}]\n```\n\n<输入文本>\n现在的阿森纳恐怕不能再给人以强队的信心，但教授的神经也真是够硬，在英超夺冠几无希望的情况下，"}
  ],
  "temperature": 0.01,
  "max_tokens": 512
}'
```
返回结果：
```
[{'label': 'organization', 'text': '阿森纳'}, {'label': 'organization', 'text': '英超'}, {'label': 'position', 'text': '教授'}]
```

#### 3.2.2 基于 transformers + PEFT 的本地推理部署
除了使用 vLLM 进行高性能服务化部署外，我们也可以通过 HuggingFace 的 transformers 框架结合 `PEFT`（Parameter-Efficient Fine-Tuning） 库实现本地加载和推理 LoRA 微调后的模型。
这种方式适用于开发调试、小规模部署或对推理过程有更细粒度控制的场景。

##### 加载模型和分词器
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from typing import List


def load_adapter_model() -> PeftModel:
    """ 加载基础模型与 LoRA 适配器模型，并返回模型和 tokenizer。
    """
    pretrain_path = "../model_hub/Qwen2.5-7B-Instruct"
    adapter_model_path = "./data/outputs/Qwen2.5-7B-Instruct-clue-ner-lora-sft-peft"
    # 加载基础模型和分词器
    model = AutoModelForCausalLM.from_pretrained(pretrain_path, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(pretrain_path)
    model = PeftModel.from_pretrained(model, adapter_model_path)
    return model, tokenizer

def predict(messages: List[dict], model: PeftModel, tokenizer):
    device = auto_device()
    # 获取当前设备（如 cuda:0）
    device = model.device

    # 构建对话模板文本（用于模型输入）
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    print("【DEBUG】构建后的 prompt 文本：")
    print(text)
    print("\n" + "-" * 80 + "\n")

    # Tokenize 输入文本并转换为张量
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    print("【DEBUG】tokenized 输入内容 (input_ids & attention_mask)：")
    print(model_inputs)
    print("\n" + "-" * 80 + "\n")

    # 模型生成 token ID 序列
    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
    print("【DEBUG】原始生成的 token IDs：")
    print(generated_ids)
    print("\n" + "-" * 80 + "\n")

    # 截取仅输出部分（去掉 prompt 部分）
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    print("【DEBUG】截断后的生成 token IDs（仅回答部分）：")
    print(generated_ids)
    print("\n" + "-" * 80 + "\n")

    # 解码生成内容为自然语言
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print("【DEBUG】最终解码结果（JSON 格式实体识别结果）：")
    print(response)
    print("\n" + "-" * 80 + "\n")

    return response


messages = [
    {"role": "system", "content": "你是一个文本实体识别领域的专家，擅长从自然语言中提取不同类别的实体名称。"},
    {"role": "user", "content": "......"}
]
model, tokenizer = load_adapter_model()
response = predict(messages, model, tokenizer)
print(response)
```
##### 输出示例

1、构建后的 Prompt 文本：
```
<|im_start|>system
你是一个文本实体识别领域的专家，擅长从自然语言中提取不同类别的实体名称。<|im_end|>
<|im_start|>user
请从给定的句子中识别并提取出以下指定类别的实体。\n\n<实体类别集合>\nname, organization, scene, company, movie, book, government, position, address, game\n\n<任务说明>\n1. 仅提取属于上述类别的实体，忽略其他类型的实体。\n2. 以json格式输出，对于每个识别出的实体，请提供：\n   - label: 实体类型，必须严格使用原始类型标识（不可更改）\n   - text: 实体在原文中的中文内容\n\n<输出格式要求>\n```json\n[{"label": "实体类别", "text": "实体名称"}]\n```\n\n<输入文本>\n现在的阿森纳恐怕不能再给人以强队的信心，但教授的神经也真是够硬，在英超夺冠几无希望的情况下，<|im_end|>
<|im_start|>assistant
```
2、生成 input_ids、attention_mask
```
{'input_ids': tensor([[151644,   8948,    198,   2610,    525,   1207,  16948,     11,   3465,
            553,  54364,  14817,     13,   1446,    525,    264,  10950,  17847,
             13, 151645,    198, 151644,    872,    198,  14880,  45181,  89012,
          22382,   9370, 109949,  15946, 102450,  62926, 107439,  20221,  87752,
         105146,  21515, 102657, 101565,   1773, 100779,  32022, 100358,  99304,
         151645,    198, 151644,  77091,    198]], device='cuda:0'), 
 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1]], device='cuda:0')}
```
3、原始生成的 Token IDs：
```
[tensor([ 99487, 109949, 102298,  87752, 101565,  48443,  73218,  13072,   5122,
        100779,  32022, 100358, 151645], device='cuda:0')]
```
4、最终解码结果（NER 实体识别）：
```
[{'label': 'organization', 'text': '阿森纳'}, {'label': 'organization', 'text': '英超'}, {'label': 'position', 'text': '教授'}]
```

## 四、模型评估结果
```
       Label Precision  Recall     F1
       scene    0.6933  0.5678 0.6243
    position    0.8054  0.7788 0.7919
organization    0.8102   0.782 0.7959
        name    0.8918  0.9135 0.9025
       movie    0.8705  0.8067 0.8374
  government    0.8132  0.8566 0.8343
        game    0.8354  0.9199 0.8756
     company    0.7931  0.8169 0.8048
        book    0.8182  0.8289 0.8235
     address    0.5956  0.6676 0.6295
  Micro Avg.    0.7907  0.8005 0.7956
  Macro Avg.         -       - 0.7920
```

## 参考引用
[1] [05-Qwen3-8B-LoRA及SwanLab可视化记录.md](https://github.com/datawhalechina/self-llm/blob/master/models/Qwen3/05-Qwen3-8B-LoRA%E5%8F%8ASwanLab%E5%8F%AF%E8%A7%86%E5%8C%96%E8%AE%B0%E5%BD%95.md)<br>
[2] [LLM Finetune-指令微调-命名实体识别/文本分类-Github](https://github.com/Zeyi-Lin/LLM-Finetune)<br>
[3] [LLM Finetune-指令微调-博客](https://blog.csdn.net/SoulmateY/article/details/139831606)<br>
[4] [vllm官方文档](https://docs.vllm.ai/en/v0.5.2/models/lora.html)<br>
[5] [利用大模型做NER实践(总结版)-Github](https://github.com/cjymz886/LLM-NER)<br>
[6] [利用大模型做NER实践(总结版)-博客](https://mp.weixin.qq.com/s/LBlzFm8wxK7Aj7YXhCoXgQ)<br>
[7] [大模型微调新手全流程友好指南](https://cloud.tencent.com/developer/article/2517177)<br>
[8] [基于 Qwen2.5-0.5B 微调训练 Ner 命名实体识别任务](https://blog.csdn.net/qq_43692950/article/details/142631780)<br>