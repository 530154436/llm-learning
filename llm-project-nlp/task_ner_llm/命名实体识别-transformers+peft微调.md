

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
原始数据：data/dataset/clue.train.jsonl
```json lines
{
	"text": "北京城.",  // 表示原始文本
	"label": {
		"NT": {"北京城": [[0, 2]]}  // "label" 表示实体类型（如 NT）及其对应的实体名称和出现的位置索引。
	}
}
```
转换为alpaca格式：data/dataset/alpaca/alpaca_clue_train.json
```
[
  {
    "instruction": "你是一个文本实体识别领域的专家，请从给定的句子中识别并提取出以下指定类别的实体。\n\n<实体类别集合>\nname, organization, scene, company, movie, book, government, position, address, game\n\n<任务说明>\n1. 仅提取属于上述类别的实体，忽略其他类型的实体。\n2. 以json格式输出，对于每个识别出的实体，请提供：\n   - label: 实体类型，必须严格使用原始类型标识（不可更改）\n   - text: 实体在原文中的中文内容\n\n<输出格式要求>\n```json\n[{\"label\": \"实体类别\", \"text\": \"实体名称\"}]\n```\n\n<输入文本>\n北京城.",
    "input": "",
    "output": "[{\"label\": \"address\", \"text\": \"北京城\"}]"
  }
]
```

## 三、整体流程
### 3.1 微调阶段
### 3.1.1 对话模板适配
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
输入
```
<|im_start|>system\n你是一个文本实体识别领域的专家，擅长从自然语言中提取不同类别的实体名称。<|im_end|>\n<|im_start|>user\n请从给定的句子中识别并提取出以下指定类别的实体。\n\n<实体类别集合>\nname, organization, scene, company, movie, book, government, position, address, game\n\n<任务说明>\n1. 仅提取属于上述类别的实体，忽略其他类型的实体。\n2. 以json格式输出，对于每个识别出的实体，请提供：\n   - label: 实体类型，必须严格使用原始类型标识（不可更改）\n   - text: 实体在原文中的中文内容\n\n<输出格式要求>\n```json\n[{\"label\": \"实体类别\", \"text\": \"实体名称\"}]\n```\n\n<输入文本>\n凯尔特人在苏格兰赛场连胜7场，不过连胜含金量要打折扣。双方首回合交锋奥尔堡客场逼平凯尔特人。<|im_end|>\n<|im_start|>assistant\n
```
输出
```
"[{\"label\": \"organization\", \"text\": \"苏格兰\"}, {\"label\": \"organization\", \"text\": \"奥尔堡\"}, {\"label\": \"organization\", \"text\": \"凯尔特人\"}]<|im_end|><|endoftext|>"
```

在实际 tokenization 阶段，该模板将被转换为特定的词元序列，例如：
```shell
<|im_start|>system
你是一个文本实体识别领域的专家，擅长从自然语言中提取不同类别的实体名称。<|im_end|>
<|im_start|>user
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
[{"label": "实体类别", "text": "实体名称"}]
\```

<输入文本>
凯尔特人在苏格兰赛场连胜7场，不过连胜含金量要打折扣。双方首回合交锋奥尔堡客场逼平凯尔特人。<|im_end|>
<|im_start|>assistant

[{"label": "organization", "text": "苏格兰"}, {"label": "organization", "text": "奥尔堡"}, {"label": "organization", "text": "凯尔特人"}]<|im_end|><|endoftext|>




input_text    : <|im_start|>system\n请从给定的句子中识别并提取出以下指定类别的实体。浙商银行企业<|im_end|>\n<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n
input_token   : ['<|im_start|>', 'system', 'Ċ', ..., 'éĵ¶è¡Į', '<|im_end|>', '<|endoftext|>']
input_ids     : [151644, 8948, 198, ..., 100358, 151645, 151643]
attention_mask: [1, 1, 1, ..., 1, 1, 1]
labels:       : [-100, -100, -100, ..., 100358, 151645, 151643]
```
训练阶段：
```
{'loss': 0.0429, 'grad_norm': 0.2003297060728073, 'learning_rate': 5.583126550868486e-07, 'epoch': 1.99}                                                          
{'loss': 0.0275, 'grad_norm': 0.2421131432056427, 'learning_rate': 3.515301902398677e-07, 'epoch': 1.99}
{'loss': 0.0352, 'grad_norm': 0.09518225491046906, 'learning_rate': 1.4474772539288668e-07, 'epoch': 2.0}
{'train_runtime': 8418.1136, 'train_samples_per_second': 2.298, 'train_steps_per_second': 0.574, 'train_loss': 0.05433076250654198, 'epoch': 2.0}                              
100%|██████████████████████| 4836/4836 [2:20:18<00:00,  1.74s/it]
Best model saved at: ./data/outputs/Qwen2.5-7B-Instruct-clue-ner-lora-sft-peft/checkpoints/checkpoint-4200


|===============================+======================+======================|
|   0  NVIDIA A100-PCI...  Off  | 00000000:65:00.0 Off |                    0 |
| N/A   66C    P0   168W / 250W |  35250MiB / 40536MiB |     61%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
```

### 3.1 推理阶段

OpenAI接口方式
```
[{'role': 'system', 'content': '你是一个文本实体识别领域的专家，擅长从自然语言中提取不同类别的实体名称。'}, {'role': 'user', 'content': '请从给定的句子中识别并提取出以下指定类别的实体。\n\n<实体类别集合>\nname, organization, scene, company, movie, book, government, position, address, game\n\n<任务说明>\n1. 仅提取属于上述类别的实体，忽略其他类型的实体。\n2. 以json格式输出，对于每个识别出的实体，请提供：\n   - label: 实体类型，必须严格使用原始类型标识（不可更改）\n   - text: 实体在原文中的中文内容\n\n<输出格式要求>\n```json\n[{"label": "实体类别", "text": "实体名称"}]\n```\n\n<输入文本>\n现在的阿森纳恐怕不能再给人以强队的信心，但教授的神经也真是够硬，在英超夺冠几无希望的情况下，'}]

[{'label': 'organization', 'text': '阿森纳'}, {'label': 'organization', 'text': '英超'}, {'label': 'position', 'text': '教授'}]
```

peft
```
<|im_start|>system
你是一个文本实体识别领域的专家，擅长从自然语言中提取不同类别的实体名称。<|im_end|>
<|im_start|>user
请从给定的句子中识别并提取出以下指定类别的实体。\n\n<实体类别集合>\nname, organization, scene, company, movie, book, government, position, address, game\n\n<任务说明>\n1. 仅提取属于上述类别的实体，忽略其他类型的实体。\n2. 以json格式输出，对于每个识别出的实体，请提供：\n   - label: 实体类型，必须严格使用原始类型标识（不可更改）\n   - text: 实体在原文中的中文内容\n\n<输出格式要求>\n```json\n[{"label": "实体类别", "text": "实体名称"}]\n```\n\n<输入文本>\n现在的阿森纳恐怕不能再给人以强队的信心，但教授的神经也真是够硬，在英超夺冠几无希望的情况下，<|im_end|>
<|im_start|>assistant

{'input_ids': tensor([[151644,   8948,    198,   2610,    525,   1207,  16948,     11,   3465,
            553,  54364,  14817,     13,   1446,    525,    264,  10950,  17847,
             13, 151645,    198, 151644,    872,    198,  14880,  45181,  89012,
          22382,   9370, 109949,  15946, 102450,  62926, 107439,  20221,  87752,
         105146,  21515, 102657, 101565,   1773, 100779,  32022, 100358,  99304,
         151645,    198, 151644,  77091,    198]], device='cuda:0'), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1]], device='cuda:0')}
[tensor([ 99487, 109949, 102298,  87752, 101565,  48443,  73218,  13072,   5122,
        100779,  32022, 100358, 151645], device='cuda:0')]
这个句子包含以下实体：

公司名：浙商银行
```



### vllm部署和调用
+ 服务部署<br>
服务端入口接受所有其他的 LoRA 配置参数（如 max_loras、`max_lora_rank`、max_cpu_loras 等），这些参数将应用于所有后续的请求。

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
```
|===============================+======================+======================|
|   0  NVIDIA A100-PCI...  Off  | 00000000:65:00.0 Off |                    0 |
| N/A   32C    P0    35W / 250W |  31312MiB / 40536MiB |      0%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
```
查询 `/models` 接口，可以看到LoRA 模型及其对应的基础模型（base model）。
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

+ 调用示例
```shell
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "clue-ner-lora-sft-peft",
  "messages": [
    {"role": "system", "content": "你是Qwen，由阿里云创建。你是一个乐于助人的助手。"},
    {"role": "user", "content": "你是谁？"}
  ],
  "temperature": 0.01,
  "max_tokens": 512
}'
```




报错：
```
Error in model execution: LoRA rank 64 is greater than max_lora_rank 16.
```
在部署时，设置允许更高的 max_lora_rank（如 64）。
```shell
python -m vllm.entrypoints.openai.api_server --max-lora-rank 64 ...
```








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




1、在构造模型输入 input_ids 时，不仅包含了用户输入（query），还把标签（response）也拼接进去了。
因为语言模型是通过上下文预测下一个 token，所以必须把完整的历史上下文（包括 input 和 response）都传给模型。

2. 为什么 labels 要用 -100 屏蔽 input 部分？
目的： 只让模型对 response（输出回答）部分 进行预测和计算损失，忽略 input（用户输入）部分。
原因： -100 是 PyTorch 中 `torch.nn.CrossEntropyLoss` 的默认 “忽略索引”（ignore index）。 当标签中出现 -100 时，PyTorch 会自动跳过该位置的损失计算。
```python
# torch.nn.CrossEntropyLoss 的构造函数
def __init__(self, weight: Optional[Tensor] = None, size_average=None, ignore_index: int = -100,
             reduce=None, reduction: str = 'mean', label_smoothing: float = 0.0)
```

3、这和推理阶段有什么不同？
推理时不需要 labels，只需要 input，模型会自回归地生成 response。


[] [vllm官方文档](https://docs.vllm.ai/en/v0.5.2/models/lora.html)<br>



大模型微调新手全流程友好指南
https://cloud.tencent.com/developer/article/2517177

### 基于Transformers+peft框架
利用大模型做NER实践(总结版)
https://mp.weixin.qq.com/s/LBlzFm8wxK7Aj7YXhCoXgQ
https://github.com/cjymz886/LLM-NER

LLM Finetune：
指令微调-文本分类
指令微调-命名实体识别
博客：https://blog.csdn.net/SoulmateY/article/details/139831606
代码：https://github.com/Zeyi-Lin/LLM-Finetune

05-Qwen3-8B-LoRA及SwanLab可视化记录.md
https://github.com/datawhalechina/self-llm/blob/master/models/Qwen3/05-Qwen3-8B-LoRA%E5%8F%8ASwanLab%E5%8F%AF%E8%A7%86%E5%8C%96%E8%AE%B0%E5%BD%95.md

chinese_ner_sft数据集
https://hf-mirror.com/datasets/qgyd2021/chinese_ner_sft

Qwen2.5大模型微调实战：医疗命名实体识别任务（完整代码）
https://zhuanlan.zhihu.com/p/19682001982

基于 Qwen2.5-0.5B 微调训练 Ner 命名实体识别任务
https://blog.csdn.net/qq_43692950/article/details/142631780
