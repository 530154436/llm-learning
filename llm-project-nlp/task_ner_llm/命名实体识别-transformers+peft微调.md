





微调输入格式：
```shell
# qwen对话模板
input_template: "<|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|><|endoftext|>"

# 模型输入特征
input_id_token: ['<|im_start|>', 'system', 'Ċ', ..., 'éĵ¶è¡Į', '<|im_end|>', '<|endoftext|>']
input_ids     : [151644, 8948, 198, ..., 100358, 151645, 151643]
attention_mask: [1, 1, 1, ..., 1, 1, 1]
labels:       : [-100, -100, -100, ..., 100358, 151645, 151643]
```
推理输入格式：
```

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
