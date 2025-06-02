
## 一、预训练模型

BERT（Bidirectional Encoder Representations from Transformers） 是 Google 提出的一种基于 Transformer 的预训练语言模型，具有以下特点：
+ 双向上下文建模能力
+ 使用 MLM（Masked Language Model）和 NSP（Next Sentence Prediction）任务进行预训练
+ 可以通过微调迁移到各种下游任务中，包括 NER、文本分类、问答系统等

在本项目中，我们使用了多个中文 BERT 预训练版本，包括：

| 版本                      | 适用场景                          | 特点                                                                                | 
|-------------------------|-------------------------------|-----------------------------------------------------------------------------------|
| bert-base-uncased       | 大小写不敏感的任务，如一般的文本分类任务          | 所有输入文本都会被转换为小写，减少了词汇表的大小                                                          |
| bert-base-cased         | 需要区分大小写的任务，例如命名实体识别(NER)或语法分析 | 保持了文本中的大小写信息，有助于捕捉更多文本结构细节                                                        |
| chinese-bert-wwm-ext    | 中文命名实体识别、文本分类、句法分析等           | 由哈工大讯飞联合实验室开发<br>通过全词掩码（wwm，Whole Word Masking）技术增强了对中文语境的理解                      |
| chinese-roberta-wwm-ext | 中文文本分类、命名实体识别、问答系统等           | 预训练阶段采用wwm策略进行mask（但没有使用dynamic masking）<br> 取消了Next Sentence Prediction（NSP）<br> |

注意：chinese-roberta-wwm-ext 模型在使用上与中文BERT系列模型完全一致，无需任何代码调整即可使用。

## 二、训练流程
### 2.1 数据预处理
原始数据格式：
```
{
	"text": "北京城.",
	"label": {
		"NT": {"北京城": [[0, 2]]}
	}
}
```

BIOS格式：
```
北 B-NT
京 I-NT
城 I-NT
.  O
```
collate_fn：
1、截断：如果输入长度超过限制，则截断至允许的最大长度（减去特殊标记长度）
2、将 token 转换为 id，并加上特殊标记 [CLS] 和 [SEP]
3、padding：填充到固定长度 min_max_seq_length

### 2.2 模型架构

### 


[哈工大讯飞联合实验室发布中文RoBERTa-wwm-ext预训练模型](https://cogskl.iflytek.com/archives/924)<br>
[RoBERTa中文预训练模型：RoBERTa for Chinese](https://mp.weixin.qq.com/s/K2zLEbWzDGtyOj7yceRdFQ)





https://github.com/ymcui/Chinese-BERT-wwm

NLP（二十三）序列标注算法评估模块seqeval的使用
https://www.cnblogs.com/jclian91/p/12913042.html

用BERT做NER？教你用PyTorch轻松入门Roberta！
https://zhuanlan.zhihu.com/p/346828049
https://github.com/hemingkx/CLUENER2020


一文看懂如何使用 Hydra 框架高效地跑各种超参数配置的深度学习实验
https://zhuanlan.zhihu.com/p/662221581?share_code=1blPCxRD9j5Z4&utm_psn=1905046521625425673

[CLUENER2020 官方Baseline](https://github.com/lemonhu/NER-BERT-pytorch)<br>
[Chinese NER Project](https://github.com/hemingkx/CLUENER2020)<br>