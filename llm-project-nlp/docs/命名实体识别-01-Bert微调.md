
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
### 2.1 数据处理
#### 2.1.1 原始数据格式
```
{
	"text": "北京城.",
	"label": {
		"NT": {"北京城": [[0, 2]]}
	}
}
```
其中：<br>
"text" 表示原始文本；<br>
"label" 是一个嵌套字典，表示实体类型（如 NT）及其对应的实体名称和出现的位置索引。<br>

#### 2.1.2 BIOS 标注格式转换
将原始标注信息转换为逐 token 的 BIOS 格式标签，BIOS 是命名实体识别任务中常用的标注方式：
```
北 B-NT
京 I-NT
城 I-NT
.  O
```
说明：<br>
B-XX：表示某个实体的开始；<br>
I-XX：表示某个实体的中间或结尾；<br>
O：非实体。<br>
S-XX：表示单独成实体的 token（适用于单字实体）。<br>

#### 2.1.3 数据加载和批处理（DataLoader.collate_fn）
在构建 DataLoader 时使用自定义 collate_fn 对 batch 数据进行处理，从而满足 Bert 的输入格式，主要包括以下步骤：<br>

1、截断（Truncation）：如果输入长度超过模型最大限制（如512），则截断至允许的最大长度（需减去 [CLS] 和 [SEP] 所占位置）。<br>
2、token 转 id：使用 tokenizer 将 token 转换为对应 token_id。
3、添加特殊标记（Special Tokens）：在序列前后分别加上 [CLS] 和 [SEP] 标记。<br>
4、padding：填充到固定长度 min_max_seq_length。
5、生成 attention_mask 和 token_type_ids。<br>

**示例1 BERT 输入格式转换**：<br>
```
(a) 对于句子对（sequence pairs）：
    tokens:          [CLS] 是 这 jack ##son ##ville ? [SEP] 不 是 的 . [SEP]
    token_type_ids:    0   0  0   0    0     0     0    0   1  1  1  1  1
(b) 对于单个句子（single sequences）：
    tokens:          [CLS] 这只 狗 很 茸毛 . [SEP]
    token_type_ids :   0   0   0  0   0  0   0
```
+ [CLS]：每个序列开头都会加入这个特殊 token，用于表示整个句子的聚合信息，常用于分类任务。
+ [SEP]：用于分隔两个句子或标记一个句子的结束。
+ token_type_ids （segment_ids）：用于区分句子对中的不同句子。第1个句子的所有 token 标记为 0；第2个句子的所有 token 标记为 1；单独使用时全部为 0。
+ attention_mask （segment_ids）： 用于指示哪些位置是真实 token（1），哪些是 padding（0）。

**示例2 BERT 输入格式转换**：<br>
```
    tokens: 		[CLS] 北   京   城  [SEP]
    input_ids: 		101 1266 776 1814  102  0  0  0  0  0
    attention_mask: 1    1    1    1    1   0  0  0  0  0
    token_type_ids: 0    0    0    0    0   0  0  0  0  0
    label_ids: 		0    1    2    2    0   0  0  0  0  0
```

### 2.2 训练模型
#### 2.2.1 模型定义
BertBiLstmCrf(
  (bert): BertModel(
    (embeddings): BertEmbeddings(
      (word_embeddings): Embedding(21128, 768, padding_idx=0)
      (position_embeddings): Embedding(512, 768)
      (token_type_embeddings): Embedding(2, 768)
      (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
    (encoder): BertEncoder(
      (layer): ModuleList(
        (0-11): 12 x BertLayer(
          (attention): BertAttention(
            (self): BertSdpaSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (output): BertSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (intermediate): BertIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
            (intermediate_act_fn): GELUActivation()
          )
          (output): BertOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
    )
    (pooler): BertPooler(
      (dense): Linear(in_features=768, out_features=768, bias=True)
      (activation): Tanh()
    )
  )
  (bilstm): LSTM(768, 64, batch_first=True, bidirectional=True)
  (dropout): Dropout(p=0.3, inplace=False)
  (linear): Linear(in_features=128, out_features=31, bias=True)
  (crf): CRF(num_tags=31)
)
#### 2.2.2 损失函数
通常使用交叉熵损失函数来计算预测标签与真实标签之间的差异。忽略 padding 部分的损失计算。

#### 2.2.3 优化器与学习率调度
使用 AdamW 优化器。 使用线性预热（warmup）和余弦/线性衰减的学习率调度器。

#### 2.2.4


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