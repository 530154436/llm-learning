
## 一、预训练模型

BERT（Bidirectional Encoder Representations from Transformers） 是 Google 提出的一种基于 Transformer 的预训练语言模型，具有以下特点：
+ 双向上下文建模能力
+ 使用 MLM（Masked Language Model）和 NSP（Next Sentence Prediction）任务进行预训练
+ 可以通过微调迁移到各种下游任务中，包括 NER、文本分类、问答系统等

在本项目中，我们使用了多个中文 BERT 预训练版本，包括：

| 版本                      | 适用场景                     | 特点                                                  | 
|-------------------------|--------------------------|-----------------------------------------------------|
| bert-base-uncased       | 大小写不敏感的任务，如一般的文本分类任务     | 所有输入文本都会被转换为小写，减少了词汇表的大小                            |
| bert-base-cased         | 需要区分大小写的任务，例如命名实体识别(NER) | 保持了文本中的大小写信息，有助于捕捉更多文本结构细节                          |
| chinese-bert-wwm-ext    | 中文命名实体识别、文本分类、句法分析等      | 通过全词掩码（wwm，Whole Word Masking）技术增强了对中文语境的理解         |
| chinese-roberta-wwm-ext | 中文文本分类、命名实体识别、问答系统等      | 预训练阶段采用wwm策略进行mask，取消了NSP（Next Sentence Prediction） |

注意：chinese-roberta-wwm-ext 模型在使用上与中文BERT系列模型完全一致，无需任何代码调整即可使用。

## 二、整体流程
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
+ attention_mask： 用于指示哪些位置是真实 token（1），哪些是 padding（0）。

**示例2 BERT 输入格式转换**：<br>
```
    tokens: 		[CLS] 北   京   城  [SEP]
    input_ids: 		101 1266 776 1814  102  0  0  0  0  0
    attention_mask: 1    1    1    1    1   0  0  0  0  0
    token_type_ids: 0    0    0    0    0   0  0  0  0  0
    label_ids: 		0    1    2    2    0   0  0  0  0  0
```

### 2.2 模型训练
#### 2.2.1 模型结构（以BertBiLstmCRF为例）
Torch 实现:
```python
class BertBiLstmCrf(BaseNerModel):

    def __init__(self, pretrain_path: str, num_labels: int, dropout: float = 0.3,
                 lstm_num_layers: int = 2, lstm_hidden_size: int = 256):

        super().__init__(pretrain_path, num_labels)
        self.bert_config = BertConfig.from_pretrained(pretrain_path)
        self.bert = BertModel.from_pretrained(pretrain_path)
        self.bilstm = nn.LSTM(input_size=self.bert_config.hidden_size,
                              bidirectional=True,
                              num_layers=lstm_num_layers,
                              hidden_size=lstm_hidden_size // 2,
                              batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(lstm_hidden_size, num_labels)
        self.crf = CRF(num_tags=num_labels, batch_first=True)

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                token_type_ids: torch.Tensor) -> torch.FloatTensor:
        """
        正向传播
        :param input_ids: [batch_size, seq_len]
        :param attention_mask: [batch_size, seq_len]
        :param token_type_ids: [batch_size, seq_len]

        :return: [batch_size, seq_len, num_labels]
        """
        # [batch_size, seq_len] => [batch_size, seq_len, embedding_size]
        output = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]

        # [batch_size, seq_len, lstm_hidden_size]
        lstm_out, _ = self.bilstm(output)
        lstm_out = self.dropout(lstm_out)

        # [batch_size, seq_len, num_labels]
        logits = self.linear(lstm_out)

        return logits
```
各模块明细： 
 
| 模块      | 子模块/组件     | 输入 Shape    | 输出 Shape    | 关键参数与功能                                                           |
|---------|------------|-------------|-------------|-------------------------------------------------------------------|
| BERT    | embeddings | (B, L)      | (B, L, 768) | - 词嵌入：21128×768<br>- 位置嵌入：512×768<br>- 分段嵌入：2×768<br>- 归一化后输出统一维度 |
|         | encoder    | (B, L, 768) | (B, L, 768) | - 共12层，每层含自注意力（768→768）<br>- 前馈网络扩展至 3072 维后降回 768<br>- 残差连接+层归一化 |
|         | pooler     | (B, 768)    | (B, 768)    | - 线性层+Tanh 激活，生成句子级表示（可能未直接使用）                                    |
| BiLSTM  | bilstm     | (B, L, 768) | (B, L, 128) | - 双向LSTM，隐藏层 64 维，双向拼接为 128<br>- batch_first=True 适配BERT输出        |
| Dropout | dropout    | (B, L, 128) | (B, L, 128) | - 丢弃概率 p=0.3，防止过拟合                                                |
| Linear  | linear     | (B, L, 128) | (B, L, 31)  | - 投影到标签空间：128→31（如NER的31类标签）                                      |
| CRF     | crf        | (B, L, 31)  | (B, L)      | - 条件随机场建模标签转移概率<br>- 输出最优标签序列（非概率分布）                              |

#### 2.2.2 损失函数
使用`负对数似然损失`（Negative Log Likelihood Loss），由 CRF 层自动计算。
```
loss = -self.crf(logits, labels, mask=attention_mask.bool(), reduction='mean')
```

#### 2.2.3 优化器与学习率调度
使用 AdamW 优化器和线性预热（warmup）和余弦/线性衰减的学习率调度器。
```python
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

optimizer = AdamW(params, lr=learning_rate)
train_steps_per_epoch = train_size // config.batch_size
scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                            num_warmup_steps=(config.epoch_num // 10) * train_steps_per_epoch,
                                            num_training_steps=config.epoch_num * train_steps_per_epoch)
```

### 2.3 模型评估
常见的评估指标包括 准确率（Accuracy） 和 F1 分数（F1 Score），尤其在类别不平衡的情况下，F1 分数更具参考价值。
使用 torchmetrics 构建评估指标集合，方便在训练、验证过程中高效计算多个指标。
```python
from torchmetrics import F1Score, MetricCollection, Accuracy

metrics = MetricCollection({
    'acc': Accuracy(task="multiclass", num_classes=num_labels),
    'f1': F1Score(task="multiclass", num_classes=num_labels)
})
```

## 三、实验结果

训练日志
```
run_bert_bilstm_crf.py[:42] 开始训练模型
run_bert_bilstm_crf.py[:43] 配置信息:
data_dir: ./data
train_data_path: ./data/dataset/clue/train.jsonl
dev_data_path: ./data/dataset/clue/dev.jsonl
test_data_path: ./data/dataset/clue/test.jsonl
label_data_path: ./data/dataset/clue/label.json
model_name: BertBiLstmCrf_chinese-roberta-wwm-ext
model_path: ./data/outputs/BertBiLstmCrf_chinese-roberta-wwm-ext.pth
device: cuda:0
batch_size: 64
dropout: 0.3
epoch_num: 50
learning_rate: 3.0e-05
pretrain_path: ../model_hub/chinese-roberta-wwm-ext
num_labels: 31
lstm_num_layers: 1
lstm_hidden_size: 128
clip_grad: 5.0
run_bert_bilstm_crf.py[:44] 加载Dataset和Tokenizer.
run_bert_bilstm_crf.py[:55] 加载DataLoader
run_bert_bilstm_crf.py[:60] 初始化模型
run_bert_bilstm_crf.py[:63] 模型训练参数: 102699678
run_bert_bilstm_crf.py[:66] 配置优化器、学习率调整器、损失函数、评估指标
run_bert_bilstm_crf.py[:78] 训练模型...
my_trainer.py[:153] epoch: 1, Current lr : 0.0, train_loss: 104.08669, val_loss: 55.19459, val_acc: 0.80742, val_f1: 0.80742
my_trainer.py[:153] epoch: 2, Current lr : 6e-06, train_loss: 43.44882, val_loss: 30.66795, val_acc: 0.88152, val_f1: 0.88152
my_trainer.py[:153] epoch: 3, Current lr : 1.2e-05, train_loss: 25.88252, val_loss: 19.61042, val_acc: 0.9322, val_f1: 0.9322
my_trainer.py[:153] epoch: 4, Current lr : 1.8e-05, train_loss: 17.6796, val_loss: 15.44608, val_acc: 0.9485, val_f1: 0.9485
my_trainer.py[:153] epoch: 5, Current lr : 2.4e-05, train_loss: 13.20995, val_loss: 13.15177, val_acc: 0.95016, val_f1: 0.95016
my_trainer.py[:153] epoch: 6, Current lr : 3e-05, train_loss: 10.24443, val_loss: 12.30117, val_acc: 0.94918, val_f1: 0.94918
my_trainer.py[:153] epoch: 7, Current lr : 2.9e-05, train_loss: 8.17817, val_loss: 11.69731, val_acc: 0.95208, val_f1: 0.95208
my_trainer.py[:153] epoch: 8, Current lr : 2.9e-05, train_loss: 6.79995, val_loss: 10.98749, val_acc: 0.95036, val_f1: 0.95036
my_trainer.py[:153] epoch: 9, Current lr : 2.8e-05, train_loss: 5.683, val_loss: 10.85063, val_acc: 0.95385, val_f1: 0.95385
my_trainer.py[:153] epoch: 10, Current lr : 2.7e-05, train_loss: 4.8795, val_loss: 11.26778, val_acc: 0.95154, val_f1: 0.95154
my_trainer.py[:153] epoch: 11, Current lr : 2.7e-05, train_loss: 4.16538, val_loss: 11.37653, val_acc: 0.95152, val_f1: 0.95152
my_trainer.py[:153] epoch: 12, Current lr : 2.6e-05, train_loss: 3.6833, val_loss: 11.83993, val_acc: 0.95148, val_f1: 0.95148
my_trainer.py[:153] epoch: 13, Current lr : 2.5e-05, train_loss: 3.18895, val_loss: 11.87034, val_acc: 0.95159, val_f1: 0.95159
my_trainer.py[:153] epoch: 14, Current lr : 2.5e-05, train_loss: 2.87335, val_loss: 11.73891, val_acc: 0.95247, val_f1: 0.95247
my_trainer.py[:153] epoch: 15, Current lr : 2.4e-05, train_loss: 2.57749, val_loss: 11.99193, val_acc: 0.95206, val_f1: 0.95206
my_trainer.py[:153] epoch: 16, Current lr : 2.3e-05, train_loss: 2.32171, val_loss: 12.20675, val_acc: 0.95113, val_f1: 0.95113
my_trainer.py[:153] epoch: 17, Current lr : 2.3e-05, train_loss: 2.08853, val_loss: 12.93478, val_acc: 0.94898, val_f1: 0.94898
my_trainer.py[:153] epoch: 18, Current lr : 2.2e-05, train_loss: 1.87674, val_loss: 12.79549, val_acc: 0.952, val_f1: 0.952
my_trainer.py[:153] epoch: 19, Current lr : 2.1e-05, train_loss: 1.74081, val_loss: 12.8339, val_acc: 0.95143, val_f1: 0.95143
my_trainer.py[:159] Current loss: 12.833900, Best val_loss: 10.850630
my_trainer.py[:162] Saved model's loss: 10.850630
```
+ Bert+BiLstm+Crf在CLUENER2020的评测结果（验证集）
```
              precision    recall  f1-score   support

     address       0.56      0.65      0.60       373
        book       0.76      0.77      0.77       154
     company       0.75      0.83      0.79       378
        game       0.77      0.84      0.80       295
  government       0.76      0.80      0.78       247
       movie       0.83      0.79      0.81       151
        name       0.73      0.89      0.80       465
organization       0.74      0.80      0.77       367
    position       0.81      0.81      0.81       433
       scene       0.65      0.63      0.64       209

   micro avg       0.73      0.79      0.76      3072
   macro avg       0.74      0.78      0.76      3072
weighted avg       0.73      0.79      0.76      3072
```

## 参考引用

[1] [Chinese NER Project](https://github.com/hemingkx/CLUENER2020)<br>
[2] [CLUENER2020 官方Baseline](https://github.com/lemonhu/NER-BERT-pytorch)<br>
[3] [Chinese-BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm)<br>
[4] [RoBERTa中文预训练模型：RoBERTa for Chinese](https://mp.weixin.qq.com/s/K2zLEbWzDGtyOj7yceRdFQ)<br>
[5] [NLP（二十三）序列标注算法评估模块seqeval的使用](https://www.cnblogs.com/jclian91/p/12913042.html)<br>
[6] [哈工大讯飞联合实验室发布中文RoBERTa-wwm-ext预训练模型](https://cogskl.iflytek.com/archives/924)<br>
[7] [用BERT做NER？教你用PyTorch轻松入门Roberta！](https://zhuanlan.zhihu.com/p/346828049)<br>
[8] [一文看懂如何使用 Hydra 框架高效地跑各种超参数配置的深度学习实验](https://zhuanlan.zhihu.com/p/662221581?share_code=1blPCxRD9j5Z4&utm_psn=1905046521625425673)<br>



