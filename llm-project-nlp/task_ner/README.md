## 一、任务与数据集概述

### 1.1 任务北京

命名实体识别（NER）是自然语言处理中的基础任务之一，其目标是从非结构化文本中识别出具有特定意义的实体，并将其分类到预定义的类别中，如人员、组织、位置等。
NER广泛应用于信息抽取、问答系统、搜索引擎、智能客服等场景。本项目聚焦于中文细粒度命名实体识别任务，在 CLUENER2020 数据集上进行模型训练与评估。

### 1.2 数据集概要

目前可公开访问获得的、高质量、细粒度的中文NER数据集较少，CLUE基于清华大学开源的文本分类数据集THUCNEWS，选出部分数据进行细粒度命名实体标注，并对数据进行清洗，得到一个细粒度的NER数据集。
CLUENER2020共有10个不同的类别，包括：

- 地址（address）: 描述地理位置，包括省、市、区、街道和门牌号等详细信息，确保定位准确到最小单位。
- 书名（book）: 包括各类纸质或电子形式的读物，如小说、杂志、教科书、习题集、地图册、食谱等，涵盖书店可购得的所有书籍类型。
- 公司（company）: 指商业实体，包括有限公司、集团公司以及除中国人民银行外的银行机构，例如新东方教育科技集团。
- 游戏（game）: 涵盖市面上所有形式的游戏作品，注意从小说或电视剧改编的游戏需要具体分析确认其是否为游戏类别。
- 政府（government）: 由中央行政机关（国务院及其组成部门）和地方行政机关构成，包含军队等国家管理机构，但不包括央行和中国人民银行。
- 电影（movie）: 主要指影视作品，对于根据书籍改编的作品需通过上下文区分是原作名称还是电影名称。
- 姓名（name）: 涵盖真实及虚构人物的名字或绰号，可以通过别名识别特定人物，如“及时雨”宋江。
- 组织机构（organization）: 包含体育队伍（篮球队、足球队）、艺术团体（乐团）、社团组织，还包括文学作品中的帮派组织如少林寺、丐帮。
- 职位（position）: 古今职务职称的总称，从古代的巡抚、知州到现代的总经理、记者等职业头衔均属于此类。
- 景点（scene）: 指著名的旅游地点，包括公园、动物园、自然景观（如黄河、长江）等适合观光游览的地方。

数据样例：
```
{
	"text": "叶老桂认为，对目前国内商业银行而言，",
	"label": {
		"name": {"叶老桂": [[0, 2]]},
		"company": {"浙商银行": [[11, 14]]}
	}
}
```
该数据集平均句子长度37.4字，最长50字，具体数量如下：
+ 训练集：10748 条
+ 验证集：1343 条（作为测试集使用）
+ 测试集：1345 条（未公开）

详情参考 [CLUENER 细粒度命名实体识别](https://github.com/CLUEbenchmark/CLUENER2020)<br>

## 二、模型设计和实验结果

本项目尝试了多种主流 NER 模型架构，包括传统深度学习模型（[命名实体识别-01-Bert系列.md](../docs/命名实体识别-01-Bert系列.md)）和大语言模型微调方案（[命名实体识别-02-大模型.md](../docs/命名实体识别-02-大模型.md)）。

| 模型名称                           | 技术架构                    | 特点说明      |
|--------------------------------|-------------------------|-----------|
| Bert + CRF                     | BERT 编码器 + 条件随机场解码      | 基线模型，性能一般 |
| Bert + BiLSTM + CRF            | BERT + 双向 LSTM + CRF    | 提升上下文建模能力 |
| Bert-WWM-ext + BiLSTM + CRF    | 中文优化版 BERT + 序列建模 + CRF | 性能较优      |
| Roberta-WWM-ext + BiLSTM + CRF | 更强的语言理解能力               | 排名靠前      |
| Qwen2.5-7B-Instruct + Lora     | 大语言模型 + 参数高效微调          | 最佳表现模型    |

整体性能对比（Micro Average F1-score）

| 模型                             | address | book  | company | game  | govern | movie | name  | org   | position | scene | micro avg |
|--------------------------------|---------|-------|---------|-------|--------|-------|-------|-------|----------|-------|-----------|
| Bert + CRF                     | 0.61    | 0.71  | 0.76    | 0.74  | 0.78   | 0.74  | 0.72  | 0.68  | 0.74     | 0.69  | 0.72      |
| Bert + BiLSTM + CRF            | 0.60    | 0.77  | 0.79    | 0.80  | 0.78   | 0.81  | 0.80  | 0.77  | 0.81     | 0.64  | 0.76      |
| Bert-WWM-ext + BiLSTM + CRF    | 0.63    | 0.73  | 0.77    | 0.80  | 0.79   | 0.80  | 0.85  | 0.80  | 0.78     | 0.68  | 0.77      |
| Roberta-WWM-ext + BiLSTM + CRF | 0.61    | 0.81  | 0.80    | 0.81  | 0.82   | 0.78  | 0.86  | 0.79  | 0.79     | 0.68  | 0.78      |
| Qwen2.5-7B-Instruct + LoRA     | 0.612   | 0.816 | 0.819   | 0.881 | 0.845  | 0.845 | 0.888 | 0.794 | 0.799    | 0.666 | 0.798     |

> Bert：bert-base-chinese <br>
> Bert-WWM-ext：chinese-bert-wwm-ext <br>
> Roberta-WWM-ext：chinese-roberta-wwm-ext <br>

### 2.1 结果明细
+ Bert+Crf
```
              precision    recall  f1-score   support

     address       0.64      0.59      0.61       373
        book       0.66      0.77      0.71       154
     company       0.73      0.80      0.76       378
        game       0.73      0.74      0.74       295
  government       0.72      0.85      0.78       247
       movie       0.70      0.80      0.74       151
        name       0.61      0.88      0.72       465
organization       0.68      0.68      0.68       367
    position       0.74      0.75      0.74       433
       scene       0.68      0.69      0.69       209

   micro avg       0.68      0.76      0.72      3072
   macro avg       0.69      0.76      0.72      3072
weighted avg       0.69      0.76      0.72      3072
```

+ Bert+BiLstm+Crf
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

+ Bert-wwm-ext+BiLstm+Crf
```
              precision    recall  f1-score   support

     address       0.62      0.65      0.63       373
        book       0.69      0.78      0.73       154
     company       0.76      0.78      0.77       378
        game       0.75      0.84      0.80       295
  government       0.75      0.83      0.79       247
       movie       0.81      0.79      0.80       151
        name       0.80      0.90      0.85       465
organization       0.81      0.79      0.80       367
    position       0.76      0.79      0.78       433
       scene       0.66      0.70      0.68       209

   micro avg       0.74      0.79      0.77      3072
   macro avg       0.74      0.79      0.76      3072
weighted avg       0.74      0.79      0.77      3072
```
+ Roberta-wwm-ext+BiLstm+Crf
```
              precision    recall  f1-score   support
     address       0.56      0.67      0.61       373
        book       0.86      0.77      0.81       154
     company       0.79      0.82      0.80       378
        game       0.76      0.87      0.81       295
  government       0.78      0.87      0.82       247
       movie       0.81      0.74      0.78       151
        name       0.84      0.89      0.86       465
organization       0.76      0.82      0.79       367
    position       0.76      0.82      0.79       433
       scene       0.68      0.67      0.68       209

   micro avg       0.75      0.80      0.78      3072
   macro avg       0.76      0.79      0.77      3072
weighted avg       0.75      0.80      0.78      3072
```

+ Qwen2.5-7B-Instruct + Lora
```
       Label Precision  Recall     F1
     address    0.6104  0.6154 0.6129
        name    0.8686  0.9091 0.8884
organization     0.814  0.7762 0.7946
        game    0.8484  0.9164 0.8811
       scene    0.6919  0.6432 0.6667
        book    0.8299  0.8026 0.8161
    position    0.8127  0.7859 0.7990
     company    0.8255  0.8142 0.8198
  government    0.8168   0.877 0.8458
       movie    0.8723    0.82 0.8454
  Macro Avg.         -       - 0.7970
  Micro Avg.    0.7986  0.7991 0.7988
```

## 参考引用
[1] [CLUENER2020：中文细粒度命名实体识别数据集来了](https://zhuanlan.zhihu.com/p/103034432)<br>
[2] [CLUENER 细粒度命名实体识别-Github](https://github.com/CLUEbenchmark/CLUENER2020)<br>
[3] [CLUENER2020 官方排行榜](https://www.cluebenchmarks.com/ner.html)<br>
[4] [CLUENER2020 官方Baseline](https://github.com/lemonhu/NER-BERT-pytorch)<br>
[5] [Chinese NER Project](https://github.com/hemingkx/CLUENER2020)<br>
