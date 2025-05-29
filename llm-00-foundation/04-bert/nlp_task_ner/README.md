
利用大模型做NER实践(总结版)
https://mp.weixin.qq.com/s/LBlzFm8wxK7Aj7YXhCoXgQ
https://github.com/cjymz886/LLM-NER

指令微调-命名实体识别
博客：https://blog.csdn.net/SoulmateY/article/details/139831606
代码：https://github.com/Zeyi-Lin/LLM-Finetune

Qwen2.5大模型微调实战：医疗命名实体识别任务（完整代码）
https://zhuanlan.zhihu.com/p/19682001982

基于 Qwen2.5-0.5B 微调训练 Ner 命名实体识别任务
https://blog.csdn.net/qq_43692950/article/details/142631780

qwen3 finetune
https://qwen.readthedocs.io/zh-cn/latest/training/llama_factory.html

RoBERTa中文预训练模型：RoBERTa for Chinese
https://mp.weixin.qq.com/s/K2zLEbWzDGtyOj7yceRdFQ
https://github.com/ymcui/Chinese-BERT-wwm

NLP（二十三）序列标注算法评估模块seqeval的使用
https://www.cnblogs.com/jclian91/p/12913042.html

用BERT做NER？教你用PyTorch轻松入门Roberta！
https://zhuanlan.zhihu.com/p/346828049
https://github.com/hemingkx/CLUENER2020

CLUE官方baseline
https://github.com/CLUEbenchmark/CLUENER2020/tree/master

CLUENER2020官方的排行榜：[传送门](https://www.cluebenchmarks.com/ner.html)。
[CLUENER2020任务详情](https://github.com/CLUEbenchmark/CLUENER2020)

https://github.com/lemonhu/NER-BERT-pytorch

本数据是在清华大学开源的文本分类数据集THUCTC基础上，选出部分数据进行细粒度命名实体标注，原数据来源于Sina News RSS.
任务详情：CLUENER2020
训练集：10748 验证集：1343
标签类别：
数据分为10个标签类别，分别为: 地址（address），书名（book），公司（company），游戏（game），政府（goverment），电影（movie），姓名（name），组织机构（organization），职位（position），景点（scene）

例子：
{"text": "浙商银行企业信贷部叶老桂博士则从另一个角度对五道门槛进行了解读。叶老桂认为，对目前国内商业银行而言，", "label": {"name": {"叶老桂": [[9, 11]]}, "company": {"浙商银行": [[0, 3]]}}}
{"text": "生生不息CSOL生化狂潮让你填弹狂扫", "label": {"game": {"CSOL": [[4, 7]]}}}

标签定义与规则：
地址（address）: **省**市**区**街**号，**路，**街道，**村等（如单独出现也标记），注意：地址需要标记完全, 标记到最细。
书名（book）: 小说，杂志，习题集，教科书，教辅，地图册，食谱，书店里能买到的一类书籍，包含电子书。
公司（company）: **公司，**集团，**银行（央行，中国人民银行除外，二者属于政府机构）, 如：新东方，包含新华网/中国军网等。
游戏（game）: 常见的游戏，注意有一些从小说，电视剧改编的游戏，要分析具体场景到底是不是游戏。
政府（goverment）: 包括中央行政机关和地方行政机关两级。 中央行政机关有国务院、国务院组成部门（包括各部、委员会、中国人民银行和审计署）、国务院直属机构（如海关、税务、工商、环保总局等），军队等。
电影（movie）: 电影，也包括拍的一些在电影院上映的纪录片，如果是根据书名改编成电影，要根据场景上下文着重区分下是电影名字还是书名。
姓名（name）: 一般指人名，也包括小说里面的人物，宋江，武松，郭靖，小说里面的人物绰号：及时雨，花和尚，著名人物的别称，通过这个别称能对应到某个具体人物。
组织机构（organization）: 篮球队，足球队，乐团，社团等，另外包含小说里面的帮派如：少林寺，丐帮，铁掌帮，武当，峨眉等。
职位（position）: 古时候的职称：巡抚，知州，国师等。现代的总经理，记者，总裁，艺术家，收藏家等。
景点（scene）: 常见旅游景点如：长沙公园，深圳动物园，海洋馆，植物园，黄河，长江等。

【训练集】标签数据分布如下：
地址（address）:2829
书名（book）:1131
公司（company）:2897
游戏（game）:2325
政府（government）:1797
电影（movie）:1109
姓名（name）:3661
组织机构（organization）:3075
职位（position）:3052
景点（scene）:1462

【验证集】标签数据分布如下：
地址（address）:364
书名（book）:152
公司（company）:366
游戏（game）:287
政府（government）:244
电影（movie）:150
姓名（name）:451
组织机构（organization）:344
职位（position）:425
景点（scene）:199

+ Bert+Crf

| class          | precision | recall | f1-score | support |
|----------------|-----------|--------|----------|---------|
| address        | 0.58      | 0.61   | 0.59     | 373     |
| book           | 0.70      | 0.81   | 0.75     | 154     |
| company        | 0.75      | 0.80   | 0.77     | 378     |
| game           | 0.68      | 0.85   | 0.75     | 295     |
| government     | 0.74      | 0.85   | 0.79     | 247     |
| movie          | 0.71      | 0.84   | 0.77     | 151     |
| name           | 0.65      | 0.88   | 0.75     | 465     |
| organization   | 0.65      | 0.79   | 0.71     | 367     |
| position       | 0.71      | 0.79   | 0.75     | 433     |
| scene          | 0.60      | 0.70   | 0.64     | 209     |

+ Bert+BiLstm+Crf

| class          | precision | recall | f1-score | support |
|----------------|-----------|--------|----------|---------|
| address        | 0.60      | 0.62   | 0.61     | 373     |
| book           | 0.74      | 0.81   | 0.77     | 154     |
| company        | 0.76      | 0.80   | 0.78     | 378     |
| game           | 0.77      | 0.85   | 0.81     | 295     |
| government     | 0.76      | 0.83   | 0.80     | 247     |
| movie          | 0.83      | 0.80   | 0.81     | 151     |
| name           | 0.83      | 0.87   | 0.85     | 465     |
| organization   | 0.76      | 0.78   | 0.77     | 367     |
| position       | 0.78      | 0.80   | 0.79     | 433     |
| scene          | 0.69      | 0.71   | 0.70     | 209     |

+ Bert-wwm-ext+BiLstm+Crf

|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| address      | 0.59      | 0.64   | 0.62     | 373     |
| book         | 0.60      | 0.78   | 0.68     | 154     |
| company      | 0.76      | 0.79   | 0.77     | 378     |
| game         | 0.67      | 0.87   | 0.75     | 295     |
| government   | 0.73      | 0.83   | 0.78     | 247     |
| movie        | 0.70      | 0.79   | 0.74     | 151     |
| name         | 0.77      | 0.89   | 0.83     | 465     |
| organization | 0.74      | 0.78   | 0.76     | 367     |
| position     | 0.76      | 0.78   | 0.77     | 433     |
| scene        | 0.67      | 0.66   | 0.67     | 209     |

+ Roberta-wwm-ext+BiLstm+Crf

|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| address      | 0.62      | 0.66   | 0.64     | 373     |
| book         | 0.78      | 0.76   | 0.77     | 154     |
| company      | 0.79      | 0.81   | 0.80     | 378     |
| game         | 0.77      | 0.84   | 0.81     | 295     |
| government   | 0.80      | 0.86   | 0.83     | 247     |
| movie        | 0.80      | 0.77   | 0.79     | 151     |
| name         | 0.85      | 0.89   | 0.87     | 465     |
| organization | 0.76      | 0.80   | 0.78     | 367     |
| position     | 0.76      | 0.80   | 0.78     | 433     |
| scene        | 0.70      | 0.65   | 0.67     | 209     |

+ 汇总（F1-score）

| 模型                         | address | book | company | game | govern | movie | name | org  | position | scene | micro avg |
|----------------------------|:-------:|:----:|:-------:|:----:|:------:|:-----:|:----:|:----:|:--------:|:-----:|:---------:| 
| Bert+Crf                   |  0.59   | 0.75 |  0.77   | 0.75 |  0.77  | 0.76  | 0.75 | 0.71 |   0.74   | 0.65  |   0.72    |
| Bert+BiLstm+Crf            |  0.62   | 0.77 |  0.78   | 0.81 |  0.79  | 0.82  | 0.84 | 0.77 |   0.79   | 0.71  |   0.77    |
| Bert-wwm-ext+BiLstm+Crf    |  0.62   | 0.68 |  0.77   | 0.75 |  0.77  | 0.74  | 0.82 | 0.76 |   0.77   | 0.68  |   0.74    |
| Roberta-wwm-ext+BiLstm+Crf |  0.64   | 0.77 |  0.80   | 0.81 |  0.83  | 0.78  | 0.87 | 0.78 |   0.79   | 0.67  |   0.78    |

Bert：bert-base-chinese
Bert-wwm-ext：chinese-bert-wwm-ext
Roberta-wwm-ext：chinese-roberta-wwm-ext
