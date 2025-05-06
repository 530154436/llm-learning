

bert-base-uncased 和 bert-base-cased 是 BERT（Bidirectional Encoder Representations from Transformers）模型的两个不同版本，它们主要在处理输入文本时对大小写的处理方式上有所不同：


| 版本                    | 适用场景                          | 特点                         | 
|-----------------------|-------------------------------|----------------------------|
| **bert-base-uncased** | 大小写不敏感的任务，如一般的文本分类任务          | 所有输入文本都会被转换为小写，减少了词汇表的大小   |
| **bert-base-cased**   | 需要区分大小写的任务，例如命名实体识别(NER)或语法分析 | 保持了文本中的大小写信息，有助于捕捉更多文本结构细节 |

论文-中英对照
https://www.yiyibooks.cn/nlp/bert/main.html

Chinese NER Project
https://github.com/hemingkx/CLUENER2020

datawhalechina的nlp学习
https://github.com/datawhalechina/learn-nlp-with-transformers

BERT再学习与代码复现
https://github.com/songyingxin/BERT-pytorch

美团BERT的探索和实践
https://tech.meituan.com/2019/11/14/nlp-bert-practice.html

飞浆 预训练模型 » BERT
https://paddlepedia.readthedocs.io/en/latest/tutorials/pretrain_model/bert.html

BERT 模型详解
https://www.cnblogs.com/nickchen121/p/15114385.html#%E5%8D%81bert-%E6%A8%A1%E5%9E%8B

CS244n
https://blog.showmeai.tech/cs224n/