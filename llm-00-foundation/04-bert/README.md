
locust 性能测试

基于 Qwen2.5-0.5B 微调训练 Ner 命名实体识别任务
https://blog.csdn.net/qq_43692950/article/details/142631780

评估指标工具库
https://blog.csdn.net/mzpmzk/article/details/134424122

bert-base-uncased 和 bert-base-cased 是 BERT（Bidirectional Encoder Representations from Transformers）模型的两个不同版本，它们主要在处理输入文本时对大小写的处理方式上有所不同：


| 版本                    | 适用场景                          | 特点                         | 
|-----------------------|-------------------------------|----------------------------|
| **bert-base-uncased** | 大小写不敏感的任务，如一般的文本分类任务          | 所有输入文本都会被转换为小写，减少了词汇表的大小   |
| **bert-base-cased**   | 需要区分大小写的任务，例如命名实体识别(NER)或语法分析 | 保持了文本中的大小写信息，有助于捕捉更多文本结构细节 |


问题：
1、进入crf层时为什么要去除[CLS]标签，直接给成 O 标签不可以吗


SENTENCE-BERT
ModernBERT

一文看懂如何使用 Hydra 框架高效地跑各种超参数配置的深度学习实验
https://zhuanlan.zhihu.com/p/662221581?share_code=1blPCxRD9j5Z4&utm_psn=1905046521625425673

论文-中英对照
https://www.yiyibooks.cn/nlp/bert/main.html

Chinese NER Project
https://github.com/intro-llm/intro-llm-code/blob/main/chs/ch2-foundations/PreTrain/main.py
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