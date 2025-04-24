
机器翻译与数据集
https://zh.d2l.ai/chapter_recurrent-modern/machine-translation-and-dataset.html


语言模型是自然语言处理的关键， 而机器翻译是语言模型最成功的基准测试。 因为机器翻译正是将输入序列转换成输出序列的 序列转换模型（sequence transduction）的核心问题。 序列转换模型在各类现代人工智能应用中发挥着至关重要的作用， 因此我们将其做为本章剩余部分和 10节的重点。 为此，本节将介绍机器翻译问题及其后文需要使用的数据集。

机器翻译（machine translation）指的是 将序列从一种语言自动翻译成另一种语言。 事实上，这个研究领域可以追溯到数字计算机发明后不久的20世纪40年代， 特别是在第二次世界大战中使用计算机破解语言编码。 几十年来，在使用神经网络进行端到端学习的兴起之前， 统计学方法在这一领域一直占据主导地位 (Brown et al., 1990, Brown et al., 1988)。 因为统计机器翻译（statistical machine translation）涉及了 翻译模型和语言模型等组成部分的统计分析， 因此基于神经网络的方法通常被称为 神经机器翻译（neural machine translation）， 用于将两种翻译模型区分开来。


任务：
（1）基于序列到序列（Seq2Seq）学习框架，设计并训练一个中英文机器翻译模型，完成中译英和英译中翻译任务。具体模型选择可以参考如 LSTM，GRU，Transformer 等，但不做限制；
（2）实验数据集为 WMT18 新闻评论数据集 News Commentary v13，整个语料库分训练集（约 252,700 条）、验证集和测试集（分别约 2,000 条）三部分，每部分包含中英文平行语料两个文件；
（3）根据指定的评价指标和测试集数据评价模型性能。

评价指标：
（1）BLEU1/2/3：BLEU（Bilingual Evaluation Understudy）计算同时出现在系统译文和参考译文中的 n 元词重叠程度，实验要求计算 n=1/2/3。
（2）Perplexity：Perplexity 是衡量语言模型生成句子能力的评价指标

数据来源：
https://www.statmt.org/wmt18/translation-task.html


对比不同分词的影响
spacy
https://www.cnblogs.com/luohenyueji/p/17584672.html
```python
import spacy

# 加载已安装的中文模型
nlp = spacy.load('zh_core_web_sm')

# 执行一些简单的NLP任务
doc = nlp("早上好!")
for token in doc:
    # token.text表示标记的原始文本，token.pos_表示标记的词性（part-of-speech），token.dep_表示标记与其他标记之间的句法依存关系
    print(token.text, token.pos_, token.dep_)

import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple is looking at buying U.K. startup for $1 billion")
for token in doc:
    print(token.text)
```

Byte-BPE
HuggingFace ByteLevelBPETokenizer： from tokenizers import ByteLevelBPETokenizer
OpenAI GPT Tokenizer: https://github.com/openai/tiktoken

Bert-wordpiece
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
tokenizer.save_pretrained(model_path)

sentencepiece
https://github.com/google/sentencepiece