
机器翻译（machine translation）指的是 将序列从一种语言自动翻译成另一种语言。 事实上，这个研究领域可以追溯到数字计算机发明后不久的20世纪40年代， 特别是在第二次世界大战中使用计算机破解语言编码。 几十年来，在使用神经网络进行端到端学习的兴起之前， 统计学方法在这一领域一直占据主导地位 (Brown et al., 1990, Brown et al., 1988)。 因为统计机器翻译（statistical machine translation）涉及了 翻译模型和语言模型等组成部分的统计分析， 因此基于神经网络的方法通常被称为 `神经机器翻译`（neural machine translation）， 用于将两种翻译模型区分开来。

+ 任务说明：<br>
（1）基于序列到序列（Seq2Seq）学习框架，设计并训练一个中英文机器翻译模型，完成中译英和英译中翻译任务。具体模型选择可以参考如 LSTM，GRU，Transformer 等；<br>
（2）根据指定的评价指标和测试集数据评价模型性能。

+ 评价指标：
（1）BLEU1/2/3：BLEU（Bilingual Evaluation Understudy）计算同时出现在系统译文和参考译文中的 n 元词重叠程度，实验要求计算 n=1/2/3。
（2）Perplexity：Perplexity 是衡量语言模型生成句子能力的评价指标

+ 数据说明<br>
实验数据集为 WMT18 新闻评论数据集 [News Commentary v13](http://data.statmt.org/wmt18/translation-task/training-parallel-nc-v13.tgz)<br>
包含中英文平行语料两个文件： `news-commentary-v13.zh-en.en`、`news-commentary-v13.zh-en.zh`<br>
整个语料库分训练集（约 176943 条）、验证集（约25278条）和测试集（约 50556 条）三部分，即按7:1:2的比例进行切分；<br>

+ 数据示例<br>
```

```




[1] [动手深度学习v2-机器翻译与数据集](https://zh.d2l.ai/chapter_recurrent-modern/machine-translation-and-dataset.html)<br>








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



### 问题记录
Windows 报错
```
Traceback (most recent call last):
  File "E:\PycharmProjects\llm-learning\llm-00-foundation\03-transformer\translation\dataset.py", line 9, in <module>
    from torchtext.data import get_tokenizer
  File "E:\PythonEnvs\llm-learning\lib\site-packages\torchtext\__init__.py", line 18, in <module>
    from torchtext import _extension  # noqa: F401
  File "E:\PythonEnvs\llm-learning\lib\site-packages\torchtext\_extension.py", line 64, in <module>
    _init_extension()
  File "E:\PythonEnvs\llm-learning\lib\site-packages\torchtext\_extension.py", line 58, in _init_extension
    _load_lib("libtorchtext")
  File "E:\PythonEnvs\llm-learning\lib\site-packages\torchtext\_extension.py", line 50, in _load_lib
    torch.ops.load_library(path)
  File "E:\PythonEnvs\llm-learning\lib\site-packages\torch\_ops.py", line 1295, in load_library
    ctypes.CDLL(path)
  File "E:\PythonEnvs\llm-learning\lib\ctypes\__init__.py", line 374, in __init__
    self._handle = _dlopen(self._name, mode)
OSError: [WinError 127] 找不到指定的程序。
```
原因：torch与torchtext版本的兼容问题
torch2.4.1版本过高，降至2.3.0，最终问题解决。
https://blog.csdn.net/bcxbdzh/article/details/144276080


