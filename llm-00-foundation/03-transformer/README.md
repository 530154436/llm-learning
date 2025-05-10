[transformer学习]
[李宏毅-自注意力机制 Self-attention（上）](https://www.bilibili.com/video/BV1Wv411h7kN?spm_id_from=333.788.videopod.episodes&vd_source=436107f586d66ab4fcf756c76eb96c35&p=38)
[李宏毅-Transformer（上）](https://www.bilibili.com/video/BV1Wv411h7kN/?p=49&share_source=copy_web&vd_source=e46571d631061853c8f9eead71bdb390)
[李沐-Transformer论文逐段精读【论文精读】](https://www.bilibili.com/video/BV1pu411o7BE/?share_source=copy_web&vd_source=e46571d631061853c8f9eead71bdb390)
[3Blue1Brown-【官方双语】GPT是什么？直观解释Transformer | 【深度学习第5章】](https://www.bilibili.com/video/BV13z421U7cs/?share_source=copy_web&vd_source=e46571d631061853c8f9eead71bdb390)
[3Blue1Brown-【官方双语】直观解释注意力机制，Transformer的核心 | 【深度学习第6章】](https://www.bilibili.com/video/BV1TZ421j7Ke/?share_source=copy_web)

⭐Transformer 论文精读
https://github.com/Hoper-J/AI-Guide-and-Demos-zh_CN/tree/master/PaperNotes

⭐《大语言模型：从理论到实践v2.2025.张奇》transformer组件实现+机器翻译实战
https://github.com/intro-llm/intro-llm-code/blob/main/chs/ch2-foundations/Transformer

哈佛NLP实现：
博客：https://nlp.seas.harvard.edu/annotated-transformer/#training
代码：https://github.com/mcxiaoxiao/annotated-transformer-Chinese/blob/main/AnnotatedTransformer%20%E4%B8%AD%E6%96%87.ipynb

PPT可以
【Transformer 的 Pytorch 代码实现讲解-哔哩哔哩】 https://b23.tv/pPamgsI
源码是github上现成的哦～链接🔗 https://github.com/wmathor/nlp-tutorial/blob/master/5-1.Transformer/Transformer_Torch.py#L122
视频中用到的 ppt 链接在这里啦：链接🔗 https://pan.quark.cn/s/6e55ac9f40d9
https://github.com/zxuu/Self-Attention/blob/main/My_Transformer.py

Transformer常见问题与回答总结
https://zhuanlan.zhihu.com/p/496012402?utm_medium=social&utm_oi=629375409599549440

比较清晰的各模块实现
https://github.com/hyunwoongko/transformer/blob/master/models/blocks/decoder_layer.py

datawhalechina
https://github.com/datawhalechina/fun-transformer/tree/main


[transformer实战-机器翻译]
博客：https://zhuanlan.zhihu.com/p/347061440
代码：https://github.com/hemingkx/ChineseNMT/tree/master
研究生实验2：基于序列模型的英文到中文翻译机：https://www.zybuluo.com/wujiaju/note/1885072

HUuggingface：https://huggingface.co/learn/llm-course/zh-CN/chapter7/4?fw=pt
PyTorch官方实现：https://pytorch.ac.cn/tutorials/beginner/torchtext_translation.html
PyTorch官方实现中文：https://github.com/apachecn/apachecn-dl-zh/blob/master/docs/pt-tut-17/32.md
PyTorch官方示例：https://github.com/pytorch/examples/blob/main/language_translation/src/model.py

（不全）https://github.com/BrightXiaoHan/MachineTranslationTutorial/blob/master/tutorials/Chapter2/Normalize.md


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

[HAMI-core Msg(813:140699094931264:libvgpu.c:836)]: Initializing.....
Fri May  2 07:22:52 2025
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 470.57.02    Driver Version: 470.57.02    CUDA Version: 12.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA A100-PCI...  Off  | 00000000:65:00.0 Off |                    0 |
| N/A   44C    P0    74W / 250W |  26940MiB / 40536MiB |     40%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+


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


### 掩码相关问题
注意力分数(batch_size, num_heads, seq_len_q, seq_len_k)
所以：
1. 编码器掩码（正确）
填充掩码形状：(batch_size, 1, 1, seq_len_src)
合理性：编码器的键（K）来自源序列，需屏蔽填充位置（如<pad>）。
实现逻辑：通过扩展掩码维度至 (batch_size, num_heads, seq_len_q, seq_len_k)，实际计算时自动广播到所有头和查询位置。

2. 解码器自注意力掩码（部分需修正）
- 填充掩码形状：(batch_size, 1, 1, seq_len_tgt)
合理性：目标序列的填充位置需屏蔽，但需广播到查询维度。
修正建议：实际应用中需将掩码扩展为 (batch_size, 1, seq_len_tgt, seq_len_tgt)，确保每个查询位置均过滤无效键。
- 前瞻掩码（因果掩码）形状：(batch_size, 1, seq_len_tgt, seq_len_tgt)
合理性：下三角矩阵严格防止未来信息泄露，与标准Transformer实现一致。
术语修正：建议使用**“因果掩码”**而非“前瞻掩码”，避免歧义（前瞻掩码可能指允许有限未来访问）。
- 序列掩码合并逻辑：填充掩码 & 因果掩码
操作步骤：
将填充掩码从 (batch_size, 1, 1, seq_len_tgt) 广播至 (batch_size, 1, seq_len_tgt, seq_len_tgt)。
与因果掩码按位与操作，生成最终掩码 (batch_size, 1, seq_len_tgt, seq_len_tgt)。

3. 解码器交叉注意力掩码（正确）
掩码形状：(batch_size, 1, 1, seq_len_src)
合理性：交叉注意力中，键（K）来自编码器输出，需应用源序列的填充掩码。
实现逻辑：掩码广播至 (batch_size, num_heads, seq_len_tgt, seq_len_src)，确保解码器每个查询仅关注有效源位置。

https://www.cvmart.net/community/detail/5137
https://avoid.overfit.cn/post/2371a9ec5eca46af81dbe23d3442a383
https://ifwind.github.io/2021/08/17/Transformer%E7%9B%B8%E5%85%B3%E2%80%94%E2%80%94%EF%BC%887%EF%BC%89Mask%E6%9C%BA%E5%88%B6/#unilm%E4%B8%AD%E7%9A%84mask

#### 问题

1. 为何不对查询（Q）掩码？
功能分离：Q代表当前需要“提问”的位置（如解码器当前生成词），K/V代表可检索的上下文信息。
计算效率：若对Q掩码，需逐个判断其有效性，增加复杂度；掩码K可一次性屏蔽所有无效上下文。
模型鲁棒性：即使Q来自填充位置（如编码器的无效词），其输出会被后续处理（如池化层）过滤，无需额外操作。
2. 为何填充掩码仅作用于K？
注意力权重本质：注意力分数表示查询与键的关联性。若K包含填充位置，需全局屏蔽这些无效关联，而非限制查询的主动性。
统一过滤逻辑：所有查询（无论有效与否）均不关注填充位置，避免模型学习到虚假依赖。
