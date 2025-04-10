#### 3.3.1 BPE（Byte-Pair Encoding）
字节对编码 (Byte-Pair Encoding，BPE) 最初是作为一种压缩文本的算法开发的，最早是由Philip Gage于1994年在《A New Algorithm for Data Compression》一文中提出，后来被 OpenAI 在预训练 GPT 模型时用于分词器（Tokenizer）。它被许多 Transformer 模型使用，包括 GPT、GPT-2、RoBERTa、BART 和 DeBERTa。

> **参考文献：**
> - [A new algorithm for data compression. 1994](http://www.pennelynn.com/Documents/CUJ/HTML/94HTML/19940045.HTM)
> - [Neural Machine Translation of Rare Words with Subword Units. 2015](https://arxiv.org/pdf/1508.07909v5)

BPE 每次的迭代目标是找到频率最高的相邻字符对，定义 Score：

$$
\text{Score}_{\text{BPE}}(x, y) = \text{freq}(x, y)
$$

其中, $\text{freq}(x, y)$ 表示字符对 $(x, y)$ 在语料库中的出现频次。 算法步骤：

1. **初始化词汇表 $V$**：
   - $V$ 包含语料库中的所有唯一字符，即单词字符的集合。
2. **统计字符对的频次**：
   - 对于每个单词的字符序列，统计相邻字符对的出现频次。
3. **找到频次（Score）最高的字符对并合并**：
   - 选择出现频率最高的字符对 $(x, y)$，将其合并为新符号 $xy$。
4. **更新词汇表并重复步骤 2 到 4**：
   - 将新符号添加到词汇表 $V = V \cup \{xy\}$。
   - 更新语料库中的单词表示，重复统计和合并过程，直到满足停止条件（例如，词汇表达到预定大小）。

计算初始词表：通过训练语料获得或者最初的英文中26个字母加上各种符号以及常见中文字符，这些作为初始词表。
构建频率统计：统计所有子词单元对（两个连续的子词）在文本中的出现频率。
合并频率最高的子词对：选择出现频率最高的子词对，将它们合并成一个新的子词单元，并更新词汇表。
重复合并步骤：不断重复步骤 2 和步骤 3，直到达到预定的词汇表大小、合并次数，或者直到不再有有意义的合并（即，进一步合并不会显著提高词汇表的效益）。
分词：使用最终得到的词汇表对文本进行分词。

#### 3.3.2 B-BPE（Byte-level BPE）
字节级别的 BPE（Byte-level BPE, B-BPE）是 BPE 算法的一种拓展。
它将字节视为合并操作的基本符号，从而可以实现更细粒度的分割，且解决了未登录词问题。
具体来说，如果将所有 Unicode 字符都视为基本字符，那么包含所有可能基本字符的基本词表会非常庞大（例如将中文的每个汉字当作一个基本字符）。
而将字节作为基本词表可以设置基本词库的大小为 256，同时确保每个基本字符都包含在词汇中。 
例如，GPT-2 的词表大小为 50,257 ，包括 256 个字节的基本词元、一个特殊的文末词元以及通过 50,000 次合并学习到的词元。


## 参考引用
[1] [transformers-BPE tokenization 算法](https://huggingface.co/learn/llm-course/zh-CN/chapter6/5?fw=pt)<br>
[2] [BPE分词原理](https://github.com/BrightXiaoHan/MachineTranslationTutorial/blob/master/tutorials/Chapter2/BPE.md)<br>
[3] [理解 Tokenizer 的工作原理与子词分割方法](https://github.com/Hoper-J/AI-Guide-and-Demos-zh_CN/blob/master/Guide/)<br>
[4] [Subword Tokenization 算法](https://www.huaxiaozhuan.com/%E5%B7%A5%E5%85%B7/huggingface_transformer/chapters/1_tokenizer.html)<br>
[5] [论文分享 Neural machine Translation of Rare Words with Subword Units](https://blog.csdn.net/Mr_tyting/article/details/91352726)<br>
[6] [subword-nmt](https://github.com/rsennrich/subword-nmt/blob/master/learn_bpe.py)<br>
[6] [LLM大语言模型之Tokenization分词方法（WordPiece，Byte-Pair Encoding (BPE)，Byte-level BPE(BBPE)原理及其代码实现）](https://zhuanlan.zhihu.com/p/652520262)<br>
