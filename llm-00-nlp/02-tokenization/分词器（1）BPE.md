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


#### 3.3.2 B-BPE（Byte-level BPE）
字节级别的 BPE（Byte-level BPE, B-BPE）是 BPE 算法的一种拓展。
它将字节视为合并操作的基本符号，从而可以实现更细粒度的分割，且解决了未登录词问题。
具体来说，如果将所有 Unicode 字符都视为基本字符，那么包含所有可能基本字符的基本词表会非常庞大（例如将中文的每个汉字当作一个基本字符）。
而将字节作为基本词表可以设置基本词库的大小为 256，同时确保每个基本字符都包含在词汇中。 
例如，GPT-2 的词表大小为 50,257 ，包括 256 个字节的基本词元、一个特殊的文末词元以及通过 50,000 次合并学习到的词元。

### 3.2 WordPiece
WordPiece 是谷歌内部非公开的分词算法，最初是由谷歌研究人员在开发语音搜索系统时提出的。
随后，在 2016 年被用于机器翻译系统，并于 2018 年被 BERT 采用作为分词器。
WordPiece 分词和 BPE 分词的想法非常相似，都是通过迭代合并连续的词元，但是合并的选择标准略有不同。
在合并前，`WordPiece`分词算法会首先训练一个语言模型，并用这个语言模型对所有可能的词元对进行评分。然后，在每次合并时，它都会选择使得训练数据的似然性增加最多的词元对。
与 BPE 类似，Word Piece 分词算法也是从一个小的词汇表开始，其中包括模型使用的特殊词元和初始词汇表。
由于它是通过添加前缀（如 BERT 的##）来识别子词的，因此每个词的初始拆分都是将前缀添加到词内的所有字符上。
举例来说，“word”会被拆分为：“w##o ##r ##d”。与 BPE 方法的另一个不同点在于，WordPiece 分词算法并不选择最频繁的词对，而是使用下面的公式为每个词对计算分数：

$$
得分=\frac{词对出现的频率}{第一个词出现的频率\times 第二个词出现的频率} \\
$$
### 3.3 SentencePiece
### 3.4 Unigram
与 BPE 分词和 WordPiece 分词不同，Unigram 分词方法从语料库的一组足够大的字符串或词元初始集合开始，迭代地删除其中的词元，直到达到预期的词表大小。
它假设从当前词表中删除某个词元，并计算训练语料的似然增加情况，以此来作为选择标准。
这个步骤是基于一个训练好的一元语言模型来进行的。
为估计一元语言模型，它采用期望最大化（Expectation–Maximization, EM）算法：
在每次迭代中，首先基于旧的语言模型找到当前最优的分词方式，然后重新估计一元概率从而更新语言模型。
这个过程中一般使用动态规划算法（即`维特比算法`，Viterbi Algorithm）来高效地找到语言模型对词汇的最优分词方式。采用这种分词方法的代表性模型包括 T5 和 mBART。


## 参考引用
[BPE vs WordPiece：理解Tokenizer的工作原理与子词分割方法](https://github.com/Hoper-J/AI-Guide-and-Demos-zh_CN/blob/master/Guide/21. BPE vs WordPiece：理解 Tokenizer 的工作原理与子词分割方法.md)<br>