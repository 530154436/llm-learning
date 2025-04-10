<nav>
<a href="#一分词的基本概念">一、分词的基本概念</a><br/>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#11-模型输入编码-encode阶段">1.1 模型输入（编码 Encode）阶段</a><br/>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#12-模型输出解码-decode阶段">1.2 模型输出（解码 Decode）阶段</a><br/>
<a href="#二分词流程">二、分词流程</a><br/>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#21-标准化normalization">2.1 标准化（normalization）</a><br/>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#22-预分词pre-tokenization">2.2 预分词（Pre-tokenization）</a><br/>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#23-模型model">2.3 模型（Model）</a><br/>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#24-分词后处理post-tokenization">2.4 分词后处理（Post-tokenization）</a><br/>
<a href="#三分词算法">三、分词算法</a><br/>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#31-按单词划分word-based">3.1 按单词划分（Word-based）</a><br/>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#32-按字符划分character-based">3.2 按字符划分（Character-based）</a><br/>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#33-按子词切分-subword-tokenization">3.3 按子词切分 (Subword tokenization)</a><br/>
<a href="#参考引用">参考引用</a><br/>
</nav>


## 一、分词的基本概念

在自然语言处理（NLP）技术中，`Tokenization`也可以被称为“word segmentation”，直译为中文是指`分词`。
具体来讲，分词是NLP的基础任务，按照特定需求能把文本中的句子、段落切分成一个字符串序列（其中的元素通常称为token 或叫词语），作为大语言模型的输入数据。
这些Tokens可以是单个字符、单词的一部分，甚至是整个单词或句子片段。

### 1.1 模型输入（编码 Encode）阶段
由于神经网络模型不能直接处理文本，因此需要先将文本转换为数字，这个过程被称为编码 (Encoding)，其包含两个步骤：

+ **分词（Tokenize）**： 将文本拆分为词元（Token），常见的分词方式包括字级、词级、子词级（如 BPE、WordPiece）、空格分词等。
```
输入: "你好世界"
分词: ["你好", "世界"]
```
+ **映射（Mapping）**：将每个词元映射为词汇表中唯一的 Token ID，生成的数字序列即为模型的输入。
```
分词: ["你好", "世界"]
映射: [1001, 1002]
```

### 1.2 模型输出（解码 Decode）阶段

解码是编码的逆过程，旨在将模型输出的数字序列重新转换为人类可读的文本格式。它也包含两个步骤：

+ **反映射（De-mapping）**：模型输出的数字序列通过词汇表映射回对应的词元，二者是一一对应的关系。
```
输出: [1001, 1002]
反映射: ["你好", "世界"]
```

+ **文本重组**：将解码后的词元以某种规则重新拼接为完整文本。
```
反映射: ["你好", "世界"]
重组: "你好世界"
```

## 二、分词流程
分词的流程通常包括标准化（Normalization）、预分词（Pre-tokenization）、Model和Post-tokenization。

<img src="../images/01-tokenization/分词流程.png" width="45%" height="45%" alt="">

### 2.1 标准化（normalization）

Normalization主要包括以下几个方面：

+ **文本清洗**<br>
 去除无用字符：移除文本中的特殊字符、非打印字符等，只保留对分词和模型训练有意义的内容。<br>
 去除额外空白：消除文本中多余的空格、制表符、换行符等，统一文本格式。<br>

+ **标准化写法**<br>
统一大小写：将所有文本转换为小写或大写，减少大小写变体带来的影响。<br>
数字标准化：将数字统一格式化，有时候会将所有数字替换为一个占位符或特定的标记，以减少模型需要处理的变量数量。<br>

+ **编码一致性**<br>
字符标准化：确保文本采用统一的字符编码（如UTF-8），处理或转换特殊字符和符号。

+ **语言规范化**<br>
词形还原（Lemmatization）：将单词还原为基本形式（lemma），例如将动词的过去式还原为一般现在时。<br>
词干提取（Stemming）：去除单词的词缀，提取词干，这是一种更粗糙的词形还原方式。<br>

```python
from tokenizers.normalizers import BertNormalizer
from transformers import AutoTokenizer, BertTokenizerFast

tokenizer: BertTokenizerFast = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
normalizer: BertNormalizer = tokenizer.backend_tokenizer.normalizer
# hello how are u?
print(normalizer.normalize_str("Héllò hôw are U?"))

tokenizer: BertTokenizerFast = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
normalizer: BertNormalizer = tokenizer.backend_tokenizer.normalizer
# Héllò hôw are U?
print(normalizer.normalize_str("Héllò hôw are U?"))
```

### 2.2 预分词（Pre-tokenization）

Pre-tokenization是基于一些简单的规则（如`空格`和`标点符号`）进行初步的文本分割，这一步骤是为了将文本初步拆解为更小的单元，如句子和词语；对于英文等使用空格分隔的语言来说，这一步相对直接，但对于中文等无空格分隔的语言，则可能直接进入下一步。通常使用 tokenizer 对象的 pre_tokenizer 属性的 `pre_tokenize_str()` 方法进行预分词。
+ `BERT` tokenizer 在空白和标点符号上进行分割，不保留空格。
+ `GPT-2` 的 tokenizer 也会在空格和标点符号上进行分割，但它保留空格，并用 Ġ 符号替换它们，使得在解码 tokens 时能够恢复原始空格，并且不会忽略双空格。
+ `T5` 使用 SentencePiece 算法进行分词。T5 tokenizer 保留空格并用特定 token 替换它们（例如 ▁），但只在空格上进行分割，不考虑标点符号。此外，它会在句子开头默认添加一个空格，并忽略 are 和 you 之间的双空格。
```python
from tokenizers.pre_tokenizers import BertPreTokenizer, ByteLevel, Sequence
from transformers import AutoTokenizer, BertTokenizerFast, GPT2TokenizerFast, T5TokenizerFast

tokenizer: BertTokenizerFast = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
pre_tokenizer: BertPreTokenizer = tokenizer.backend_tokenizer.pre_tokenizer
# [('Hello', (0, 5)), (',', (5, 6)), ('how', (7, 10)), ('are', (11, 14)), ('you', (16, 19)), ('?', (19, 20))]
print(pre_tokenizer.pre_tokenize_str("Hello, how are  you?"))

tokenizer: GPT2TokenizerFast = AutoTokenizer.from_pretrained("openai-community/gpt2")
pre_tokenizer: ByteLevel = tokenizer.backend_tokenizer.pre_tokenizer
# [('Hello', (0, 5)), (',', (5, 6)), ('Ġhow', (6, 10)), ('Ġare', (10, 14)), ('Ġ', (14, 15)), ('Ġyou', (15, 19)), ('?', (19, 20))]
print(pre_tokenizer.pre_tokenize_str("Hello, how are  you?"))

tokenizer: T5TokenizerFast = AutoTokenizer.from_pretrained("AI-ModelScope/t5-small")
pre_tokenizer: Sequence = tokenizer.backend_tokenizer.pre_tokenizer
# [('▁Hello,', (0, 6)), ('▁how', (7, 10)), ('▁are', (11, 14)), ('▁you?', (16, 20))]
print(pre_tokenizer.pre_tokenize_str("Hello, how are  you?"))
```

### 2.3 模型（Model）

Model是分词的核心部分，在Pre-tokenization的基础上，根据选定的模型或算法（BPE、WordPiece、Unigram或SentencePiece等）进行更细致的处理，包括通过大量文本数据，根据算法规则生成词汇表(Vocabulary)，然后依据词汇表，将文本拆分为Token。 在自然语言处理模型中，确定合适的词汇表大小是一个关键步骤，它直接影响模型的性能、效率以及适应性。理想的词汇表应该在保证模型性能和效率的同时，满足特定任务和数据集的需求。

+ **较大词汇表**：可以提高模型覆盖不同词汇和表达的能力，有助于更好地理解和生成文本。然而，它也会增加计算负担和资源需求，可能导致训练和推理过程变慢，甚至引起词嵌入空间的稀疏性问题。
+ **较小词汇表**：减少计算资源的需求，但可能限制模型的表现力，使其难以捕捉复杂的语言结构。

不同的NLP任务对词汇表大小有不同的要求。例如，精细的文本生成任务可能需要较大的词汇量来捕捉更多细节，而一些简单的分类任务则可能只需较小的词汇表就能高效运行。此外，考虑到不同语言间的结构差异及数据集中文本的多样性，设计理想的词汇表也需要考虑具体的应用场景和需求。在实际应用中，可能需要通过实验和调整来找到最适合特定模型和任务的词汇表大小，下面是各大LLM词汇表的大小和性能对比：

| 模型                | 分词器                      | 词汇表大小   | 主要特点/性能概述                                                              |
|-------------------|--------------------------|---------|------------------------------------------------------------------------|
| BERT-base-cased   | WordPiece                | 28,996  | 区分大小写的预训练模型，适用于多种NLP任务。                                                |
| BERT-base-uncased | WordPiece                | 30,522  | 不区分大小写的版本，同样广泛应用于各类NLP任务。                                              |
| T5-small          | SentencePiece            | 32,100  | 采用统一框架处理所有NLP任务，包括文本生成、翻译等。                                            |
| GPT-2             | Byte Pair Encoding (BPE) | 50,257  | 扩展了词汇表和参数规模，显著提升了生成能力和理解复杂度。                                           |
| Qwen              | Byte Pair Encoding (BPE) | 151,646 | https://qwen.readthedocs.io/zh-cn/latest/getting_started/concepts.html |

### 2.4 分词后处理（Post-tokenization）

Post-tokenization主要包括：

- **序列填充与截断**：为保证输入序列的长度一致，对过长的序列进行截断，对过短的序列进行填充。
- **特殊Token添加**：根据模型需求，在序列的适当位置添加特殊Token（如[CLS], [SEP]）。
- **构建注意力掩码**：对于需要的模型，构建注意力掩码以区分实际Token和填充Token。

## 三、分词算法

根据切分粒度的不同，分词策略可以分为以下几种：

### 3.1 按单词划分（Word-based）
按照词进行分词，根据空格或标点进行分割。例如："Today is sunday"→[today, is, sunday, .]<br>

优点：
+ 简单直观，易于理解和实现。<br>

缺点：
+ `词汇表过大`：由于需要涵盖语料库中的所有词语，这会导致词汇表异常庞大。<br>
+ `未登陆词问题`（Out-Of-Vocabulary, OOV）：对于不在词汇表中的词，模型通常无法识别，只能将其标记为未知符号（[UNK]），影响了对新词的适应能力。<br>
+ `低频词训练不足`：受限于词汇表大小，一些出现频率较低的词可能被排除在外，导致这些词在模型训练中得不到充分学习。<br>
+ `形态学信息丢失`：对于英语等语言，不同形式的同一个词（如"look", "looks", "looking", "looked"）被视为不同的词，增加了词汇量，并且使得模型难以学习到这些词之间的关系，既增加了训练冗余，也加剧了大词汇量的问题。<br>

### 3.2 按字符划分（Character-based）
按照单字符进行分词，将每个字符（包括标点符号）视为一个单独的单元。例如："Today is sunday"→[t， o， d，a，y，i, s, s，u，n，d，a，y，.]<br>

优点：
+ 词汇量小：与基于词的方法相比，基于字符的词汇表要小得多，因为只需要考虑构成语言的所有字符，而不是所有的单词组合。<br>
+ 减少未知标记（OOV）问题：由于每个单词都可以从字符构建，因此减少了遇到词汇外单词的情况，提高了模型对未见过单词的适应能力。<br>

缺点：
+ `粒度过细`：一个完整的词被分解成多个字符，这导致每个字符或token的信息密度较低，不利于捕捉词或短语的整体意义。<br>
+ `训练成本高、解码效率低`：由于需要处理更多的token（即字符），模型训练时间增加，同时解码过程也会变得更慢、效率更低。<br>
+ `在某些语言中字符的意义有限`：特别是在那些字符本身不携带完整意义的语言中（如英语等使用字母文字的语言），单个字符无法传达有效的信息，必须结合其他字符才能表达完整的意思。<br>


### 3.3 按子词切分 (Subword tokenization)
基于子词的分词方法旨在结合基于词和基于字符的分词法的优点，同时克服它们各自的缺点。这种方法通过将单词切分成更小但具有一定语义意义的子词来实现。
+ 高频词保持完整：如"dog"这样的高频词不会被分割。
+ 低频词被拆分为有意义的子词：例如"dogs"会被拆分为["dog", "##s"]。这种方法允许用一个有限的词汇表解决所有单词的分词问题，同时尽量减少词汇表的大小。

优点：
+ `降低词汇表大小`：可以通过组合少量的sub-word构建更大的词，例如“unfortunately”可以被分解为“un-”，“for-”，“tun-”，“ate-”，“ly”。这与英语中的词根词缀拼词法类似，使得这些片段也可以用于构造其他词。
+ `学习词之间的关系`：相比传统分词方法，能够更好地捕捉词缀间的关系，比如"old"、"older"和"oldest"之间的关系。
+ `平衡OOV问题`：解决了传统分词法无法很好处理未知或罕见词汇的问题，同时避免了基于字符的方法粒度过细的问题。Subword方法通过“拼词”的方式处理许多罕见词汇，其粒度介于词和字符之间，提供了较好的OOV处理能力。


常见的主流子词切分算法如下：

| 模型   | BPE/BBPE                            | WordPiece                                    | Unigram                      |
|------|-------------------------------------|----------------------------------------------|------------------------------|
| 训练   | 从一个小的词汇表开始，学习合并token的规则             | 从一个小的词汇表开始，学习合并token的规则                      | 从一个大的词汇表开始，学习移除token的规则      |
| 训练步骤 | 合并对应最常见的token对                      | 根据频率对得分最高的token对进行合并<br>优先考虑单个token出现频率较低的组合 | 移除那些在整个语料库上会导致最小化损失的所有tokens |
| 学习   | 合并规则和词汇表                            | 只有词汇表                                        | 包含每个token分数的词汇表              |
| 编码   | 将单词分割成字符，并应用在训练过程中学到的合并             | 从开头开始查找属于词汇表中最长的子词，然后对其余部分重复此过程              | 使用训练中学到的分数找到最可能的token分割方式    |
| 代表模型 | GPT,GPT-2,RoBERTa,BART,和DeBERTa<br>开源大规模语言模型如Llama系列、Mistral、Qwen | BERT        | T5            |


详细参考：
+ [分词器（1）BPE.md](分词器（1）BPE.md)
+ [分词器（2）WordPiece.md](分词器（2）WordPiece.md)
+ [分词器（3）Unigram.md](分词器（3）Unigram.md)


## 参考引用
1. [第三篇：ChatGPT背后强大而神秘的力量，用最简单的语言讲解Transformer架构之Tokenizer](https://zhuanlan.zhihu.com/p/686444517)<br>
2. [【LLM基础知识】LLMs-Tokenizer知识总结笔记v2.0](https://www.53ai.com/news/LargeLanguageModel/2024080625837.html)<br>
3. [全网最全的大模型分词器（Tokenizer）总结.md](https://github.com/luhengshiwo/LLMForEverybody/blob/main/01-第一章-预训练/全网最全的大模型分词器（Tokenizer）总结.md)<br>
4. [transformers-tokenizer](https://huggingface.co/learn/nlp-course/zh-CN/chapter6/1?fw=pt)<br>
5. [LLM中的Tokenizers-bilibili视频](https://www.bilibili.com/video/BV1q94y1a7NU)<br>
6. [tiktokenizer-可视化](https://tiktokenizer.vercel.app/)<br>
7. [千问-tokenizer](https://github.com/QwenLM/Qwen/blob/main/tokenization_note_zh.md)<br>
8. [FastTokenizer：高性能文本处理库](https://github.com/GreatV/fast_tokenizer)<br>
