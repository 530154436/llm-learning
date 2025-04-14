<nav>
<a href="#字节对编码-bpe">字节对编码 (BPE)</a><br/>
<a href="#字节级别的-bpeb-bpe">字节级别的 BPE（B-BPE）</a><br/>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#通义千问使用的分词方式">通义千问使用的分词方式</a><br/>
<a href="#参考引用">参考引用</a><br/>
</nav>

## 字节对编码 (BPE)
字节对编码 (Byte-Pair Encoding，BPE) 最初是作为一种压缩文本的算法开发的，最早是由Philip Gage于1994年在《A New Algorithm for Data Compression》一文中提出，后来被 OpenAI 在预训练 GPT 模型时用于分词器（Tokenizer）。它被许多 Transformer 模型使用，包括 GPT、GPT-2、RoBERTa、BART 和 DeBERTa。

> **参考文献：**
> - [A new algorithm for data compression. 1994](http://www.pennelynn.com/Documents/CUJ/HTML/94HTML/19940045.HTM)
> - [Neural Machine Translation of Rare Words with Subword Units. 2015](https://arxiv.org/pdf/1508.07909v5)

BPE 每次的迭代目标是找到频率最高的相邻字符对，定义 Score：

$$
\text{Score}_{\text{BPE}}(x, y) = \text{freq}(x, y)
$$

其中, $\text{freq}(x, y)$ 表示字符对 $(x, y)$ 在语料库中的出现频次。 算法的训练步骤：

1. **初始化词汇表 $V$**：
   - 语料库标准化、预分词后，统计所有单词的频率 -> word_freqs
   - 所有单词按字符切分得到对应的子词序列（初始是一个个字符）-> word_splits
   - 构建初始词表（英文中26个字母加上各种符号以及常见中文字符）-> $V$
2. **统计子词对的频次**：
   - 对于每个单词的子词序列，统计相邻子词对（两个连续的子词）的出现频次。
3. **合并频率最高的子词对并更新词汇表**：
   - 选择出现频率最高的子词对 $(x, y)$，将其合并为新符号 $xy$。
   - 将新符号添加到词汇表 $V = V \cup \{xy\}$。
4. **重复步骤 2 到 4**：
   - 重复统计和合并过程，直到满足停止条件（例如，词汇表达到预定大小）。


```python
from data.download_model import DATA_DIR
from transformers import AutoTokenizer, GPT2TokenizerFast
from collections import defaultdict
from typing import Dict, Tuple, List, Iterable, Union


def init_vocab(corpus: Iterable,
               special_tokens: List[str] = None) -> Tuple[Dict[str, int], Dict[str, list], list]:
    """
    计算初始词表
    :param corpus: 语料库(标准化、预分词)
    :param special_tokens: 特殊token标记
    :return:
    """
    word_freqs = defaultdict(int)    # 单词-频率
    word_splits = defaultdict(list)  # 单词-对应切分的子词序列
    vocab = special_tokens if special_tokens else []  # 对于 GPT-2，唯一的特殊 tokens 是 "<|endoftext|>"

    # 计算语料库中每个单词的频率
    for sentence in corpus:
        for word in sentence:
            word_freqs[word] += 1

    # 将每个单词拆分为字符序列，并计算初始词表
    for word in word_freqs.keys():
        for letter in word:
            word_splits.setdefault(word, []).append(letter)
            if letter not in vocab:
                vocab.append(letter)
    # [',', '.', 'C', 'F', 'H', 'T', 'a'...]
    print("初始词表: ", vocab)
    return word_freqs, word_splits, vocab


def compute_pair_freqs(word_splits: Dict[str, List[str]],
                       word_freq: Dict[str, int]) -> Dict[Tuple[str], int]:
    """
    统计相邻子词对的频次
    :param word_splits: 单词对应的子词序列, {'This': ['T','h','i','s']}
    :param word_freq: 单词对应的频次, {'This': 3}
    :return 字符对和频次
    """
    pair_freqs = defaultdict(int)
    for word, splits in word_splits.items():
        if len(splits) == 1:
            continue
        for i in range(len(splits) - 1):
            pair = (splits[i], splits[i + 1])
            pair_freqs[pair] += word_freq[word]
    return pair_freqs


def find_max_freq_pair(pair_freq: Dict[Tuple[str], int]) -> Tuple[Tuple[str], int]:
    """
    找到频次最高的子词对和频率
    :param pair_freq: 字符对和频次的字典
    :return:
    """
    best_pair = ""
    max_freq = None
    for pair, freq in pair_freq.items():
        if max_freq is None or max_freq < freq:
            best_pair = pair
            max_freq = freq
    return best_pair, max_freq


def merge_pair(a: Union[str, bytes],
               b: Union[str, bytes],
               word_splits: Dict[str, List[Union[str, bytes]]]) -> Dict[str, List[str]]:
    """
    选择频次最高的子词对并合并

    兼容字节的合并，eg.
    word_splits = {'喜欢': [b'\xe5', b'\x96', b'\x9c', b'\xe6', b'\xac', b'\xa2']}
    merge_pair(b'\xe5', b'\x96', word_splits)
    => {'喜欢': [b'\xe5\x96', b'\x9c', b'\xe6', b'\xac', b'\xa2']}

    :param a: 字符对开始
    :param b: 字符对结束
    :param word_splits: 单词对应的子词序列, {'This': ['T','h','i','s']}
    :return: 单词的合并后子词序列
    """
    for word in word_splits:
        splits = word_splits[word]
        if len(splits) == 1:
            continue
        i = 0
        while i < len(splits) - 1:
            if splits[i] == a and splits[i + 1] == b:
                splits = splits[:i] + [a + b] + splits[i + 2:]  # 合并相邻子词对并生成新的子词序列
            else:
                i += 1
        word_splits[word] = splits
    return word_splits


def tokenize(sentence: Iterable, merges: Dict[Tuple[str], str]) -> List[str]:
    """
    使用学到的所有合并规则进行分词
    :param sentence:  句子
    :param merges:  合并规则
    :return:
    """
    splits = [[l for l in word] for word in sentence]
    for pair, merge in merges.items():
        for idx, split in enumerate(splits):
            i = 0
            while i < len(split) - 1:
                if split[i] == pair[0] and split[i + 1] == pair[1]:
                    split = split[:i] + [merge] + split[i + 2:]
                else:
                    i += 1
            splits[idx] = split
    return sum(splits, [])


def prepare_corpus(model: str = "openai-community/gpt2"):
    corpus = [
        "This is the Hugging Face Course.",
        "This chapter is about tokenization.",
        "This section shows several tokenizer algorithms.",
        "Hopefully, you will be able to understand how they are trained and generate tokens.",
    ]
    # 使用gpt2 标准化+预分词
    filepath = DATA_DIR.joinpath(model)
    tokenizer: GPT2TokenizerFast = AutoTokenizer.from_pretrained(filepath)

    # 计算语料库中每个单词的频率
    for text in corpus:
        # [('This', (0, 4)), ('Ġis', (4, 7)), ('Ġthe', (7, 11))..]，Ġ是空空格
        words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
        sentence = []
        for word, offset in words_with_offsets:
            sentence.append(word)
        yield sentence


def main(vocab_size: int = 50):
    # 初始化
    word_freqs, word_splits, vocab = init_vocab(prepare_corpus(), special_tokens=["<|endoftext|>"])

    # 训练BPE
    merges: Dict[Tuple[str], str] = defaultdict(str)
    iteration = 1
    while len(vocab) < vocab_size:
        pair_freqs = compute_pair_freqs(word_splits, word_freqs)
        best_pair, max_freq = find_max_freq_pair(pair_freqs)
        word_splits = merge_pair(*best_pair, word_splits)
        merges[best_pair] = best_pair[0] + best_pair[1]
        vocab.append(best_pair[0] + best_pair[1])
        print(f"第{iteration}迭代：频率最高的子词对={best_pair}->{''.join(best_pair)}，频率={max_freq}")
        iteration += 1
    print(f"最终词表({len(vocab)}): {vocab}")
    print(f"最终学习了{len(merges)}条合并规则: ")
    for k, v in merges.items():
        print(f"    {k} -> {v}""")

    # 预测
    sentence = list(prepare_corpus())[0]
    predict = tokenize(sentence=sentence, merges=merges)
    print(f"预测输出: {predict}")
```

输出结果：
```
初始词表:  ['<|endoftext|>', 'T', 'h', 'i', 's', 'Ġ', 't', 'e', 'H', 'u', 'g', 'n', 'F', 'a', 'c', 'C', 
          'o', 'r', '.', 'p', 'b', 'k', 'z', 'w', 'v', 'l', 'm', 'f', 'y', ',', 'd']
第1迭代：频率最高的子词对=('Ġ', 't')->Ġt，频率=7
第2迭代：频率最高的子词对=('i', 's')->is，频率=5
第3迭代：频率最高的子词对=('e', 'r')->er，频率=5
第4迭代：频率最高的子词对=('Ġ', 'a')->Ġa，频率=5
第5迭代：频率最高的子词对=('Ġt', 'o')->Ġto，频率=4
第6迭代：频率最高的子词对=('e', 'n')->en，频率=4
第7迭代：频率最高的子词对=('T', 'h')->Th，频率=3
第8迭代：频率最高的子词对=('Th', 'is')->This，频率=3
第9迭代：频率最高的子词对=('o', 'u')->ou，频率=3
第10迭代：频率最高的子词对=('s', 'e')->se，频率=3
第11迭代：频率最高的子词对=('Ġto', 'k')->Ġtok，频率=3
第12迭代：频率最高的子词对=('Ġtok', 'en')->Ġtoken，频率=3
第13迭代：频率最高的子词对=('n', 'd')->nd，频率=3
第14迭代：频率最高的子词对=('Ġ', 'is')->Ġis，频率=2
第15迭代：频率最高的子词对=('Ġt', 'h')->Ġth，频率=2
第16迭代：频率最高的子词对=('Ġth', 'e')->Ġthe，频率=2
第17迭代：频率最高的子词对=('i', 'n')->in，频率=2
第18迭代：频率最高的子词对=('Ġa', 'b')->Ġab，频率=2
第19迭代：频率最高的子词对=('Ġtoken', 'i')->Ġtokeni，频率=2
最终词表(50): ['<|endoftext|>', 'T', 'h', 'i', 's', 'Ġ', 't', 'e', 'H', 'u', 'g', 'n', 'F', 'a', 'c', 'C', 
  'o', 'r', '.', 'p', 'b', 'k', 'z', 'w', 'v', 'l', 'm', 'f', 'y', ',', 'd', 'Ġt', 'is', 'er', 'Ġa', 'Ġto', 
  'en', 'Th', 'This', 'ou', 'se', 'Ġtok', 'Ġtoken', 'nd', 'Ġis', 'Ġth', 'Ġthe', 'in', 'Ġab', 'Ġtokeni']
预测输出: ['This', 'Ġis', 'Ġthe', 'Ġ', 'H', 'u', 'g', 'g', 'in', 'g', 'Ġ', 'F', 'a', 'c', 'e', 'Ġ', 'C', 'ou', 'r', 'se', '.']
```

BPE理论上还是会出现OOV的，当词汇表的大小受限时，一些较少频繁出现的子词和没有在训练过程中见过的子词，就会无法进入词汇表出现OOV，而Byte-level BPE(BBPE)理论上是不会出现这个情况的。

## 字节级别的 BPE（B-BPE）
字节级别的 BPE（Byte-level BPE, B-BPE）是 BPE 算法的一种拓展。
它将字节视为合并操作的基本符号，从而可以实现更细粒度的分割，且解决了未登录词问题。
具体来说，如果将所有 Unicode 字符都视为基本字符，那么包含所有可能基本字符的基本词表会非常庞大（例如将中文的每个汉字当作一个基本字符）。
而将字节作为基本词表可以设置基本词库的大小为 256，同时确保每个基本字符都包含在词汇中。 
例如，GPT-2 的词表大小为 50,257 ，包括 256 个字节的基本词元、一个特殊的文末词元以及通过 50,000 次合并学习到的词元。

> **参考文献：**
> - [Neural Machine Translation with Byte-Level Subwords](https://arxiv.org/pdf/1909.03341)
>
> **前置知识**<br>
>Unicode 是一种旨在涵盖几乎所有书写系统和字符的字符集。它为每个字符分配了一个唯一的代码点（code point），用于标识字符。<br>
>UTF-8 是一种变长的字符编码方案，它将 Unicode 代码点转换为字节序列以便于存储和传输，所有标准ASCII字符在UTF-8中使用相同的单字节表示。<br>
>字节（Byte）是计算机存储和处理信息的基本单位，包含8个比特，每个位可以是0或1，可表示256种不同的值。<br>
> 
>编码规则示例：<br>
>英文字母“A”的Unicode代码点是U+0041，在UTF-8中表示为0x41（与ASCII相同）。<br>
>中文汉字“你”的Unicode代码点是U+4F60，在UTF-8中表示为三个字节：0xE4 0xBD 0xA0。<br>


BBPE从性能和原理上和BPE差异不大，最主要区别是BPE基于字符粒度去执行合并的过程生成词表，而BBPE是基于256个不同的字节编码（Byte) 去执行合并过程生成词表。

1. **初始化词汇表 $V$**：
   - 语料库标准化、预分词后，统计所有单词的频率 -> word_freqs
   - 所有单词按字节切分得到对应的子词序列（初始是一个个字符对应的**字节**）-> word_splits
   - 构建初始词表（特殊标记+**256个字节**）-> $V$
2. **统计子词对的频次**：
   - 对于每个单词的子词序列，统计相邻子词对（两个连续的子词）的出现频次。
3. **合并频率最高的子词对并更新词汇表**：
   - 选择出现频率最高的子词对 $(x, y)$，将其合并为新符号 $xy$。
   - 将新符号添加到词汇表 $V = V \cup \{xy\}$。
4. **重复步骤 2 到 4**：
   - 重复统计和合并过程，直到满足停止条件（例如，词汇表达到预定大小）。


```python
from collections import defaultdict
from typing import Dict, Tuple, List, Iterable
from bpe_tokenizer import compute_pair_freqs, find_max_freq_pair, merge_pair
from bpe_demo import prepare_corpus


def init_vocab(corpus: Iterable,
               special_tokens: List[str] = None) -> Tuple[Dict[str, int], Dict[str, list], list]:
    """
    计算初始词表
    :param corpus: 语料库(标准化、预分词)
    :param special_tokens: 特殊token标记
    :return:
    """
    word_freqs = defaultdict(int)    # 单词-频率
    word_splits = defaultdict(list)  # 单词-对应切分的子词序列
    vocab = [_.encode("utf8") for _ in special_tokens] if special_tokens else []
    vocab += [bytes([byte]) for byte in range(256)]

    # 计算语料库中每个单词的频率
    for sentence in corpus:
        for word in sentence:
            word_freqs[word] += 1

    # 将每个单词拆分为子词序列（字节列表）:
    # {'喜欢': [b'\xe5', b'\x96', b'\x9c', b'\xe6', b'\xac', b'\xa2']}
    for word in word_freqs.keys():
        word_splits.setdefault(word, []).extend([bytes([byte]) for byte in word.encode("utf8")])

    # 256个字节
    print(f"初始词表({len(vocab)}): {vocab}")
    return word_freqs, word_splits, vocab


def tokenize(sentence: Iterable, merges: Dict[Tuple[str], str]) -> List[bytes]:
    """
    使用学到的所有合并规则进行分词
    :param sentence:  句子
    :param merges:  合并规则
    :return:
    """
    splits = [[bytes([byte]) for byte in word.encode("utf8")] for word in sentence]
    for pair, merge in merges.items():
        for idx, split in enumerate(splits):
            i = 0
            while i < len(split) - 1:
                if split[i] == pair[0] and split[i + 1] == pair[1]:
                    split = split[:i] + [merge] + split[i + 2:]
                else:
                    i += 1
            splits[idx] = split
    return sum(splits, [])


def main(vocab_size: int = 280):
    # 初始化
    word_freqs, word_splits, vocab = init_vocab(prepare_corpus(), special_tokens=["<|endoftext|>"])

    # 训练B-BPE
    merges: Dict[Tuple[str], str] = defaultdict(str)
    iteration = 1
    while len(vocab) < vocab_size:
        pair_freqs = compute_pair_freqs(word_splits, word_freqs)
        best_pair, max_freq = find_max_freq_pair(pair_freqs)
        word_splits = merge_pair(*best_pair, word_splits)
        merges[best_pair] = best_pair[0] + best_pair[1]
        vocab.append(best_pair[0] + best_pair[1])
        print(f"第{iteration}迭代：频率最高的子词对={best_pair}->{b''.join(best_pair)}，频率={max_freq}")
        iteration += 1
    print(f"最终词表({len(vocab)}): {vocab}")
    print(f"最终学习了{len(merges)}条合并规则.")

    # =预测
    sentence = list(prepare_corpus())[0]
    predict = tokenize(sentence=sentence, merges=merges)
    print(f"预测输出【字节】: {predict}")
    print(f"预测输出【文本】: {[byte.decode('utf8') for byte in predict]}")
```
输出结果：
```
初始词表(257): [b'<|endoftext|>', b'\x00', b'\x01', b'\x02', ..., b'\xfc', b'\xfd', b'\xfe', b'\xff']
第1迭代：频率最高的子词对=(b'\xc4', b'\xa0')->b'\xc4\xa0'，频率=27
第2迭代：频率最高的子词对=(b'\xc4\xa0', b't')->b'\xc4\xa0t'，频率=7
第3迭代：频率最高的子词对=(b'i', b's')->b'is'，频率=5
第4迭代：频率最高的子词对=(b'e', b'r')->b'er'，频率=5
第5迭代：频率最高的子词对=(b'\xc4\xa0', b'a')->b'\xc4\xa0a'，频率=5
第6迭代：频率最高的子词对=(b'\xc4\xa0t', b'o')->b'\xc4\xa0to'，频率=4
第7迭代：频率最高的子词对=(b'e', b'n')->b'en'，频率=4
第8迭代：频率最高的子词对=(b'T', b'h')->b'Th'，频率=3
第9迭代：频率最高的子词对=(b'Th', b'is')->b'This'，频率=3
第10迭代：频率最高的子词对=(b'o', b'u')->b'ou'，频率=3
第11迭代：频率最高的子词对=(b's', b'e')->b'se'，频率=3
第12迭代：频率最高的子词对=(b'\xc4\xa0to', b'k')->b'\xc4\xa0tok'，频率=3
第13迭代：频率最高的子词对=(b'\xc4\xa0tok', b'en')->b'\xc4\xa0token'，频率=3
第14迭代：频率最高的子词对=(b'n', b'd')->b'nd'，频率=3
第15迭代：频率最高的子词对=(b'\xc4\xa0', b'is')->b'\xc4\xa0is'，频率=2
第16迭代：频率最高的子词对=(b'\xc4\xa0t', b'h')->b'\xc4\xa0th'，频率=2
第17迭代：频率最高的子词对=(b'\xc4\xa0th', b'e')->b'\xc4\xa0the'，频率=2
第18迭代：频率最高的子词对=(b'i', b'n')->b'in'，频率=2
第19迭代：频率最高的子词对=(b'\xc4\xa0a', b'b')->b'\xc4\xa0ab'，频率=2
第20迭代：频率最高的子词对=(b'\xc4\xa0token', b'i')->b'\xc4\xa0tokeni'，频率=2
第21迭代：频率最高的子词对=(b'\xc4\xa0tokeni', b'z')->b'\xc4\xa0tokeniz'，频率=2
第22迭代：频率最高的子词对=(b'a', b't')->b'at'，频率=2
第23迭代：频率最高的子词对=(b'i', b'o')->b'io'，频率=2
最终词表(280): [b'<|endoftext|>', b'\x00', b'\x01', b'\x02', ...,  b'\xc4\xa0tokeni', b'\xc4\xa0tokeniz', b'at', b'io']
最终学习了23条合并规则.
预测输出【字节】: [b'This', b'\xc4\xa0is', b'\xc4\xa0the', ..., b'\xc4\xa0', b'C', b'ou', b'r', b'se', b'.']
预测输出【文本】: ['This', 'Ġis', 'Ġthe', 'Ġ', ... 'Ġ', 'C', 'ou', 'r', 'se', '.']
```

### 通义千问使用的分词方式

Qwen 使用基于字节的BPE (`BBPE`) 对UTF-8编码的文本进行处理。它开始时将每个字节视为一个 token ，然后迭代地将文本中最频繁出现的 token 对合并成更大的 token，直到达到所需的词表大小。
在基于字节的BPE中，至少需要256个 token 来对每段文本进行 tokenization，并避免未登录词（out of vocabulary, OOV）问题。
基于字节的BPE的一个限制是，词表中的个别 token 可能看似没有语义意义，甚至不是有效的 UTF-8 字节序列，在某些方面，它们应该被视为一种文本压缩方案。

## 参考引用
[1] [transformers-BPE tokenization 算法](https://huggingface.co/learn/llm-course/zh-CN/chapter6/5?fw=pt)<br>
[2] [BPE分词原理](https://github.com/BrightXiaoHan/MachineTranslationTutorial/blob/master/tutorials/Chapter2/BPE.md)<br>
[3] [理解 Tokenizer 的工作原理与子词分割方法](https://github.com/Hoper-J/AI-Guide-and-Demos-zh_CN/blob/master/Guide/)<br>
[4] [Subword Tokenization 算法](https://www.huaxiaozhuan.com/%E5%B7%A5%E5%85%B7/huggingface_transformer/chapters/1_tokenizer.html)<br>
[5] [论文分享 Neural machine Translation of Rare Words with Subword Units](https://blog.csdn.net/Mr_tyting/article/details/91352726)<br>
[6] [subword-nmt](https://github.com/rsennrich/subword-nmt/blob/master/learn_bpe.py)<br>
[7] [LLM大语言模型之Tokenization分词方法（WordPiece，Byte-Pair Encoding (BPE)，Byte-level BPE(BBPE)原理及其代码实现）](https://zhuanlan.zhihu.com/p/652520262)<br>
[8] [通义千问 (Qwen)](https://qwen.readthedocs.io/zh-cn/latest/getting_started/concepts.html)<br>
