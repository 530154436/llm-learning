
### WordPiece
WordPiece 是谷歌内部非公开的分词算法，最初是由谷歌研究人员在开发语音搜索系统时提出的。
随后，在 2016 年被用于机器翻译系统，并于 2018 年被 BERT 采用作为分词器。

WordPiece核心思想是将单词拆分成`多个前缀符号`（比如BERT中的##，将前缀添加到单词内的每个字符，单词的首字符不添加前缀）最小单元，再通过子词合并规则将最小单元进行合并为子词级别。例如对于单词"word"，拆分如下：
```
w ##o ##r ##d
```
WordPiece 分词和 BPE 分词的想法非常相似，都是通过迭代合并连续的词元，但是合并的选择标准略有不同。 与 BPE 方法的另一个不同点在于，WordPiece 分词算法并不选择最频繁的词对，而是使用下面的公式为每个词对计算分数（两个子词之间的互信息，条件概率）：

$$
得分=\frac{\text{子词对出现的频率}}{\text{第一个子词出现的频率}\times \text{第二个子词出现的频率}} \\
$$


该算法的训练步骤：
1. **初始化词汇表 $V$**：
   - 语料库标准化、预分词后，统计所有单词的频率 -> word_freqs
   - 将所有单词拆分成子词序列：通过添加前缀（如 BERT 中的 ## ）来识别子词，例如 "word"→["w","##o","##r","##d"]-> word_splits
   - 构建初始词表：特殊标记["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]+英文中26个字母+各种符号以及常见中文字符-> $V$
2. **统计单个子词、相邻子词对的频次和计算 Score**：
   - 对于每个单词的子词序列，统计单个子词、相邻子词对（两个连续的子词）的出现频次，并根据定义计算得分。
3. **合并得分最高的子词对并更新词汇表**：
   - 选择出现得分最高的子词对 $(x, y)$，将其合并为新符号 $xy$。<br>
     如果第二个子词号以 ## 开头，合并时去掉 "##" 前缀再进行连接。<br>
     新符号是否以 ## 开头，取决于第一个子词是否以 "##" 开头。<br>
   - 将新符号添加到词汇表 $V = V \cup \{xy\}$。
4. **重复步骤 2 到 4**：
   - 重复统计和合并过程，直到满足停止条件（例如，词汇表达到预定大小）。

示例：
```
假设语料库预分词后得到的单词及其频次的集合：
("hug", 10), ("pug", 5), ("pun", 12), ("bun", 4), ("hugs", 5) 

一.初始化词汇表
-将所有单词拆分成子词序列
("h" "##u" "##g", 10), ("p" "##u" "##g", 5), ("p" "##u" "##n", 12), ("b" "##u" "##n", 4), ("h" "##u" "##g" "##s", 5)

-构建初始词表
["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "b", "h", "p", "##g", "##n", "##s", "##u"]

二. 统计单个子词、相邻子词对的频次和计算 Score；合并得分最高的子词对并更新词汇表：
计算出 pair("##g", "##s")的分数最高为(5)/(20*5) = 1/20，所以最先合并的pair是("##g", "##s")→("##gs")。
此时词表和拆分后的的频率将变成以下：
词汇表: ["b", "h", "p", "##g", "##n", "##s", "##u", "##gs"]
语料库: ("h" "##u" "##g", 10), ("p" "##u" "##g", 5), ("p" "##u" "##n", 12), ("b" "##u" "##n", 4), ("h" "##u" "##gs", 5)

重复上述操作
```

代码实现：
```python
from collections import defaultdict
from typing import Dict, Tuple, List, Iterable, Union
from transformers import GPT2TokenizerFast, AutoTokenizer


def init_vocab(corpus: Iterable,
               special_tokens: List[str] = None) -> Tuple[Dict[str, int], Dict[str, list], list]:
    """
    计算初始词表
    :param corpus: 语料库(标准化、预分词)
    :param special_tokens: 特殊token标记
        对于 GPT-2，唯一的特殊 tokens 是 "<|endoftext|>"
        对于 Bert，特殊tokens 是 "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"
    :return:
    """
    word_freqs = defaultdict(int)    # 单词-频率
    word_splits = defaultdict(list)  # 单词-对应切分的子词序列
    vocab = special_tokens if special_tokens else []  #

    # 计算语料库中每个单词的频率
    for sentence in corpus:
        for word in sentence:
            word_freqs[word] += 1

    # wordpiece通过添加前缀（如 BERT 中的 ## ）来识别子词，例如 "word" 将被这样分割：w ##o ##r ##d
    for word in word_freqs.keys():
        for i, char in enumerate(word):
            letter = char if i == 0 else f"##{char}"
            word_splits.setdefault(word, []).append(letter)
            if letter not in vocab:
                vocab.append(letter)

    print(f"初始词表({len(vocab)}): ", vocab)
    return word_freqs, word_splits, vocab


def compute_pair_score(word_splits: Dict[str, List[str]],
                       word_freq: Dict[str, int]) -> tuple:
    """
    统计单个子词、相邻子词对的频次和计算 Score。

    score=(freq_of_pair)/(freq_of_first_element×freq_of_second_element)
    :param word_splits: 单词对应的子词序列, {'This': ['T','##h','##i','##s']}
    :param word_freq: 单词对应的频次, {'This': 3}
    :return 字符对和频次
    """
    letter_freqs = defaultdict(int)
    pair_freqs = defaultdict(int)

    # 统计频率：单个子词、相邻子词对
    for word, splits in word_splits.items():
        for letter in splits:
            letter_freqs[letter] += word_freq[word]
        if len(splits) == 1:
            continue
        for i in range(len(splits) - 1):
            pair = (splits[i], splits[i + 1])
            pair_freqs[pair] += word_freq[word]
    # 计算分数
    scores = {
        pair: freq / (letter_freqs[pair[0]] * letter_freqs[pair[1]])
        for pair, freq in pair_freqs.items()
    }

    return letter_freqs, pair_freqs, scores


def find_max_score_pair(pair_freq: Dict[Tuple[str], int]) -> Tuple[Tuple[str], int]:
    """
    找到得分最高的子词对和得分
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
    选择得分最高的子词对并合并
    :param a: 字符对开始
    :param b: 字符对结束
    :param word_splits: 单词对应的子词序列, {'This': ['T','##h','##i','##s']}
    :return: 单词的合并后子词序列
    """
    for word in word_splits:
        splits = word_splits[word]
        if len(splits) == 1:
            continue
        i = 0
        while i < len(splits) - 1:
            if splits[i] == a and splits[i + 1] == b:
                # 如果第二个子词号以 ## 开头，合并时去掉 "##" 前缀再进行连接
                merge = a + b[2:] if b.startswith("##") else a + b
                splits = splits[:i] + [merge] + splits[i + 2:]
            else:
                i += 1
        word_splits[word] = splits
    return word_splits


def encode_word(word: str, vocab: list) -> str:
    """
    将一个单词编码为基于词汇表 vocab 的一系列子词（subword）标记。
    如果词汇表中没有完全匹配的单词，则尝试找到最长的前缀进行匹配，并将剩余部分继续分割，直到整个单词被处理完毕。
    如果没有找到任何匹配项（即找不到已知的前缀），则返回一个特殊的未知标记 "[UNK]"。
    """
    tokens = []
    while len(word) > 0:
        i = len(word)
        while i > 0 and word[:i] not in vocab:
            i -= 1
        if i == 0:
            return ["[UNK]"]
        tokens.append(word[:i])  # 找到的最长前缀
        word = word[i:]  # 后半部分需要增加"##"前缀
        if len(word) > 0:
            word = f"##{word}"
    return tokens


def tokenize(sentence: Iterable, vocab: list) -> List[str]:
    """
    使用学到的所有合并规则进行分词
    :param sentence:  句子
    :param vocab:  子词表
    :return:
    """
    encoded_words = [encode_word(word, vocab) for word in sentence]
    return sum(encoded_words, [])


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

        
def main(vocab_size: int = 70):
    # 初始化
    sentences = prepare_corpus("google-bert/bert-base-cased")
    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    word_freqs, word_splits, vocab = init_vocab(sentences, special_tokens=special_tokens)

    # 训练
    iteration = 1
    while len(vocab) < vocab_size:
        letter_freqs, pair_freqs, scores = compute_pair_score(word_splits, word_freqs)
        best_pair, max_score = find_max_score_pair(scores)
        word_splits = merge_pair(*best_pair, word_splits)
        new_token = (
            best_pair[0] + best_pair[1][2:]
            if best_pair[1].startswith("##")
            else best_pair[0] + best_pair[1]
        )
        vocab.append(new_token)
        print(f"第{iteration}迭代：得分最高的子词对={best_pair}->{new_token}，得分={max_score}")
        iteration += 1
    print(f"最终词表({len(vocab)}): {vocab}")

    # 预测
    # ['This', 'is', 'the', 'Hugging', 'Face', 'Course', '.']
    sentence = list(prepare_corpus("google-bert/bert-base-cased"))[0]
    # ['Th', '##i', '##s', 'is', 'th', '##e', 'Hugg', '##i', '##n', '##g', 'Fac', '##e', 'C', '##o', '##u', '##r', '##s', '##e', '.']
    predict = tokenize(sentence=sentence, vocab=vocab)
    print(f"预测输出: {predict}")
```

输出结果：
```
初始词表(45):  ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', 'T', '##h', '##i', '##s', 'i', 't', '##e', 'H', 
               '##u', '##g', '##n', 'F', '##a', '##c', 'C', '##o', '##r', '.', 'c', '##p', '##t', 'a', '##b',
               '##k', '##z', 's', '##w', '##v', '##l', '##m', '##f', '##y', ',', 'y', 'w', 'b', 'u', '##d', 'h', 'g']
第1迭代：得分最高的子词对=('a', '##b')->ab，得分=0.2
第2迭代：得分最高的子词对=('##f', '##u')->##fu，得分=0.2
第3迭代：得分最高的子词对=('F', '##a')->Fa，得分=0.14285714285714285
第4迭代：得分最高的子词对=('Fa', '##c')->Fac，得分=0.5
第5迭代：得分最高的子词对=('##c', '##t')->##ct，得分=0.14285714285714285
第6迭代：得分最高的子词对=('##fu', '##l')->##ful，得分=0.14285714285714285
第7迭代：得分最高的子词对=('##ful', '##l')->##full，得分=0.16666666666666666
第8迭代：得分最高的子词对=('##full', '##y')->##fully，得分=0.5
第9迭代：得分最高的子词对=('T', '##h')->Th，得分=0.125
第10迭代：得分最高的子词对=('c', '##h')->ch，得分=0.2
第11迭代：得分最高的子词对=('##h', '##m')->##hm，得分=0.25
第12迭代：得分最高的子词对=('ch', '##a')->cha，得分=0.16666666666666666
第13迭代：得分最高的子词对=('cha', '##p')->chap，得分=0.5
第14迭代：得分最高的子词对=('chap', '##t')->chapt，得分=0.16666666666666666
第15迭代：得分最高的子词对=('##t', '##hm')->##thm，得分=0.2
第16迭代：得分最高的子词对=('H', '##u')->Hu，得分=0.125
第17迭代：得分最高的子词对=('Hu', '##g')->Hug，得分=0.25
第18迭代：得分最高的子词对=('Hug', '##g')->Hugg，得分=0.3333333333333333
第19迭代：得分最高的子词对=('s', '##h')->sh，得分=0.1111111111111111
第20迭代：得分最高的子词对=('t', '##h')->th，得分=0.14285714285714285
第21迭代：得分最高的子词对=('i', '##s')->is，得分=0.1
第22迭代：得分最高的子词对=('##thm', '##s')->##thms，得分=0.125
第23迭代：得分最高的子词对=('##z', '##a')->##za，得分=0.1
第24迭代：得分最高的子词对=('##za', '##t')->##zat，得分=0.25
第25迭代：得分最高的子词对=('##u', '##t')->##ut，得分=0.1111111111111111
最终词表(70): ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', 'T', '##h', '##i', '##s', 'i', 't', '##e', 'H', 
               '##u', '##g', '##n', 'F', '##a', '##c', 'C', '##o', '##r', '.', 'c', '##p', '##t', 'a', '##b', 
               '##k', '##z', 's', '##w', '##v', '##l', '##m', '##f', '##y', ',', 'y', 'w', 'b', 'u', '##d', 'h', 
               'g', 'ab', '##fu', 'Fa', 'Fac', '##ct', '##ful', '##full', '##fully', 'Th', 'ch', '##hm', 'cha', 
               'chap', 'chapt', '##thm', 'Hu', 'Hug', 'Hugg', 'sh', 'th', 'is', '##thms', '##za', '##zat', '##ut']
预测输出: ['Th', '##i', '##s', 'is', 'th', '##e', 'Hugg', '##i', '##n', '##g', 'Fac', '##e', 'C', '##o', '##u',
         '##r', '##s', '##e', '.']
```

## 参考引用
[1] [transformers-WordPiece tokenization 算法](https://huggingface.co/learn/llm-course/zh-CN/chapter6/6?fw=pt)<br>
[2] [理解 Tokenizer 的工作原理与子词分割方法](https://github.com/Hoper-J/AI-Guide-and-Demos-zh_CN/blob/master/Guide/)<br>
[3] [Subword Tokenization 算法](https://www.huaxiaozhuan.com/%E5%B7%A5%E5%85%B7/huggingface_transformer/chapters/1_tokenizer.html)<br>
[4] [LLM大语言模型之Tokenization分词方法（WordPiece，Byte-Pair Encoding (BPE)，Byte-level BPE(BBPE)原理及其代码实现）](https://zhuanlan.zhihu.com/p/652520262)<br>
