#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2025/4/10 15:43
# @function:
from collections import defaultdict
from typing import Dict, Tuple, List, Iterable, Union


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
