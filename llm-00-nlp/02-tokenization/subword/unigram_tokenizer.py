#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2025/4/10 15:43
# @function:
from collections import defaultdict
from typing import Dict, Tuple, List, Iterable, Union


def init_vocab(corpus: Iterable) -> Tuple[Dict[str, int], Dict[str, list]]:
    """
    计算初始词表
    """
    word_freqs = defaultdict(int)       # 单词-频率
    char_freqs = defaultdict(int)       # 字符-频率
    subwords_freqs = defaultdict(int)   # 子词-频率

    # 计算语料库中每个单词的频率
    for sentence in corpus:
        for word in sentence:
            word_freqs[word] += 1

    # 计算语料库中每个字符、子词的频率
    for word, freq in word_freqs.items():
        for i in range(len(word)):
            char_freqs[word[i]] += freq
            # 循环遍历长度至少为2的子词
            for j in range(i + 2, len(word) + 1):
                subwords_freqs[word[i:j]] += freq

    # 按频率对子词排序，用最优的子词对字符进行分组，获得大小为 300 的初始词汇表。
    sorted_subwords = sorted(subwords_freqs.items(), key=lambda x: x[1], reverse=True)
    token_freqs = list(char_freqs.items()) + sorted_subwords[:300 - len(char_freqs)]
    token_freqs = {token: freq for token, freq in token_freqs}

    print(f"初始词表({len(token_freqs)}): ", token_freqs)
    return word_freqs, token_freqs


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
