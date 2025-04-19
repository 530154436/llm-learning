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
