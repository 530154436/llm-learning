#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2025/4/10 15:43
# @function:
from collections import defaultdict
from typing import Dict, Tuple, List


def compute_pair_freqs(word_splits: Dict[str, List[str]],
                       word_freq: Dict[str, int]) -> Dict[Tuple[str], int]:
    """
    统计相邻字符对的频次
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


def find_max_freq_pair(pair_freq: Dict[str, int]) -> Tuple[Tuple[str], int]:
    """
    找到频次最高的字符对和频率
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


def merge_pair(a: str, b: str, word_splits: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    选择频次最高的字符对并合并
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
                splits = splits[:i] + [a + b] + splits[i + 2:]  # 合并相邻字符对并生成新的子词序列
            else:
                i += 1
        word_splits[word] = splits
    return word_splits
