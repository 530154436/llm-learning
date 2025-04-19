#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2025/4/10 15:43
# @function:
from collections import defaultdict
from typing import Dict, Tuple, List, Iterable


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
