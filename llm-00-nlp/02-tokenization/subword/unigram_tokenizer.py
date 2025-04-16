#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2025/4/10 15:43
# @function:
import copy
from collections import defaultdict
from typing import Dict, Tuple, List, Iterable, Union
from unigram_viterbi import viterbi


def init_vocab(corpus: Iterable, max_vocab_size: int = 300) -> Tuple[Dict[str, int], Dict[str, list]]:
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
    token_freqs = list(char_freqs.items()) + sorted_subwords[:max_vocab_size - len(char_freqs)]
    token_freqs = {token: freq for token, freq in token_freqs}

    print(f"初始词表({len(token_freqs)}): ", token_freqs)
    return word_freqs, token_freqs


def compute_loss(model: Dict[str, float], word_freqs: Dict[str, int]) -> float:
    """
    计算语料库上的分词损失
    """
    loss = 0
    for word, freq in word_freqs.items():
        _, word_loss = viterbi(word, model)
        loss += freq * word_loss
    return loss


def compute_scores(model: Dict[str, float], word_freqs: Dict[str, int]):
    """
    计算每个子词的分数：移除该子词后模型对数似然的下降量。
    """
    scores = {}
    model_loss = compute_loss(model, word_freqs)
    for token, score in model.items():
        # 我们将保留长度为 1 的 tokens
        if len(token) == 1:
            continue
        model_without_token = copy.deepcopy(model)
        _ = model_without_token.pop(token)
        scores[token] = compute_loss(model_without_token, word_freqs) - model_loss
    return scores


def tokenize(sentence: Iterable, model: Dict[str, float]):
    encoded_words = [viterbi(word, model)[0] for word in sentence]
    return sum(encoded_words, [])

