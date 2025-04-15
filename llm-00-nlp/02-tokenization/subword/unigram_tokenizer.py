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


def encode_word(word: str, model: Dict[str, float]) -> Tuple[list, int]:
    """
    为词的每一个位置（从 0 开始，一直到词的总长度）都保存一个字典，字典里有两个键：最好的分割中最后一个词的起始位置，以及最好的分割的得分。
    一个主循环用来遍历每个可能的开始位置，第二个循环则试着找出所有以这个开始位置开始的子串。

    Character 0 (u): "u" (score 0.171429)
    Character 1 (n): "un" (score 0.076191)
    Character 2 (h): "un" "h" (score 0.005442)
    Character 3 (u): "un" "hu" (score 0.005442)
    Character 4 (g): "un" "hug" (score 0.005442)
    """
    best_segmentations = \
        [{"start": 0, "score": 1}] \
        + [{"start": None, "score": None} for _ in range(len(word))]

    for start_idx in range(len(word)):
        # best_score_at_start应该由循环的前面的步骤计算和填充
        best_score_at_start = best_segmentations[start_idx]["score"]
        print(f"start_idx={start_idx}: ", end="")
        for end_idx in range(start_idx + 1, len(word) + 1):
            token = word[start_idx: end_idx]
            if token in model and best_score_at_start is not None:
                score = model[token] + best_score_at_start
                # 如果我们发现以 end_idx 结尾的更好分段,我们会更新
                if (
                    best_segmentations[end_idx]["score"] is None
                    or best_segmentations[end_idx]["score"] > score
                ):
                    best_segmentations[end_idx] = {"start": start_idx, "score": score}

    segmentation = best_segmentations[-1]
    if segmentation["score"] is None:
        # 我们没有找到单词的 tokens  -> unknown
        return ["<unk>"], None

    score = segmentation["score"]
    start = segmentation["start"]
    end = len(word)
    tokens = []
    while start != 0:
        tokens.insert(0, word[start:end])
        next_start = best_segmentations[start]["start"]
        end = start
        start = next_start
    tokens.insert(0, word[start:end])
    return tokens, score
