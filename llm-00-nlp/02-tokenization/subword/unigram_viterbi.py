#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2025/4/16 14:16
# @function:
from typing import Dict, Tuple, List


def viterbi_v1(word: str, model: Dict[str, float]) -> Tuple[list, int]:
    """
    Viterbi 算法对单词进行分词
    """
    best_segmentations = \
        [{"start": 0, "score": 0}] \
        + [{"start": None, "score": None} for _ in range(len(word))]

    for start_idx in range(len(word)):
        # best_score_at_start应该由循环的前面的步骤计算和填充
        best_score_at_start = best_segmentations[start_idx]["score"]
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
    # print(best_segmentations)

    # 回溯
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


def viterbi(word: str, model: Dict[str, float]) -> Tuple[List[str], int]:
    """
    Viterbi算法（动态规划）
    核心思路：
        将问题转化为一个路径搜索问题，把字符串的每个位置看作一个节点，从字符串的起始位置开始，逐步尝试不同的子词切分，记录到达每个位置的最小得分和对应的切分方式。
    具体步骤如下：
        初始化一个数组 dp，用于记录到达每个位置的最小得分，以及一个数组 path 用于记录对应的切分方式。
        遍历字符串的每个位置，对于每个位置，尝试所有可能的子词切分，更新 dp 数组和 path 数组。
        最后，根据 path 数组回溯得到最优切分方式。
    :param word: 字符串
    :param model:  子词对应的概率
    :return:
    """
    n = len(word)
    dp = [float("inf")] * (n + 1)
    path = [-1] * (n + 1)
    dp[0] = 0

    # 动态规划求解
    for i in range(1, n + 1):
        for j in range(i):
            token = word[j: i]
            if token in model:
                score = dp[j] + model[token]
                if score < dp[i]:
                    dp[i] = score
                    path[i] = j
    if path[-1] == -1:
        return ["<unk>"], None

    # 回溯得到最优切分方式
    optimal_split = []
    i = n
    while i > 0:
        prev = path[i]
        optimal_split.append(word[prev:i])
        i = prev
    optimal_split.reverse()
    return optimal_split, dp[n]


if __name__ == "__main__":
    _word = "thought"
    _model = {'▁': 2.21, 'T': 4.55, 'h': 3.45, 'i': 3.08, 's': 3.08, 't': 3.01, 'e': 2.6, 'H': 4.95, 'u': 3.85,
              'g': 4.04, 'n': 3.25, 'F': 5.65, 'a': 3.16, 'c': 4.55, 'C': 5.65, 'o': 3.08, 'r': 3.45, '.': 4.26,
              'p': 4.95, 'b': 4.55, 'k': 4.55, 'z': 4.95, 'w': 4.55, 'v': 5.65, 'l': 3.7, 'm': 5.65, 'f': 5.65,
              'y': 4.55, ',': 5.65, 'd': 4.26, '▁t': 3.7, 'er': 4.04, '▁a': 4.04, '▁to': 4.26, 'en': 4.26,
              '▁This': 4.55, 'th': 4.55, 'ou': 4.55, '▁token': 4.55, 'ra': 4.55, 'nd': 4.55, '▁is': 4.95, '▁the': 4.95,
              '▁H': 4.95, 'in': 4.95, 'te': 4.95, '▁ab': 4.95, '▁tokeniz': 4.95, 'how': 4.95, 'era': 4.95, 'al': 4.95,
              's.': 4.95, 'll': 4.95, 'and': 4.95, '▁Hugging': 5.65, '▁Face': 5.65, '▁Course.': 5.65, '▁chapter': 5.65,
              '▁about': 5.65, '▁tokenization.': 5.65, '▁section': 5.65, '▁shows': 5.65, '▁several': 5.65,
              'sever': 5.65, 'severa': 5.65, 'several': 5.65}
    _optimal_split, _score = viterbi(_word, _model)
    print("最优切分方式:", _optimal_split)
    print("最小得分之和:", _score)

    _optimal_split, _score = viterbi_v1(_word, _model)
    print("最优切分方式:", _optimal_split)
    print("最小得分之和:", _score)
