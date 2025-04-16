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
        通过动态规划（DP）维护每个位置的最优分词路径及累积概率，结合子词概率进行计算。
        数组dp[i]: 表示到达位置 i 时的最大概率。
        数组path[i]: 表示到位置 i 时的分词路径。例如，path[i+1]=j表示以j为开始下标、i为结束下标的词元。
    具体步骤如下：
        初始化：dp[0] 初始概率为0，路径为空。
        递推计算：遍历每个位置 i，尝试所有可能的前驱位置 j，计算候选词的条件概率。
        路径记录：保存最大概率对应的分词路径。
        回溯最优解：最终 dp[n] 中的路径即为最优分词结果。
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
    _model = {'h': 3.45, 't': 3.01, 'u': 3.85, 'g': 4.04, 'o': 3.08, 'th': 4.55, 'ou': 4.55}
    _optimal_split, _score = viterbi(_word, _model)
    print("最优切分方式:", _optimal_split)
    print("最小得分之和:", _score)

    _optimal_split, _score = viterbi_v1(_word, _model)
    print("最优切分方式:", _optimal_split)
    print("最小得分之和:", _score)
