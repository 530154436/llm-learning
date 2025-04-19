#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2025/4/10 15:43
# @function:
# 流程：
# 将单词拆分为单个字符
# 根据学习的合并规则，按顺序合并拆分的字符
from collections import defaultdict
from typing import Dict, Tuple
from corpus import prepare_corpus
from bpe_tokenizer import init_vocab, compute_pair_freqs, find_max_freq_pair, merge_pair, tokenize


def main(vocab_size: int = 50):
    # 初始化
    word_freqs, word_splits, vocab = init_vocab(prepare_corpus(), special_tokens=["<|endoftext|>"])

    # 训练BPE
    merges: Dict[Tuple[str], str] = defaultdict(str)
    iteration = 1
    while len(vocab) < vocab_size:
        pair_freqs = compute_pair_freqs(word_splits, word_freqs)
        best_pair, max_freq = find_max_freq_pair(pair_freqs)
        word_splits = merge_pair(*best_pair, word_splits)
        merges[best_pair] = best_pair[0] + best_pair[1]
        vocab.append(best_pair[0] + best_pair[1])
        print(f"第{iteration}迭代：频率最高的子词对={best_pair}->{''.join(best_pair)}，频率={max_freq}")
        iteration += 1
    print(f"最终词表({len(vocab)}): {vocab}")
    print(f"最终学习了{len(merges)}条合并规则.")
    # for k, v in merges.items():
    #     print(f"    {k} -> {v}""")

    # 预测
    sentence = list(prepare_corpus())[0]
    predict = tokenize(sentence=sentence, merges=merges)
    print(f"预测输出: {predict}")


if __name__ == "__main__":
    main()
