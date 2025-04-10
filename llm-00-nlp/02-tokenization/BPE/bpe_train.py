#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2025/4/10 15:43
# @function:
# 流程：
# 标准化
# 预分词
# 将单词拆分为单个字符
# 根据学习的合并规则，按顺序合并拆分的字符
from data.download_model import DATA_DIR
from transformers import AutoTokenizer, GPT2TokenizerFast
from collections import defaultdict
from bpe_tokenizer import compute_pair_freqs, find_max_freq_pair, merge_pair


def prepare_corpus():
    corpus = [
        "This is the Hugging Face Course.",
        "This chapter is about tokenization.",
        "This section shows several tokenizer algorithms.",
        "Hopefully, you will be able to understand how they are trained and generate tokens.",
    ]
    # 使用gpt2 标准化+预分词
    filepath = DATA_DIR.joinpath("openai-community/gpt2")
    tokenizer: GPT2TokenizerFast = AutoTokenizer.from_pretrained(filepath)
    word_freqs = defaultdict(int)     # 单词-频率

    # 计算语料库中每个单词的频率
    for text in corpus:
        # [('This', (0, 4)), ('Ġis', (4, 7)), ('Ġthe', (7, 11))..]，Ġ是空空格
        words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
        for word, offset in words_with_offsets:
            word_freqs[word] += 1
    return word_freqs


def prepare_corpus_v2():
    corpus = [
        "我",
        "喜欢",
        "吃",
        "苹果",
        "他",
        "不",
        "喜欢",
        "吃",
        "苹果派",
        "I like to eat apples",
        "She has a cute cat",
        "you are very cute",
        "give you a hug",
    ]
    # 计算语料库中每个单词的频率
    word_freqs = defaultdict(int)  # 单词-频率
    for sentence in corpus:
        for word in sentence.split():
            word_freqs[word] += 1
    return word_freqs


def main(vocab_size: int = 50):
    # word_freqs = prepare_corpus()
    word_freqs = prepare_corpus_v2()

    word_splits = defaultdict(list)  # 单词-对应切分的子词序列
    alphabet = []

    # 将每个单词拆分为字符序列，并计算初始词表
    for word in word_freqs.keys():
        word_splits[word] = list(word)
        for letter in word:
            if letter not in alphabet:
                alphabet.append(letter)
    alphabet.sort()
    # 对于 GPT-2，唯一的特殊 tokens 是 "<|endoftext|>"
    # [',', '.', 'C', 'F', 'H', 'T', 'a'...]
    vocab = ["<|endoftext|>"] + alphabet.copy()
    print("初始词表: ", vocab)

    # 训练BPE
    merges = dict()
    iteration = 1
    while len(vocab) < vocab_size:
        pair_freqs = compute_pair_freqs(word_splits, word_freqs)
        best_pair, max_freq = find_max_freq_pair(pair_freqs)
        word_splits = merge_pair(*best_pair, word_splits)
        merges[best_pair] = best_pair[0] + best_pair[1]
        vocab.append(best_pair[0] + best_pair[1])
        print(f"第{iteration}迭代：频率最高的子词对={best_pair}，频率={max_freq}")
        iteration += 1
    print(f"最终词表({len(vocab)}): {vocab}")
    print(f"最终学习了{len(merges)}条合并规则: {merges}")


if __name__ == "__main__":
    main()
