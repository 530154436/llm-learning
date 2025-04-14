#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2025/4/10 15:43
# @function:
from corpus import prepare_corpus
from unigram_tokenizer import init_vocab, encode_word, tokenize


def main(vocab_size: int = 70):
    # 初始化
    sentences = prepare_corpus("xlnet-base-cased")
    word_freqs, token_freqs = init_vocab(sentences)

    # # 训练
    # iteration = 1
    # while len(vocab) < vocab_size:
    #     letter_freqs, pair_freqs, scores = compute_pair_score(word_splits, word_freqs)
    #     best_pair, max_score = find_max_score_pair(scores)
    #     word_splits = merge_pair(*best_pair, word_splits)
    #     new_token = (
    #         best_pair[0] + best_pair[1][2:]
    #         if best_pair[1].startswith("##")
    #         else best_pair[0] + best_pair[1]
    #     )
    #     vocab.append(new_token)
    #     print(f"第{iteration}迭代：得分最高的子词对={best_pair}->{new_token}，得分={max_score}")
    #     iteration += 1
    # print(f"最终词表({len(vocab)}): {vocab}")
    #
    # # 预测
    # # ['This', 'is', 'the', 'Hugging', 'Face', 'Course', '.']
    # sentence = list(prepare_corpus("google-bert/bert-base-cased"))[0]
    # # ['Th', '##i', '##s', 'is', 'th', '##e', 'Hugg', '##i', '##n', '##g', 'Fac', '##e', 'C', '##o', '##u', '##r', '##s', '##e', '.']
    # predict = tokenize(sentence=sentence, vocab=vocab)
    # print(f"预测输出: {predict}")


if __name__ == "__main__":
    main(vocab_size=70)
