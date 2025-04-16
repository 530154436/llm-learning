#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2025/4/10 15:43
# @function:
from math import log
from corpus import prepare_corpus
from unigram_tokenizer import init_vocab, compute_scores, tokenize


def main(vocab_size: int = 100, percent_to_remove: float = 0.1):
    # 构建初始词汇表
    sentences = prepare_corpus("xlnet-base-cased")
    word_freqs, token_freqs = init_vocab(sentences)

    # 训练: EM算法迭代优化概率
    # 初始化子词概率
    total_sum = sum([freq for token, freq in token_freqs.items()])
    model = {token: -log(freq / total_sum) for token, freq in token_freqs.items()}
    iteration = 1
    while len(model) > vocab_size:
        scores = compute_scores(model, word_freqs)
        sorted_scores = sorted(scores.items(), key=lambda x: x[1])  # loss减少量=移除后loss-移除前loss
        # 删除分数最低（loss减少量最小）的percent_to_remov tokens 。
        remove_sub_words = []
        for i in range(int(len(model) * percent_to_remove)):
            remove_sub_word = sorted_scores[i][0]
            token_freqs.pop(remove_sub_word)
            remove_sub_words.append(remove_sub_word)

        print(f"第{iteration}迭代：删除子词（{len(remove_sub_words)}）={remove_sub_words}")
        # 更新子词概率
        total_sum = sum([freq for token, freq in token_freqs.items()])
        model = {token: -log(freq / total_sum) for token, freq in token_freqs.items()}

        iteration += 1
    print(f"最终词表({len(model)}): {list(model.keys())}")

    # 预测
    # ['▁This', '▁is', '▁the', '▁Hugging', '▁Face', '▁Course.']
    sentence = ['▁This', '▁is', '▁the', '▁Hugging', '▁Face', '▁course.']
    predict = tokenize(sentence=sentence, model=model)
    # ['▁This', '▁is', '▁the', '▁Hugging', '▁Face', '▁', 'c', 'ou', 'r', 's', 'e', '.']
    print(f"预测输出: {predict}")


if __name__ == "__main__":
    main(vocab_size=70)
