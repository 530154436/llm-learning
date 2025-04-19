#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2025/4/14 13:45
# @function:
from transformers import GPT2TokenizerFast, AutoTokenizer
from data.download_model import DATA_DIR


def prepare_corpus(model: str = "openai-community/gpt2", use_cache: bool = True):
    """
    预分词：
    openai-community/gpt2
    google-bert/bert-base-cased
    xlnet-base-cased:
    """

    # 已预分词的结果
    if use_cache:
        if model == "openai-community/gpt2":
            pre_tokens = [
                ['This', 'Ġis', 'Ġthe', 'ĠHugging', 'ĠFace', 'ĠCourse', '.'],
                ['This', 'Ġchapter', 'Ġis', 'Ġabout', 'Ġtokenization', '.'],
                ['This', 'Ġsection', 'Ġshows', 'Ġseveral', 'Ġtokenizer', 'Ġalgorithms', '.'],
                ['Hopefully', ',', 'Ġyou', 'Ġwill', 'Ġbe', 'Ġable', 'Ġto', 'Ġunderstand', 'Ġhow', 'Ġthey', 'Ġare', 'Ġtrained', 'Ġand', 'Ġgenerate', 'Ġtokens', '.']
            ]

        elif model == "google-bert/bert-base-cased":
            pre_tokens = [
                ['This', 'is', 'the', 'Hugging', 'Face', 'Course', '.'],
                ['This', 'chapter', 'is', 'about', 'tokenization', '.'],
                ['This', 'section', 'shows', 'several', 'tokenizer', 'algorithms', '.'],
                ['Hopefully', ',', 'you', 'will', 'be', 'able', 'to', 'understand', 'how', 'they', 'are', 'trained', 'and', 'generate', 'tokens', '.']
            ]
        elif model == "xlnet-base-cased":
            pre_tokens = [
                ['▁This', '▁is', '▁the', '▁Hugging', '▁Face', '▁Course.'],
                ['▁This', '▁chapter', '▁is', '▁about', '▁tokenization.'],
                ['▁This', '▁section', '▁shows', '▁several', '▁tokenizer', '▁algorithms.'],
                ['▁Hopefully,', '▁you', '▁will', '▁be', '▁able', '▁to', '▁understand', '▁how', '▁they', '▁are', '▁trained', '▁and', '▁generate', '▁tokens.']
            ]
        else:
            raise NotImplementedError(f"该模型({model})不支持读取缓存数据")
        for tokens in pre_tokens:
            yield tokens

    corpus = [
        "This is the Hugging Face Course.",
        "This chapter is about tokenization.",
        "This section shows several tokenizer algorithms.",
        "Hopefully, you will be able to understand how they are trained and generate tokens.",
    ]
    # 使用gpt2 标准化+预分词
    filepath = DATA_DIR.joinpath(model)
    tokenizer: GPT2TokenizerFast = AutoTokenizer.from_pretrained(filepath)

    # 计算语料库中每个单词的频率
    for text in corpus:
        # [('This', (0, 4)), ('Ġis', (4, 7)), ('Ġthe', (7, 11))..]，Ġ是空空格
        words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
        sentence = []
        for word, offset in words_with_offsets:
            sentence.append(word)
        yield sentence


if __name__ == "__main__":
    # _sentences = prepare_corpus("openai-community/gpt2")
    # _sentences = prepare_corpus("google-bert/bert-base-cased")
    _sentences = prepare_corpus("xlnet-base-cased", use_cache=False)
    for _sentence in _sentences:
        print(_sentence)
