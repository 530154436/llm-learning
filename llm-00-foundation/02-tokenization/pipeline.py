#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2025/4/8 9:15
# @function: 分词流程
from tokenizers.normalizers import BertNormalizer
from tokenizers.pre_tokenizers import BertPreTokenizer, ByteLevel, Sequence
from transformers import AutoTokenizer, BertTokenizerFast, GPT2TokenizerFast, T5TokenizerFast
from data.download_model import DATA_DIR

# -------------------------------------------------------------------------------------------
# 标准化（normalization）
# -------------------------------------------------------------------------------------------
filepath = DATA_DIR.joinpath("google-bert/bert-base-uncased")
tokenizer: BertTokenizerFast = AutoTokenizer.from_pretrained(filepath)
normalizer: BertNormalizer = tokenizer.backend_tokenizer.normalizer
print(f"bert-base-uncased 词汇表大小：{len(tokenizer.vocab)}")
# hello how are u?
print(normalizer.normalize_str("Héllò hôw are U?"))

filepath = DATA_DIR.joinpath("google-bert/bert-base-cased")
tokenizer: BertTokenizerFast = AutoTokenizer.from_pretrained(filepath)
normalizer: BertNormalizer = tokenizer.backend_tokenizer.normalizer
print(f"bert-base-cased 词汇表大小：{len(tokenizer.vocab)}")
# Héllò hôw are U?
print(normalizer.normalize_str("Héllò hôw are U?"))


# -------------------------------------------------------------------------------------------
# 预分词（Pre-tokenization）
# -------------------------------------------------------------------------------------------
filepath = DATA_DIR.joinpath("google-bert/bert-base-cased")
tokenizer: BertTokenizerFast = AutoTokenizer.from_pretrained(filepath)
pre_tokenizer: BertPreTokenizer = tokenizer.backend_tokenizer.pre_tokenizer
# [('Hello', (0, 5)), (',', (5, 6)), ('how', (7, 10)), ('are', (11, 14)), ('you', (16, 19)), ('?', (19, 20))]
print(pre_tokenizer.pre_tokenize_str("Hello, how are  you?"))


filepath = DATA_DIR.joinpath("openai-community/gpt2")
tokenizer: GPT2TokenizerFast = AutoTokenizer.from_pretrained(filepath)
pre_tokenizer: ByteLevel = tokenizer.backend_tokenizer.pre_tokenizer
print(f"gpt2 词汇表大小：{len(tokenizer.vocab)}")
# [('Hello', (0, 5)), (',', (5, 6)), ('Ġhow', (6, 10)), ('Ġare', (10, 14)), ('Ġ', (14, 15)), ('Ġyou', (15, 19)), ('?', (19, 20))]
print(pre_tokenizer.pre_tokenize_str("Hello, how are  you?"))


filepath = DATA_DIR.joinpath("AI-ModelScope/t5-small")
tokenizer: T5TokenizerFast = AutoTokenizer.from_pretrained(filepath)
print(f"t5-small 词汇表大小：{len(tokenizer.vocab)}")
pre_tokenizer: Sequence = tokenizer.backend_tokenizer.pre_tokenizer
# [('▁Hello,', (0, 6)), ('▁how', (7, 10)), ('▁are', (11, 14)), ('▁you?', (16, 20))]
print(pre_tokenizer.pre_tokenize_str("Hello, how are  you?"))
