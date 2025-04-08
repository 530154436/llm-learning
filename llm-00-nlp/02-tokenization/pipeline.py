#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2025/4/8 9:15
# @function: 分词流程
from tokenizers.normalizers import BertNormalizer
from tokenizers.pre_tokenizers import BertPreTokenizer
from transformers import AutoTokenizer, BertTokenizerFast, GPT2TokenizerFast
from data.download_model import DATA_DIR

# -------------------------------------------------------------------------------------------
# 标准化（normalization）
# -------------------------------------------------------------------------------------------
filepath = DATA_DIR.joinpath("google-bert/bert-base-uncased")
tokenizer: BertTokenizerFast = AutoTokenizer.from_pretrained(filepath)
normalizer: BertNormalizer = tokenizer.backend_tokenizer.normalizer
# hello how are u?
print(normalizer.normalize_str("Héllò hôw are U?"))

filepath = DATA_DIR.joinpath("google-bert/bert-base-cased")
tokenizer: BertTokenizerFast = AutoTokenizer.from_pretrained(filepath)
normalizer: BertNormalizer = tokenizer.backend_tokenizer.normalizer
# Héllò hôw are U?
print(normalizer.normalize_str("Héllò hôw are U?"))


# -------------------------------------------------------------------------------------------
# 预分词（Pre-tokenization）
# -------------------------------------------------------------------------------------------
pre_tokenizer: BertPreTokenizer = tokenizer.backend_tokenizer.pre_tokenizer
# [('Hello', (0, 5)), (',', (5, 6)), ('how', (7, 10)), ('are', (11, 14)), ('you', (16, 19)), ('?', (19, 20))]
print(pre_tokenizer.pre_tokenize_str("Hello, how are  you?"))


filepath = DATA_DIR.joinpath("openai-community/gpt2")
tokenizer: GPT2TokenizerFast = AutoTokenizer.from_pretrained(filepath)
# [('Hello', (0, 5)), (',', (5, 6)), ('Ġhow', (6, 10)), ('Ġare', (10, 14)), ('Ġyou', (14, 18)), ('?', (18, 19))]
print(tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str("Hello, how are  you?"))


filepath = DATA_DIR.joinpath("AI-ModelScope/t5-small")
tokenizer: GPT2TokenizerFast = AutoTokenizer.from_pretrained(filepath)
#
print(type(tokenizer))
print(tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str("Hello, how are  you?"))
