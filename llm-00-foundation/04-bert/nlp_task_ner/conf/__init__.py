#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2025/5/10 22:29
# @function:
from pathlib import Path

from transformers import RobertaTokenizer

BASE_DIR = Path(__file__).parent.parent
# print(BASE_DIR)
tokenizer = RobertaTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')