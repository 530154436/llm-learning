#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2024/11/9 13:49
# @function:
import sys
from pathlib import Path
from modelscope import snapshot_download

if sys.platform == 'win32':
    DATA_DIR = Path('./')
else:
    DATA_DIR = Path('./')

# snapshot_download("AI-ModelScope/bge-large-zh-v1.5",
#                   revision='master',
#                   cache_dir='models')
# snapshot_download("Qwen/Qwen2.5-7B-Instruct",
#                   revision='master',
#                   cache_dir='models')

# snapshot_download("google-bert/bert-base-uncased",
#                   revision='master',
#                   cache_dir=DATA_DIR)
# snapshot_download("google-bert/bert-base-cased",
#                   revision='master',
#                   cache_dir=DATA_DIR)
# snapshot_download("google-bert/bert-base-chinese",
#                   revision='master',
#                   cache_dir=DATA_DIR)

# snapshot_download("dienstag/chinese-roberta-wwm-ext",
#                   revision='master',
#                   cache_dir=DATA_DIR)

# snapshot_download("openai-community/gpt2",
#                   revision='master',
#                   cache_dir=DATA_DIR)
# snapshot_download("AI-ModelScope/t5-small",
#                   revision='master',
#                   cache_dir=DATA_DIR)
snapshot_download("Qwen/Qwen2.5-0.5B-Instruct",
                  revision='master',
                  cache_dir=DATA_DIR)
