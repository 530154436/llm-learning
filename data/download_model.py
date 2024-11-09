#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2024/11/9 13:49
# @function:
from modelscope import snapshot_download


snapshot_download("AI-ModelScope/bge-large-zh-v1.5",
                  revision='master',
                  cache_dir='models')

snapshot_download("Qwen/Qwen2.5-7B-Instruct",
                  revision='master',
                  cache_dir='models')
