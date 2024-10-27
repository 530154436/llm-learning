#!/usr/bin/env python3
# -*- coding:utf-8 -*--
from modelscope import snapshot_download
model_dir = snapshot_download("AI-ModelScope/bge-large-zh-v1.5",
                              revision='master',
                              cache_dir='models')
