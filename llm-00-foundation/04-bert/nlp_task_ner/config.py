#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2025/5/10 15:30
# @function:
import json
import os
from pathlib import Path
from util import logger

data_dir = Path(__file__).parent.joinpath('data')

# 设置日志格式
LOGGER = logger.get_logger(path=f'{data_dir}/log')

# 数据集
train_data_path = f'{data_dir}/dataset/train.jsonl'
dev_data_path = f'{data_dir}/dataset/dev.jsonl'
test_data_path = f'{data_dir}/dataset/test.jsonl'

# 实体标签
label_data_path = f'{data_dir}/dataset/label.json'
label2id = json.load(open(label_data_path, 'r'))
id2label = {_id: _label for _label, _id in list(label2id.items())}

# 模型
bert_model = f'{data_dir}/pretrained/bert-base-chinese/'
roberta_model = f'{data_dir}/pretrained/chinese_roberta_wwm_large_ext/'
model_path = f'{data_dir}/experiment/ner.pth'

for sub_dir in ["dataset", "experiment", "log", "pretrained"]:
    _dir = f'{data_dir}/{sub_dir}'
    if not os.path.exists(_dir):
        os.mkdir(_dir)

