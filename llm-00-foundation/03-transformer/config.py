#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2025/4/27 19:30
# @function:
import os
from pathlib import Path
import torch
from util import logger

data_dir = Path(__file__).parent.joinpath('data')

# 设置日志格式
LOGGER = logger.get_logger(path=f'{data_dir}/log')

# 数据集
train_data_path = f'{data_dir}/dataset/train.jsonl'
dev_data_path = f'{data_dir}/dataset/dev.jsonl'
test_data_path = f'{data_dir}/dataset/test.jsonl'

# model_path = f'{data_dir}/experiment/model.pth'
model_path = f'{data_dir}/experiment/model-0502.pth'

for sub_dir in ["dataset", "experiment", "log"]:
    _dir = f'{data_dir}/{sub_dir}'
    if not os.path.exists(_dir):
        os.mkdir(_dir)

# 模型配置
d_model = 512
n_heads = 8
n_layers = 6
d_k = 64
d_v = 64
d_ff = 2048

src_vocab_size = 32000
tgt_vocab_size = 32000
dropout = 0.1
padding_idx = 0
bos_idx = 2
eos_idx = 3

batch_size = 32
epoch_num = 20
early_stop = 5
lr = 3e-4
warmup_steps = 50

# 解码配置
max_len = 60
# beam size for bleu
beam_size = 3
# Label Smoothing
use_smoothing = False
# NoamOpt
use_noamopt = False

# gpu_id and device id is the relative id
# thus, if you wanna use os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'
# you should set CUDA_VISIBLE_DEVICES = 2 as main -> gpu_id = '0', device_id = [0, 1]
gpu_id = '0'
device_id = [0, 1]

# set device
if torch.cuda.is_available():
    device = torch.device(f"cuda:{gpu_id}")
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
