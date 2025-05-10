#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2025/5/6 16:29
# @function:
from omegaconf import OmegaConf, DictConfig
from pathlib import Path
from util import logger


# 设置日志格式
LOGGER = logger.get_logger(path=f'{data_dir}/log')
BASE_DIR = Path(__file__).parent.joinpath('data')


# 加载 YAML 配置
config: DictConfig = OmegaConf.load("config/bert_crf.yaml")
print(config)
print(config.data_dir)