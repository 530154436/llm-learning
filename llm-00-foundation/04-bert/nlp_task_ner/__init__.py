#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2025/5/6 16:29
# @function:
import logging

from omegaconf import OmegaConf, DictConfig
from pathlib import Path

BASE_DIR = Path(__file__).parent

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(filename)s[:%(lineno)d] %(message)s')


# # 加载 YAML 配置
# config: DictConfig = OmegaConf.load("data/conf/config.yaml")
# print(config)
# print(config.data_dir)
