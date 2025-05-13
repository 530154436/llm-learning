#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2025/5/10 14:38
# @function:
import hydra
import logging
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="conf", config_name="config.yaml")
def main(config: DictConfig):
    logging.info(f"开始训练模型:{config.model.name}, 数据集: {config.dataset.name}")
    print(config)
    print(type(config.dataset), config.dataset)
    print(type(config.model), config.model)
    # TODO


if __name__ == '__main__':
    main()
