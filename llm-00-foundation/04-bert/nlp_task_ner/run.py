#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2025/5/10 14:38
# @function:
import hydra
import logging
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig


@hydra.main(version_base=None, config_path="conf", config_name="config.yaml")
def main(config: DictConfig):
    # print(OmegaConf.to_yaml(HydraConfig.get()))
    logging.info("Running BERT CRF")
    logging.info("\n" + OmegaConf.to_yaml(config))


if __name__ == '__main__':
    main()
