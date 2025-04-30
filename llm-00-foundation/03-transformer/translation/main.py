#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2025/4/29 10:41
# @function:
from torch.utils.data import DataLoader
from modules.models import Transformer
from translation import config
from translation.config import LOGGER
from translation.data_loader import MTDataset


def train():
    LOGGER.info("加载Dataset和Tokenizer.")

    train_dataset = MTDataset(config.train_data_path)
    dev_dataset = MTDataset(config.dev_data_path)
    test_dataset = MTDataset(config.test_data_path)

    LOGGER.info("加载DataLoader")
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size,
                                  collate_fn=train_dataset.collate_fn)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=config.batch_size,
                                collate_fn=dev_dataset.collate_fn)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=config.batch_size,
                                 collate_fn=test_dataset.collate_fn)
    LOGGER.info("初始化模型")
    model = Transformer(config.src_vocab_size, config.tgt_vocab_size,
                        d_model=config.d_model, num_heads=config.n_heads, d_ff=config.d_ff,
                        dropout=config.dropout, N=config.n_layers)


if __name__ == '__main__':
    train()
