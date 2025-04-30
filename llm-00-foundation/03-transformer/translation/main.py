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

    # https://github.com/intro-llm/intro-llm-code/blob/main/chs/ch2-foundations/Transformer/main.py

    # https://zhuanlan.zhihu.com/p/347061440
    # https://link.zhihu.com/?target=https%3A//github.com/hemingkx/ChineseNMT

    # https://github.com/hyunwoongko/transformer/blob/master/train.py
    # https://github.com/Hoper-J/AI-Guide-and-Demos-zh_CN/blob/master/PaperNotes/Transformer%20%E8%AE%BA%E6%96%87%E7%B2%BE%E8%AF%BB.md#q2-%E4%BB%80%E4%B9%88%E6%98%AF%E8%87%AA%E5%9B%9E%E5%BD%92%E4%B8%8E%E9%9D%9E%E8%87%AA%E5%9B%9E%E5%BD%92

    # https://nlp.seas.harvard.edu/annotated-transformer/#part-1-model-architecture
    # https://github.com/mcxiaoxiao/annotated-transformer-Chinese/tree/main


if __name__ == '__main__':
    train()
