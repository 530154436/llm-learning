#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2025/4/29 10:41
# @function:
import time
from functools import partial

import torch
import tqdm
from typing import Callable
from torch import nn
from torch.optim import AdamW, lr_scheduler, Optimizer
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from modules.models import Transformer
from translation import config
from translation.config import LOGGER
from translation.data_loader import MTDataset, MTBatch


def initialize_weights(model: nn.Module):
    """ 初始化模型权重
    """
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
            # nn.init.kaiming_uniform_(p)

def count_trainable_parameters(model: nn.Module):
    """ 计算模型参数
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_epoch(data: DataLoader[MTBatch],
              model: Transformer,
              optimizer: Optimizer,
              criterion: Callable):
    model.train()
    epoch_loss = 0.0
    tk0 = tqdm.tqdm(data, desc="train", smoothing=0, mininterval=1.0)
    for i, item in enumerate(tk0, start=1):
        # 模型预测
        batch: MTBatch = item
        y_pred = model(batch.src_input, batch.tgt_input, batch.src_mask, batch.tgt_mask)
        y_pred = y_pred.contiguous().view(-1, y_pred.size(-1))
        y_true = batch.tgt_output.contiguous().view(-1)

        # 计算损失
        optimizer.zero_grad()
        loss = criterion(y_pred, y_true)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        tk0.set_postfix(loss=round(epoch_loss / i, 5))
    return epoch_loss / len(data)


@torch.no_grad()
def evaluate(data: DataLoader[MTBatch],
             model: Transformer,
             criterion: Callable):
    """ 在data上用训练好的模型进行预测，打印模型翻译结果
    """
    model.eval()
    epoch_loss = 0
    tk0 = tqdm.tqdm(data, desc="eval", smoothing=0, mininterval=1.0)
    for i, item in enumerate(tk0, start=1):
        # 计算loss
        batch: MTBatch = item
        y_pred = model(batch.src_input, batch.tgt_input, batch.src_mask, batch.tgt_mask)
        y_pred = y_pred.contiguous().view(-1, y_pred.size(-1))
        y_true = batch.tgt_output.contiguous().view(-1)
        loss = criterion(y_pred, y_true)
        # 计算blue分值

        epoch_loss += loss.item()
        tk0.set_postfix(loss=round(epoch_loss / i, 5))
    return epoch_loss / len(data)

def train(model: nn.Module,
          train_dataloader: DataLoader,
          dev_dataloader: DataLoader):
    # 初始化模型权重，定义优化器、学习率调整器、损失函数
    model.apply(initialize_weights)
    optimizer = AdamW(model.parameters(), lr=config.lr)
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer=optimizer)
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                num_warmup_steps=config.warmup_steps,
                                                num_training_steps=len(train_dataloader) * config.epoch_num)
    criterion = nn.CrossEntropyLoss(ignore_index=config.padding_idx)

    start = time.time()
    model.to(config.device)
    for epoch in range(1, config.epoch_num + 1):
        # 模型训练
        train_loss = run_epoch(train_dataloader, model, optimizer, criterion)
        # 验证集
        valid_loss = evaluate(dev_dataloader, model, criterion)

        current_lr = optimizer.state_dict()['param_groups'][0]['lr']
        LOGGER.info("Epoch: {}, train_loss: {}, valid_loss: {}, lr: {}"
                    .format(epoch, round(train_loss, 6), round(valid_loss, 6), round(current_lr, 6)))
        scheduler.step()



def main():
    LOGGER.info("加载Dataset和Tokenizer.")
    train_dataset = MTDataset(config.train_data_path)
    dev_dataset = MTDataset(config.dev_data_path)
    test_dataset = MTDataset(config.test_data_path)
    LOGGER.info("加载DataLoader")
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size,
                                  collate_fn=partial(train_dataset.collate_fn, device=config.device))
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=config.batch_size,
                                collate_fn=partial(train_dataset.collate_fn, device=config.device))
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=config.batch_size,
                                 collate_fn=partial(train_dataset.collate_fn, device=config.device))

    LOGGER.info("初始化模型")
    model = Transformer(config.src_vocab_size, config.tgt_vocab_size,
                        d_model=config.d_model, num_heads=config.n_heads, d_ff=config.d_ff,
                        dropout=config.dropout, N=config.n_layers)
    # print(model)
    LOGGER.info(f'模型训练参数: {count_trainable_parameters(model)}')
    LOGGER.info("训练模型...")
    train(model, train_dataloader, dev_dataloader)


if __name__ == '__main__':
    main()
