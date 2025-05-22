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
# from torch.nn import Transformer
from modules.models import Transformer
import config
from config import LOGGER
from data_loader import MTDataset, MTBatch
from util.modeling_util import initialize_weights, count_trainable_parameters


def run_epoch(data: DataLoader[MTBatch],
              model: Transformer,
              criterion: Callable,
              optimizer: Optimizer = None,
              epoch: int = 1) -> float:
    model.train() if optimizer is not None else model.eval()
    desc = "train" if optimizer is not None else "eval"
    epoch_loss = 0.0
    tk0 = tqdm.tqdm(data, desc=f"{desc} {epoch}/{config.epoch_num}", smoothing=0, mininterval=1.0)
    for i, item in enumerate(tk0, start=1):
        # 模型预测
        batch: MTBatch = item
        logits = model(batch.src_input, batch.tgt_input, batch.src_mask, batch.tgt_mask)

        # 转换为 nn.CrossEntropyLoss 的输入格式
        # [batch_size, tgt_seq_len, tgt_vocab_size] => [batch_size * tgt_seq_len, tgt_vocab_size]
        logits = logits.contiguous().view(-1, logits.size(-1))
        # [batch_size, tgt_seq_len] => [batch_size * tgt_seq_len]
        y_true = batch.tgt_output.contiguous().view(-1)

        # 计算损失
        if optimizer is not None:
            optimizer.zero_grad()
            loss = criterion(logits, y_true)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                loss = criterion(logits, y_true)

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
    criterion = nn.CrossEntropyLoss(ignore_index=config.padding_idx)  # => nn.LogSoftmax()+nn.NLLLoss()

    start = time.time()
    model.to(config.device)
    for epoch in range(1, config.epoch_num + 1):
        # 模型训练
        train_loss = run_epoch(train_dataloader, model, criterion, optimizer, epoch=epoch)
        # 验证集
        valid_loss = run_epoch(dev_dataloader, model, criterion, epoch=epoch)
        # blue_score = evaluate(dev_dataloader, model, criterion, epoch=epoch)

        current_lr = optimizer.state_dict()['param_groups'][0]['lr']
        LOGGER.info("Epoch: {}, train_loss: {}, valid_loss: {}, lr: {}"
                    .format(epoch, round(train_loss, 6), round(valid_loss, 6), round(current_lr, 6)))
        scheduler.step()

    # 保存模型
    torch.save(model.state_dict(), config.model_path)


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
