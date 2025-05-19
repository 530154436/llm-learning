#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2025/5/10 14:38
# @function:
import logging
import hydra
import torch
import tqdm
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from typing import Dict, Callable
from nlp_task_ner.data_loader import NERDataset
from nlp_task_ner.model.bert_crf import BertCrf
from nlp_task_ner.util.modeling_util import count_trainable_parameters


def train_epoch(data: DataLoader[Dict[str, torch.Tensor]],
                model: nn.Module,
                criterion: Callable,
                optimizer: Optimizer = None,
                epoch: int = 1,
                epoch_num: int = None) -> float:
    model.train()
    desc = "train" if optimizer is not None else "eval"
    epoch_loss = 0.0
    tk0 = tqdm.tqdm(data, desc=f"{desc} {epoch}/{epoch_num}", smoothing=0, mininterval=1.0)
    for i, item in enumerate(tk0, start=1):
        # 模型预测

        logits = y_pred.contiguous().view(-1, y_pred.size(-1))
        y_true = batch.tgt_output.contiguous().view(-1)

        # 计算损失
        optimizer.zero_grad()
        loss = criterion(logits, y_true)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        tk0.set_postfix(loss=round(epoch_loss / i, 5))
    return epoch_loss / len(data)


def train(model: nn.Module,
          train_dataloader: DataLoader,
          dev_dataloader: DataLoader,
          learning_rate: float,
          ):
    # 初始化模型权重，定义优化器、学习率调整器、损失函数
    optimizer = AdamW(model.parameters(), lr=learning_rate)
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


@hydra.main(version_base=None, config_path="conf", config_name="config.yaml")
def main(config: DictConfig):
    logging.info(f"开始训练模型")
    logging.info("配置信息:\n{}".format(OmegaConf.to_yaml(config, resolve=True)))
    logging.info("加载Dataset和Tokenizer.")
    _tokenizer = BertTokenizer.from_pretrained(
        pretrained_model_name_or_path=config.model.pretrain_path,
        do_lower_case=True
    )
    train_dataset = NERDataset(config.dataset.train_data_path, config.dataset.label_data_path, tokenizer=_tokenizer)
    dev_dataset = NERDataset(config.dataset.dev_data_path, config.dataset.label_data_path, tokenizer=_tokenizer)
    test_dataset = NERDataset(config.dataset.test_data_path, config.dataset.label_data_path, tokenizer=_tokenizer)

    logging.info("加载DataLoader")
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=config.model.batch_size,
                                  collate_fn=train_dataset.collate_fn)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=config.model.batch_size,
                                collate_fn=dev_dataset.collate_fn)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=config.model.batch_size,
                                 collate_fn=test_dataset.collate_fn)
    # logging.info("初始化模型")
    model = BertCrf(pretrain_path=config.model.pretrain_path,
                    num_labels=config.dataset.label_data_path,
                    dropout=config.model.dropout)
    print(model)
    logging.info(f'模型训练参数: {count_trainable_parameters(model)}')
    logging.info("训练模型...")
    train(model, train_dataloader, dev_dataloader, learning_rate=config.model.lr)


if __name__ == '__main__':
    main()
