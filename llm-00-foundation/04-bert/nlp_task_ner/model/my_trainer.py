#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2025/5/22 11:09
# @function:
import logging
import tqdm
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from nlp_task_ner.model import BaseModel
from nlp_task_ner.model.my_loss_func import CRFLoss


class MyTrainer(object):
    """A general trainer for single task learning.

    Args:
        model (nn.Module): any multi task learning model.
        optimizer (torch.optim): optimizer function of pytorch (default = `torch.optim.Adam`).
        scheduler (torch.optim.lr_scheduler) : torch scheduling class, eg. `torch.optim.lr_scheduler.StepLR`.
        n_epoch (int): epoch number of training.
        early_stop_patience (int): how long to wait after last time validation auc improved (default=10).
        device (str): `"cpu"` or `"cuda:0"`
        model_path (str): the path you want to save the model (default="./"). Note only save the best weight in the validation data.
    """

    def __init__(
        self,
        model: BaseModel,
        loss_fn: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        n_epoch: int = 10,
        early_stop_patience=10,
        device: str = "cpu",
        model_path: str = "model.pth",
        metrics=None
    ):
        self.model: BaseModel = model.to(device)
        self.device = torch.device(device)
        self.n_epoch = n_epoch
        self.model_path = model_path

        # 配置：损失函数、优化器、学习率调度器(差分学习率)
        self.loss_fn = loss_fn
        self.optimizer: Optimizer = optimizer
        self.scheduler: LRScheduler = scheduler

    def train_one_epoch(self, data_loader, log_interval: int = None, epoch: int = 1) -> float:
        self.model.train()
        step, total_loss = 1, 0
        tk0 = tqdm.tqdm(data_loader, desc=f"train {epoch}/{self.n_epoch}", smoothing=0, mininterval=1.0)
        for step, xy in enumerate(tk0, start=1):
            xy_tuple = tuple(x.to(self.device) for x in xy)

            # Logits：是模型最后一层未经任何激活函数变换的输出。
            # 序列标注任务: [batch_size, seq_len, num_classes]
            # 分类任务：[batch_size, num_classes]
            logits: torch.Tensor = self.model(*xy_tuple[:-1])  # 不传label
            y_true: torch.Tensor = xy_tuple[-1]
            # print(logits.shape, y_true.shape)

            if isinstance(self.loss_fn, CRFLoss):
                # CRF 的情况比较特殊
                loss = self.loss_fn(logits=logits, labels=y_true, input_ids=xy_tuple[0])
            else:
                # 非CRF情况：展平成 CrossEntropyLoss 要求的格式
                # 序列标注任务:
                # [batch_size, seq_len, num_classes] => [batch_size * seq_len, num_classes]
                # [batch_size, seq_len] => [batch_size * seq_len]
                # 分类任务：
                # [batch_size, num_classes]
                # [batch_size]
                logits = logits.contiguous().view(-1, logits.size(-1))
                y_true = y_true.contiguous().view(-1)
                loss = self.loss_fn(logits, y_true)

            self.model.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()  # 更新lr

            total_loss += loss.item()
            if isinstance(log_interval, int) and (step + 1) % log_interval == 0:
                tk0.set_postfix(loss=round(total_loss/step, 5))
        return round(total_loss / step, 5)

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader):
        self.model.eval()
        step, total_loss = 1, 0
        for step, xy in enumerate(data_loader, start=1):
            xy_tuple = tuple(x.to(self.device) for x in xy)
            logits: torch.Tensor = self.model(*xy_tuple[:-1])  # 不传label
            y_true: torch.Tensor = xy_tuple[-1]

            # 计算 loss
            if isinstance(self.loss_fn, CRFLoss):
                loss = self.loss_fn(logits=logits, labels=y_true, input_ids=xy_tuple[0])
            else:
                logits = logits.contiguous().view(-1, logits.size(-1))
                y_true = y_true.contiguous().view(-1)
                loss = self.loss_fn(logits, y_true)

            total_loss += loss.item()
        return round(total_loss / step, 5)

    def fit(self,
            train_dataloader: DataLoader,
            val_dataloader: DataLoader,
            test_dataloader: DataLoader = None) -> float:
        for epoch in range(1, self.n_epoch + 1):
            lr = self.scheduler.get_lr()[0]  # 获取当前的学习率
            train_loss = self.train_one_epoch(train_dataloader, epoch=epoch)
            val_loss = self.evaluate(val_dataloader)
            logging.info(f'epoch: {epoch}, Current lr : {round(lr, 6)}, train_loss: {train_loss}, val_loss: {val_loss}')

            # if self.early_stopper.stop_training(val_loss, self.model.state_dict(), mode='min'):
            #     self.model.load_state_dict(self.early_stopper.best_weights)
            #     logging.info('Current loss: %.6f, Best Value: %.6f\n' % (val_loss, self.early_stopper.best_value))
            #     break
        torch.save(self.model.state_dict(), self.model_path)
        # logging.info('Saved model\'s loss: %.6f' % self.early_stopper.best_value)
