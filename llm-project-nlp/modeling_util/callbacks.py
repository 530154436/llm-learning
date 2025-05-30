#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2025/5/27 11:18
# @function:
import copy


class EarlyStopper(object):
    """Early stops the training if validation loss doesn't improve after a given patience.

    Args:
        patience (int): How long to wait after last time validation auc improved.
    """

    def __init__(self, patience, delta: float = 0, mode: str = 'min'):
        self.patience = patience
        self.delta = delta

        # 初始化
        self.no_improvement_count = 0
        self.mode = mode
        self.best_value = 0 if mode == 'max' else 999999
        self.best_weights = None

    def stop_training(self, value, weights) -> bool:
        """whether to stop training.

        Args:
            value (float): auc score in val data. auc、loss
            weights (tensor): the weights of model
        """
        if (value > self.best_value and self.mode == 'max') or \
                (value < self.best_value and self.mode == 'min'):
            self.best_value = value
            self.no_improvement_count = 0
            self.best_weights = copy.deepcopy(weights)
            return False
        elif self.no_improvement_count + 1 < self.patience:
            self.no_improvement_count += 1
            return False
        else:
            return True
