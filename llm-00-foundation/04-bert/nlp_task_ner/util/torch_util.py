#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2025/5/6 17:39
# @function:
from torch import nn


def initialize_weights(model: nn.Module):
    """ 初始化模型权重
    """
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)


def count_trainable_parameters(model: nn.Module):
    """ 计算模型参数
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
