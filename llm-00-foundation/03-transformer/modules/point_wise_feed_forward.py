#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2025/4/22 19:34
# @function:
import torch
import torch.nn as nn


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        位置前馈网络。
        $$
        \text{FFN}(x_i) = \text{max}(0, x_i W_1 + b_1) W_2 + b_2
        $$

        :param d_model: 输入和输出向量的维度
        :param d_ff: FFN 隐藏层的维度，或者说中间层
        :param dropout: 随机失活率（Dropout），即随机屏蔽部分神经元的输出，用于防止过拟合

        （实际上论文并没有确切地提到在这个模块使用 dropout，所以注释）
        """
        super(PositionWiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)  # 第一个线性层
        self.w_2 = nn.Linear(d_ff, d_model)  # 第二个线性层
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_2(self.dropout(self.w_1(x).relu()))
