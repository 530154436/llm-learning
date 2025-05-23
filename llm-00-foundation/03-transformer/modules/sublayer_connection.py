#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2025/4/22 20:01
# @function:
import torch
from torch import nn


class ResidualConnection(nn.Module):
    def __init__(self, dropout=0.1):
        """
       残差连接，用于在每个子层后添加残差连接和 Dropout。

       残差连接是一种跳跃连接（Skip Connection），它将层的输入直接加到子层的输出上，对应的公式如下：
        $$
        \text{Output} = \text{SubLayer}(x) + x
        $$

       :param dropout: Dropout 概率，用于在残差连接前应用于子层输出，防止过拟合。
       """
        super(ResidualConnection, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, sublayer_output: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。

        :param x: 子层模块的输入张量，形状为 (batch_size, seq_len, d_model)。
        :param sublayer_output: 子层模块(注意力机制、前馈网络)的输出张量 (batch_size, seq_len, d_model)
        :return 经过残差连接和 Dropout 处理后的张量，形状为 (batch_size, seq_len, d_model)。
        """
        return x + self.dropout(sublayer_output)


class LayerNorm(nn.Module):

    def __init__(self, d_model, epsilon=1e-9):
        """
        BatchNorm（层归一化）
        LayerNorm 基于每个样本的所有特征，针对样本自身（行内所有特征）进行归一化，即在每一行（一个样本的 embed_size 个特征）上计算均值和方差。

        :param d_model: 输入特征的维度大小，即归一化的特征维度。
        :param epsilon: 防止除零的小常数。
        """
        super(LayerNorm, self).__init__()
        self.d_model = d_model
        self.epsilon = epsilon

        self.gamma = nn.Parameter(torch.ones(d_model))  # (d_model, )
        self.beta = nn.Parameter(torch.zeros(d_model))  # (d_model, )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.epsilon) + self.beta


class SublayerConnection(nn.Module):
    def __init__(self, d_model, dropout=0.1, epsilon=1e-9):
        """
        Add & Norm（残差连接和层归一化，也称为子层连接），包括残差连接和层归一化，应用于 Transformer 的每个子层。

        **操作步骤**：
        1. **残差连接**：将输入直接与子层的输出相加。
        2. **层归一化**：对相加后的结果进行归一化。

        :param d_model: 输入特征的维度大小，即归一化的特征维度。
        :param dropout: 残差连接中的 Dropout 概率。
        :param epsilon: 防止除零的小常数。
        """
        super(SublayerConnection, self).__init__()
        self.residual = ResidualConnection(dropout)   # 使用 ResidualConnection 进行残差连接
        self.norm = LayerNorm(d_model, epsilon)       # 层归一化

    def forward(self, x: torch.Tensor, sublayer_output: torch.Tensor) -> torch.Tensor:
        """
        将子层输出应用 dropout 后经过残差连接后再进行归一化
        :param x: 子层模块的输入张量，形状为 (batch_size, seq_len, d_model)。
        :param sublayer_output: 子层模块(注意力机制、前馈网络)的输出张量 (batch_size, seq_len, d_model)
        :return: (batch_size, seq_len, d_model)
        """
        return self.norm(self.residual(x, sublayer_output))
