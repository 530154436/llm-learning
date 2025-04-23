#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2025/4/23 9:53
# @function:
import math
import torch
from torch import nn


class Embedding(nn.Module):

    def __init__(self, vocab_size: int, d_model: int):
        """
        嵌入层，将 token ID 转换为固定维度的嵌入向量，并进行缩放。
        Embedding权值矩阵 (vocab_size, d_model)

        :param vocab_size: 词汇表大小。
        :param d_model: 嵌入向量的维度。
        """
        super(Embedding, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.scale_factor = math.sqrt(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。

        :param x: 输入张量，形状为 (batch_size, seq_len)，其中每个元素是 token ID。
        :return 缩放后的嵌入向量，形状为 (batch_size, seq_len, d_model)。
        """
        return self.embedding(x) * self.scale_factor
        # return self.embedding(x)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        Transformer 使用的是固定位置编码（Positional Encoding），其公式如下：
        $$
        \begin{aligned}
        PE_{(pos, 2i)} &= \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right), \\
        PE_{(pos, 2i+1)} &= \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right).
        \end{aligned}
        $$

        其中：
        - $pos$ 表示位置索引（Position）。
        - $i$ 表示维度索引。
        - $d_{\text{model}}$ 是嵌入向量的维度。

        :param d_model: 嵌入维度，即每个位置的编码向量的维度。
        :param dropout: 位置编码后应用的 Dropout 概率。
        :param max_len: 位置编码的最大长度，适应不同长度的输入序列。
        """
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)

        # unsqueeze: 对数据维度进行扩充，给指定位置加上维数为一的维度。
        # 位置索引：(max_len) => (max_len, 1), 即torch.Size([5000, 1])
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # 计算位置编码每个维度对应的频率(分母部分)
        # 其中0/1、2/3、...、d_model-2/d_model-1位置的频率相同，所以shape是(d_model/2) 即torch.Size([256])
        div_term = torch.exp(
            (torch.arange(0, d_model, 2) / d_model) * (-math.log(10000.0))
        )

        # 将位置和频率结合，计算 sin 和 cos
        # 广播机制 position * div_term: (max_len, d_model/2) 即torch.Size([5000, 256])
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度

        # 增加一个维度，方便后续与输入相加，形状变为 (1, max_len, d_model)
        pe = pe.unsqueeze(0)

        # 将位置编码注册为模型的缓冲区，不作为参数更新
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor):
        """
        前向传播函数。
        :param x: 输入序列的嵌入向量，形状为 (batch_size, seq_len, d_model)。
        :return 加入位置编码和 Dropout 后的嵌入向量，形状为 (batch_size, seq_len, d_model)。
        """
        # 取出与输入序列长度相同的对应位置编码，并与输入相加
        x = x + self.pe[:, :x.size(1), :]

        # 应用 dropout
        return self.dropout(x)
