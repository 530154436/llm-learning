#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2025/4/21 16:56
# @function:
from typing import Tuple
import torch
import torch.nn.functional as F


def scaled_dot_product_attention(Q: torch.Tensor,
                                 K: torch.Tensor,
                                 V: torch.Tensor,
                                 mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    缩放点积注意力计算。

    给定查询矩阵 $Q$、键矩阵 $K$ 和值矩阵 $V$, 其注意力输出的数学表达式如下：
    $$
    \text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V
    $$

    :param Q: 查询矩阵 (batch_size, seq_len_q, embed_size)
    :param K: 键矩阵 (batch_size, seq_len_k, embed_size)
    :param V: 值矩阵 (batch_size, seq_len_v, embed_size)
    :param mask: 掩码矩阵，用于屏蔽不应该关注的位置 (可选)
    :return
        output: 注意力加权后的输出矩阵
        attention_weights: 注意力权重矩阵
    """
    # 键或查询向量的维度
    d_k = torch.tensor(K.size(dim=-1), dtype=torch.float32)

    # 计算点积并进行缩放
    # Q     :       (batch_size, embed_size, seq_len_k)
    # K^\top:       (batch_size, seq_len_q, embed_size)
    # Q * K^\top:   (batch_size, seq_len_q, seq_len_k)
    scores = F.softmax(torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(d_k), dim=-1)

    # 如果提供了掩码矩阵，则将掩码对应位置的分数设为 -inf
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # 对缩放后的分数应用 Softmax 函数，得到注意力权重
    attention_weights = F.softmax(scores, dim=-1)

    # 加权求和，计算输出
    # attention_weights: (batch_size, seq_len_q, seq_len_k)
    # V:                 (batch_size, seq_len_v, embed_size)
    # output:            (batch_size, seq_len_q, embed_size)
    # 自注意力机制（Self-Attention）： q=k=v
    # 交叉注意力（Cross-Attention）： q≠k=v
    output = torch.matmul(attention_weights, V)

    return output, attention_weights


if __name__ == '__main__':
    # 示例：“I love dogs” 被拆分为三个词元（tokens）：
    # 词元列表：["I", "love", "dogs"]
    # 对应索引：  0      1      2
    # Q/K/V 的维度均为 (batch_size=1, seq_len=3, embed_size=4)
    # 每个词元对应的向量：（随机生成）：
    # "I"    → Q向量：[0.21, -0.53, 0.72, 0.15]
    # "love" → Q向量：[-0.12, 0.84, 0.03, -0.42]
    # "dogs" → Q向量：[0.34, -0.11, 0.89, 0.07]

    # 1、q=k=v
    _Q = torch.randn(1, 3, 4)
    _K = torch.randn(1, 3, 4)
    _V = torch.randn(1, 3, 4)
    _output, _attention_weights = scaled_dot_product_attention(_Q, _K, _V)
    # [1, 3, 4], [1, 3, 3]
    print(_output.shape, _attention_weights.shape)

    # 2、q≠k=v
    _Q = torch.randn(1, 8, 4)
    _output, _attention_weights = scaled_dot_product_attention(_Q, _K, _V)
    # [1, 8, 4], [1, 8, 3]
    print(_output.shape, _attention_weights.shape)
