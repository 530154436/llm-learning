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
                                 mask: torch.Tensor = None,
                                 dropout=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    缩放点积注意力计算。

    给定查询矩阵 $Q$、键矩阵 $K$ 和值矩阵 $V$, 其注意力输出的数学表达式如下：
    $$
    \text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V
    $$

    单头注意力：
    Q: 查询矩阵 (batch_size, seq_len_q, embed_size)
    K: 键矩阵 (batch_size, seq_len_k, embed_size)
    V: 值矩阵 (batch_size, seq_len_v, embed_size)
    mask: 掩码矩阵 (batch_size, seq_len_q, seq_len_k)

    多头注意力：
    Q: 查询矩阵 (batch_size, num_heads, seq_len_q, head_dim)
    K: 键矩阵 (batch_size, num_heads, seq_len_k, head_dim)
    V: 值矩阵 (batch_size, num_heads, seq_len_v, head_dim)
    mask: 掩码矩阵 (batch_size, 1, 1, seq_len)

    :return
        output: 注意力加权后的输出矩阵 (batch_size, seq_len_q, embed_size)
                                   (batch_size, num_heads, seq_len_q, head_dim)
        attention_weights: 注意力权重矩阵 (batch_size, seq_len_q, seq_len_k)
                                       (batch_size, num_heads, seq_len_q, seq_len_k)
    """
    # 键或查询向量的维度
    d_k = torch.tensor(K.size(dim=-1), dtype=torch.float32)

    # 计算点积并进行缩放
    # Q     :       (..., seq_len_q, embed_size)
    # K^\top:       (..., embed_size, seq_len_k)
    # Q * K^\top:   (..., seq_len_q, seq_len_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(d_k)

    # 如果提供了掩码矩阵，则将掩码对应位置的分数设为 -inf
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # 对缩放后的分数应用 Softmax 函数，得到注意力权重
    attention_weights = F.softmax(scores, dim=-1)
    if dropout is not None:
        attention_weights = dropout(attention_weights)

    # 加权求和，计算输出, k和v的输入都是一样的(seq_len_k==seq_len_v)
    # attention_weights: (..., seq_len_q, seq_len_k)
    # V:                 (..., seq_len_v, embed_size)
    # output:            (..., seq_len_q, embed_size)
    # 自注意力机制（Self-Attention）： q=k=v（q、k、v分别表示Q、K、V的输入序列）
    # 交叉注意力（Cross-Attention）： q≠k=v
    output = torch.matmul(attention_weights, V)

    return output, attention_weights

