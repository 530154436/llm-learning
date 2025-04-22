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
    mask: 掩码矩阵 (batch_size, num_heads, seq_len_q, seq_len_k)

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

    # 加权求和，计算输出
    # attention_weights: (..., seq_len_q, seq_len_k)
    # V:                 (..., seq_len_v, embed_size)
    # output:            (..., seq_len_q, embed_size)
    # 自注意力机制（Self-Attention）： q=k=v（q、k、v分别表示Q、K、V的输入序列）
    # 交叉注意力（Cross-Attention）： q≠k=v
    output = torch.matmul(attention_weights, V)

    return output, attention_weights


def test_single_head():
    # 示例：“I love dogs” 被拆分为三个词元（tokens）：
    # 词元列表：["I", "love", "dogs"]
    # Q/K/V 的维度均为 (batch_size=1, seq_len=3, embed_size=4)
    # 每个词元对应的向量：（随机生成）：
    # "I"    → [0.21, -0.53, 0.72, 0.15]
    # "love" → [-0.12, 0.84, 0.03, -0.42]
    # "dogs" → [0.34, -0.11, 0.89, 0.07]

    # 1、q=k=v
    _seq_len_q, _seq_len_k, _seq_len_v, _dim_size = 3, 3, 3, 4
    _Q = torch.randn(1, _seq_len_q, _dim_size)
    _K = torch.randn(1, _seq_len_k, _dim_size)
    _V = torch.randn(1, _seq_len_v, _dim_size)
    _output, _attention_weights = scaled_dot_product_attention(_Q, _K, _V)
    # torch.Size([1, 3, 4]) torch.Size([1, 3, 3])
    print(_output.shape, _attention_weights.shape)

    # 2、q≠k=v
    _seq_len_q, _seq_len_k, _seq_len_v, _dim_size = 8, 3, 3, 4
    _Q = torch.randn(1, _seq_len_q, _dim_size)
    _output, _attention_weights = scaled_dot_product_attention(_Q, _K, _V)
    # torch.Size([1, 8, 4]) torch.Size([1, 8, 3])
    print(_output.shape, _attention_weights.shape)


def test_multi_head():
    # 多头注意力
    # 1、q=k=v
    _head_num, _seq_len_q, _seq_len_k, _seq_len_v, _dim_size = 2, 3, 3, 3, 4
    _Q = torch.randn(1, _head_num, _seq_len_q, _dim_size)
    _K = torch.randn(1, _head_num, _seq_len_k, _dim_size)
    _V = torch.randn(1, _head_num, _seq_len_v, _dim_size)
    _output, _attention_weights = scaled_dot_product_attention(_Q, _K, _V)
    # torch.Size([1, 2, 3, 4]) torch.Size([1, 2, 3, 3])
    print(_output.shape, _attention_weights.shape)

    # 2、q≠k=v
    _head_num, _seq_len_q, _seq_len_k, _seq_len_v, _dim_size = 2, 3, 3, 3, 4
    _Q = torch.randn(1, _head_num, _seq_len_q, _dim_size)
    _output, _attention_weights = scaled_dot_product_attention(_Q, _K, _V)
    # torch.Size([1, 2, 8, 4]) torch.Size([1, 2, 8, 3])
    print(_output.shape, _attention_weights.shape)

    # 生成下三角掩码矩阵 (1, 1, seq_len_q, seq_len_k)，通过广播应用到所有头
    # mask.shape (seq_len_q, seq_len_k) -> (1, 1, seq_len_q, seq_len_k)
    mask = torch.tril(torch.ones(_seq_len_q, _seq_len_k))
    _output, _attention_weights = scaled_dot_product_attention(_Q, _K, _V, mask=mask)

    # tensor([[1., 0., 0.],
    #         [1., 1., 0.],
    #         [1., 1., 1.]])
    # tensor([[[[1.0000, 0.0000, 0.0000],
    #           [0.5896, 0.4104, 0.0000],
    #           [0.0632, 0.0905, 0.8463]],
    #
    #          [[1.0000, 0.0000, 0.0000],
    #           [0.0269, 0.9731, 0.0000],
    #           [0.2973, 0.1114, 0.5913]]]])
    print(_output.shape, _attention_weights.shape)
    print(mask)
    print(_attention_weights)


if __name__ == '__main__':
    test_single_head()
    test_multi_head()
