#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2025/4/22 16:10
# @function:
import torch
from modules.attention_func import scaled_dot_product_attention


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
