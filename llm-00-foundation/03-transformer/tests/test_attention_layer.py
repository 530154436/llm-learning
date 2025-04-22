#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2025/4/22 16:14
# @function:
import torch
import torch.nn as nn
from modules.attention_layer import SelfAttention, CrossAttention, MultiHeadAttention, MultiHeadAttentionV1


def test_self_attention_layer():
    embed_size = 128
    batch_size = 32
    sequence_length = 10
    input_tensor = torch.randn(batch_size, sequence_length, embed_size)
    out, attention_weights = SelfAttention(embed_size).forward(input_tensor)
    # torch.Size([32, 10, 128]) torch.Size([32, 10, 10])
    print(out.shape, attention_weights.shape)


def test_cross_attention_layer():
    embed_size = 128
    batch_size = 32
    seq_len_q = 20
    seq_len_x = 10
    q = torch.randn(batch_size, seq_len_q, embed_size)
    x = torch.randn(batch_size, seq_len_x, embed_size)
    out, attention_weights = CrossAttention(embed_size).forward(q, x)
    # torch.Size([32, 20, 128]) torch.Size([32, 20, 10])
    print(out.shape, attention_weights.shape)


def test_multi_head_attention_v1_layer():
    embed_size = 128
    num_heads = 4
    batch_size = 32
    seq_len_q = 20
    seq_len_x = 10
    q = torch.randn(batch_size, seq_len_q, embed_size)
    x = torch.randn(batch_size, seq_len_x, embed_size)
    out = MultiHeadAttentionV1(embed_size, num_heads).forward(q, x, x)
    # torch.Size([32, 20, 128])
    print(out.shape)


def test_multi_head_attention_layer():
    embed_size = 128
    num_heads = 4
    batch_size = 32
    seq_len_q = 20
    seq_len_x = 10
    q = torch.randn(batch_size, seq_len_q, embed_size)
    x = torch.randn(batch_size, seq_len_x, embed_size)
    out = MultiHeadAttention(embed_size, num_heads).forward(q, x, x)
    # torch.Size([32, 20, 128])
    print(out.shape)


if __name__ == '__main__':
    # test_self_attention_layer()
    # test_cross_attention_layer()
    test_multi_head_attention_v1_layer()
    # test_multi_head_attention_layer()
