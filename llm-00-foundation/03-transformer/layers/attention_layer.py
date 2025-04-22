#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2025/4/21 16:56
# @function:
import torch
from torch import nn
from attention_func import scaled_dot_product_attention


class SelfAttention(nn.Module):
    def __init__(self, embed_size):
        """
        自注意力机制（Self-Attention）
        查询、键和值矩阵来自同一输入序列，模型通过自注意力机制学习输入序列的全局依赖关系。

        :param embed_size: 输入序列的嵌入维度（每个向量的特征维度）。
        """
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size

        # 定义线性层，用于生成查询、键和值矩阵
        self.w_q = nn.Linear(embed_size, embed_size)
        self.w_k = nn.Linear(embed_size, embed_size)
        self.w_v = nn.Linear(embed_size, embed_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        前向传播函数。

        :param x: 输入序列 (batch_size, seq_len_x, embed_size)
        :param mask: 掩码矩阵 (batch_size, seq_len_x, seq_len_x)
        :return
            out: 自注意力加权后的输出 (batch_size, seq_len_x, embed_size)
            attention_weights: 注意力权重矩阵 (batch_size, seq_len_x, seq_len_x)
        """
        # 在自注意力机制中，q, k, v 都来自同一输入序列（q = k = v = x）
        # 将输入序列通过线性变换生成 Q, K, V
        Q = self.w_q(x)  # (batch_size, seq_len_q, embed_size)
        K = self.w_k(x)  # (batch_size, seq_len_k, embed_size)
        V = self.w_v(x)  # (batch_size, seq_len_v, embed_size)

        # 使用缩放点积注意力函数计算输出和权重
        out, attention_weights = scaled_dot_product_attention(Q, K, V, mask)
        return out, attention_weights


class CrossAttention(nn.Module):
    def __init__(self, embed_size):
        """
        交叉注意力机制（Cross-Attention）
        查询矩阵来自解码器的输入，而键和值矩阵来自编码器的输出，解码器的第二个 Attention 模块就是 Cross-Attention，用于从编码器输出中获取相关的上下文信息。

        :param embed_size: 输入序列的嵌入维度（每个向量的特征维度）。
        """
        super(CrossAttention, self).__init__()
        self.embed_size = embed_size

        # 定义线性层，用于生成查询、键和值矩阵
        self.w_q = nn.Linear(embed_size, embed_size)
        self.w_k = nn.Linear(embed_size, embed_size)
        self.w_v = nn.Linear(embed_size, embed_size)

    def forward(self, q: torch.Tensor, x: torch.Tensor, mask: torch.Tensor = None):
        """
        前向传播函数。

        :param q: 查询矩阵的输入，来自解码器 (batch_size, seq_len_q, embed_size)
        :param x: 键和值矩阵的输入，来自编码器 (batch_size, seq_len_x, embed_size)
        :param mask: 掩码矩阵 (batch_size, seq_len_q, seq_len_kv)
        :return
            out: 自注意力加权后的输出 (batch_size, seq_len_q, embed_size)
            attention_weights: 注意力权重矩阵 (batch_size, seq_len_q, seq_len_x)  => Q与K匹配程度矩阵
        """
        # 在交叉注意机制中，q, k 来自同一输入序列（q = k = x）、v 来自另一个输入序列
        # q≠k=v：q 来自解码器，k 和 v 来自编码器
        Q = self.w_q(q)  # (batch_size, seq_len_q, embed_size)
        K = self.w_k(x)  # (batch_size, seq_len_k, embed_size)
        V = self.w_v(x)  # (batch_size, seq_len_v, embed_size)

        # 使用缩放点积注意力函数计算输出和权重
        out, attention_weights = scaled_dot_product_attention(Q, K, V, mask)
        return out, attention_weights
