#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2025/4/23 20:33
# @function:
import torch
from torch import nn
from modules.attention_layer import MultiHeadAttention
from modules.point_wise_feed_forward import PositionWiseFeedForward
from modules.sublayer_connection import SublayerConnection


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout=0.1):
        """
        编码器层。
        :param d_model: 嵌入维度
        :param num_heads: 多头注意力的头数
        :param d_ff: 前馈神经网络的隐藏层维度
        :param dropout: Dropout 概率
        """
        super(EncoderLayer, self).__init__()
        self.d_model = d_model

        # 多头自注意力（Multi-Head Self-Attention）
        self.self_attn = MultiHeadAttention(embed_size=d_model, num_heads=num_heads)
        # 前馈神经网络
        self.feed_forward = PositionWiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)

        # Add & Norm（残差连接和层归一化）
        self.sublayer_conn1 = SublayerConnection(d_model=d_model, dropout=dropout)
        self.sublayer_conn2 = SublayerConnection(d_model=d_model, dropout=dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播函数。
        :param x: 输入张量，形状为 (batch_size, seq_len, d_model)。
        :param mask: 源序列掩码，用于自注意力。 (batch_size, 1, 1, seq_len)
        :return: 编码器层的输出，形状为 (batch_size, seq_len, d_model)
        """
        # 自注意力子层
        attn_output = self.self_attn(x, x, x, mask=mask)
        x = self.sublayer_conn1(x, attn_output)

        # 前馈子层
        ff_output = self.feed_forward(x)
        x = self.sublayer_conn2(x, ff_output)

        del attn_output
        del ff_output
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout=0.1):
        """
        解码器层。
        :param d_model: 嵌入维度
        :param num_heads: 多头注意力的头数
        :param d_ff: 前馈神经网络的隐藏层维度
        :param dropout: Dropout 概率
        """
        super(DecoderLayer, self).__init__()
        self.d_model = d_model

        # 掩码多头自注意力（Masked Multi-Head Self-Attention）
        self.self_attn = MultiHeadAttention(embed_size=d_model, num_heads=num_heads)
        # 多头交叉注意力（Multi-Head Cross-Attention）
        self.cross_attn = MultiHeadAttention(embed_size=d_model, num_heads=num_heads)
        # 前馈神经网络
        self.feed_forward = PositionWiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)

        # Add & Norm（残差连接和层归一化）
        self.sublayer_conn1 = SublayerConnection(d_model=d_model, dropout=dropout)
        self.sublayer_conn2 = SublayerConnection(d_model=d_model, dropout=dropout)
        self.sublayer_conn3 = SublayerConnection(d_model=d_model, dropout=dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None) -> torch.Tensor:
        """
        前向传播函数。

        :param x: 解码器输入 (batch_size, seq_len_tgt, d_model)
        :param encoder_output: 编码器输出 (batch_size, seq_len_src, d_model)
        :param src_mask: 源序列掩码,用于编码器的自注意力、解码器的交叉注意力
                         即填充掩码  (batch_size, 1, 1, seq_len_src)
        :param tgt_mask: 目标序列掩码，用于解码器输入部分的自注意力
                         填充掩码 & 因果掩码  (batch_size, 1, seq_len_tgt, seq_len_tgt)
        :return: 解码器输出 (batch_size, seq_len_tgt, d_model)
        """
        # 掩码多头自注意力
        self_attn_output = self.self_attn(x, x, x, mask=tgt_mask)
        x = self.sublayer_conn1(x, self_attn_output)

        # 多头交叉注意力（q来自解码器输入，k、v来自编码器输入）
        cross_attn_output = self.cross_attn(q=x, k=encoder_output, v=encoder_output, mask=src_mask)
        x = self.sublayer_conn2(x, cross_attn_output)

        # 前馈子层
        ff_output = self.feed_forward(x)
        x = self.sublayer_conn3(x, ff_output)

        del self_attn_output
        del cross_attn_output
        del ff_output
        return x
