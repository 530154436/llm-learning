#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2025/4/23 20:33
# @function:
import torch
from torch import nn
from modules.embedding import Embedding, PositionalEncoding
from modules.layers import EncoderLayer, DecoderLayer
from modules.sublayer_connection import LayerNorm


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, d_ff: int, dropout=0.1, N: int = 6):
        """
        编码器，由 N 个 EncoderLayer 堆叠而成。

        :param vocab_size: 词汇表大小
        :param d_model: 嵌入维度
        :param num_heads: 多头注意力的头数
        :param d_ff: 前馈神经网络的隐藏层维度
        :param dropout: Dropout 概率
        :param N: EncoderLayer个数
        """
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.embed = Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model, dropout=dropout)

        self.layers = nn.ModuleList(
            EncoderLayer(d_model, num_heads, d_ff, dropout=dropout) for _ in range(N)
        )
        self.norm = LayerNorm(d_model)  # 最后层归一化

    def forward(self, x: torch.LongTensor, mask=None):
        """
        前向传播函数。

        :param x: 输入张量 (batch_size, seq_len)，一个批次的多个序列，序列元素是 token_id
        :param mask: 输入掩码
        :return: 编码器的输出 (batch_size, seq_len, d_model)
        """
        # (batch_size, seq_len_tgt) => (batch_size, seq_len_tgt, d_model)
        x = self.embed(x)
        x = self.pe(x)

        # (batch_size, seq_len, d_model) => (batch_size, seq_len, d_model)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)  # 最后层归一化


class Decoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, d_ff: int, dropout=0.1, N: int = 6):
        """
        解码器，由 N 个 DecoderLayer 堆叠而成。

        :param vocab_size: 词汇表大小
        :param d_model: 嵌入维度
        :param num_heads: 多头注意力的头数
        :param d_ff: 前馈神经网络的隐藏层维度
        :param dropout: Dropout 概率
        :param N: DecoderLayer个数
        """
        super(Decoder, self).__init__()
        self.embed = Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model, dropout=dropout)
        self.layers = nn.ModuleList(
            DecoderLayer(d_model, num_heads, d_ff, dropout=dropout) for _ in range(N)
        )
        self.norm = LayerNorm(d_model)  # 最后层归一化

    def forward(self, x: torch.LongTensor, encoder_output, src_mask=None, tgt_mask=None) -> torch.Tensor:
        """
        前向传播函数。

        :param x: 解码器输入 (batch_size, seq_len_tgt)，一个批次的多个序列，序列元素是 token_id
        :param encoder_output: 编码器输出 (batch_size, seq_len_src, d_model)
        :param src_mask: 源序列掩码，用于交叉注意力
        :param tgt_mask: 目标序列掩码，用于自注意力
        :return (batch_size, seq_len_tgt, d_model)
        """
        # (batch_size, seq_len_tgt) => (batch_size, seq_len_tgt, d_model)
        x = self.embed(x)
        x = self.pe(x)

        # (batch_size, seq_len_tgt, d_model)
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)  # 最后层归一化


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, d_ff, dropout=0.1, N: int = 6):
        """
        Transformer 模型，由编码器和解码器组成。

        :param src_vocab_size:  源语言词汇表大小
        :param tgt_vocab_size:  目标语言词汇表大小
        :param d_model: 嵌入维度
        :param num_heads: 多头注意力的头数
        :param d_ff: 前馈神经网络的隐藏层维度
        :param dropout: Dropout 概率
        :param N: 编码器和解码器的层数
        """
        super(Transformer, self).__init__()

        # 编码器和解码器
        self.encoder = Encoder(src_vocab_size, d_model, num_heads, d_ff, dropout, N)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_heads, d_ff, dropout, N)

        # 输出线性层
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        前向传播函数。

        :param src: 源序列输入 (batch_size, seq_len_src)
        :param tgt: 目标序列输入 (batch_size, seq_len_tgt)
        :param src_mask: 源序列掩码，用于交叉注意力
        :param tgt_mask: 目标序列掩码，用于自注意力
        :return Transformer 的输出（未经过 Softmax） (batch_size, seq_len_tgt, tgt_vocab_size)
        """
        # 编码器
        enc_output = self.encoder(src, src_mask)

        # 解码器
        dec_output = self.decoder(tgt, enc_output, src_mask, tgt_mask)

        # 输出层
        output = self.fc_out(dec_output)

        return output
