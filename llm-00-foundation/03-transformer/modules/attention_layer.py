#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2025/4/21 16:56
# @function:
import torch
from torch import nn
from modules.attention_func import scaled_dot_product_attention


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
        :param mask: 掩码矩阵 (batch_size, seq_len_q, seq_len_x)
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


class MultiHeadAttentionV1(nn.Module):

    def __init__(self, embed_size, num_heads):
        """
        多头注意力机制：每个头单独定义线性层。

        :param embed_size: 对应原论文的 d_model，输入序列的嵌入维度，即Transformer 中每个位置的特征向量维度。
        :param num_heads: 对应原论文的 h，注意力头的数量，即将输入序列拆分为多少个并行的注意力头。
        """
        super(MultiHeadAttentionV1, self).__init__()
        assert embed_size % num_heads == 0, "embed_size 必须能被 num_heads 整除。"

        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads  # 每个头的维度

        # 为每个头单独定义 Q, K, V 的线性层
        self.w_q = nn.ModuleList([nn.Linear(embed_size, self.head_dim) for _ in range(num_heads)])
        self.w_k = nn.ModuleList([nn.Linear(embed_size, self.head_dim) for _ in range(num_heads)])
        self.w_v = nn.ModuleList([nn.Linear(embed_size, self.head_dim) for _ in range(num_heads)])

        # 输出线性层，将多头拼接后的输出映射回 embed_size
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, q, k, v, mask=None):
        """
        前向传播函数。

        :param q: 查询矩阵的输入，来自解码器 (batch_size, seq_len_q, embed_size)
        :param k: 键矩阵的输入，来自编码器 (batch_size, seq_len_k, embed_size)
        :param v: 值矩阵的输入，来自编码器 (batch_size, seq_len_v, embed_size)，其中seq_len_k=seq_len_v
        :param mask: 掩码矩阵 (batch_size, seq_len_q, seq_len_kv)
        :return
            out: 自注意力加权后的输出 (batch_size, seq_len_q, embed_size)
            attention_weights: 注意力权重矩阵 (batch_size, seq_len_q, seq_len_k)  => Q与K匹配程度矩阵
        """
        multi_head_outputs = []

        # 针对每个头独立计算 Q, K, V，并执行缩放点积注意力
        for i in range(self.num_heads):
            Q = self.w_q[i](q)  # (batch_size, seq_len_q, head_dim)
            K = self.w_k[i](k)  # (batch_size, seq_len_k, head_dim)
            V = self.w_v[i](v)  # (batch_size, seq_len_v, head_dim)

            # 执行缩放点积注意力
            scaled_attention, _ = scaled_dot_product_attention(Q, K, V, mask)
            multi_head_outputs.append(scaled_attention)

        # 将所有头的输出拼接起来
        concat_out = torch.cat(multi_head_outputs, dim=-1)  # (batch_size, seq_len_q, embed_size)

        # 通过输出线性层
        out = self.fc_out(concat_out)  # (batch_size, seq_len_q, embed_size)
        return out


class MultiHeadAttention(nn.Module):

    def __init__(self, embed_size: int, num_heads: int):
        """
        多头注意力机制（MultiHead-Attention） 优化版本

        不再循环遍历每个头来单独计算查询、键和值，而是一次性计算 Q、K 和 V，
        然后使用重塑（reshape）和转置（transpose）将这些矩阵拆分为多头的格式，有些代码实现将这些操作统一称为拆分（split）。

        :param embed_size: 对应原论文的 d_model，输入序列的嵌入维度，即Transformer 中每个位置的特征向量维度。
        :param num_heads: 对应原论文的 h，注意力头的数量，即将输入序列拆分为多少个并行的注意力头。
        """
        super(MultiHeadAttention, self).__init__()
        assert embed_size % num_heads == 0, "embed_size 必须能被 h 整除。"

        self.embed_size = embed_size
        self.num_heads = num_heads

        # 每个注意力头的维度，对应原论文中的 d_k
        self.head_dim = self.embed_size // self.num_heads

        self.w_q = nn.Linear(embed_size, embed_size)
        self.w_k = nn.Linear(embed_size, embed_size)
        self.w_v = nn.Linear(embed_size, embed_size)
        # 输出线性层，将多头拼接后的输出做映射
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None):
        """
        前向传播函数。

        :param q: 查询矩阵的输入，来自解码器 (batch_size, seq_len_q, embed_size)
        :param k: 键矩阵的输入，来自编码器 (batch_size, seq_len_k, embed_size)
        :param v: 值矩阵的输入，来自编码器 (batch_size, seq_len_v, embed_size)，其中seq_len_k=seq_len_v
        :param mask: 掩码矩阵 (batch_size, seq_len_q, seq_len_kv)
        :return
            out: 自注意力加权后的输出 (batch_size, seq_len_q, embed_size)
            attention_weights: 注意力权重矩阵 (batch_size, seq_len_q, seq_len_k)  => Q与K匹配程度矩阵
        """
        batch_size = q.size(0)
        seq_len_q, seq_len_k = q.size(1), k.size(1)

        # 1、拆分：将线性变换后的矩阵拆分为独立的多头，调整维度为 (batch_size, num_heads, seq_len, head_dim)
        # (batch_size, seq_len_q, embed_size) => (batch_size, seq_len_q, num_heads, head_dim) => (batch_size, num_heads, seq_len_q, head_dim)
        # (batch_size, seq_len_k, embed_size) => (batch_size, seq_len_k, num_heads, head_dim) => (batch_size, num_heads, seq_len_k, head_dim)
        Q = self.w_q(q).view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.w_k(k).view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.w_v(v).view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)

        # 2、使用缩放点积注意力函数计算输出和权重
        # (batch_size, num_heads, seq_len_q, head_dim)
        scaled_attention, _ = scaled_dot_product_attention(Q, K, V, mask)

        # 3、合并：合并多头并还原为输入的形状，目的是为了和输入嵌入矩阵做残差连接
        #    (batch_size, num_heads, seq_len_q, head_dim)
        # => (batch_size, seq_len_q, num_heads, head_dim)
        # => (batch_size, seq_len_q, embed_size)
        concat_out = scaled_attention.transpose(1, 2).contiguous().view(batch_size, seq_len_q, -1)

        # 输出线性层作用
        # 特征变换与适配：将多个子空间的输出通过线性变换映射到适合下游任务的新特征空间，使得特征表示更符合后续层的需求。
        # 信息融合与统一：将分散在不同子空间的多头输出进行有效融合（不同头提取的特征可能具有不同的语义和侧重点），从而形成一个综合的特征向量，更好地捕捉输入序列中的复杂模式。
        # 增加模型表达能力：通过学习权值矩阵自动调整以适应不同任务和数据分布，增强模型灵活性和性能，优于简单拼接多头输出的方法。
        output = self.fc_out(concat_out)
        return output
