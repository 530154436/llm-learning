#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2025/4/23 16:00
# @function:
#
# 注意力机制中三种掩码技术详解和Pytorch实现
# https://avoid.overfit.cn/post/2371a9ec5eca46af81dbe23d3442a383
import torch


def create_padding_mask(seq: torch.Tensor, pad_token_id: int = 0) -> torch.Tensor:
    """
    填充掩码（Padding Mask）
    用来指示哪些数据是真实的，哪些是填充的（<PAD>）。在模型处理这些数据时，掩码会用来避免在计算损失或者梯度时考虑填充的部分，确保模型的学习只关注于有效的数据。

    :param seq: 分词（tokenize）然后映射为 Token ID 序列, (batch_size, seq_len)
    :param pad_token_id: 填充对应的token_id，如<PAD>对应0
    :return: 掩码处理后的序列 (batch_size, 1, 1, seq_len)
    """
    # seq 的形状为 (batch_size, seq_len)
    mask = seq.not_equal(pad_token_id)
    mask = mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
    return mask  # 在注意力计算时，填充值为 0 的位置会被屏蔽


def create_casual_mask(size):
    """
    因果掩码（Causal Mask）
    也称为前瞻掩码或未来掩码（Look-ahead Mask）
    屏蔽未来时间步的信息（即设置为一个非常小的负值，如负无穷大），这确保了在计算每个元素的输出时，模型只能使用到当前和之前的信息，而不能使用后面的信息。

    :param size: 序列长度
    :return: (seq_len, seq_len)
    """
    mask = torch.tril(torch.ones(size, size)).type(torch.bool)  # 下三角矩阵
    return mask


def create_sequence_mask(seq: torch.Tensor, pad_token_id: int = 0) -> torch.Tensor:
    """
    生成每个序列的掩码：填充掩码+未来信息掩码
    :param seq: (batch_size, seq_len)
    :param pad_token_id: 填充对应的token_id，如<PAD>对应0
    :return: mask处理后的序列 (batch_size, 1, seq_len, seq_len)
    """
    # (batch_size, 1, 1, seq_len)
    padding_mask = create_padding_mask(seq, pad_token_id)

    # (seq_len, seq_len)
    look_ahead_mask = create_casual_mask(seq.size(1)).to(seq.device)

    # look_ahead_mask: (seq_len, seq_len) => (1, seq_len, seq_len)
    # (batch_size, 1, seq_len, seq_len): (1, seq_len, seq_len) & (batch_size, 1, 1, seq_len)
    combined_mask = look_ahead_mask.unsqueeze(0) & padding_mask
    return combined_mask
