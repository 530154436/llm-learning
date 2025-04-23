#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2025/4/23 16:00
# @function:
import torch


def create_padding_mask(seq: torch.Tensor, pad_token_id: int = 0) -> torch.Tensor:
    """
    填充掩码（Padding Mask）
    填充掩码用于在注意力计算时屏蔽填充 <PAD> 位置，防止模型计算注意力权重的时候考虑这些无意义的位置，在编码器的自注意力中使用。

    :param seq: 分词（tokenize）然后映射为 Token ID 序列, (batch_size, seq_len)
    :param pad_token_id: 填充对应的token_id，如<PAD>对应0
    :return: 掩码处理后的序列 (batch_size, 1, 1, seq_len)
    """
    # seq 的形状为 (batch_size, seq_len)
    mask = seq.not_equal(pad_token_id)
    mask = mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
    return mask  # 在注意力计算时，填充值为 0 的位置会被屏蔽


def create_look_ahead_mask(size):
    """
    未来信息掩码（Look-ahead Mask）
    未来信息掩码用于在解码器中屏蔽未来的位置，防止模型在预测下一个词时“偷看”答案（训练时），在解码器中使用。

    :param size: 序列长度
    :return: (seq_len, seq_len)
    """
    mask = torch.tril(torch.ones(size, size)).type(torch.bool)  # 下三角矩阵
    return mask


def create_decoder_mask(seq: torch.Tensor, pad_token_id: int = 0) -> torch.Tensor:
    """
    生成每个序列的 填充掩码+未来信息掩码
    :param seq: (batch_size, seq_len)
    :param pad_token_id: 填充对应的token_id，如<PAD>对应0
    :return: mask处理后的序列 (batch_size, 1, seq_len, seq_len)
    """
    # (batch_size, 1, 1, seq_len)
    padding_mask = create_padding_mask(seq, pad_token_id)

    # (seq_len, seq_len)
    look_ahead_mask = create_look_ahead_mask(seq.size(1)).to(seq.device)

    # look_ahead_mask: (seq_len, seq_len) => (1, seq_len, seq_len)
    # (batch_size, 1, seq_len, seq_len): (1, seq_len, seq_len) & (batch_size, 1, 1, seq_len)
    combined_mask = look_ahead_mask.unsqueeze(0) & padding_mask
    return combined_mask
