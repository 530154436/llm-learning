#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2025/4/22 16:14
# @function:
import torch
from modules.mask import create_padding_mask, create_casual_mask, create_sequence_mask


def test_create_padding_mask():
    seq = torch.tensor([[5, 7, 9, 0, 0],
                        [8, 6, 0, 0, 0]])  # 0 表示 <PAD>
    mask = create_padding_mask(seq)
    print(f"原始序列({seq.shape})：")  # torch.Size([2, 5])
    print(seq)
    print(f"填充掩码({mask.shape})：")  # torch.Size([2, 1, 1, 5])
    print(mask)

    # 自注意力
    batch_size, num_heads, seq_len_q, seq_len_k = 1, 1, 5, 5
    scores = torch.randn(batch_size, num_heads, seq_len_q, seq_len_k)
    scores = scores.masked_fill(mask == 0, float('-inf'))
    print(f"注意力分数({scores.shape})：")
    print(scores)
    print()


def test_create_look_ahead_mask():
    seq = torch.tensor([[5, 7, 9, 0, 0],
                        [8, 6, 0, 0, 0]])
    res = create_casual_mask(seq.size(1))
    print(f"序列长度：{seq.size(1)}")
    print(f"未来信息掩码({res.shape})：\n{res}")  # torch.Size([5, 5])
    print()


def test_create_decoder_mask():
    seq = torch.tensor([[5, 7, 9, 0, 0],
                        [8, 6, 0, 0, 0]])
    padding_mask = create_padding_mask(seq)
    look_ahead_mask = create_casual_mask(seq.size(1))
    combined_mask1 = look_ahead_mask.unsqueeze(0) & padding_mask
    print(f"原始序列({seq.shape})：\n{seq}")  # torch.Size([2, 5])
    print(f"填充掩码({padding_mask.shape})：\n{padding_mask}")               # torch.Size([2, 1, 1, 5])
    print(f"未来信息掩码({look_ahead_mask.shape})：\n{look_ahead_mask}")      # torch.Size([5, 5])
    print(f"组合掩码1({combined_mask1.shape})：\n{combined_mask1}")          # torch.Size([2, 1, 5, 5])

    combined_mask2 = create_sequence_mask(seq)
    print(f"组合掩码2({combined_mask2.shape})：\n{combined_mask2}")          # torch.Size([2, 1, 5, 5])


if __name__ == '__main__':
    test_create_padding_mask()
    # test_create_look_ahead_mask()
    # test_create_decoder_mask()
