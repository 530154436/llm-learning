#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2025/4/22 16:14
# @function:
import torch
from modules.layers import EncoderLayer, DecoderLayer


def test_EncoderLayer():
    embed_size = 4
    batch_size = 1
    sequence_length = 2
    input_tensor = torch.randn(batch_size, sequence_length, embed_size)
    out = EncoderLayer(embed_size, num_heads=2, d_ff=4).forward(input_tensor)
    print(f"输入Token嵌入 {input_tensor.shape}: \n{input_tensor}")  # torch.Size([1, 2, 4])
    print(f"编码器输出 {out.shape}: \n{out}")                        # torch.Size([1, 2, 4])
    return out


def test_DecoderLayer():
    embed_size = 4
    batch_size = 1
    sequence_length = 2
    input_tensor = torch.randn(batch_size, sequence_length, embed_size)
    encoder_output = test_EncoderLayer()

    out = DecoderLayer(embed_size, num_heads=2, d_ff=4).forward(input_tensor, encoder_output)
    print(f"输入Token嵌入 {input_tensor.shape}: \n{input_tensor}")  # torch.Size([1, 2, 4]):
    print(f"解码器输出 {out.shape}: \n{out}")                        # torch.Size([1, 2, 4]):


if __name__ == '__main__':
    # test_EncoderLayer()
    test_DecoderLayer()
