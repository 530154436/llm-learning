#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2025/4/22 16:14
# @function:
import torch
from modules.models import Encoder, Decoder, Transformer


def test_Encoder():
    vocab_size = 10
    embed_size = 4
    batch_size = 1
    sequence_length = 2
    input_tensor = torch.randint(0, vocab_size, (batch_size, sequence_length))
    out = Encoder(vocab_size, embed_size, num_heads=2, d_ff=4).forward(input_tensor)
    print(f"输入批次 {input_tensor.shape}: \n{input_tensor}")  # torch.Size([1, 2])
    print(f"编码器输出 {out.shape}: \n{out}")                  # torch.Size([1, 2, 4])
    print()
    return out


def test_Decoder():
    vocab_size = 10
    embed_size = 4
    batch_size = 1
    sequence_length = 2
    input_tensor = torch.randint(0, vocab_size, (batch_size, sequence_length))
    encoder_output = test_Encoder()

    out = Decoder(vocab_size, embed_size, num_heads=2, d_ff=4).forward(input_tensor, encoder_output)
    print(f"输入批次   {input_tensor.shape}: \n{input_tensor}")  # torch.Size([1, 2])
    print(f"解码器输出 {out.shape}: \n{out}")                  # torch.Size([1, 2, 4])


def test_Transformer():
    embed_size = 4
    batch_size = 1
    sequence_length = 2
    src_vocab_size, tgt_vocab_size = 8, 6
    src_tensor = torch.randint(0, src_vocab_size, (batch_size, sequence_length))
    tgt_tensor = torch.randint(0, tgt_vocab_size, (batch_size, sequence_length + 1))

    model = Transformer(src_vocab_size, tgt_vocab_size, embed_size, num_heads=2, d_ff=4)
    out = model.forward(src_tensor, tgt_tensor)
    print(f"源输入批次   {src_tensor.shape}: \n{src_tensor}")  # torch.Size([1, 2])
    print(f"目标输入批次 {tgt_tensor.shape}: \n{tgt_tensor}")  # torch.Size([1, 3])
    print(f"解码器输出   {out.shape}: \n{out}")                # torch.Size([1, 3, 6])


if __name__ == '__main__':
    # test_Encoder()
    # test_Decoder()
    test_Transformer()
