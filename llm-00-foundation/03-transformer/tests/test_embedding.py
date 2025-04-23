#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2025/4/22 16:14
# @function:
import torch
from modules.embedding import Embedding, PositionalEncoding


def test_embedding():
    vocab_size = 3
    d_model = 4
    module = Embedding(vocab_size, d_model)
    # torch.Size([3, 4])
    print(f"Embedding权重矩阵({module.embedding.weight.shape})：\n {module.embedding.weight.data}")
    print()

    batch_size = 2
    seq_len = 5
    input_tensor = torch.randint(0, vocab_size, (batch_size, seq_len))
    out = module.forward(input_tensor)
    # torch.Size([2, 5, 4])
    print(f"输入序列({input_tensor.shape}): \n{input_tensor}")
    print(f"输出Token嵌入({out.shape}): \n{out}")  # 缩放因子是 sqrt(d_model)

    return vocab_size, d_model, out


def test_PositionalEncoding():
    # + 位置编码
    vocab_size, d_model, out = test_embedding()

    pos_encoder = PositionalEncoding(d_model, max_len=5000)
    out = pos_encoder.forward(out)
    # torch.Size([2, 5, 4])
    print(f"输出Token嵌入+位置嵌入({out.shape}): \n{out}")


if __name__ == '__main__':
    # test_embedding()
    test_PositionalEncoding()
