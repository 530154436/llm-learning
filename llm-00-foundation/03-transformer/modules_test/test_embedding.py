#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2025/4/22 16:14
# @function:
import pandas as pd
import torch
import matplotlib
from matplotlib import pyplot as plt
from modules.embedding import Embedding, PositionalEncoding, TransformerEmbedding

matplotlib.use('TkAgg')  # 或者 'Qt5Agg', 'GTK3Agg', 等等


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


def example_positional_matplotlib():
    pe = PositionalEncoding(20, 0)
    y = pe.forward(torch.zeros(1, 100, 20))

    # 准备数据
    data = []
    for dim in [4, 5, 6, 7]:
        for position in range(100):
            data.append({"embedding": y[0, position, dim].item(), "dimension": dim, "position": position})
    df = pd.DataFrame(data)

    # 使用Matplotlib绘图
    plt.figure(figsize=(10, 6))
    for dim in [4, 5, 6, 7]:
        plt.plot(df[df['dimension'] == dim]['position'],
                 df[df['dimension'] == dim]['embedding'],
                 label=f'dim {dim}')

    plt.title('Positional Encoding with Matplotlib')
    plt.xlabel('Position')
    plt.ylabel('Embedding Value')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    test_embedding()
    test_PositionalEncoding()
    # example_positional_matplotlib()
