#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2025/4/22 16:14
# @function:
import torch
from modules.point_wise_feed_forward import PositionWiseFeedForward
from modules.sublayer_connection import ResidualConnection, LayerNorm, SublayerConnection


def test_residual_connection():
    batch_size = 10
    sql_len = 10
    d_model = 512
    d_ff = 2048
    input_tensor = torch.randn(batch_size, sql_len, d_model)
    module = PositionWiseFeedForward(d_model, d_ff)

    out = ResidualConnection().forward(input_tensor, module)
    # torch.Size([10, 10, 512])
    print(out.shape)


def test_layer_norm():
    batch_size = 10
    sql_len = 10
    d_model = 512
    input_tensor = torch.randn(batch_size, sql_len, d_model)
    out = LayerNorm(feature_size=d_model).forward(input_tensor)
    # torch.Size([10, 10, 512])
    print(out.shape)
    # print(out)


def test_sublayer_connection():
    batch_size = 10
    sql_len = 10
    d_model = 512
    d_ff = 2048
    input_tensor = torch.randn(batch_size, sql_len, d_model)
    module = PositionWiseFeedForward(d_model, d_ff)

    out = SublayerConnection(feature_size=d_model).forward(input_tensor, module)
    # torch.Size([10, 10, 512])
    print(out.shape)
    # print(out)


if __name__ == '__main__':
    # test_residual_connection()
    # test_layer_norm()
    test_sublayer_connection()
