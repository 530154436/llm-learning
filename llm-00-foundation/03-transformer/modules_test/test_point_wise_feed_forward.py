#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2025/4/22 16:14
# @function:
import torch
from modules.point_wise_feed_forward import PositionWiseFeedForward


def test_test_point_wise_feed_forward():
    batch_size = 10
    sql_len = 10
    d_model = 512
    d_ff = 2048
    input_tensor = torch.randn(batch_size, sql_len, d_model)
    out = PositionWiseFeedForward(d_model, d_ff).forward(input_tensor)
    # torch.Size([10, 10, 512])
    print(out.shape)


if __name__ == '__main__':
    test_test_point_wise_feed_forward()
