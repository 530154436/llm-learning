#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2025/5/19 9:46
# @function:
import torch
from torch import nn


class BaseModel(nn.Module):

    def forward(self, *inputs: torch.Tensor) -> torch.FloatTensor:
        raise NotImplementedError

    def predict(self, *inputs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
