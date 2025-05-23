#!/usr/bin/env python3
# -*- coding:utf-8 -*--
import torch
from torch import nn


class BaseModel(nn.Module):

    def forward(self, *inputs: torch.Tensor) -> torch.FloatTensor:
        raise NotImplementedError

    def predict(self, *inputs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
