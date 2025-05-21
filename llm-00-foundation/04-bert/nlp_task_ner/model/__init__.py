#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2025/5/19 9:46
# @function:
import torch
from torch import nn


class BaseNerModel(nn.Module):

    def predict_proba(self,
                      input_ids: torch.Tensor,
                      attention_mask: torch.Tensor,
                      token_type_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def loss_fn(self,
                logits: torch.Tensor,
                labels: torch.Tensor,
                attention_mask: torch.Tensor = None) -> torch.FloatTensor:
        raise NotImplementedError

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                token_type_ids: torch.Tensor) -> torch.FloatTensor:
        raise NotImplementedError
