#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2025/5/19 9:46
# @function:
import torch
from modeling_util import BaseModel


class BaseNerModel(BaseModel):

    def __init__(self, pretrain_path: str, num_labels: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pretrain_path = pretrain_path
        self.num_labels = num_labels
