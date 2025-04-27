#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2025/4/27 19:18
# @function:
from collections import Counter
from pathlib import Path
from torch.utils.data import Dataset


class MTDataset(Dataset):

    def __init__(self, file: Path):
        self.data = []

    def build_vocab(filepath, tokenizer):
        pass

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


