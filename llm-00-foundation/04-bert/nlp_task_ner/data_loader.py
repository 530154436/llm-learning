#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2025/5/10 14:35
# @function:
from typing import List, Tuple
from transformers import BertTokenizer
from torch.utils.data import Dataset


class NERBatch(object):
    def __init__(self, words, labels, config, word_pad_idx=0, label_pad_idx=-1):
        pass


class NERDataset(Dataset):
    def __init__(self, data_path: str, label_path: str, tokenizer: BertTokenizer):
        self.data_path = data_path
        self.label_path = label_path
        self.tokenizer = tokenizer

        self.dataset = self.load_raw()
        # self.word_pad_idx = word_pad_idx
        # self.label_pad_idx = label_pad_idx
        # self.device = config.device

    def load_raw(self) -> List[Tuple[List[str], List[str]]]:
        """
        加载原始数据: bio格式
        :return
        [
           (['藏', '家', '1', '2', '条', '收', '藏', '秘', '籍'],
            ['B-position', 'I-position', 'O', 'O', 'O', 'O', 'O', 'O', 'O'])
        ]
        """
        with open(self.data_path, 'r', encoding='utf-8') as f:
            samples = []
            words, labels = [], []
            for line in f:
                line = line.strip()
                if line == "":
                    samples.append((words, labels))
                    words, labels = [], []
                else:
                    word, label = line.split(' ')
                    words.append(word)
                    labels.append(label)
        return samples

    def __getitem__(self, idx):
        """sample data to get batch"""
        words = self.dataset[idx][0]
        labels = self.dataset[idx][1]
        return [words, labels]

    def __len__(self):
        """get dataset size"""
        return len(self.dataset)

    def collate_fn(self, batch: List[Tuple[list]]):
        pass


if __name__ == '__main__':
    _tokenizer = BertTokenizer.from_pretrained(
        pretrained_model_name_or_path="data/pretrain/bert-base-chinese",
        do_lower_case=True
    )
    _dataset = NERDataset("data/dataset/clue/train.jsonl",
                          "data/dataset/clue/label.jsonl",
                          tokenizer=_tokenizer)
    for i in range(len(_dataset)):
        print(i, _dataset[i])

