#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2025/5/10 14:35
# @function:
import json
from typing import List, Tuple, Dict
import torch
from functorch.dim import Tensor
from transformers import BertTokenizer
from torch.utils.data import Dataset, TensorDataset
from nlp_task_ner.data_process import convert_examples_to_feature


class NERDataset(Dataset):
    def __init__(self, data_path: str, label_path: str, tokenizer: BertTokenizer):
        self.data_path = data_path
        self.label_path = label_path
        self.tokenizer = tokenizer

        self.label2id = json.load(open(label_path, 'r', encoding='utf-8'))
        self.dataset = self.load_raw()

    def load_raw(self) -> List[Tuple[List[str], List[str]]]:
        """
        加载原始数据: bios格式
        :return
        [
           (['藏', '家', '1', '2', '条', '收', '藏', '秘', '籍'],
            ['B-position', 'I-position', 'O', 'O', 'O', 'O', 'O', 'O', 'O'])
        ]
        """
        with open(self.data_path, 'r', encoding='utf-8') as f:
            examples = []
            words, labels = [], []
            for line in f:
                line = line.strip()
                if line == "":
                    examples.append((words, labels))
                    words, labels = [], []
                else:
                    word, label = line.split(' ')
                    words.append(word)
                    labels.append(label)
        return examples

    def __getitem__(self, idx):
        """sample data to get batch"""
        words = self.dataset[idx][0]
        labels = self.dataset[idx][1]
        return [words, labels]

    def __len__(self):
        """get dataset size"""
        return len(self.dataset)

    def collate_fn(self, batch: List[Tuple[list]]) -> Dict[str, Tensor]:
        """
        将训练数据转化为 Bert 的输入格式
        :param batch: [batch_size, sql_len]
        :return:  dict([batch_size, sql_len],
                       [batch_size, sql_len],
                       [batch_size, sql_len],
                       [batch_size, sql_len])
        """
        all_input_ids, all_input_mask, all_token_type_ids, all_label_ids = [], [], [], []
        for feature in convert_examples_to_feature(batch, label2id=self.label2id, tokenizer=self.tokenizer):
            all_input_ids.append(feature.input_ids)
            all_input_mask.append(feature.input_mask)
            all_token_type_ids.append(feature.token_type_ids)
            all_label_ids.append(feature.label_ids)
        return {
            "input_ids": torch.LongTensor(all_input_ids),
            "input_mask": torch.LongTensor(all_input_mask),
            "token_type_ids": torch.LongTensor(all_token_type_ids),
            "label_ids": torch.LongTensor(all_label_ids),
        }


if __name__ == '__main__':
    _tokenizer = BertTokenizer.from_pretrained(
        pretrained_model_name_or_path="data/pretrain/bert-base-chinese",
        do_lower_case=True
    )
    _dataset = NERDataset("data/dataset/clue/train.jsonl",
                          "data/dataset/clue/label.json",
                          tokenizer=_tokenizer)
    _input_batch = _dataset.collate_fn([_dataset[i] for i in range(32)])
    print(_input_batch.get("input_ids").shape)
    print(_input_batch.get("input_mask").shape)
    print(_input_batch.get("token_type_ids").shape)
    print(_input_batch.get("label_ids").shape)

