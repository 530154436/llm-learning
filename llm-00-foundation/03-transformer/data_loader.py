#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2025/4/27 19:18
# @function:
import json
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from util.sentencepiece_tokenizer import SentencePieceTokenizerWithLang
from modules.mask import create_padding_mask, create_sequence_mask


class MTBatch(object):
    def __init__(self, src_text: List[str], src_input: torch.Tensor, src_mask: torch.Tensor,
                 tgt_text: List[str], tgt_input: torch.Tensor, tgt_mask: torch.Tensor,
                 tgt_output: torch.Tensor, device: str = "cpu"):
        self.src_text = src_text
        self.src_input = src_input.to(device)
        self.src_mask = src_mask.to(device)

        self.tgt_text = tgt_text
        self.tgt_input = tgt_input.to(device)
        self.tgt_mask = tgt_mask.to(device)
        self.tgt_output = tgt_output.to(device)


class MTDataset(Dataset):
    """
    翻译数据集Loader, <src, tgt>
    """
    def __init__(self, path: str, src_lang: str = "en", tgt_lang: str = "zh"):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.src_sentences, self.tgt_sentences = self.load_raw(path, sort=True)
        self.src_tokenizer = SentencePieceTokenizerWithLang(lang=src_lang)
        self.tgt_tokenizer = SentencePieceTokenizerWithLang(lang=tgt_lang)

    def load_raw(self, path, sort=False) -> tuple:
        """
        加载原始数据: jsonline格式
        :param path: 训练数据路径
        :param sort: 是否按英文句子排序
        :return Tuple[英文句子, 中文句子]
        """
        src_sent = []
        tgt_sent = []
        for idx, line in enumerate(open(path, 'r', encoding='utf-8')):
            line = line.strip()
            if not line:
                continue
            json_data: dict = json.loads(line)
            assert self.src_lang in json_data and self.tgt_lang in json_data

            src_sent.append(json_data.get(self.src_lang))
            tgt_sent.append(json_data.get(self.tgt_lang))

        # 按照句子长度从小到大排序
        if sort:
            zipped = sorted(zip(src_sent, tgt_sent, map(lambda x: len(x), src_sent)), key=lambda x: x[-1])
            src_sent = list(map(lambda x: x[0], zipped))
            tgt_sent = list(map(lambda x: x[1], zipped))
        return src_sent, tgt_sent

    def __getitem__(self, idx):
        src_sentence = self.src_sentences[idx]
        tgt_sentence = self.tgt_sentences[idx]
        return [src_sentence, tgt_sentence]

    def __len__(self):
        return len(self.src_sentences)

    def collate_fn(self, batch: List[Tuple[str]], device: str = "cpu"):
        """
        预处理流程：
        1、把原始语料中的中英文句对按照英文句子的长度排序，使得每个batch中的句子长度相近。
        2、利用训练好的分词模型分别对中英文句子进行分词，利用词表将其转换为id。
        3、在每个 id sequence 的首尾加上起始符和终止符，并将其转换为Tensor。
        4、分别生成编码器、解码器的掩码；右移（shifted right）操作，对目标序列划分成输入、输出两部分。
        """
        # 1、分别对中英文句子进行分词、转换为token id， 并在每个 id sequence 的首尾加上起始符和终止符
        src_text, src_token_ids, tgt_text, tgt_token_ids = [], [], [], []
        for src_sentence, tgt_sentence in batch:
            src_text.append(src_sentence)
            src_token_ids.append(
                [self.src_tokenizer.bos_id()] + self.src_tokenizer.encode_as_id(src_sentence) + [self.src_tokenizer.eos_id()]
            )
            tgt_text.append(tgt_sentence)
            tgt_token_ids.append(
                [self.tgt_tokenizer.bos_id()] + self.tgt_tokenizer.encode_as_id(tgt_sentence) + [self.tgt_tokenizer.eos_id()]
            )

        # 2、Token id 序列转换为 Tensor，并将该批次不同长度的 Tensor 填充到等长 => (batch_size, seq_len)
        src_tensor = pad_sequence([torch.LongTensor(np.array(l_)) for l_ in src_token_ids],
                                  batch_first=True, padding_value=self.src_tokenizer.pad_id())
        tgt_tensor = pad_sequence([torch.LongTensor(np.array(l_)) for l_ in tgt_token_ids],
                                  batch_first=True, padding_value=self.tgt_tokenizer.pad_id())

        # 3、生成掩码
        # Encoder训练时的输入部分
        src_input = src_tensor
        src_mask = create_padding_mask(src_input, pad_token_id=self.src_tokenizer.pad_id())

        # shift-right构造标签
        # Decoder训练时的输入部分
        tgt_input = tgt_tensor[:, :-1]  # 去除每个序列的最后一个token
        tgt_mask = create_sequence_mask(tgt_input, pad_token_id=self.tgt_tokenizer.pad_id())
        # Decoder训练时的输出部分
        tgt_output = tgt_tensor[:, 1:]  # 去除每个序列的第一个token

        return MTBatch(src_text=src_text, src_input=src_input, src_mask=src_mask,
                       tgt_text=tgt_text, tgt_input=tgt_input, tgt_mask=tgt_mask,
                       tgt_output=tgt_output, device=device)


if __name__ == '__main__':
    _dataset = MTDataset("data/dataset/dev.jsonl")
    # for i in range(len(_dataset)):
    #     print(i, _dataset[i])



