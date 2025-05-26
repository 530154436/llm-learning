#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2025/5/19 9:46
# @function:
from typing import List
import torch
from torch import nn
from transformers import BertModel, BertConfig
from torchcrf import CRF
from nlp_task_ner.model import BaseNerModel
import warnings
warnings.filterwarnings("ignore", message="where received a uint8 condition tensor")


class BertCrf(BaseNerModel):

    def __init__(self, pretrain_path: str, num_labels: int, dropout: float = 0.3):

        super().__init__(pretrain_path, num_labels)
        self.bert_config = BertConfig.from_pretrained(pretrain_path)
        self.bert = BertModel.from_pretrained(pretrain_path)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(self.bert_config.hidden_size, num_labels)
        self.crf = CRF(num_tags=num_labels, batch_first=True)

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                token_type_ids: torch.Tensor) -> torch.FloatTensor:
        """
        正向传播
        :param input_ids: [batch_size, seq_len]
        :param attention_mask: [batch_size, seq_len]
        :param token_type_ids: [batch_size, seq_len]

        :return: [batch_size, seq_len, num_labels]
        """
        # [batch_size, seq_len] => [batch_size, seq_len, embedding_size]
        output = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]
        output = self.dropout(output)

        # [batch_size, seq_len, num_labels]
        logits = self.linear(output)

        return logits

    def predict(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                token_type_ids: torch.Tensor) -> torch.Tensor:
        """
        :param input_ids: [batch_size, seq_len, num_labels]
        :param attention_mask: [batch_size, seq_len, num_labels]
        :param token_type_ids: [batch_size, seq_len, num_labels]
        :return: torch.Tensor [batch_size, seq_len]
        """
        # [batch_size, seq_len, num_labels]
        logits = self.forward(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # [batch_size, seq_len]
        # predicts: List[list] = self.crf.decode(logits, mask=attention_mask.bool())
        predicts: List[list] = self.crf.decode(logits)
        return torch.tensor(predicts, dtype=torch.int32)


if __name__ == "__main__":
    _pretrain_path = "../data/pretrain/bert-base-chinese"
    _model = BertCrf(_pretrain_path, num_labels=31)
    # for name, param in list(_model.named_parameters()):
    #     print(name)
    print(_model)
    for name, module in _model.named_modules():
        print(name, type(module))
