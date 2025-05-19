#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2025/5/19 9:46
# @function:
from typing import List
import torch
from torch import nn
from transformers import BertModel, BertPreTrainedModel, BertConfig
from torchcrf import CRF


class BertCrf(BertPreTrainedModel):

    def __init__(self,
                 pretrain_path: str,
                 num_labels: int,
                 dropout: float = 0.3):
        self.bert_config = BertConfig.from_pretrained(pretrain_path)
        super().__init__(self.bert_config)
        self.bert = BertModel.from_pretrained(pretrain_path)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert_config.hidden_size, num_labels)
        self.crf = CRF(num_tags=num_labels, batch_first=True)

        self.init_weights()

    def predict_proba(self,
                      input_ids: torch.Tensor,
                      attention_mask: torch.Tensor,
                      token_type_ids: torch.Tensor) -> torch.Tensor:
        logits = self.forward(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # 预测解码
        predicts: List[list] = self.crf.decode(logits, mask=attention_mask.bool())
        return predicts

    def loss_fn(self,
                logits: torch.Tensor,
                labels: torch.Tensor,
                attention_mask: torch.Tensor = None) -> torch.FloatTensor:
        """
        损失函数: CRF负对数似然损失
        :param logits: [batch_size, seq_len, num_labels]
        :param labels: [batch_size, seq_len]
        :param attention_mask: [batch_size, seq_len]
        """
        if attention_mask is not None:
            loss = -self.crf(logits, labels, attention_mask.bool(), reduction="mean")
        else:
            loss = -self.crf(logits, labels, reduction="mean")
        return loss

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
        logits = self.classifier(output)

        return logits
