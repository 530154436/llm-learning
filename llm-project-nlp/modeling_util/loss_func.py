#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2025/5/22 17:35
# @function:
import torch
import torch.nn as nn
from typing import Optional
from torchcrf import CRF


class CRFLoss(nn.Module):

    def __init__(self, crf: CRF, pad_token_id: int = 0):
        super(CRFLoss, self).__init__()
        self.crf = crf
        self.pad_token_id = pad_token_id

    def forward(self,
                logits: torch.Tensor,
                labels: torch.Tensor,
                input_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        负对数似然损失
        logits: [batch_size, seq_len, num_labels]
        labels: [batch_size, seq_len]
        input_ids: [batch_size, seq_len]，通过原始输入计算出 attention_mask
        """
        if input_ids is not None:
            attention_mask: torch.Tensor = (input_ids != self.pad_token_id)
            loss = -self.crf(logits, labels, mask=attention_mask.bool(), reduction='mean')
        else:
            loss = -self.crf(logits, labels, reduction='mean')
        return loss
