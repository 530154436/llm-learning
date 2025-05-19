#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2025/5/6 17:39
# @function:
from torch import nn
from torch.optim import AdamW


def initialize_weights(model: nn.Module):
    """ 初始化模型权重
    """
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)


def count_trainable_parameters(model: nn.Module):
    """ 计算模型参数
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# model.named_parameters(): [bert, bilstm, classifier, crf]
bert_optimizer = list(model.bert.named_parameters())
lstm_optimizer = list(model.bilstm.named_parameters())
classifier_optimizer = list(model.classifier.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in bert_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay': weight_decay},
    {'params': [p for n, p in bert_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay': 0.0},
    {'params': [p for n, p in lstm_optimizer if not any(nd in n for nd in no_decay)],
     'lr': learning_rate * 5, 'weight_decay': weight_decay},
    {'params': [p for n, p in lstm_optimizer if any(nd in n for nd in no_decay)],
     'lr': learning_rate * 5, 'weight_decay': 0.0},
    {'params': [p for n, p in classifier_optimizer if not any(nd in n for nd in no_decay)],
     'lr': learning_rate * 5, 'weight_decay': weight_decay},
    {'params': [p for n, p in classifier_optimizer if any(nd in n for nd in no_decay)],
     'lr': learning_rate * 5, 'weight_decay': 0.0},
    {'params': model.crf.parameters(), 'lr': learning_rate * 5}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
return optimizer

def build_optimizer(self,
                    model: nn.Module,
                    learning_rate: float = 3e-5,
                    weight_decay: float = 0.01):
    """

    """
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = []

    def param_groups_from_module(module: nn.Module):
        params = list(module.named_parameters())
        decay_params = [p for n, p in params if not any(nd in n for nd in no_decay)]
        no_decay_params = [p for n, p in params if any(nd in n for nd in no_decay)]
        return [
            {'params': decay_params, 'weight_decay': weight_decay, 'lr': learning_rate * lr_multiplier},
            {'params': no_decay_params, 'weight_decay': 0.0, 'lr': learning_rate * lr_multiplier}
        ]

    # 遍历所有子模块并根据类型添加参数分组
    for name, module in model.named_modules():
        if isinstance(module, (nn.modules.transformer.TransformerEncoderLayer, nn.Module)) and 'Bert' in name:
            optimizer_grouped_parameters.extend(param_groups_from_module(module))
        elif isinstance(module, nn.LSTM):
            optimizer_grouped_parameters.extend(param_groups_from_module(module, lr_multiplier=5))
        elif isinstance(module, nn.Linear):
            optimizer_grouped_parameters.extend(param_groups_from_module(module, lr_multiplier=5))
        elif hasattr(module, '__class__') and 'CRF' in module.__class__.__name__:
            optimizer_grouped_parameters.append({'params': module.parameters(), 'lr': learning_rate * 5})

    # 如果没有匹配到任何模块，直接使用默认的AdamW设置整个模型参数
    if not optimizer_grouped_parameters:
        optimizer_grouped_parameters = [{'params': model.parameters(), 'lr': learning_rate}]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, correct_bias=False)
    return optimizer