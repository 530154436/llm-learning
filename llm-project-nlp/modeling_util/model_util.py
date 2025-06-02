#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2025/5/6 17:39
# @function:
import torch
from torch import nn
from torch.optim import AdamW, Optimizer


def auto_device() -> str:
    """ 根据设备获取设备
    """
    if torch.cuda.is_available():
        return "cuda:0"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


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


def build_optimizer(model: nn.Module,
                    learning_rate: float = 3e-5,
                    weight_decay: float = 0.01) -> Optimizer:
    """
    L2 正则化（也称为权重衰减）
    权重衰减 是一种正则化技术（等价于L2正则化），通过惩罚大权重值防止模型过拟合。 然而，某些参数（如偏置项、归一化层的参数）通常不需要应用权重衰减，因为它们对模型的正则化效果贡献较小，甚至可能损害训练稳定性。
    1、精细控制不同模块的学习率
    2、避免对特定参数（如 bias、LayerNorm）进行正则化
    • 偏置项（bias）： 偏置参数通常用于调整神经元的激活阈值，其数值大小对模型的表达能力影响较小，正则化反而可能抑制灵活性。
    • 归一化层参数（LayerNorm）： 层归一化的 weight（缩放参数）和 bias（偏移参数）本质上是学习数据分布的均值和方差，对其进行正则化会破坏归一化的统计特性，可能导致训练不稳定。
    3、提升模型训练效率与性能
    :param model: 模型对象
    :param learning_rate: 基础学习率（用于BERT）
    :param weight_decay: 权重衰减系数
    :return optimizer: 配置好的 AdamW 优化器
    """
    params = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight', 'BatchNorm.bias', 'BatchNorm.weight']

    # 需要权重衰减的参数（排除 no_decay 列表中的参数），比如: bert.embeddings.word_embeddings.weight
    decay_params = [p for n, p in params if not any(nd in n for nd in no_decay)]
    # 不需要权重衰减的参数（no_decay 列表中的参数），比如 bert.embeddings.LayerNorm.weight
    no_decay_params = [p for n, p in params if any(nd in n for nd in no_decay)]
    params = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]
    optimizer = AdamW(params, lr=learning_rate)
    return optimizer
