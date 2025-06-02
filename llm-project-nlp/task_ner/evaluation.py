#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2025/5/24 9:45
# @function:
import json
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from task_ner.data_loader import NERDataset
from task_ner.inference import load_model_by_name
from task_ner.model import BaseNerModel
from seqeval.metrics.v1 import classification_report


def evaluate(config_path: str):
    config: DictConfig = OmegaConf.load(config_path)
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.pop("hydra")  # 评估阶段不需要
    logger.info("配置信息:\n{}".format(OmegaConf.to_yaml(config, resolve=True)))

    # 加载模型
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=config.pretrain_path, do_lower_case=True)
    label2id: dict = json.load(open(config.label_data_path, 'r', encoding='utf-8'))
    id2label: dict = {v: k for k, v in label2id.items()}
    model: BaseNerModel = load_model_by_name(config, device=config.device)

    # 加载数据集
    test_dataset = NERDataset(config.test_data_path, config.label_data_path, tokenizer=tokenizer)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=config.batch_size,
                                 collate_fn=test_dataset.collate_fn)

    y_trues, y_preds = [], []
    for step, xy in enumerate(test_dataloader, start=1):
        xy_tuple: tuple = tuple(x.to(config.device) for x in xy)
        y_pred: torch.Tensor = model.predict(*xy_tuple[:-1])  # 不传label
        y_true: torch.Tensor = xy_tuple[-1]
        print(step, y_true.shape, y_pred.shape)
        assert y_pred.shape == y_true.shape
        # 转为实体标签
        for y_pred_i, y_true_i in zip(y_pred.tolist(), y_true.tolist()):
            y_preds.append([id2label.get(j) for j in y_pred_i])
            y_trues.append([id2label.get(j) for j in y_true_i])
    print(classification_report(y_trues, y_preds))


if __name__ == '__main__':
    evaluate("conf/BertCrf.yaml")
    # evaluate("conf/BertBiLstmCrf.yaml")
    # evaluate("conf/BertBiLstmCrf_chinese-bert-wwm-ext.yaml")
    # evaluate("conf/BertBiLstmCrf_chinese-roberta-wwm-ext.yaml")

