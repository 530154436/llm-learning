#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2025/5/10 14:38
# @function:
import logging
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torchmetrics import F1Score, MetricCollection, Accuracy
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from nlp_task_ner.data_loader import NERDataset
from nlp_task_ner.model.bert_bilstm_crf import BertBiLstmCrf
from modeling_util.loss_func import CRFLoss
from modeling_util.my_trainer import MyTrainer
from modeling_util.model_util import count_trainable_parameters, build_optimizer


# @hydra.main(version_base=None, config_path="conf", config_name="BertBiLstmCrf.yaml")
# @hydra.main(version_base=None, config_path="conf", config_name="BertBiLstmCrf_chinese-bert-wwm-ext.yaml")
@hydra.main(version_base=None, config_path="conf", config_name="BertBiLstmCrf_chinese-roberta-wwm-ext.yaml")
def train(config: DictConfig):
    """ 模型训练
    data_dir: ./data
    train_data_path: ./data/dataset/clue/train.jsonl
    dev_data_path: ./data/dataset/clue/dev.jsonl
    test_data_path: ./data/dataset/clue/test.jsonl
    label_data_path: ./data/dataset/clue/label.json
    num_labels: 31
    model_name: BertBiLstmCrf
    model_path: ./data/experiment/BertBiLstmCrf.pth
    device: cuda:0
    pretrain_path: ./data/pretrain/bert-base-chinese
    lstm_num_layers: 1
    lstm_hidden_size: 128
    batch_size: 64
    dropout: 0.3
    epoch_num: 50
    learning_rate: 3.0e-05
    """
    logging.info(f"开始训练模型")
    logging.info("配置信息:\n{}".format(OmegaConf.to_yaml(config, resolve=True)))
    logging.info("加载Dataset和Tokenizer.")
    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    tokenizer = BertTokenizer.from_pretrained(
        pretrained_model_name_or_path=config.pretrain_path,
        do_lower_case=True
    )
    train_dataset = NERDataset(config.train_data_path, config.label_data_path, tokenizer=tokenizer)
    dev_dataset = NERDataset(config.dev_data_path, config.label_data_path, tokenizer=tokenizer)
    train_size = len(train_dataset)
    num_labels = len(train_dataset.label2id)

    logging.info("加载DataLoader")
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size,
                                  collate_fn=train_dataset.collate_fn)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=config.batch_size,
                                collate_fn=dev_dataset.collate_fn)
    logging.info("初始化模型")
    model: BertBiLstmCrf = BertBiLstmCrf(pretrain_path=config.pretrain_path, num_labels=num_labels, dropout=config.dropout,
                                         lstm_num_layers=config.lstm_num_layers, lstm_hidden_size=config.lstm_hidden_size)
    logging.info(f'模型训练参数: {count_trainable_parameters(model)}')
    print(model)

    logging.info("配置优化器、学习率调整器、损失函数、评估指标")
    optimizer = build_optimizer(model, learning_rate=config.learning_rate)
    train_steps_per_epoch = train_size // config.batch_size
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                num_warmup_steps=(config.epoch_num // 10) * train_steps_per_epoch,
                                                num_training_steps=config.epoch_num * train_steps_per_epoch)
    loss_fn = CRFLoss(model.crf, pad_token_id=train_dataset.pad_token_id)
    metrics = MetricCollection({
        'acc': Accuracy(task="multiclass", num_classes=num_labels),
        'f1': F1Score(task="multiclass", num_classes=num_labels)
    })

    logging.info("训练模型...")
    trainer = MyTrainer(model=model, loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler,
                        n_epoch=config.epoch_num, device=device, model_path=config.model_path,
                        metrics=metrics)
    trainer.fit(train_dataloader, dev_dataloader)


if __name__ == '__main__':
    train()
