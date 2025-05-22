#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2025/5/10 14:38
# @function:
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from nlp_task_ner.data_loader import NERDataset
from nlp_task_ner.model.bert_crf import BertCrf
from nlp_task_ner.model.my_loss_func import CRFLoss
from nlp_task_ner.model.my_trainer import MyTrainer
from nlp_task_ner.util.modeling_util import count_trainable_parameters, build_optimizer


@hydra.main(version_base=None, config_path="conf", config_name="config.yaml")
def main(config: DictConfig):
    logging.info(f"开始训练模型")
    logging.info("配置信息:\n{}".format(OmegaConf.to_yaml(config, resolve=True)))
    logging.info("加载Dataset和Tokenizer.")
    _tokenizer = BertTokenizer.from_pretrained(
        pretrained_model_name_or_path=config.model.pretrain_path,
        do_lower_case=True
    )
    train_dataset = NERDataset(config.dataset.train_data_path, config.dataset.label_data_path, tokenizer=_tokenizer)
    dev_dataset = NERDataset(config.dataset.dev_data_path, config.dataset.label_data_path, tokenizer=_tokenizer)
    test_dataset = NERDataset(config.dataset.test_data_path, config.dataset.label_data_path, tokenizer=_tokenizer)
    train_size = len(train_dataset)
    num_labels = len(train_dataset.label2id)

    logging.info("加载DataLoader")
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=config.model.batch_size,
                                  collate_fn=train_dataset.collate_fn)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=config.model.batch_size,
                                collate_fn=dev_dataset.collate_fn)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=config.model.batch_size,
                                 collate_fn=test_dataset.collate_fn)

    logging.info("初始化模型")
    model: BertCrf = BertCrf(pretrain_path=config.model.pretrain_path,
                             num_labels=num_labels,
                             dropout=config.model.dropout)
    logging.info(f'模型训练参数: {count_trainable_parameters(model)}')
    print(model)

    logging.info("配置优化器、学习率调整器、损失函数")
    optimizer = build_optimizer(model, learning_rate=config.model.learning_rate)
    train_steps_per_epoch = train_size // config.model.batch_size
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                num_warmup_steps=(config.model.epoch_num // 10) * train_steps_per_epoch,
                                                num_training_steps=config.model.epoch_num * train_steps_per_epoch)
    loss_fn = CRFLoss(model.crf, pad_token_id=train_dataset.pad_token_id)

    logging.info("训练模型...")
    trainer = MyTrainer(model=model, loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler,
                        n_epoch=config.model.epoch_num, device=config.model.device,
                        model_path=config.model.model_path)
    trainer.fit(train_dataloader, dev_dataloader)


if __name__ == '__main__':
    main()
