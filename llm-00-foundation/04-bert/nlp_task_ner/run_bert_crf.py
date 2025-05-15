#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2025/5/10 14:38
# @function:
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from nlp_task_ner.data_loader import NERDataset


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

    logging.info("加载DataLoader")
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=config.model.batch_size,
                                  collate_fn=train_dataset.collate_fn)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=config.model.batch_size,
                                collate_fn=dev_dataset.collate_fn)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=config.model.batch_size,
                                 collate_fn=test_dataset.collate_fn)

    # logging.info("初始化模型")
    # model = Transformer(config.src_vocab_size, config.tgt_vocab_size,
    #                     d_model=config.d_model, num_heads=config.n_heads, d_ff=config.d_ff,
    #                     dropout=config.dropout, N=config.n_layers)
    # # print(model)
    # logging.info(f'模型训练参数: {count_trainable_parameters(model)}')
    # logging.info("训练模型...")
    # train(model, train_dataloader, dev_dataloader)


if __name__ == '__main__':
    main()
