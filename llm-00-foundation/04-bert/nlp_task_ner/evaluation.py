#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2025/5/22 19:49
# @function:
import logging
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from nlp_task_ner.data_loader import NERDataset
from nlp_task_ner.model.bert_crf import BertCrf
from nlp_task_ner.model.my_loss_func import CRFLoss
from nlp_task_ner.model.my_trainer import MyTrainer
from nlp_task_ner.util.modeling_util import count_trainable_parameters, build_optimizer


@hydra.main(version_base=None, config_path="conf", config_name="bert-crf.yaml")
def eval_bert_crf(config: DictConfig):
    logging.info(f"开始训练模型")
    logging.info("配置信息:\n{}".format(OmegaConf.to_yaml(config, resolve=True)))
    logging.info("加载Dataset和Tokenizer.")
    _tokenizer = BertTokenizer.from_pretrained(
        pretrained_model_name_or_path=config.pretrain_path,
        do_lower_case=True
    )
    test_dataset = NERDataset(config.test_data_path, config.label_data_path, tokenizer=_tokenizer)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=config.batch_size,
                                 collate_fn=test_dataset.collate_fn)
    model: BertCrf = BertCrf(pretrain_path=config.pretrain_path, num_labels=config.num_labels, dropout=config.dropout)
    model.load_state_dict(torch.load(config.model_path,
                                     weights_only=False,
                                     map_location=torch.device("cpu")))



if __name__ == '__main__':
    eval_bert_crf()
