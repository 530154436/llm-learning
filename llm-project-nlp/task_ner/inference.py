#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2025/5/23 13:40
# @function:
import json
import torch
from typing import List, Dict
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from transformers import BertTokenizer
from seqeval.metrics.sequence_labeling import get_entities
from task_ner.data_process import convert_text_to_features
from task_ner.model import BaseNerModel
from task_ner.model.bert_crf import BertCrf
from task_ner.model.bert_bilstm_crf import BertBiLstmCrf


def load_model_by_name(config: DictConfig,
                       device: str = "cpu") -> BaseNerModel:
    # 命名格式：模型名称_预训练模型
    model_name = config.model_name.split("_")[0]
    if model_name.__contains__("BertCrf"):
        model = BertCrf(pretrain_path=config.pretrain_path, num_labels=config.num_labels)
    elif model_name.__contains__("BertBiLstmCrf"):
        model = BertBiLstmCrf(pretrain_path=config.pretrain_path, num_labels=config.num_labels,
                              lstm_num_layers=config.lstm_num_layers, lstm_hidden_size=config.lstm_hidden_size)
    else:
        raise ValueError(f"Unknown model name: {config.model_name}")
    model.load_state_dict(torch.load(config.model_path,
                                     weights_only=False,
                                     map_location=torch.device(device)))
    return model


class Predictor(object):
    """ 序列标注模型推理
    """
    def __init__(self, config_path: str):
        config: DictConfig = OmegaConf.load(config_path)
        logger.info("加载配置信息:\n{}".format(OmegaConf.to_yaml(config, resolve=True)))

        self.config = config
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=config.pretrain_path,
                                                       do_lower_case=True)
        self.label2id: dict = json.load(open(config.label_data_path, 'r', encoding='utf-8'))
        self.id2label: dict = {v: k for k, v in self.label2id.items()}
        self.model: BaseNerModel = load_model_by_name(config, device=self.device)

    @torch.no_grad()
    def predict(self, sentences: List[str]) -> List[dict]:
        """
        序列标注预测
        """
        if not sentences:
            return []
        input_ids, input_mask, segment_ids = convert_text_to_features(sentences,
                                                                      label2id=self.label2id,
                                                                      tokenizer=self.tokenizer)
        input_ids = input_ids.to(self.device)
        input_mask = input_mask.to(self.device)
        segment_ids = segment_ids.to(self.device)
        preds = self.model.predict(input_ids, input_mask, segment_ids).tolist()

        del input_ids, input_mask, segment_ids

        # 获取真实的标签名称
        result = []
        for sentence, y_pred in zip(sentences, preds):
            # 构造输入为 [CLS] xxx [SEP]，所以需要去掉前后的特殊字符
            labels = [self.id2label.get(y) for y in y_pred[1: len(y_pred) - 1]]
            entities = get_entities(labels)
            result_sub = []
            for (label, start, end) in entities:
                result_sub.append({
                    "text": sentence[start: end+1],
                    "label": label,
                    "start": start,
                    "end": end,
                })
            result.append(result_sub)
        return result


if __name__ == "__main__":
    _sentences = [
        "北京城",
        "价格高昂的大钻和翡翠消费为何如此火？通灵珠宝总裁沈东军认为，这与原料稀缺有直接关系。"
    ]
    # predictor = Predictor("conf/BertCrf.yaml")
    # predictor = Predictor("conf/BertBiLstmCrf.yaml")
    predictor = Predictor("conf/BertBiLstmCrf_chinese-bert-wwm-ext.yaml")
    # predictor = Predictor("conf/BertBiLstmCrf_chinese-roberta-wwm-ext.yaml")
    for item in predictor.predict(_sentences):
        print(item)
