#!/usr/bin/env python3
# -*- coding:utf-8 -*--
import copy
import json
import logging
import torch
from typing import List, Tuple, Iterable, Dict
from transformers import PreTrainedTokenizer, AutoTokenizer

# 配置 logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(filename)s[:%(lineno)d] %(message)s')


class InputFeatures(object):
    """
    Bert输入特征
    """

    def __init__(self, input_ids, input_mask, input_len, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.input_len = input_len

    def __repr__(self):
        return str(self.to_json())

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def convert_text_to_features(sentences: List[Tuple[str, List[str]]],
                             tokenizer: PreTrainedTokenizer,
                             max_seq_length: int = 512) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pass


def convert_examples_to_feature(examples: Iterable[Tuple[List[str], List[str]]],
                                label2id: Dict[str, int],
                                tokenizer: PreTrainedTokenizer,
                                max_seq_length: int = 512,
                                pad_token_id: int = 0,
                                cls_token: str = "[CLS]",
                                sep_token: str = "[SEP]") -> Tuple[torch.Tensor]:
    """
    将训练数据转换为 BERT 的输入格式：
    (a) 对于句子对（sequence pairs）：
        tokens:       [CLS] 是 这 jack ##son ##ville ? [SEP] 不 是 的 . [SEP]
        segment_ids:    0   0  0   0    0     0     0    0   1  1  1  1  1
    (b) 对于单个句子（single sequences）：
        tokens:       [CLS] 这只 狗 很 茸毛 . [SEP]
        segment_ids :   0   0   0  0   0  0   0

    + [CLS]：每个序列开头都会加入这个特殊 token，用于表示整个句子的聚合信息，常用于分类任务。
    + [SEP]：用于分隔两个句子或标记一个句子的结束。
    + segment_ids （token_type_ids）：用于区分句子对中的不同句子。第1个句子的所有 token 标记为 0；第2个句子的所有 token 标记为 1；单独使用时全部为 0。

    输入示例
    输入：[
           (["北", "京", "城"], ["B-NT", "I-NT", "I-NT"])
         ]
    输出：
        tokens: 		[CLS] 北   京   城  [SEP]
        input_ids: 		101 1266 776 1814  102  0  0  0  0  0
        input_mask: 	1    1    1    1    1   0  0  0  0  0
        segment_ids: 	0    0    0    0    0   0  0  0  0  0
        label_ids: 		0    1    2    2    0   0  0  0  0  0
    示例：
        _examples = [("北京城", ["B-NT", "I-NT", "I-NT"])]
        _label_list = ["O", "B-NT", "I-NT"]
        _tokenizer = AutoTokenizer.from_pretrained(BASE_DIR.joinpath("data", "modelfiles", "bert-base-chinese"))
        f = convert_example2features(_examples, _label_list, max_seq_length=10, tokenizer=_tokenizer)
        print(len(list(f)))
    """
    # 找到满足所有文本的最小最大长度
    special_tokens_count = 2  # [CLS]、[SEP]
    min_max_seq_length = min(max(map(lambda x: len(x[0]), examples)) + special_tokens_count, max_seq_length)
    print(min_max_seq_length)
    idx = 0
    for idx, (tokens, labels) in enumerate(examples, start=1):
        print(tokens)
        # 1、
        if len(tokens) > min_max_seq_length - special_tokens_count:
            tokens = tokens[:min_max_seq_length - special_tokens_count]
            labels = labels[:min_max_seq_length - special_tokens_count]

        # 2、
        input_ids = tokenizer.convert_tokens_to_ids([cls_token] + tokens + [sep_token])
        label_ids = [label2id['O']] + [label2id[label] for label in labels] + [label2id['O']]
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(tokens)
        input_len = len(input_ids)

        # 3、
        pad_length = min_max_seq_length - len(input_ids)
        input_ids += [pad_token_id] * pad_length
        label_ids += [pad_token_id] * pad_length
        input_mask += [pad_token_id] * pad_length
        segment_ids += [pad_token_id] * pad_length

        assert len(input_ids) == min_max_seq_length
        assert len(input_mask) == min_max_seq_length
        assert len(segment_ids) == min_max_seq_length
        assert len(label_ids) == min_max_seq_length

        if idx <= 3:
            logging.info("*** Example ***")
            logging.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
        if idx % 1000 == 0:
            logging.info("processing example: no.%s", idx)

        yield InputFeatures(input_ids=input_ids,
                            input_mask=input_mask,
                            segment_ids=segment_ids,
                            label_ids=label_ids,
                            input_len=input_len)
    logging.info("processing done, total = %s", idx)


if __name__ == '__main__':
    _examples = [
        (["北", "京", "城"], ["B-NT", "I-NT", "I-NT"]),
        (['藏', '家', '1', '2', '条', '收', '藏', '秘', '籍'], ['B-p', 'I-p', 'O', 'O', 'O', 'O', 'O', 'O', 'O'])
    ]
    _label2id = {"O": 0, "B-NT": 1, "I-NT": 2}
    _tokenizer = AutoTokenizer.from_pretrained("data/pretrain/bert-base-chinese")
    for item in convert_examples_to_feature(_examples, _label2id, _tokenizer, max_seq_length=10):
        print(item)
