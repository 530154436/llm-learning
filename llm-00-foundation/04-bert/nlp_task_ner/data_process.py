#!/usr/bin/env python3
# -*- coding:utf-8 -*--
import copy
import json
import logging as logger
from typing import List, Tuple, Iterable, Dict
from transformers import PreTrainedTokenizer, AutoTokenizer


class InputFeatures(object):
    """
    Bert输入特征
    """

    def __init__(self, input_ids: List[int], input_mask: List[int], token_type_ids: List[int],
                 input_len: int, label_ids: List[int]):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.token_type_ids = token_type_ids
        self.input_len = input_len
        self.label_ids = label_ids

    def __repr__(self):
        return str(self.to_json())

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def convert_text_to_features(sentences: List[Tuple[str, List[str]]],
                             label2id: Dict[str, int],
                             tokenizer: PreTrainedTokenizer) -> InputFeatures:
    """
    将输入文本转换为 BERT 的输入格式
    """
    examples = [(list(sent), ["O"] * len(sent)) for sent in sentences]
    return convert_examples_to_feature(examples, label2id, tokenizer)


def convert_examples_to_feature(examples: Iterable[Tuple[List[str], List[str]]],
                                label2id: Dict[str, int],
                                tokenizer: PreTrainedTokenizer,
                                max_seq_length: int = 512,
                                pad_token_id: int = 0,
                                cls_token: str = "[CLS]",
                                sep_token: str = "[SEP]") -> InputFeatures:
    """
    将训练数据转换为 BERT 的输入格式：
    (a) 对于句子对（sequence pairs）：
        tokens:       [CLS] 是 这 jack ##son ##ville ? [SEP] 不 是 的 . [SEP]
        token_type_ids:    0   0  0   0    0     0     0    0   1  1  1  1  1
    (b) 对于单个句子（single sequences）：
        tokens:       [CLS] 这只 狗 很 茸毛 . [SEP]
        token_type_ids :   0   0   0  0   0  0   0

    + [CLS]：每个序列开头都会加入这个特殊 token，用于表示整个句子的聚合信息，常用于分类任务。
    + [SEP]：用于分隔两个句子或标记一个句子的结束。
    + token_type_ids （segment_ids）：用于区分句子对中的不同句子。第1个句子的所有 token 标记为 0；第2个句子的所有 token 标记为 1；单独使用时全部为 0。

    输入示例
    输入：[
           (["北", "京", "城"], ["B-NT", "I-NT", "I-NT"])
         ]
    输出：
        tokens: 		[CLS] 北   京   城  [SEP]
        input_ids: 		101 1266 776 1814  102  0  0  0  0  0
        input_mask: 	1    1    1    1    1   0  0  0  0  0
        token_type_ids: 	0    0    0    0    0   0  0  0  0  0
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
    idx = 0
    for idx, (tokens, labels) in enumerate(examples, start=1):

        # 1、截断：如果输入长度超过限制，则截断至允许的最大长度（减去特殊标记长度）
        if len(tokens) > min_max_seq_length - special_tokens_count:
            tokens = tokens[:min_max_seq_length - special_tokens_count]
            labels = labels[:min_max_seq_length - special_tokens_count]

        # 2、将 token 转换为 id，并加上特殊标记 [CLS] 和 [SEP]
        input_ids = tokenizer.convert_tokens_to_ids([cls_token] + tokens + [sep_token])
        label_ids = [label2id['O']] + [label2id[label] for label in labels] + [label2id['O']]
        input_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)
        input_len = len(input_ids)

        # 3、padding：填充到固定长度 min_max_seq_length
        pad_length = min_max_seq_length - len(input_ids)
        input_ids += [pad_token_id] * pad_length
        label_ids += [pad_token_id] * pad_length
        input_mask += [pad_token_id] * pad_length
        token_type_ids += [pad_token_id] * pad_length

        assert len(input_ids) == min_max_seq_length
        assert len(input_mask) == min_max_seq_length
        assert len(token_type_ids) == min_max_seq_length
        assert len(label_ids) == min_max_seq_length
        if idx <= 2:
            logger.info("*** Example ***")
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
        if idx % 1000 == 0:
            logger.info("processing example: no.%s", idx)

        yield InputFeatures(input_ids=input_ids,
                            input_mask=input_mask,
                            token_type_ids=token_type_ids,
                            label_ids=label_ids,
                            input_len=input_len)
    logger.info("processing done, total = %s", idx)


if __name__ == '__main__':
    _examples = [
        (["北", "京", "城"], ["B-NT", "I-NT", "I-NT"]),
        (['藏', '家', '1', '2', '条', '收', '藏', '秘', '籍'], ['B-p', 'I-p', 'O', 'O', 'O', 'O', 'O', 'O', 'O'])
    ]
    _label2id = {"O": 0, "B-NT": 1, "I-NT": 2, "B-p": 3, "I-p": 4}
    _tokenizer = AutoTokenizer.from_pretrained("data/pretrain/bert-base-chinese")
    for item in convert_examples_to_feature(_examples, _label2id, _tokenizer, max_seq_length=10):
        print(item.to_dict())

    _sentences = ["北京城", "藏家12条收藏秘籍"]
    for item in convert_text_to_features(_sentences, _label2id, _tokenizer):
        print(item.to_dict())
