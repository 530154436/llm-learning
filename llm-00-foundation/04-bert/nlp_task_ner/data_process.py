#!/usr/bin/env python3
# -*- coding:utf-8 -*--
import json
from typing import Tuple


def process_line(line: str) -> Tuple[str, str]:
    """ 处理每一行，转为 BISO 格式
    """
    # loads()：用于处理内存中的json对象，strip去除可能存在的空格
    json_line: dict = json.loads(line.strip())
    word_list, label_list = [], []

    text = json_line['text']
    words = list(text)
    # 如果没有label，则返回None
    label_entities: dict = json_line.get('label', None)
    if label_entities is None:
        return None
    labels = ['O'] * len(words)
    for key, value in label_entities.items():
        for sub_name, sub_index in value.items():
            for start_index, end_index in sub_index:
                assert ''.join(words[start_index:end_index + 1]) == sub_name
                if start_index == end_index:
                    labels[start_index] = 'S-' + key
                else:
                    labels[start_index] = 'B-' + key
                    labels[start_index + 1:end_index + 1] = ['I-' + key] * (len(sub_name) - 1)
    assert len(words) == len(labels)
    return words, labels
