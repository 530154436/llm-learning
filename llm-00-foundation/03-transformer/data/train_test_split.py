#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2025/4/27 20:25
# @function:
import json
import os.path
import random

random.seed(1024)


def merge_and_split_files(
        src_file: str,
        tgt_file: str,
        train_ratio: float = 0.7,
        dev_ratio: float = 0.1,
        limit: int = None,
):
    # 读取源语言和目标语言的文件内容
    with open(src_file, 'r', encoding='utf-8') as src, open(tgt_file, 'r', encoding='utf-8') as tgt:
        src_lines = src.readlines() if limit is None else src.readlines()[:limit]
        tgt_lines = tgt.readlines() if limit is None else tgt.readlines()[:limit]

    # 确保两个文件的行数相同
    assert len(src_lines) == len(tgt_lines), "源文件和目标文件的行数不匹配"

    # 合并成元组列表
    data = list(zip(src_lines, tgt_lines))

    # 打乱数据
    random.shuffle(data)

    # 计算分割点
    total = len(data)
    train_size = int(total * train_ratio)
    dev_size = int(total * dev_ratio)

    # 分割数据集
    train_data = data[:train_size]
    dev_data = data[train_size:train_size + dev_size]
    test_data = data[train_size + dev_size:]

    # 写入文件
    for dataset, name, suffix in zip([train_data, dev_data, test_data],
                                     ["train", "dev", "test"],
                                     ['jsonl', 'jsonl', 'jsonl']):
        if not os.path.exists("./dataset"):
            os.mkdir("./dataset")
        with open(f"./dataset/{name}.{suffix}", 'w', encoding='utf-8') as outfile:
            for zh, en in dataset:
                _dict = {"zh": zh, "en": en}
                outfile.write(json.dumps(_dict, ensure_ascii=False))
                outfile.write("\n")


if __name__ == '__main__':
    # 整个语料库共 252777 条
    # 按7:1:2的比例进行划分: 训练集（约 176943 条）、验证集（约25278条）和测试集（约 50556 条）三部分；
    merge_and_split_files('news-commentary-v13.zh-en.zh',
                          'news-commentary-v13.zh-en.en')
