#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2025/5/10 14:13
# @function:
import json
import os.path
import random
from data_process import process_line
random.seed(1024)


def pipeline(
    train_file: str,
    dev_file: str
):
    """
    命名实体数据集构造
    1、原始数据集划分为训练集、验证集，原始验证集作为测试集
    2、文本格式转换为命名实体识别BIOS格式
    3、记录所有实体标签
    """
    if not os.path.exists("./dataset"):
        os.mkdir("./dataset")
    if not os.path.exists("./dataset/clue"):
        os.mkdir("./dataset/clue")
    # 标签全集
    all_labels = []

    # 原始训练集按 9:1 拆分成训练集合验证
    with open(train_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        # 打乱数据
        random.shuffle(lines)

        # 计算分割点
        total = len(lines)
        train_size = int(total * 0.9)
        train_data = lines[:train_size]
        dev_data = lines[train_size:]

        # 写入文件
        for dataset, name, suffix in zip([train_data, dev_data],
                                         ["train", "dev"],
                                         ['jsonl', 'jsonl']):
            with open(f"./dataset/clue/{name}.{suffix}", 'w', encoding='utf-8') as writer:
                for line in dataset:
                    words, labels = process_line(line)
                    for label in labels:
                        if label not in all_labels:
                            all_labels.append(label)
                    writer.write("\n".join(f"{w} {l}" for w, l in zip(words, labels)))
                    writer.write("\n\n")
                    writer.flush()
        with open(f"./dataset/clue/label.json", 'w', encoding='utf-8') as writer:
            _dict = dict()
            for i, label in enumerate(all_labels):
                _dict[label] = i
            json.dump(_dict, writer, ensure_ascii=False, indent=4)

    # 原始验证集作为测试集
    with open(dev_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        with open("./dataset/clue/test.jsonl", 'w', encoding='utf-8') as writer:
            for line in lines:
                words, labels = process_line(line)
                writer.write("\n".join(f"{w} {l}" for w, l in zip(words, labels)))
                writer.write("\n\n")


if __name__ == '__main__':
    # 任务详情：CLUENER2020
    # 训练集：10748 验证集：1343
    pipeline('clue.train.jsonl', 'clue.dev.jsonl')
