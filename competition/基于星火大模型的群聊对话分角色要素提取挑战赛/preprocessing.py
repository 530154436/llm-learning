#!/usr/bin/env python3
# -*- coding:utf-8 -*--
# 预处理步骤（正则表达式）
# 1、删除群聊对话记录中的重复内容。
# 2、删除特殊符号，如"[链接]"、"[图片]"、"[文件]"、"[撤回消息]"、"[抱拳]"、"[未知消息]"、"[微笑]"等以及一些分隔线、引用信息。
# 3、对于一个人连续的对话将其合并成一个对话
import json

import pandas as pd
import re

from tqdm import tqdm

from src.utils.io import read_json, write_json
# |(https?://\S+)
DIRTY_REGEX = re.compile('(\[[^\[\]]{2,10}\])|(【.*?】)|(data-online.*)|((- -)+)|((--)+)|(「.*?」)'
                         '|(这是一条引用/回复消息：)', re.DOTALL)
NAME_REGEX = re.compile(r"(^[\u4e00-\u9fa5]+\d+)[： ]")


def drop_duplication(chat_text: str):
    """
    实现了从一个包含多段文本的字典中去重重复段落的功能，同时保留了原有的顺序。
    https://blog.csdn.net/qq_44511981/article/details/140043813?csdn_share_tail=%7B%22type%22%3A%22blog%22%2C%22rType%22%3A%22article%22%2C%22rId%22%3A%22140043813%22%2C%22source%22%3A%22qq_44511981%22%7D
    """
    chat_list = chat_text.split("\n")
    # 用于存储需要移除的段落的索引
    res = []
    s = 0
    # 循环遍历 chat_list 中的每个段落。
    while s < len(chat_list):
        i, j = s, s + 1
        start_j = j
        while i < len(chat_list) and j < len(chat_list):
            if chat_list[i] == chat_list[j]:
                i += 1
            else:
                if i != s:
                    if j - start_j > 10:
                        res += list(range(start_j, j))
                    i = s
                start_j = j
            j += 1
        s += 1

    texts = []
    for i in range(len(chat_list)):
        if i not in res:
            texts.append(chat_list[i])
    return "\n".join(texts)


def clean_data_by_regex(chat_text: str):
    """
    清洗数据
    删除特殊符号，如"[链接]"、"[图片]"、"[文件]"、"[撤回消息]"、"[抱拳]"、"[未知消息]"、"[微笑]"等以及一些分隔线。
    """
    texts = []
    chat_list = chat_text.split("\n")
    for chat in chat_list:
        cleaned = DIRTY_REGEX.sub(" ", chat)
        texts.append(cleaned.strip())
    return "\n".join(texts)


def merge_chat(chat_text: str):
    """
    对于一个人连续的对话将其合并成一个对话
    """
    texts = []
    chat_list = chat_text.split("\n")
    last_name, name = "", ""
    conversations = []
    for index, chat in enumerate(chat_list, start=1):
        names = NAME_REGEX.findall(chat)
        # if len(names) == 0 and index == 1:
        #     raise ValueError("用户不能为空。index=%s, chat=%s" % (index, chat))
        if len(names) >= 1:
            if index == 1:
                last_name = names[0]
            name = names[0]
            conversation = chat[len(names[0])+1:].strip()
        else:
            name = ""
            conversation = chat.strip()

        if name != "" and name != last_name:
            if not last_name:
                last_name = name
            if conversations:
                texts.append(f"{last_name}：{' '.join(conversations)}")
            conversations.clear()
            last_name = name
        if conversation:
            conversations.append(conversation)

    if conversations:
        texts.append(f"{last_name}：{' '.join(conversations)}")
    return "\n".join(texts)


def pipeline(file: str):
    data = read_json(file)
    for index, row in tqdm(enumerate(data, start=1), total=len(data)):
        chat_text = row.get('chat_text')
        dropped_text = drop_duplication(chat_text)
        cleaned_text = clean_data_by_regex(dropped_text)
        result = merge_chat(cleaned_text)
        if len(result) < len(chat_text) * 0.05:
            # print(chat_text)
            # print(dropped_text)
            # print(cleaned_text)
            print(result)
            raise Exception("删除太多了。 index=%s" % index)
        print(index, result)
        print("###########################################################################################")
        print()
        row.update({"chat_text": result})
    return data


if __name__ == "__main__":
    # print(clean_data_by_regex("李勇7：[链接]"))
    # print(clean_data_by_regex("哈哈哈https://blog.csdn.net/qq_44511981/"))
    # print(clean_data_by_regex("哈哈哈https://blog.csdn.net/qq_44511981/"))
    # pipeline("dataset/example.json")
    # pipeline("dataset/test_data.json")

    for _file in ["dataset/train.json",
                  "dataset/test_data.json"]:
        _result = pipeline(_file)
        f_name = _file.split(".")[0] + "_pp.json"
        write_json(f_name, _result)
