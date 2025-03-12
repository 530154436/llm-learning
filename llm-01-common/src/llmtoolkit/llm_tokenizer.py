#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2025/3/12 17:48
# @function: 计算字符串的Token数量
import tiktoken
import dashscope
from typing import List, Union
from dashscope import Tokenizer
from tiktoken import Encoding


def get_token_ids_for_qwen(
    input_str: str,
    model: str = "qwen-plus",
    verbose: bool = True,
) -> List[int]:
    """
    千问模型系列-字符串对应的Token id
    :param input_str: 需要进行标记化的输入字符串。
    :param model: 要使用的模型名称。
    :param verbose: 是否打印详细信息
    :return int: 输入字符串中的标记数量。

    https://help.aliyun.com/zh/model-studio/billing-for-model-studio
    https://github.com/QwenLM/Qwen/blob/main/tokenization_note_zh.md
    """
    tokenizer: Tokenizer = dashscope.get_tokenizer(model=model)
    token_ids = tokenizer.encode(input_str)
    if verbose:
        for token_id in token_ids:
            print(f"Token id={token_id}, 对应的字符串为：{tokenizer.decode(token_id)}")
    return token_ids


def get_token_ids_for_openai(
    input_str: str,
    model: str = "gpt-3.5-turbo",
    verbose: bool = True,
) -> List[int]:
    """
    OpenAI模型系列-字符串对应的Token id
    :param input_str: 需要进行标记化的输入字符串。
    :param model: 要使用的模型名称。
    :param verbose: 是否打印详细信息
    :return int: 输入字符串中的标记数量。

    https://github.com/openai/tiktoken/blob/main/tiktoken
    """
    # gpt-3.5-turbo -> cl100k_base
    encoding_name: str = tiktoken.encoding_name_for_model(model_name=model)
    tokenizer: Encoding = tiktoken.get_encoding(encoding_name=encoding_name)
    token_ids = tokenizer.encode(input_str)
    if verbose:
        for token_id in token_ids:
            print(f"Token id={token_id}, 对应的字符串为：{tokenizer.decode([token_id])}")
    return token_ids


def num_token_in_string(
    input_str: str,
    model: str = "qwen-plus",
) -> int:
    """计算给定字符串中的token数量。"""
    if model.startswith('qwen'):
        return len(get_token_ids_for_qwen(input_str, model=model, verbose=False))
    else:
        return len(get_token_ids_for_openai(input_str, model=model, verbose=False))


def num_token_in_message(
    messages: List[dict],
    model: str = "qwen-plus"
) -> int:
    """
    计算API调用的token数量（忽略模型名称、输入和输出的特殊token标记）
    :return:
    """
    num_tokens = 0
    for message in messages:
        for key, value in message.items():
            num_tokens += num_token_in_string(value, model=model)
    return num_tokens


if __name__ == "__main__":
    print(get_token_ids_for_qwen("你好，你怎么样？"))               # [108386, 3837, 56568, 104472, 11319]
    print(get_token_ids_for_openai("Hello, how are you?"))       # [9906, 11, 1268, 527, 499, 30]
    print(num_token_in_string("你好，你怎么样？", model="qwen-plus"))  # 5
    print(num_token_in_string("Hello, how are you?", model="gpt-3.5-turbo"))  # 6
    _message = [
        {"role": "system", "content": "你好，你怎么样？"},
        {"role": "user", "content": "我很好，谢谢."}
    ]
    print(num_token_in_message(_message, model="qwen-plus"))  # 12
