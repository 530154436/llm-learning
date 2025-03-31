#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2025/3/24 15:52
# @function:
from typing import List


class BPETokenizer:

    @classmethod
    def get_token_ids(cls,
                      input_str: str,
                      model: str = "qwen-plus",
                      verbose: bool = True) -> List[int]:
        raise NotImplementedError

    @classmethod
    def num_token_in_string(cls,
                            input_str: str,
                            model: str = "qwen-plus") -> int:
        """计算给定字符串中的token数量。"""
        return len(cls.get_token_ids(input_str=input_str, model=model, verbose=False))

    @classmethod
    def num_token_in_message(cls,
                             messages: List[dict],
                             model: str = "qwen-plus") -> int:
        """
        计算API调用的token数量（忽略模型名称、输入和输出的特殊token标记）
        :return:
        """
        num_tokens = 0
        for message in messages:
            for key, value in message.items():
                num_tokens += cls.num_token_in_string(value, model=model)
        return num_tokens
