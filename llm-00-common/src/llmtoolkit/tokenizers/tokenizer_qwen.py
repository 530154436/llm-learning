#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2025/3/12 17:48
# @function: 计算字符串的Token数量
import dashscope
from typing import List
from dashscope import Tokenizer
from tokenizers import BPETokenizer


class QwenTokenizer(BPETokenizer):

    @classmethod
    def get_token_ids(cls,
                      input_str: str,
                      model: str = "qwen-plus",
                      verbose: bool = True) -> List[int]:
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


if __name__ == "__main__":
    print(QwenTokenizer.get_token_ids("你好，你怎么样？"))               # [108386, 3837, 56568, 104472, 11319]
    print(QwenTokenizer.num_token_in_string("你好，你怎么样？", model="qwen-plus"))  # 5
    _message = [
        {"role": "system", "content": "你好，你怎么样？"},
        {"role": "user", "content": "我很好，谢谢."}
    ]
    print(QwenTokenizer.num_token_in_message(_message, model="qwen-plus"))  # 12
