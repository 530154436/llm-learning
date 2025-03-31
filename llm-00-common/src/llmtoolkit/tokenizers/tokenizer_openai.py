#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2025/3/12 17:48
# @function: 计算字符串的Token数量
import tiktoken
from typing import List
from tiktoken import Encoding
from tokenizers import BPETokenizer


class OpenaiTokenizer(BPETokenizer):

    @classmethod
    def get_token_ids(cls,
                      input_str: str,
                      model: str = "gpt-3.5-turbo",
                      verbose: bool = True) -> List[int]:
        """
        OpenAI模型系列-字符串对应的Token id
        :param input_str: 需要进行标记化的输入字符串。
        :param model: 要使用的模型名称。
        :param verbose: 是否打印详细信息
        :return int: 输入字符串中的标记数量。
        """
        # gpt-3.5-turbo -> cl100k_base
        encoding_name: str = tiktoken.encoding_name_for_model(model_name=model)
        tokenizer: Encoding = tiktoken.get_encoding(encoding_name=encoding_name)
        token_ids = tokenizer.encode(input_str)
        if verbose:
            for token_id in token_ids:
                print(f"Token id={token_id}, 对应的字符串为：{tokenizer.decode([token_id])}")
        return token_ids


if __name__ == "__main__":
    print(OpenaiTokenizer.get_token_ids("Hello, how are you?"))               # [108386, 3837, 56568, 104472, 11319]
    print(OpenaiTokenizer.num_token_in_string("Hello, how are you?", model="gpt-3.5-turbo"))  # 5
    _message = [
        {"role": "system", "content": "Hello, how are you?"},
        {"role": "user", "content": "我很好，谢谢."}
    ]
    print(OpenaiTokenizer.num_token_in_message(_message, model="gpt-3.5-turbo"))  # 12
