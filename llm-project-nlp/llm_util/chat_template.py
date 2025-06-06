#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2025/6/5 20:02
# @function:
from transformers import AutoTokenizer, Qwen2Tokenizer


def process_func(example):
    """
    将数据集进行预处理
    """
    MAX_LENGTH = 384
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(
        f"<|im_start|>system\n{example['instruction']}<|im_end|>\n<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False,
    )
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids = (
            instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    )
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = (
            [-100] * len(instruction["input_ids"])
            + response["input_ids"]
            + [tokenizer.pad_token_id]
    )
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


QWEN_QUERY_TEMPLATE = "<|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n"
QWEN_RESPONSE_TEMPLATE = "{output}<|im_end|><|endoftext|>"

def convert_alpaca_to_qwen_chat_template(data: dict, tokenizer: Qwen2Tokenizer , max_len: int = 512) -> dict:
    query = QWEN_QUERY_TEMPLATE.format(instruction=data["instruction"], input=data["input"])
    response = QWEN_RESPONSE_TEMPLATE.format(output=data["output"])
    feature = tokenizer.__call__(query + response, add_special_tokens=False)
    print(type(feature))
    print(feature)


if __name__ == "__main__":
    _tokenizer = AutoTokenizer.from_pretrained("../model_hub/Qwen2.5-0.5B-Instruct",
                                               use_fast=False, trust_remote_code=True)
    print(type(_tokenizer))
    for _token, _token_id in sorted(zip(_tokenizer.all_special_tokens,
                                        _tokenizer.all_special_ids), key=lambda x: x[1], reverse=False):
        print(_token, _token_id)

    _data = {
        "instruction": "请从给定的句子中识别并提取出以下指定类别的实体。",
        "input": "浙商银行企业",
        "output": "浙商银行"
    }
    convert_alpaca_to_qwen_chat_template(_data, _tokenizer)
