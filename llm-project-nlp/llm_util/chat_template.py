#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2025/6/5 20:02
# @function:
from transformers import AutoTokenizer, Qwen2Tokenizer

QWEN_QUERY_TEMPLATE = "<|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n"
QWEN_RESPONSE_TEMPLATE = "{output}<|im_end|><|endoftext|>"


def convert_alpaca_to_qwen_chat_template(example: dict,
                                         tokenizer: Qwen2Tokenizer,
                                         special_tokens_count: int = 15,
                                         max_token_len: int = 1024) -> dict:
    """
    根据alpaca格式的数据集合构造微调输入特征
    :param example:  {
                    "instruction": "xxxx",
                    "input": "xx",
                    "output": "yyyy"
                  }
    :param tokenizer: Qwen2Tokenizer,
        tokenizer.pad_token_id <|endoftext|> 151643
        tokenizer.eos_token_id <|im_end|> 151645
    :param special_tokens_count: 组装的微调输入共有15个特殊token,即 <|im_start|>、system、\n、<|im_end|>、user、assistant、<|endoftext|>。
    :param max_token_len: 最大输入token数量
    :return:
    """
    query = QWEN_QUERY_TEMPLATE.format(instruction=example["instruction"], input=example["input"])
    response = QWEN_RESPONSE_TEMPLATE.format(output=example["output"])
    import json
    print(json.dumps(query, ensure_ascii=False))
    print(json.dumps(response, ensure_ascii=False))
    query_feature = tokenizer.__call__(query, add_special_tokens=False)
    response_feature = tokenizer.__call__(response, add_special_tokens=False)

    input_ids = query_feature["input_ids"] + response_feature["input_ids"]
    attention_mask = query_feature["attention_mask"] + response_feature["attention_mask"]
    labels = [-100] * len(query_feature["input_ids"]) + response_feature["input_ids"]

    if len(input_ids) > max_token_len:  # 截断
        input_ids = input_ids[:max_token_len]
        attention_mask = attention_mask[:max_token_len]
        labels = labels[:max_token_len]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


if __name__ == "__main__":
    _tokenizer = AutoTokenizer.from_pretrained("../model_hub/Qwen2.5-0.5B-Instruct",
                                               use_fast=False, trust_remote_code=True)
    print(type(_tokenizer))
    for _token, _token_id in sorted(zip(_tokenizer.all_special_tokens,
                                        _tokenizer.all_special_ids), key=lambda x: x[1], reverse=False):
        print(_token, _token_id)

    _example = {
        "instruction": "你是一个文本实体识别领域的专家，擅长从自然语言中提取不同类别的实体名称。",
        "input": "请从给定的句子中识别并提取出以下指定类别的实体。\n\n<实体类别集合>\nname, organization, scene, company, movie, book, government, position, address, game\n\n<任务说明>\n1. 仅提取属于上述类别的实体，忽略其他类型的实体。\n2. 以json格式输出，对于每个识别出的实体，请提供：\n   - label: 实体类型，必须严格使用原始类型标识（不可更改）\n   - text: 实体在原文中的中文内容\n\n<输出格式要求>\n```json\n[{\"label\": \"实体类别\", \"text\": \"实体名称\"}]\n```\n\n<输入文本>\n凯尔特人在苏格兰赛场连胜7场，不过连胜含金量要打折扣。双方首回合交锋奥尔堡客场逼平凯尔特人。",
        "output": "[{\"label\": \"organization\", \"text\": \"苏格兰\"}, {\"label\": \"organization\", \"text\": \"奥尔堡\"}, {\"label\": \"organization\", \"text\": \"凯尔特人\"}]"
    }
    convert_alpaca_to_qwen_chat_template(_example, _tokenizer)
