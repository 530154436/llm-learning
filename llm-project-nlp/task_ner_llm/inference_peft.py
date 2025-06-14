#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2025/5/24 9:45
# @function:
from typing import List
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from modeling_util.model_util import auto_device


def predict(messages: List[dict], model: PeftModel, tokenizer):
    device = auto_device()
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    print(text)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    print(model_inputs)
    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    print(generated_ids)
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)
    return response


def load_adapter_model() -> PeftModel:
    pretrain_path = "../model_hub/Qwen2.5-7B-Instruct"
    adapter_model_path = "./data/outputs/Qwen2.5-7B-Instruct-clue-ner-lora-sft-peft"
    # 加载基础模型和分词器
    model = AutoModelForCausalLM.from_pretrained(pretrain_path, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(pretrain_path)
    model = PeftModel.from_pretrained(model, adapter_model_path)
    print(type(model), type(tokenizer))
    return model, tokenizer


if __name__ == '__main__':
    _model, _tokenizer = load_adapter_model()
    _messages = [
        {'role': 'system', 'content': '你是一个文本实体识别领域的专家，擅长从自然语言中提取不同类别的实体名称。'},
        {'role': 'user', 'content': '请从给定的句子中识别并提取出以下指定类别的实体。\n\n<实体类别集合>\nname, organization, '
                                    'scene, company, movie, book, government, position, address, game\n\n<任务说明>\n1. '
                                    '仅提取属于上述类别的实体，忽略其他类型的实体。\n2. 以json格式输出，对于每个识别出的实体，请提供：\n '
                                    '  - label: 实体类型，必须严格使用原始类型标识（不可更改）\n   - text: 实体在原文中的中文内容\n\n'
                                    '<输出格式要求>\n```json\n[{"label": "实体类别", "text": "实体名称"}]\n```\n\n<输入文本>\n'
                                    '现在的阿森纳恐怕不能再给人以强队的信心，但教授的神经也真是够硬，在英超夺冠几无希望的情况下，'}]
    print(predict(messages=_messages, model=_model, tokenizer=_tokenizer))
