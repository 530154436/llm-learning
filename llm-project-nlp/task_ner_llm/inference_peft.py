#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2025/5/24 9:45
# @function:
from typing import List
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def predict(messages: List[dict], model: PeftModel, tokenizer):
    device = "cuda"
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


def load_adapter_model() -> PeftModel:
    pretrain_path = "../model_hub/Qwen2.5-7B-Instruct"
    adapter_model_path = "./data/outputs/Qwen2.5-7B-Instruct-clue-ner-lora-sft-peft"
    # 加载基础模型和分词器
    model = AutoModelForCausalLM.from_pretrained(pretrain_path)
    tokenizer = AutoTokenizer.from_pretrained(pretrain_path)
    model = PeftModel.from_pretrained(model, adapter_model_path)
    print(type(model), type(tokenizer))
    return model, tokenizer


if __name__ == '__main__':
    _model = load_adapter_model()
    _messages = [
        {"role": "user", "content": "你是谁"},
    ]

