#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2025/6/7 14:51
# @function:
import json
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, DataCollatorForSeq2Seq, Trainer
from functools import partial
from llm_util.chat_template import convert_alpaca_to_qwen_chat_template


def train(pretrain_path: str):
    # 加载源数据 (10748, 3)
    train_raw = json.load(open("data/dataset/alpaca/alpaca_clue_train.json", mode='r', encoding='utf-8'))
    train_ds: Dataset = Dataset.from_list(train_raw)
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(pretrain_path, use_fast=False, trust_remote_code=True)
    # 生成数据集
    train_dataset: Dataset = train_ds.map(partial(convert_alpaca_to_qwen_chat_template, tokenizer=tokenizer),
                                          remove_columns=train_ds.column_names)

    # Lora配置
    # bfloat16 是一种 16 位浮点数格式（Brain Floating Point），相比标准的 float32：
    # 占用内存更小、运算更快（尤其在支持 bfloat16 的硬件上，如 Intel CPU、NVIDIA Ampere 架构 GPU）
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        inference_mode=False,  # 训练模式
        r=64,  # Lora 秩
        lora_alpha=16,  # Lora alaph，具体作用参见 Lora 原理
        lora_dropout=0.1,  # Dropout 比例
    )
    # 加载训练模型
    model = AutoModelForCausalLM.from_pretrained(pretrain_path, device_map="auto", torch_dtype=torch.bfloat16)
    model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法
    model = get_peft_model(model, lora_config)

    # 训练参数配置
    args = TrainingArguments(
        output_dir="./output/Qwen2.5-7b",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        logging_steps=10,
        num_train_epochs=2,
        save_steps=100,
        learning_rate=1e-4,
        save_on_each_node=True,
        gradient_checkpointing=True,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
    )
    trainer.train()


if __name__ == '__main__':
    train("../model_hub/Qwen2.5-0.5B-Instruct")
