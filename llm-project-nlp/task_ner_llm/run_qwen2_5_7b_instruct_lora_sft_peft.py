#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2025/6/7 14:51
# @function:
import json
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, DataCollatorForSeq2Seq, Trainer
from functools import partial
from llm_util.chat_template import convert_alpaca_to_qwen_chat_template


def train(pretrain_path: str, model_path: str):
    # 加载源数据 (10748, 3)
    data_raw = json.load(open("data/dataset/alpaca/alpaca_clue_train.json", mode='r', encoding='utf-8'))
    data_ds: Dataset = Dataset.from_list(data_raw)
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(pretrain_path, use_fast=False, trust_remote_code=True)
    # 生成数据集
    dataset: Dataset = data_ds.map(partial(convert_alpaca_to_qwen_chat_template, tokenizer=tokenizer),
                                   remove_columns=data_ds.column_names)
    # 拆分数据集
    split_dataset = dataset.train_test_split(test_size=0.1, shuffle=True, seed=1024)
    # 获取训练集和验证集
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']
    print("Train dataset:", train_dataset.shape)  # (8598, 3)
    print("Eval dataset:", eval_dataset.shape)  # (2150, 3)

    # Lora配置
    # bfloat16 是一种 16 位浮点数格式（Brain Floating Point），相比标准的 float32：
    # 占用内存更小、运算更快（尤其在支持 bfloat16 的硬件上，如 Intel CPU、NVIDIA Ampere 架构 GPU）
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,   # 因果语言建模任务
        inference_mode=False,           # 训练模式
        r=64,                           # Lora 权重矩阵的秩
        lora_alpha=16,                  # LoRA 的缩放因子
        lora_dropout=0.1,               # 在LoRA层施加的dropout比例
        target_modules=[                # 应用LoRA的子模块
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    # 加载训练模型
    model = AutoModelForCausalLM.from_pretrained(pretrain_path, device_map="auto", torch_dtype=torch.bfloat16)
    model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法
    model: PeftModel = get_peft_model(model, lora_config)

    # 训练参数配置
    args = TrainingArguments(
        output_dir=f"{model_path}/checkpoints",
        per_device_train_batch_size=2,      # 每块GPU上的批次大小
        gradient_accumulation_steps=2,      # 累积梯度步数
        num_train_epochs=2,                 # 训练轮数
        logging_steps=10,                   # 日志记录间隔
        learning_rate=1e-4,                 # 微调学习率，一般比预训练时小
        gradient_checkpointing=True,        # 开启梯度检查点，节省显存
        report_to="none",                   # 不启用默认的日志报告（如TensorBoard）
        save_on_each_node=True,
        save_steps=200,                     # 模型保存间隔
        save_total_limit=2,                 # 最多保留的检查点数
        eval_strategy="steps",              # 评估触发方式（steps或epoch）
        eval_steps=50,                      # 评估步数
        metric_for_best_model="eval_loss",  # 使用验证 loss 判断最佳模型
        load_best_model_at_end=True
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
    )
    trainer.train()

    # 保存LoRA微调适配器
    model.save_pretrained(f"{model_path}")
    best_checkpoint = trainer.state.best_model_checkpoint
    print(f"Best model saved at: {best_checkpoint}")


if __name__ == '__main__':
    train("../model_hub/Qwen2.5-7B-Instruct",
          model_path="./data/outputs/Qwen2.5-7B-Instruct-clue-ner-lora-sft-peft/")
