#!/usr/bin/env python3
# -*- coding:utf-8 -*--
# 基于星火大模型的群聊对话分角色要素提取挑战赛-Lora微调与prompt构造
# https://blog.csdn.net/qq_44511981/article/details/140043813?csdn_share_tail=%7B%22type%22%3A%22blog%22%2C%22rType%22%3A%22article%22%2C%22rId%22%3A%22140043813%22%2C%22source%22%3A%22qq_44511981%22%7D
import json
from tqdm import tqdm
from src.utils.io import read_json


def make_finetune_dataset(file: str, prompt_file: str, max_length: int = 4000):
    data = read_json(file)
    PROMPT = ''.join(open(prompt_file).readlines())
    save_file = file.split(".")[0] + "_finetune_v2.jsonl"
    with open(save_file, 'w', encoding='utf-8') as writer:
        for index, row in tqdm(enumerate(data, start=1), total=len(data)):
            chat_text = row["chat_text"]
            prompt = PROMPT.format(content=chat_text)
            answer = json.dumps(row["infos"], ensure_ascii=False)
            delta = max_length - len(prompt + answer)
            if delta < 0:
                prompt = PROMPT.format(content=chat_text[:delta])

            line_write = {"input": prompt, "target": answer}
            # 因为数据共有130行，为了能满足训练需要的1500条及以上，我们将正常训练数据扩充12倍。
            if file.__contains__('train'):
                for time in range(12):
                    writer.write(json.dumps(line_write, ensure_ascii=False) + '\n')  # '\n' 用于在每行末尾添加换行符
            else:
                writer.write(json.dumps(line_write, ensure_ascii=False) + '\n')  # '\n' 用于在每行末尾添加换行符
            print(line_write)
            # break


if __name__ == "__main__":
    # 0703
    make_finetune_dataset("dataset/train_pp.json", "prompts/zero_shot.tmpl", max_length=4000)  # 官网说最长4k，否则会截断
    make_finetune_dataset("dataset/test_data_pp.json", "prompts/zero_shot.tmpl", max_length=8000)
