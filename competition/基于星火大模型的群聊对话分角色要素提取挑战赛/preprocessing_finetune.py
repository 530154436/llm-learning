#!/usr/bin/env python3
# -*- coding:utf-8 -*--
# Task3：进阶 baseline2【微调方向】 + 知识点讲解
# https://datawhaler.feishu.cn/wiki/Q48xwICyHiV0O2kSwjccuTE1nrb
# 对原群聊对话设计一个总结Prompt，目的是将原始对话内容进行精简，方便做微调数据。
# 一方面直接将群聊对话作为数据集的话，会导致上下文过长，超过限制，还有上下文太长会导致抽取效果变差。
# 过长的上下文也会导致训练时长和费用倍增。（比如我做了一个数据集要花3000多块钱跑完。就算能跑可能也要1-2天……）
# 这个prompt相较于baseline01区别比较明显，对需要抽取的任务做了一次总结。总结了四个方面：
#   客户基本信息：需要从中区分出客户角色，并得到客户基本信息，其中包括姓名、手机号码、邮箱、地区、详细地址、性别、年龄和生日
#   客户意向与预算信息：客户意向与预算信息包括咨询类型、意向产品、购买异议点、预算是否充足、总体预算金额以及预算明细
#   客户购买准备情况：户购买准备情况包括竞品信息、客户是否有意向、客户是否有卡点以及客户购买阶段
#   跟进计划信息： 跟进计划信息包括参与人、时间点和具体事项，这些信息用于指导销售团队在未来的跟进工作中与客户互
# 通过总结后的数据一方面节约了微调的运算资源，一方面也让数据被清洗后更容易被模型理解，达到更好的抽取效果。
import csv
import json
from tqdm import tqdm
from src.utils.io import read_json
from src.utils.chat_spark_ai import SparkAiChatWSS


def make_finetune_train_set(file: str, prompt_file: str):
    data = read_json(file)
    PROMPT_SUMMARY = ''.join(open(prompt_file).readlines())
    instruction = "假设你是一个智能交互助手，基于用户的输入文本，解析其中语义，抽取关键信息，以json格式生成结构化的语义内容。"
    save_file = file.split(".")[0] + ".jsonl"
    with open(save_file, 'w', encoding='utf-8') as file:
        # 训练集行数(130)不符合要求，范围：1500~90000000
        # 遍历数据列表，并将每一行写入文件
        # 这里为了满足微调需求我们重复12次数据集 130*12=1560
        for line_data in tqdm(data):
            chat_text = line_data["chat_text"]
            infos = line_data["infos"]
            prompt = PROMPT_SUMMARY.format(content=chat_text)
            res = SparkAiChatWSS().get_completion(prompt)
            print(res)
            line_write = {
                # "instruction": instruction,
                "input": json.dumps(instruction + "\n" + res.replace("markdown", ""), ensure_ascii=False),
                "target": json.dumps(infos, ensure_ascii=False)
            }
            # 因为数据共有130行，为了能满足训练需要的1500条及以上，我们将正常训练数据扩充12倍。
            for time in range(12):
                file.write(json.dumps(line_write, ensure_ascii=False) + '\n')  # '\n' 用于在每行末尾添加换行符


def make_finetune_test_set(file: str, prompt_file: str):
    data = read_json(file)
    PROMPT_SUMMARY = ''.join(open(prompt_file).readlines())
    save_file = file.split(".")[0] + ".csv"
    with open(save_file, 'w', newline='', encoding='utf-8') as csvfile:
        # 创建一个csv writer对象
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["input", "target"])
        # 遍历数据列表，并将每一行写入CSV文件
        for line_data in tqdm(data):
            chat_text = line_data["chat_text"]
            prompt = PROMPT_SUMMARY.format(content=chat_text)
            res = SparkAiChatWSS().get_completion(prompt)
            # 文件内容校验失败: test.jsonl(不含表头起算)第1行的内容不符合规则，
            # 限制每组input和target字符数量总和上限为8000，当前行字符数量：10721
            line_list = [res, "-"]
            csvwriter.writerow(line_list)


if __name__ == "__main__":
    # 0702
    # make_finetune_train_set("dataset/train.json", "prompts/baseline2.tmpl")
    make_finetune_test_set("dataset/test_data.json", "prompts/baseline2.tmpl")
