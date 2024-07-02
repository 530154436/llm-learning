#!/usr/bin/env python3
# -*- coding:utf-8 -*--
# 零基础入门大模型技术竞赛
# https://datawhaler.feishu.cn/wiki/VIy8ws47ii2N79kOt9zcXnbXnuS
import json
from typing import List
from tqdm import tqdm
from src.utils.io import read_json, write_json
from src.utils.spark_ai_chat import SparkAiChatWSS


class JsonFormatError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class JsonEmptyError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


def convert_all_json_in_text_to_dict(text):
    """提取LLM输出文本中的json字符串"""
    dicts, stack = [], []
    for i in range(len(text)):
        if text[i] == '{':
            stack.append(i)
        elif text[i] == '}':
            begin = stack.pop()
            if not stack:
                dicts.append(json.loads(text[begin:i+1]))
    return dicts


def print_json_format(data):
    """格式化输出json格式"""
    print(json.dumps(data, indent=4, ensure_ascii=False))


def check_and_complete_json_format(data):
    """确保数据输出格式与要求的一致"""
    required_keys = {
        "基本信息-姓名": str,
        "基本信息-手机号码": str,
        "基本信息-邮箱": str,
        "基本信息-地区": str,
        "基本信息-详细地址": str,
        "基本信息-性别": str,
        "基本信息-年龄": str,
        "基本信息-生日": str,
        "咨询类型": list,
        "意向产品": list,
        "购买异议点": list,
        "客户预算-预算是否充足": str,
        "客户预算-总体预算金额": str,
        "客户预算-预算明细": str,
        "竞品信息": str,
        "客户是否有意向": str,
        "客户是否有卡点": str,
        "客户购买阶段": str,
        "下一步跟进计划-参与人": list,
        "下一步跟进计划-时间点": str,
        "下一步跟进计划-具体事项": str
    }
    is_all_value_empty = True
    if not isinstance(data, list):
        raise JsonFormatError("Data is not a list")

    for item in data:
        if not isinstance(item, dict):
            raise JsonFormatError("Item is not a dictionary")
        for key, value_type in required_keys.items():
            if key not in item:
                item[key] = [] if value_type == list else ""
            else:
                if is_all_value_empty:
                    is_all_value_empty = item[key] in ("", [])
            if not isinstance(item[key], value_type):
                raise JsonFormatError(f"Key '{key}' is not of type {value_type.__name__}")
            if value_type == list and not all(isinstance(i, str) for i in item[key]):
                raise JsonFormatError(f"Key '{key}' does not contain all strings in the list")
    if is_all_value_empty:
        raise JsonEmptyError(f"all fields is empty")
    return data


def core_run(dataset: List[dict], prompt_template: str,
             model: str = "generalv3.5",):
    """
    调用星火大模型进行推理
    """
    retry_count = 5  # 重试次数
    result = []
    error_data = []
    for index, data in tqdm(enumerate(dataset)):
        index += 1
        is_success = False
        chat_text = data["chat_text"]
        for i in range(retry_count):
            try:
                prompt = prompt_template.format(content=chat_text)
                res = SparkAiChatWSS(model=model).get_completion(prompt)
                print("index:", index, ", result:", res.replace("\n", ""))
                infos = convert_all_json_in_text_to_dict(res)
                infos = check_and_complete_json_format(infos)
                if infos:
                    result.append({
                        "infos": infos,
                        "index": index
                    })
                    is_success = True
                    break
            except Exception as e:
                print("index:", index, ", error:", e)
                # continue
        if not is_success:
            data["index"] = index
            error_data.append(data)
            result.append({
                "infos": [],
                "index": index
            })
    # 保存输出
    write_json("output.json", result)


if __name__ == "__main__":
    # test_data = read_json("dataset/test_data.json")
    test_data = read_json("dataset/test_data_pp.json")

    # 提示工程（非微调）
    # PROMPT_EXTRACT = ''.join(open("prompts/baseline.tmpl").readlines())
    PROMPT_EXTRACT = ''.join(open("prompts/zero_shot.tmpl").readlines())
    # PROMPT_EXTRACT = ''.join(open("prompts/zero_shot_v2.tmpl").readlines())
    core_run(test_data, PROMPT_EXTRACT)

    # 提示工程（微调）

