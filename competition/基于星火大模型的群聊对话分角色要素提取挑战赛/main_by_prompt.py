#!/usr/bin/env python3
# -*- coding:utf-8 -*--
import json
from copy import deepcopy

from tqdm import tqdm
from src.utils.io import read_json, write_json
from src.utils.chat_robot import SparkAiChat, OpenAiChat
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


if __name__ == "__main__":
    train_data = read_json("dataset/train.json")
    test_data = read_json("dataset/test_data.json")
    PROMPT_PP = ''.join(open("prompts/preprocessing.tmpl").readlines())
    PROMPT_EXTRACT = ''.join(open("prompts/zero_shot.tmpl").readlines())
    retry_count = 5  # 重试次数
    result = []
    error_data = []

    new_datas = []
    for index, data in tqdm(enumerate(test_data)):
        index += 1
        is_success = False
        chat_text = data["chat_text"]

        prompt = PROMPT_PP.format(content=chat_text, delimiter="####")
        _messages = [
            {'role': 'system', 'content': '你是一个自然语言处理工程师，请处理群聊对话中的文本信息。'},
            {'role': 'user', 'content': prompt}
        ]
        # res = SparkAiChat().get_completion_from_messages(_messages)
        # res = OpenAiChat().get_completion_from_messages(_messages)
        res = SparkAiChatWSS().get_completions_from_message(_messages)
        new_data = deepcopy(data)
        new_data["chat_text"] = res.replace("```", "")
        new_datas.append(new_data)
        print("index:", index)
        print(res)
        print()
        print()
        # break
    with open("test_data_new.json", encoding="utf8", mode='w') as f:
        json.dump(new_datas, f, ensure_ascii=False)

    #
    #     for i in range(retry_count):
    #         try:
    #
    #             prompt = PROMPT_EXTRACT.format(content=)
    #             res = SparkAiChat().get_completion(prompt)
    #             print("index:", index, ", result:", res.replace("\n", ""))
    #             infos = convert_all_json_in_text_to_dict(res)
    #             infos = check_and_complete_json_format(infos)
    #             if infos:
    #                 result.append({
    #                     "infos": infos,
    #                     "index": index
    #                 })
    #                 is_success = True
    #                 break
    #         except Exception as e:
    #             print("index:", index, ", error:", e)
    #             # continue
    #     if not is_success:
    #         data["index"] = index
    #         error_data.append(data)
    #         result.append({
    #             "infos": [],
    #             "index": index
    #         })
    #
    # # 保存输出
    # write_json("output.json", result)
