#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2024/11/9 14:57
# @function:
import json
import logging
import re
import traceback
from typing import List, Any, Optional

JSON_PATTERN = re.compile(r"""(\{.*})""", re.DOTALL)
JSON_INTERNAL_PATTERN = re.compile(r"""(\{.*?})""", re.DOTALL)
JSON_ARRAY_PATTERN = re.compile(r"""(\[.*])""", re.DOTALL)


def convert_text_to_json(text: str, raise_exception: bool = False) -> str:
    _dict = convert_text_to_json(text, raise_exception=raise_exception)
    if _dict:
        return json.dumps(_dict, ensure_ascii=False)


def convert_text_to_dict(text: str, raise_exception: bool = False) -> dict:
    """
    提取LLM输出文本中的json字符串
    json.loads: 反斜杠 "\" 会导致报错，但实际是正确的json数据（JSONDecodeError）
    > {"content": "w_x(K) = c / \sigma_x(K) (11)"}
    """
    try:
        text: str = JSON_PATTERN.findall(text)[0]
        # eval(text)
        return json.loads(text)
    except Exception as e:
        # logging.warning("%s\n解析错误的文本：%s", traceback.format_exc(limit=5), text)
        if raise_exception:
            raise e


def convert_json_array_in_text_to_list(text: str) -> List[Any]:
    """
    提取LLM输出文本中的jsonArray字符串
    """
    result = []
    if not isinstance(text, str):
        return result
    matches = JSON_ARRAY_PATTERN.findall(text)
    try:
        rsp = json.loads(matches[0])
        if len(rsp) > 0 and isinstance(rsp[0], str):
            matches = JSON_INTERNAL_PATTERN.match(rsp[0])
            for match in matches:
                result.append(convert_text_to_json(match))
        else:
            result.extend(rsp)
    except Exception as e:
        logging.warning("%s\n解析错误的文本：%s", traceback.format_exc(limit=5), text)
    return result


if __name__ == '__main__':
    s = r"""
    {"description": [{"tag": "BackgroundArt", "content": [{"num": "", "text": "3-羟基丁酸。"}]}]}
    """
    _json = convert_text_to_json(s, raise_exception=True)
    print(s)
    print(_json)
