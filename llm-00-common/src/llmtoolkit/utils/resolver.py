#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2024/11/9 14:57
# @function:
import json
import re
import traceback
from typing import List, Any
from llm_common.src.utils.logger import LOGGER

JSON_PATTERN = re.compile(r"""(\{.*})""", re.DOTALL)
JSON_INTERNAL_PATTERN = re.compile(r"""(\{.*?})""", re.DOTALL)
JSON_ARRAY_PATTERN = re.compile(r"""(\[.*])""", re.DOTALL)


def convert_json_in_text_to_dict(text: str) -> dict:
    """
    提取LLM输出文本中的json字符串
    """
    if not isinstance(text, str):
        return dict()
    matches = JSON_PATTERN.findall(text)
    try:
        return json.loads(matches[0])
    except Exception as e:
        LOGGER.warning("%s\n解析错误的文本：%s", traceback.format_exc(limit=5), text)
    return dict()


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
                result.append(convert_json_in_text_to_dict(match))
        else:
            result.extend(rsp)
    except Exception as e:
        LOGGER.warning("%s\n解析错误的文本：%s", traceback.format_exc(limit=5), text)
    return result
