#!/usr/bin/env python3
# -*- coding:utf-8 -*--
import json
import re
# https://watchsound.medium.com/handle-invalid-json-output-for-small-size-llm-a2dc455993bd

EXTRACT_AND_FIX_JSON_PROMPT = \
"""输入是一个错误的JSON数据。你的任务是:

1、通过定位第一个开括号{{和相应的最后一个闭括号}}来识别和纠正JSON数据。.
2、从输入中提取JSON子字符串。
3、验证并修复JSON数据，确保其符合标准JSON格式，解决常见问题如：
中文逗号、中文双引号或单引号。
缺失或多余的逗号。
不平衡的括号{{}}或方括号[]。
在闭括号/方括号前的尾随逗号。
JSON结构中的任何无效字符。
4、只输出修正后的JSON，不包括其他的文本，并且不能改变原本的内容。
5、如果输入中没有有效的JSON，返回错误信息：
"Error: Unable to extract valid JSON from the input."

示例：
输入1：
{{"key1": "xxx"，"key2": "xxx。”}}

输出1:
{{"key1": "xxx", "key2": "xxxx。"}}

输入2：
{{"key1": "xxx", "key2": "xxx。",}}

输出2:
{{"key1": "xxx", "key2": "xxxx。"}}

# 输入:
{input}

# 注意:
我不需要代码实现，只需要JSON数据。你必须只生成有效的JSON。不要包含任何额外的文本、解释或注释。
不要在JSON前添加任何文本。也不要在JSON后添加任何文本。
只提供JSON对象。
"""


def _extract_json_string(llm_response):
    # Step 1: Find the first '{' as the start of the JSON
    start_idx = llm_response.find('{')
    if start_idx == -1:
        return None

        # Step 2: Determine the end of the JSON
    # Check if the last '}' is the last character in the string
    end_idx = llm_response.rfind('}')
    if end_idx == -1 or end_idx < len(llm_response.strip()) - 1:
        # If no closing brace or text follows the last '}', take the whole string to the end
        end_idx = len(llm_response)

    # Extract the potential JSON substring
    json_string = llm_response[start_idx:end_idx].strip()
    return json_string


def extract_and_fix_json(llm_response):
    json_string = _extract_json_string(llm_response)
    if json_string is None:
        return None, "No valid JSON found in the response."

    # print(f"json_string = {json_string}")

    # Step 3: Attempt to parse the JSON string directly
    def safe_json_parse(json_string):
        try:
            return json.loads(json_string), None
        except json.JSONDecodeError as e:
            return None, str(e)

    result, error = safe_json_parse(json_string)
    if result:
        return result, None

    # Step 4: Attempt auto-fix on the extracted JSON string
    def auto_fix_json(json_string):
        # Remove trailing commas or invalid characters
        json_string = re.sub(r',\s*([}\]])', r'\1', json_string)  # Remove commas before braces/brackets
        json_string = re.sub(r',\s*$', '', json_string.strip())  # Remove trailing commas

        # Fix unbalanced braces
        open_braces = json_string.count('{')
        close_braces = json_string.count('}')
        if open_braces > close_braces:
            json_string += '}' * (open_braces - close_braces)

        # print(f"json_string (2) = {json_string}")
        # Attempt to parse again
        try:
            return json.loads(json_string), None
        except json.JSONDecodeError as e:
            return None, str(e)

    result, error = auto_fix_json(json_string)
    if result:
        return result, None

    # Step 5: Log and return error if all attempts fail
    return None, f"Failed to parse JSON after auto-fixing: {error}"


def ask_llm_to_fix_json(llm, json_string):
    prompt = EXTRACT_AND_FIX_JSON_PROMPT.format(input=json_string)
    fixed_json = llm.invoke(prompt)
    return fixed_json


def robust_json_parser(llm_response, llm=None):
    # Step 1: Try `extract_and_fix_json` first
    parsed_data, error = extract_and_fix_json(llm_response)
    if parsed_data:
        return parsed_data, None

    # Step 2: If it fails, try asking LLM to fix JSON
    if llm is not None:
        json_part = _extract_json_string(llm_response)
        fixed_response = ask_llm_to_fix_json(llm, json_part)
        fixed_response = fixed_response.content.replace('```', '').replace('json', '')
        print(fixed_response)
        # Attempt parsing again
        parsed_data, error = extract_and_fix_json(fixed_response)
        if parsed_data:
            return parsed_data, None

    # Step 3: If all else fails, return error
    return None, f"Failed to parse JSON: {error}"


if __name__ == '__main__':
    import os
    from langchain_openai.chat_models import ChatOpenAI
    from llm_common.src.utils.chat_openai import OpenAiChat
    api_key = ""
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    chat_robot = ChatOpenAI(model_name="qwen-plus-latest", openai_api_key=api_key, base_url=base_url)

    patent_json = r'''{"patent_name": "阀门布置", "patent_brief": "在柱塞处设置有密封材料的密封部分（19）。”}'''
    result = robust_json_parser(patent_json, llm=chat_robot)
    print(result)
