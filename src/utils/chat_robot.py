#!/usr/bin/env python3
# -*- coding:utf-8 -*--
# Chat GPT 工具类
import os
from typing import Union
from openai import OpenAI


class OpenAiChat(object):
    # api_key = os.environ.get("OPENAI_API_KEY")
    # base_url = "https://api.openai.com/v1"
    api_key = os.environ.get("CHAT_ANYWHERE_API_KEY")
    base_url = "https://api.chatanywhere.com.cn/v1"
    # api_key = os.environ.get("IDATARIVER_API_KEY")
    # base_url = "https://apiok.us/api/openai/v1"

    @classmethod
    def get_completion(cls, prompt: str, model: str = "gpt-3.5-turbo",
                       timeout: int = 60, temperature: int = 0) -> Union[str]:
        """
        封装 OpenAI 接口的函数，参数为 Prompt，返回对应结果
        文档：https://platform.openai.com/docs/quickstart
        prompt: 对应的提示词
        model: 调用的模型，默认为 gpt-3.5-turbo(ChatGPT)，有内测资格的用户可以选择 gpt-4
        temperature: 模型输出的温度系数，控制输出的随机程度
        """
        client = OpenAI(api_key=cls.api_key, base_url=cls.base_url)
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            timeout=timeout,
            temperature=temperature,
        )
        # 调用 OpenAI 的 ChatCompletion 接口
        if len(completion.choices) > 0:
            return completion.choices[0].message.content
        return None


if __name__ == "__main__":
    _text = f"""
    您应该提供尽可能清晰、具体的指示，以表达您希望模型执行的任务。\
    这将引导模型朝向所需的输出，并降低收到无关或不正确响应的可能性。\
    不要将写清晰的提示词与写简短的提示词混淆。\
    在许多情况下，更长的提示词可以为模型提供更多的清晰度和上下文信息，从而导致更详细和相关的输出。
    """
    _prompt = f"""
    请将三个反引号之间的文本总结成一句话。
    ```{_text}```
    """
    # 指令内容，使用 ``` 来分隔指令和待总结的内容
    response = OpenAiChat.get_completion(_prompt)
    print(response)
