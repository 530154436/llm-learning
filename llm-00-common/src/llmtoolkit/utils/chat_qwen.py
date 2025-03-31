#!/usr/bin/env python3
# -*- coding:utf-8 -*--
# 千问通义
import os
from llm_common.src.utils.chat_openai import OpenAiChat


class QwenLocal(OpenAiChat):
    """
    本地部署大模型
    """
    api_key = "0"
    base_url = "http://localhost:8000/v1"  # 虚拟机本地环境
    if os.environ.get('ENV') == "local":
        base_url = "http://172.19.190.6:30769/v1"

    def __init__(self, model: str = "Qwen2.5-7B-Instruct", temperature: int = 0,
                 max_tokens: int = 2048, timeout: int = 60, trace: bool = False):
        super(QwenLocal, self).__init__(model=model, temperature=temperature, max_tokens=max_tokens,
                                        timeout=timeout, trace=trace)


class QwenLong(OpenAiChat):
    """
    大模型接口-千问long
    """
    api_key = os.environ.get("QWEN_API_KEY")
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"  # 虚拟机本地环境

    def __init__(self, model: str = "qwen-long", temperature: int = 0,
                 max_tokens: int = 2048, timeout: int = 60, trace: bool = False):
        super(QwenLong, self).__init__(model=model, temperature=temperature, max_tokens=max_tokens,
                                       timeout=timeout, trace=trace)


if __name__ == "__main__":
    _text = "中国的首都是哪里"
    _prompt = f"""
    请将三个反引号之间的文本总结成一句话。
    ```{_text}```
    """
    _messages = [
        {'role': 'system', 'content': '你是一个助理， 并以 Seuss 苏斯博士的风格作出回答。'},
        {'role': 'user', 'content': '就快乐的小鲸鱼为主题给我写一首短诗'}
    ]

    for chat_robot in [QwenLocal(trace=True), QwenLong(trace=True)]:
        print(chat_robot.get_completion(_prompt))
        print(chat_robot.get_completion_with_usage(_messages))
        print(chat_robot.tracer)
