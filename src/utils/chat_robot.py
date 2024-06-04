#!/usr/bin/env python3
# -*- coding:utf-8 -*--
# Chat GPT 工具类
import os
from typing import List, Tuple, Union
from openai import OpenAI
from openai.types import CompletionUsage, ModerationCreateResponse
from openai.types.chat import ChatCompletion


class OpenAiChat(object):
    # api_key = os.environ.get("OPENAI_API_KEY")
    # base_url = "https://api.openai.com/v1"
    # api_key = os.environ.get("IDATARIVER_API_KEY")
    # base_url = "https://apiok.us/api/openai/v1"
    api_key = os.environ.get("CHAT_ANYWHERE_API_KEY")
    base_url = "https://api.chatanywhere.com.cn/v1"

    def __init__(self, model: str = "gpt-3.5-turbo", temperature: int = 0,
                 max_tokens: int = 500, timeout: int = 60):
        """
        文档：https://platform.openai.com/docs/quickstart
        :param model: 调用的模型，默认为 gpt-3.5-turbo(ChatGPT)，有内测资格的用户可以选择 gpt-4
        :param temperature: 模型输出的温度系数，控制输出的随机程度；值越大，生成的回复越随机。默认为 0。
        :param max_tokens: 生成回复的最大 token 数量。默认为 500。
        :param timeout: 超时时间
        """
        self.model: str = model
        self.temperature: int = temperature
        self.max_tokens: int = max_tokens
        self.timeout: int = timeout
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def create_completion(self, messages: List[dict]) -> ChatCompletion:
        """
        1、系统消息(system message)：以系统身份发送消息，提供了一个总体的指示。有助于设置助手的行为和角色，并作为对话的高级指示。
        2、用户消息(user message)：在 ChatGPT 网页界面中，用户发送的消息称为用户消息
        3、助手消息(assistant message)：ChatGPT 的消息称为助手消息。
        :example
            messages =  [
                {'role':'system', 'content':'You are an assistant that speaks like Shakespeare.'},
                {'role':'user', 'content':'tell me a joke'},
                {'role':'assistant', 'content':'Why did the chicken cross the road'},
                {'role':'user', 'content':'I don\'t know'}
            ]
        """
        completion = self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            timeout=self.timeout,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return completion

    def get_completion_from_messages(self, messages: List[dict]) -> str:
        """
        返回生成的回复内容
        """
        completion = self.create_completion(messages)
        if len(completion.choices) > 0:
            return completion.choices[0].message.content
        return None

    def get_completion(self, prompt: str) -> str:
        """
        返回生成的回复内容
        """
        message = [{"role": "user", "content": prompt}]
        return self.get_completion_from_messages(message)

    def get_completion_and_token_count(self, messages: List[dict]) -> Tuple[str, dict]:
        """
        返回生成的回复内容以及使用的 token 数量。
        :return
        content: 生成的回复内容。
        token_dict: 包含'prompt_tokens'、'completion_tokens'和'total_tokens'的字典，
                    分别表示提示的 token 数量、生成的回复的 token 数量和总的 token 数量。
        """
        completion = self.create_completion(messages)
        if len(completion.choices) > 0:
            content = completion.choices[0].message.content
            usage: CompletionUsage = completion.usage
            token_dict = {
                'prompt_tokens': usage.prompt_tokens,
                'completion_tokens': usage.completion_tokens,
                'total_tokens': usage.total_tokens,
            }
            return content, token_dict
        return None

    def moderation(self, _input: Union[str, List[str]]) -> List[dict]:
        """
        Moderation API 是一个有效的内容审查工具。
        目标：确保内容符合 OpenAI 的使用政策，帮助开发人员识别和过滤各种类别的违禁内容，例如仇恨、自残、色情和暴力等。
        文档：https://platform.openai.com/docs/guides/moderation/quickstart
        :return
        flagged：true表示如果模型将内容归类为潜在有害内容，否则为false。
        categories：包含每个类别违规标记的字典。对于每个类别，如果模型将相应类别标记为违规，则为true，否则为false。
        category_scores：包含模型输出的每个类别的原始分数的字典，表示模型对输入违反 OpenAI 针对该类别的政策的置信度。
        """
        response: ModerationCreateResponse = self.client.moderations.create(
            input=_input,
            timeout=self.timeout
        )
        data = []
        if response:
            for result in response.results:
                data.append(result.to_dict())
            return data
        return None


if __name__ == "__main__":
    _text = "中国的首都是哪里"
    _prompt = f"""
    请将三个反引号之间的文本总结成一句话。
    ```{_text}```
    """
    chat_robot = OpenAiChat()
    # print(chat_robot.get_completion(_prompt))

    # _messages = [
    #     {'role': 'system', 'content': '你是一个助理， 并以 Seuss 苏斯博士的风格作出回答。'},
    #     {'role': 'user', 'content': '就快乐的小鲸鱼为主题给我写一首短诗'}
    # ]
    # print(chat_robot.get_completion_and_token_count(_messages))

    print(chat_robot.moderation("你是个傻屌。"))
