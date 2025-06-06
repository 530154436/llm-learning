#!/usr/bin/env python3
# -*- coding:utf-8 -*--
import os
from typing import List, Union
from sparkai.core.messages import ChatMessage
from sparkai.llm.llm import ChatSparkLLM, ChunkPrintHandler
from llm_common.src.utils.chat_openai import OpenAiChat
from llm_common.src.utils.decorators import handle_exception


class SparkAiChatWSS(object):

    SPARKAI_URL = 'wss://spark-api.xf-yun.com/v3.5/chat'
    SPARKAI_APP_ID = os.environ.get("SPARKAI_APP_ID")
    SPARKAI_API_SECRET = os.environ.get("SPARKAI_API_SECRET")
    SPARKAI_API_KEY = os.environ.get("SPARKAI_API_KEY")

    def __init__(self, model: str = "generalv3.5", temperature: float = 0.1,
                 max_tokens: int = 8192, timeout: int = 60):
        """
        星火认知大模型Spark3.5 Max的URL值，其他版本大模型URL值请前往文档（https://www.xfyun.cn/doc/spark/Web.html）查看
        星火认知大模型调用秘钥信息，请前往讯飞开放平台控制台（https://console.xfyun.cn/services/bm35）查看
        星火认知大模型Spark3.5 Max的domain值，其他版本大模型domain值请前往文档（https://www.xfyun.cn/doc/spark/Web.html）查看
        :param model: 调用的模型
        :param temperature: 模型输出的温度系数，控制输出的随机程度；值越大，生成的回复越随机。默认为 0。
        :param max_tokens: 生成回复的最大 token 数量。默认为 500。
        :param timeout: 超时时间
        """
        self.model: str = model
        self.temperature: int = temperature
        self.max_tokens: int = max_tokens
        self.timeout: int = timeout
        self.client = ChatSparkLLM(
            spark_api_url=self.SPARKAI_URL,
            spark_app_id=self.SPARKAI_APP_ID,
            spark_api_key=self.SPARKAI_API_KEY,
            spark_api_secret=self.SPARKAI_API_SECRET,
            spark_llm_domain=self.model,
            streaming=False,
            request_timeout=timeout,
            temperature=temperature,
            max_tokens=max_tokens
        )

    @handle_exception(max_retry=5)
    def get_completion(self, prompt: str):
        messages = [
            ChatMessage(role="user", content=prompt)
        ]
        return self.get_completion_from_message(messages)

    @handle_exception(max_retry=5)
    def get_completion_from_message(self, messages: List[Union[ChatMessage, dict]]):
        chat_messages = []
        for message in messages:
            if isinstance(message, dict):
                chat_messages.append(ChatMessage(role=message.get("role"),
                                                 content=message.get("content")))
            elif isinstance(message, ChatMessage):
                chat_messages.append(message)

        handler = ChunkPrintHandler()
        a = self.client.generate([chat_messages], callbacks=[handler])
        return a.generations[0][0].text


class SparkAiChatWSS4Finetune(SparkAiChatWSS):

    SPARKAI_URL = 'wss://spark-api-n.xf-yun.com/v3.1/chat'
    SPARKAI_APP_ID = os.environ.get("SPARKAI_APP_ID")
    SPARKAI_API_SECRET = os.environ.get("SPARKAI_API_SECRET")
    SPARKAI_API_KEY = os.environ.get("SPARKAI_API_KEY")

    def __init__(self, model: str = "patchv3", temperature: float = 0.1,
                 max_tokens: int = 8192, timeout: int = 60):
        super().__init__(model, temperature, max_tokens, timeout)


class SparkAiChat(OpenAiChat):
    base_url = "https://spark-api-open.xf-yun.com/v1"
    api_key = f'{os.environ.get("SPARKAI_API_KEY")}:{os.environ.get("SPARKAI_API_SECRET")}'

    def __init__(self, model: str = "generalv3.5", temperature: int = 0,
                 max_tokens: int = 8192, timeout: int = 60):
        """
        文档：https://www.xfyun.cn/doc/spark/Web.html#_1-接口说明
        :param model: 调用的模型，默认为 generalv3.5指向Max版本;
        :param temperature: 模型输出的温度系数，控制输出的随机程度；值越大，生成的回复越随机。默认为 0。
        :param max_tokens: 生成回复的最大 token 数量。默认为 500。
        :param timeout: 超时时间
        """
        super().__init__(model, temperature=temperature, max_tokens=max_tokens, timeout=timeout)


if __name__ == "__main__":
    # 测试模型配置是否正确
    _prompt = "中国的首都是哪里"
    _messages = [
        # ChatMessage(role="system", content="你是一个智能客服。"),
        # ChatMessage(role="user", content=_text)
        {'role': 'system', 'content': '你是一个智能客服.'},
        {'role': 'user', 'content': _prompt}
    ]

    # print(SparkAiChatWSS().get_completion(_prompt))
    # print(SparkAiChatWSS().get_completion_from_message(_messages))
    # print(SparkAiChatWSS4Finetune().get_completion_from_message(_messages))

    chat_robot = SparkAiChat()
    print(chat_robot.get_completion(_prompt))
    print(chat_robot.get_completion_from_messages(_messages))
    print(chat_robot.get_completion_with_usage(_messages))

