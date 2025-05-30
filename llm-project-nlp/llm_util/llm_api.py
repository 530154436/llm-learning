#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2024/12/26 15:01
# @function:
import time
from typing import Dict, Any, Union
from openai import OpenAI, AsyncOpenAI, Stream, NOT_GIVEN
from openai.types.chat import ChatCompletion
# from tenacity import stop_after_attempt, wait_fixed, retry, after_log


class ChatClient(object):
    """
    封装 OpenAi 的客户端
    model (str, optional): _description_. Defaults to "gpt-3.5-turbo".
    temperature (int, optional): _description_. Defaults to 1.
    max_tokens (int, optional): 最大的输出tokens. Defaults to 1024.
    frequency_penalty, presence_penalty:  (-2.0, 2.0)  正值代表惩罚
    mu[j] -> mu[j] - c[j] * alpha_frequency - float(c[j] > 0) * alpha_presence
    presence_penalty: 对已经出现在生成内容中的词施加惩罚，降低它们再次被选中的概率。
    frequency_penalty: 对已经多次出现在生成内容中的词施加惩罚，根据词出现的频率调整它们再次被选中的概率。
    temperature: 控制生成文本的随机性或创造性。取值范围: [0, +∞)，通常在 [0, 1] 之间
    top_p: 使用**核采样（Nucleus Sampling）**来控制生成的多样性。
           当 top_p=0.9 时，模型会选择前 90% 概率的词，并忽略概率最低的 10% 的词。
    stream: 控制生成文本的输出方式。 True:模型以流式方式输出文本，即逐字逐句生成和传递。

    重试机制：https://github.com/jd/tenacity

    token  通常 1 个中文词语、1 个英文单词、1 个数字或 1 个符号计为 1 个 token。
        1个Token通常对应1.5-1.8个汉字, 约等于 3~4个字母,
        1 个英文字符 ≈ 0.3 个 token。
        1 个中文字符 ≈ 0.6 个 token。

    各模型版本上下文
    https://help.aliyun.com/zh/model-studio/getting-started/models

    | 模型名称     | 上下文长度   | 最大输入      | 最大输出   |
    |------------|-------------|-------------|-----------|
    | qwen-long  | 10,000,000  | 10,000,000  | 6,000     |
    | qwen-plus  | 131,072     | 129,024     | 8,192     |
    """

    def __init__(self,
                 base_url: str,
                 api_key: str,
                 model="qwen-plus",
                 temperature=0.01,
                 top_p=0.9,
                 max_tokens=8192,
                 frequency_penalty=1.1,
                 presence_penalty=0,
                 stream=False,
                 timeout=3600,
                 system_message: str = None):

        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        if model.__contains__("qwen-plus"):
            headers.update({"X-DashScope-DataInspection": '{"input":"disable", "output":"disable"}'})

        self.client: OpenAI = OpenAI(api_key=api_key, base_url=base_url, default_headers=headers)
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.stream = stream
        self.timeout = timeout
        self.system_message = system_message

    # @retry(reraise=True,
    #        stop=stop_after_attempt(3),
    #        wait=wait_fixed(5),
    #        after=after_log(logging.getLogger(), logging.WARNING))
    def create(self, prompt: str) -> ChatCompletion:
        assert prompt is not None
        messages = []
        if self.system_message:
            messages.append({'role': 'system', 'content': self.system_message})
        messages.append({'role': 'user', 'content': prompt})

        stream_options = NOT_GIVEN
        if self.stream:
            stream_options = {"include_usage": True}
        extra_body = None
        if self.model.startswith("qwen3"):
            extra_body = {"enable_thinking": False}

        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=None,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            stream=self.stream,
            stream_options=stream_options,  # 仅对流式有效
            timeout=self.timeout,
            extra_body=extra_body
        )

    def completion(self, prompt: str) -> Dict[str, Any]:
        start_time = time.time()
        completion: Union[ChatCompletion, Stream] = self.create(prompt)
        result = {
            "content": "",
            "usage": {
                "prompt_length": len(prompt),
            }
        }

        # 可能会出现输出不完整，不如 first_reason: length
        if self.stream:
            rsp_content = ""
            finish_reason = None  # 记录最后一条数据的状态
            for chunk in completion:  # chunk: ChatCompletionChunk
                # 记录流式消费的token、原因
                if chunk.choices[0].finish_reason is not None:
                    finish_reason = chunk.choices[0].finish_reason
                if chunk.usage is not None:
                    result["usage"].update({
                        "request_id": chunk.id,
                        "total_tokens": chunk.usage.total_tokens,
                        "prompt_tokens": chunk.usage.prompt_tokens,
                        "completion_tokens": chunk.usage.completion_tokens,
                    })

                # 记录输出内容
                event_text = chunk.choices[0].delta.content  # extract the text
                if not event_text: continue
                rsp_content += event_text
                # print(event_text, end='', flush=True)  # 流式输出
            result.update({'content': rsp_content, "finish_reason": finish_reason})
        else:
            result["usage"].update({
                "request_id": completion.id,
                "total_tokens": completion.usage.total_tokens,
                "prompt_tokens": completion.usage.prompt_tokens,
                "completion_tokens": completion.usage.completion_tokens,
            })
            rst = completion.choices[0].message.content
            result.update({'content': rst, "finish_reason": completion.choices[0].finish_reason})

        end_time = time.time()
        result["usage"].update({"delay": round(end_time - start_time, 3)})
        return result


class AsyncChatClient(object):
    """
    封装 OpenAi 的异步客户端
    """

    def __init__(self,
                 base_url: str,
                 api_key: str,
                 model="qwen-plus",
                 temperature=0.01,
                 top_p=0.9,
                 max_tokens=8192,
                 frequency_penalty=1.1,
                 presence_penalty=0,
                 stream=False,
                 timeout=3600,
                 system_message: str = None):

        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        if model.__contains__("qwen-plus"):
            headers.update({"X-DashScope-DataInspection": '{"input":"disable", "output":"disable"}'})

        self.client: AsyncOpenAI = AsyncOpenAI(api_key=api_key, base_url=base_url, default_headers=headers)
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.stream = stream
        self.timeout = timeout
        self.system_message = system_message

    # @retry(reraise=True,
    #        stop=stop_after_attempt(3),
    #        wait=wait_fixed(1),
    #        after=after_log(logging.getLogger(), logging.WARNING))
    async def create(self, prompt: str) -> ChatCompletion:
        assert prompt is not None
        messages = []
        if self.system_message:
            messages.append({'role': 'system', 'content': self.system_message})
        messages.append({'role': 'user', 'content': prompt})

        stream_options = NOT_GIVEN
        if self.stream:
            stream_options = {"include_usage": True}

        extra_body = None
        if self.model.startswith("qwen3"):
            extra_body = {"enable_thinking": False}

        return await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            stream=self.stream,
            stream_options=stream_options,  # 仅对流式有效
            timeout=self.timeout,
            extra_body=extra_body
        )

    async def completion(self, prompt: str) -> Dict[str, Any]:
        start_time = time.time()
        completion: Union[ChatCompletion, Stream] = await self.create(prompt)
        result = {
            "content": "",
            "usage": {
                "prompt_length": len(prompt),
            }
        }

        # 可能会出现输出不完整，不如 first_reason: length
        if self.stream:
            rsp_content = ""
            finish_reason = None  # 记录最后一条数据的状态
            async for chunk in completion:  # chunk: ChatCompletionChunk
                # 记录流式消费的token、原因
                if chunk.choices[0].finish_reason is not None:
                    result.update({
                        "finish_reason": chunk.choices[0].finish_reason
                    })
                if chunk.usage is not None:
                    result["usage"].update({
                        "request_id": chunk.id,
                        "total_tokens": chunk.usage.total_tokens,
                        "prompt_tokens": chunk.usage.prompt_tokens,
                        "completion_tokens": chunk.usage.completion_tokens
                    })

                # 记录输出内容
                event_text = chunk.choices[0].delta.content  # extract the text
                if not event_text: continue
                rsp_content += event_text
                # print(event_text, end='', flush=True)  # 流式输出
            result.update({'content': rsp_content})
        else:
            result["usage"].update({
                "request_id": completion.id,
                "total_tokens": completion.usage.total_tokens,
                "prompt_tokens": completion.usage.prompt_tokens,
                "completion_tokens": completion.usage.completion_tokens,
            })
            rst = completion.choices[0].message.content
            result.update({'content': rst, "finish_reason": completion.choices[0].finish_reason})

        end_time = time.time()
        result["usage"].update({"delay": round(end_time - start_time, 3)})
        return result


if __name__ == '__main__':
    import asyncio

    _base_url = "http://172.19.190.6:31833/v1"
    _api_key = "<KEY>"
    _model = "Qwen2.5-7B-Instruct"
    chat_client = ChatClient(stream=False,
                             base_url=_base_url,
                             api_key=_api_key,
                             model=_model,
                             timeout=60,
                             temperature=0.01,
                             max_tokens=512,
                             top_p=None,
                             frequency_penalty=None,
                             presence_penalty=None)
    _prompt = "你是谁？"
    print(chat_client.completion(_prompt))

    async def main_async():
        chatgpt = AsyncChatClient(stream=False,
                                  base_url=_base_url,
                                  api_key=_api_key,
                                  model=_model,
                                  timeout=60,
                                  temperature=0.01,
                                  max_tokens=512,
                                  top_p=None,
                                  frequency_penalty=None,
                                  presence_penalty=None)
        response = await chatgpt.completion(prompt=_prompt)
        print(response)
    asyncio.run(main_async())
