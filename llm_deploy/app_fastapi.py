#!/usr/bin/env python3
# -*- coding:utf-8 -*--
import gc
from pathlib import Path

from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import uvicorn
import json
import datetime
import torch

# 设置设备参数
DEVICE = "cuda"  # 使用CUDA
DEVICE_ID = "0"  # CUDA设备ID，如果未设置则为空
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE  # 组合CUDA设备信息

# 加载预训练的分词器和模型: 15112MiB
BASE_DIR = Path(__file__).parent.parent
model_name_or_path = BASE_DIR.joinpath('data/models/Qwen2.5-7B-Instruct').__str__()
print(model_name_or_path)
TOKENIZER = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
MODEL = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto", torch_dtype=torch.bfloat16)


def clear_torch_cache():
    gc.collect()

    if torch.has_mps:
        try:
            from torch.mps import empty_cache
            empty_cache()
        except Exception as e:
            print(e)
            print("如果您使用的是 macOS 建议将pytorch版本升级至 2.0.0 或更高版本，以支持及时清理 torch 产生的内存占用。")
    elif torch.has_cuda:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# 创建FastAPI应用
app = FastAPI()


# 处理POST请求的端点
@app.post("/")
async def create_item(request: Request):
    json_post_raw = await request.json()  # 获取POST请求的JSON数据
    json_post = json.dumps(json_post_raw)  # 将JSON数据转换为字符串
    json_post_list = json.loads(json_post)  # 将字符串转换为Python对象
    prompt = json_post_list.get('prompt')  # 获取请求中的提示
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    # 调用模型进行对话生成
    input_ids = TOKENIZER.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = TOKENIZER([input_ids], return_tensors="pt").to('cuda')
    generated_ids = MODEL.generate(model_inputs.input_ids, max_new_tokens=512)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = TOKENIZER.batch_decode(generated_ids, skip_special_tokens=True)[0]
    now = datetime.datetime.now()  # 获取当前时间
    time = now.strftime("%Y-%m-%d %H:%M:%S")  # 格式化时间为字符串
    # 构建响应JSON
    answer = {
        "response": response,
        "status": 200,
        "time": time
    }
    # 构建日志信息
    log = "[" + time + "] " + '", prompt:"' + prompt + '", response:"' + repr(response) + '"'
    print(log)  # 打印日志
    clear_torch_cache()  # 执行GPU内存清理
    return answer  # 返回响应


if __name__ == '__main__':
    # 启动FastAPI应用
    # 用6006端口可以将autodl的端口映射到本地，从而在本地使用api
    uvicorn.run(app, host='0.0.0.0', port=6006, workers=1)  # 在指定端口和主机上启动应用

    # 测试用例
    # curl http://localhost:6006/ -H "Content-Type: application/json" -d '{
    #   "prompt": "你是谁？"
    # }'
