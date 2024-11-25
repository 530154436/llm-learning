#!/bin/bash

nohup python -m vllm.entrypoints.openai.api_server \
--host 0.0.0.0 \
--port 8000 \
--model data/models/Qwen2.5-7B-Instruct \
--served-model-name Qwen2.5-7B-Instruct \
--gpu-memory-utilization 0.9 \
--max-model-len 2400 > server.log &

# 测试服务
#curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
#  "model": "Qwen2.5-7B-Instruct",
#  "messages": [
#    {"role": "system", "content": "你是Qwen，由阿里云创建。你是一个乐于助人的助手。"},
#    {"role": "user", "content": "你是谁？"}
#  ],
#  "temperature": 0.7,
#  "top_p": 0.8,
#  "repetition_penalty": 1.05,
#  "max_tokens": 512
#}'