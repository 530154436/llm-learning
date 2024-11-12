#!/bin/bash

#conda activate base

nohup python -m vllm.entrypoints.openai.api_server \
--host 0.0.0.0 \
--port 8000 \
--model data/models/Qwen2.5-7B-Instruct \
--served-model-name Qwen2.5-7B-Instruct \
--gpu-memory-utilization 0.8 \
--max-model-len 2400 > server.log &
