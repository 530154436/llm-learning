


| 镜像名称                      | 大小      | Python版本 | 备注 |
|---------------------------|---------|----------|----|
| 15521147129/llm:vllm0.6.3 | 10.38GB | 3.10.12  |    |



加载预训练的分词器和模型: 15112MiB


cd llm-learning/llm_app_deploy/
python -u app_fastapi.py


cd llm-learning/
/bin/bash app_vllm.sh


vLLM 源码解读系列
https://github.com/PaddleJitLab/CUDATutorial/blob/develop/docs/16_vllm_source_code/01_vllm_arch.md


Qwen-TensorRT-LLM
https://github.com/Tlntin/Qwen-TensorRT-LLM/tree/main

TensorRT-LLM-Qwen
https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/qwen

LLM 推理 - Nvidia TensorRT-LLM 与 Triton Inference Server
https://www.cnblogs.com/zackstang/p/18269743

一文探秘LLM应用开发(18)-模型部署与推理(框架工具-Triton Server、RayLLM、OpenLLM)
https://developer.volcengine.com/articles/7382246474320445490
