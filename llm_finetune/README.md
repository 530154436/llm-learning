


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
