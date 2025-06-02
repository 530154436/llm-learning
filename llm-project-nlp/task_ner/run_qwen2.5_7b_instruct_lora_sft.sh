llamafactory-cli train conf/Qwen2.5-7B-Instruct-lora-sft.yaml

nohup python -m vllm.entrypoints.openai.api_server \
--host 0.0.0.0 \
--port 8000 \
--model /data/dev-nfs/zhengchubin/models/Qwen2.5-7B-Instruct \
--served-model-name Qwen2.5-7B-Instruct \
--enable-lora \
--gpu-memory-utilization 0.8 \
--max-model-len 1024 \
--disable-log-requests \
--lora-modules clue-ner-lora-sft=data/experiment/Qwen2.5-7B-Instruct-clue-ner-lora-sft \
> server.log &
