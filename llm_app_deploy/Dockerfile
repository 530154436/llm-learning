# FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime
FROM mix-docker-registry.qizhidao.net/library/cuda.12.2.2-runtime-ubuntu22.04.llmsvr:v1.0.0
ARG conda_env=base
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

RUN mkdir /data/logs

RUN /app/anaconda3/bin/pip3 install vllm==0.5.2 -i http://maven.qizhidao.net:8081/repository/algo-public/simple/  --trusted-host  maven.qizhidao.net

COPY requirements.txt requirements.txt
RUN /app/anaconda3/bin/pip3 install --no-cache -r requirements.txt -i http://maven.qizhidao.net:8081/repository/algo-public/simple/  --trusted-host  maven.qizhidao.net

