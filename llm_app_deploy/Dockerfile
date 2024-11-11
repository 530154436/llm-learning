# 15521147129/llm:vllm0.6.3
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHON_VERSION=3.11

# 实际工作目录：通过外部挂载代码+模型作为工作目录（docker run -w /data/devNFS/zhengchubin/llm-learning/llm_app_deploy）
WORKDIR /
RUN echo `pwd`
RUN ln -fs /usr/share/zoneinfo/Asia/Shanghai /etc/localtime

# 安装必要软件: Python3.10.12
COPY sources.list /etc/apt/sources.list
RUN apt-get update && \
    apt-get install -y \
    git \
    python3 \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.10 /usr/bin/python
RUN echo `which python`

# 安装Python依赖
COPY requirements.txt /requirements.txt
RUN python -m pip install --no-cache --upgrade pip setuptools wheel -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com
RUN python -m pip install --no-cache -r requirements.txt -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com

# /usr/local/bin/gunicorn
RUN echo `which gunicorn`
