#FROM mix-docker-registry.qizhidao.net/library/pytorch:2.1.2-cuda12.1-cudnn8-runtime
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# 实际工作目录：通过外部挂载代码+模型作为工作目录（docker run -w /data/devNFS/zhengchubin/llm-learning/llm_app_deploy）
RUN pwd
#
COPY requirements.txt /workspace

RUN ln -fs /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
RUN python -m pip install --upgrade pip setuptools wheel -i http://maven.qizhidao.net:8081/repository/algo-public/simple/ --trusted-host maven.qizhidao.net
RUN python -m pip install --no-cache -r requirements.txt -i http://maven.qizhidao.net:8081/repository/algo-public/simple/ --trusted-host maven.qizhidao.net

EXPOSE ${APP_PORT}
RUN which gunicorn
#
