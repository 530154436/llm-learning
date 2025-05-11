#!/usr/bin/env python3
# -*- coding:utf-8 -*--
import logging
import sys
from logging import Logger, getLogger, StreamHandler, Formatter
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path


LEVEL_MAP = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL,
}


def get_logger(
    path: str = "/data/log",
    level: str = "INFO",
    name='mylogger'
) -> Logger:
    """
    获取日志记录器
    Args:
        path: 日志存储路径（默认为当前目录 /data/log）
        level: 日志级别（DEBUG/INFO/WARNING/ERROR/CRITICAL）
        name: 日志记录器名称
    Returns:
        logging.Logger 实例
    """
    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

    # 日志格式：[时间] [文件名-行号] [类型] [信息]
    _format = '%(asctime)s - %(levelname)s - %(filename)s[:%(lineno)d] %(message)s'

    # 创建 logger
    logger = getLogger(name)
    logger.setLevel(LEVEL_MAP.get(level.upper(), logging.INFO))

    # 定义多个输出位置及其日志级别
    handlers = [
        (TimedRotatingFileHandler(path/"all.log", when="midnight", interval=1, backupCount=7, encoding="utf-8"), level),
        (TimedRotatingFileHandler(path/"error.log", when="midnight", interval=1, backupCount=7, encoding="utf-8"), 'ERROR'),
        (StreamHandler(sys.stdout), 'INFO'),
    ]

    formatter = Formatter(_format)
    for handler, log_level in handlers:
        handler.setLevel(LEVEL_MAP.get(log_level.upper(), logging.INFO))
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


# 防止日志消息从 logger 传播到父级 logger（包括 root logger），避免日志重复打印的问题
# LOGGER = get_logger()
# LOGGER.propagate = False
