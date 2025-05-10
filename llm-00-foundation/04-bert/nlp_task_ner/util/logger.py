#!/usr/bin/env python3
# -*- coding:utf-8 -*--
import os
import sys
from logging import Logger, getLogger, StreamHandler, Formatter, ERROR, INFO
from logging.handlers import TimedRotatingFileHandler


def get_logger(
    path: str = "/data/log",
    level: str = "INFO",
    name='mylogger'
) -> Logger:
    """
    获取日志实例
    Args:
        path:    日志目录
        level:   日志级别
        name:        日志记录器的名字
    Returns:
    """
    logger = getLogger(name)
    logger.setLevel(level)

    if not os.path.exists(path):
        os.makedirs(path)

    # 可以将日志写到任意位置(handlers)
    default_handlers = {
        # 设置文件名、滚动间隔和保留日志文件个数
        TimedRotatingFileHandler(f'{path}/all.log', when="midnight", interval=1, backupCount=7, encoding="utf8"): level,
        TimedRotatingFileHandler(f'{path}/error.log', when="midnight", interval=1, backupCount=7, encoding="utf8"): ERROR,
        # 控制台
        StreamHandler(sys.stdout): INFO
    }

    # 日志格式：[时间] [文件名-行号] [类型] [信息]
    # _format = '%(asctime)s - %(levelname)s - %(message)s'
    # _format = '%(asctime)s - %(filename)s[:%(lineno)d] - %(levelname)s - %(message)s'
    _format = '%(asctime)s - %(levelname)s - %(filename)s[:%(lineno)d] %(message)s'

    # 添加多个位置
    for handler, level in default_handlers.items():
        handler.setFormatter(Formatter(_format))
        logger.addHandler(handler)
    return logger

