#!/usr/bin/env python3
# -*- coding:utf-8 -*--
import os
import sys
import logging.handlers
import logging
from logging import FileHandler, StreamHandler, Formatter
from pathlib import Path

logging_name_to_Level = {
    'CRITICAL': logging.CRITICAL,
    'FATAL': logging.FATAL,
    'ERROR': logging.ERROR,
    'WARN': logging.WARNING,
    'WARNING': logging.WARNING,
    'INFO': logging.INFO,
    'DEBUG': logging.DEBUG,
    'NOTSET': logging.NOTSET,
}


def get_my_logger(log_path: str = None, log_level: str = "INFO", name='mylogger'):
    """
    获取日志实例
    Args:
        log_path:    日志目录
        log_level:   日志级别
        name:        日志记录器的名字
    Returns:
    """
    # log 路径
    if not log_path:
        log_path: Path = Path(os.path.realpath(__file__)).parent.parent.parent
        log_path = log_path.joinpath('logs')
    if not log_path.exists():
        log_path.mkdir(parents=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging_name_to_Level.get(log_level.upper(), logging.INFO))

    # 可以将日志写到任意位置(handlers)
    default_handlers = {
        logging.handlers.WatchedFileHandler(log_path.joinpath('all.log')): logging.INFO,   # 所有日志
        logging.handlers.WatchedFileHandler(log_path.joinpath('error.log')): logging.ERROR,   # 错误日志
        StreamHandler(sys.stdout): logging.INFO                         # 控制台
    }

    # 日志格式：[时间] [文件名-行号] [类型] [信息]
    # _format = '%(asctime)s - %(levelname)s - %(message)s'
    # _format = '%(asctime)s - %(filename)s[:%(lineno)d] - %(levelname)s - %(message)s'
    _format = '%(asctime)s - %(levelname)s - %(filename)s[:%(lineno)d] %(message)s'

    # 添加多个位置
    for handler, level in default_handlers.items():
        handler.setFormatter(Formatter(_format))
        if level is not None:
            handler.setLevel(level)
        logger.addHandler(handler)
    return logger


LOGGER = get_my_logger()
