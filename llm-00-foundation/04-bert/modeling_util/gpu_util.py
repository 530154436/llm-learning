#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2025/5/23 13:47
# @function:
import functools
import subprocess
import torch
from loguru import logger


def print_gpu_memory_stats(gpu_index: int = None):
    if torch.cuda.is_available():
        with_gpu_index = f"-i {gpu_index}" if gpu_index else ""
        cmd = f"nvidia-smi {with_gpu_index} --query-gpu=memory.used,memory.total --format=csv"
        gpu_info = subprocess.check_output(cmd.split()).decode("utf-8").strip().split("\n")
        for i, msg in enumerate(gpu_info):
            prefix = f"{i}, " if i != 0 else "GPU, "
            msg = prefix + msg
            logger.info(msg)


def print_gpu_memory_usage(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # 打印加载前显存使用情况
        if torch.cuda.is_available():
            logger.info("显存使用情况（加载前）:")
            print_gpu_memory_stats()

        # 加载模型
        res = func(*args, **kwargs)

        # 打印加载后显存使用情况
        if torch.cuda.is_available():
            logger.info("显存使用情况（加载后）:")
            print_gpu_memory_stats()

        return res

    return wrapper


def clear_torch_cache():
    gc.collect()

    if torch.has_mps:
        try:
            from torch.mps import empty_cache
            empty_cache()
        except Exception as e:
            print(e)
            print("如果您使用的是 macOS 建议将pytorch版本升级至 2.0.0 或更高版本，以支持及时清理 torch 产生的内存占用。")
    elif torch.has_cuda:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
