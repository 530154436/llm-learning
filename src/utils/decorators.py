#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2021/5/18 9:21
# @function:    一些常用的装饰器
import time
import functools
import traceback
from src.utils.logger import LOGGER

print_success_info = '[触发重试机制-第%s次成功]: 调用方法 --> [%s]'
print_error_info = '[触发重试机制-第%s次失败]: 调用方法 --> [%s], \n%s'


def handle_exception(max_retry=None, timeout=300, interval=5,
                     error_detail_level=1, is_throw_error=False):
    """
    捕获函数错误的装饰器,重试并打印日志
    :param max_retry: 最大尝试次数
    :param timeout : 超时时间
    :param interval: 重试间隔时间
    :param error_detail_level :为0打印exception提示，为1打印3层深度的错误堆栈，为2打印所有深度层次的错误堆栈
    :param is_throw_error : 在达到最大次数时候是否重新抛出错误
    :type error_detail_level: int
    """
    def get_error_info(e: Exception):
        """ 根据等级返回错误信息 """
        error_info = ''
        if error_detail_level == 0:
            error_info = '错误类型是：' + str(e.__class__) + '  ' + str(e)
        elif error_detail_level == 1:
            error_info = '错误类型是：' + str(e.__class__) + '  ' + traceback.format_exc(limit=3)
        elif error_detail_level == 2:
            error_info = '错误类型是：' + str(e.__class__) + '  ' + traceback.format_exc()
        return error_info

    if error_detail_level not in [0, 1, 2]:
        raise Exception('error_detail_level参数必须设置为0 、1 、2')

    def _handle_exception(func):
        @functools.wraps(func)
        def __handle_exception(*args, **kwargs):
            cnt = 0
            t0 = time.time()
            while True:
                # 达到最大重试次数或超时就退出
                if (not(max_retry is None) and cnt > max_retry) or (time.time()-t0) > timeout:
                    break
                try:
                    result = func(*args, **kwargs)
                    if cnt >= 1:
                        LOGGER.warning(print_success_info, cnt, func.__name__)
                    return result

                except Exception as e:
                    LOGGER.error(print_error_info, cnt, func.__name__, get_error_info(e))
                    cnt += 1
                    if (not(max_retry is None) and cnt > max_retry) or (time.time()-t0) > timeout:  # 达到超时时间，
                        if is_throw_error:  # 重新抛出错误
                            raise e
                    time.sleep(interval)

        return __handle_exception

    return _handle_exception
