#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2024/10/24 19:02
# @function:
import time
import pandas as pd
import requests
from typing import List, Tuple, Iterable
from concurrent.futures import ThreadPoolExecutor
from llm_common.src.utils.decorators import handle_exception
from llm_common.src.utils.logger import LOGGER


class InterfacePerformanceEvaluator(object):
    """
    服务接口性能测试
    """
    def __init__(self, url: str, workers: Tuple[int] = tuple([1]),
                 output: str = None):
        self.url = url
        self.workers = workers
        self.costs: List[Tuple] = []
        self.results: List[dict] = []
        self.output = output

    def request(self, payload: dict) -> dict:
        """ 请求服务接口 """

        @handle_exception(max_retry=5, interval=0.5, timeout=3)
        def job():
            headers = {
                'Content-Type': 'application/json'
            }
            rsp = requests.post(self.url, json=payload, headers=headers, timeout=3)
            if rsp is None:
                print("Request failed return None.")
                raise Exception("Request failed None.")
            elif rsp.status_code != 200:
                print(f'Request failed with status code {rsp.status_code}')
                raise Exception(rsp.text)

        start = time.time()
        job()
        delay = time.time() - start
        return delay

    def gen_report(self) -> List[dict]:
        total = 1
        for i, (worker, total_delay, total_num) in enumerate(self.costs):
            result = {
                "序号": i + 1,
                "数据量": total_num,
                "并发数（请求的线程数）": worker,
                "平均每次请求耗时/秒": f"{round(total_delay / total_num, 3) if total_num > 0 else 'NA'}",
                "总耗时/秒": f"{round(total_delay, 3) if total_delay > 0 else 'NA'}",
            }
            self.results.append(result)
            total = total_num
        if self.output is not None:
            name = self.output.split(".")[0]
            subfix = self.output.split(".")[-1]
            with pd.ExcelWriter(f"{name}-{total}.{subfix}") as writer:
                pd.DataFrame(self.results).to_excel(writer, index=False, sheet_name="结果")
        return self.results

    def run(self, params: Iterable[dict]) -> List[dict]:
        """ 并发请求接口并计算相关指标 """
        assert params is not None
        # 并发请求服务接口
        futures = []
        tasks_total = len(self.workers)
        for num, worker in enumerate(self.workers, start=1):
            start = time.time()
            pool = ThreadPoolExecutor(worker)
            for i, payload in enumerate(params, start=1):
                futures.append(pool.submit(self.request, payload=payload))
            for future in futures:
                future.result()
            pool.shutdown()
            total_delay = time.time() - start
            self.costs.append((worker, total_delay, len(params)))

            del pool
            LOGGER.info("已完成: %s/%s, 并发数=%s.", num, tasks_total, worker)
            futures.clear()
        # 生成报告
        self.gen_report()
        return self.results


def main(size: int = 10):
    _workers = [1, 2, 4, 8, 10, 16, 20, 30, 40, 50, 60]
    _url = ""
    print("url: ", _url, "数据量: ", size)
    evaluator = InterfacePerformanceEvaluator(url=_url, workers=_workers, output="接口压测.xlsx")
    _payloads = [{}]
    evaluator.run(_payloads)
