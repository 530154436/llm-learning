#!/usr/bin/env python3
# -*- coding:utf-8 -*--
import json


def read_json(json_file_path: str):
    """读取json文件"""
    if json_file_path.endswith("jsonl"):
        data = []
        with open(json_file_path, 'r') as f:
            for row in f:
                data.append(json.loads(row))
    else:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
    return data


def write_json(json_file_path, data):
    """写入json文件"""
    with open(json_file_path, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
