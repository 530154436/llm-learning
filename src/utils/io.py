#!/usr/bin/env python3
# -*- coding:utf-8 -*--
import json


def read_json(json_file_path):
    """读取json文件"""
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    return data


def write_json(json_file_path, data):
    """写入json文件"""
    with open(json_file_path, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
