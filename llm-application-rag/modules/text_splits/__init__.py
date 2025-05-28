#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2025/5/28 19:36
# @function:
from typing import List


def merge_small_chunks(chunks: List[str], chunk_size: int = 4000) -> List[str]:
    """
    在使用 RecursiveCharacterTextSplitter 时，当文本无法按语义边界（如空格、换行）完整切分而不超过 chunk_size 时，
    就会被迫降级到字符级切割，导致出现像 'a '、'\n'、'.' 这样的“小 chunk”。
    """
    merged = []
    i = 0
    n = len(chunks)

    while i < n:
        current = chunks[i]
        # 尝试合并到前一个
        if merged and len(merged[-1] + current) <= chunk_size:
            merged[-1] += current
            i += 1
        # 尝试合并到后一个
        elif i + 1 < n and len(current + chunks[i + 1]) <= chunk_size:
            merged.append(current + chunks[i + 1])
            i += 2
        else:
            merged.append(current)
            i += 1
    return merged
