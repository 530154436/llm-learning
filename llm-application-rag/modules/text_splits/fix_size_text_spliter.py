#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2025/5/28 20:08
# @function:
from typing import Callable, List
from modules.text_splits.base import TextSplitter


class FixedSizeCharacterTextSplitter(TextSplitter):
    """按照固定的字符数（Character Count）或 Token 数（Token Count）来切割文本。
    `核心思想`：
    （1）设定一个 chunk_size（如 500 个字符）和一个 chunk_overlap（如 50 个字符）。<br>
    （2）从文本开头取 chunk_size 个字符作为第一个块，然后下一次从 start_index + chunk_size - chunk_overlap 的位置开始取下一个块，依此类推。
    """

    def __init__(
        self,
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        length_function: Callable[[str], int] = len
    ) -> None:
        """Initialize the splitter."""
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function
        )

    def split_text(self, text: str) -> List[str]:
        """Split text into fixed-size chunks with overlap.
        :param text: Input text to be split.
        :return List[str]: List of text chunks.
        """
        if self._strip_whitespace:
            text = text.strip()

        total_length = self._length_function(text)
        if total_length == 0:
            return []

        chunks = []
        start_idx = 0
        while start_idx < total_length:
            end_idx = min(start_idx + self._chunk_size, total_length)
            chunk = text[start_idx:end_idx]
            if self._strip_whitespace:
                chunk = chunk.strip()
            if chunk:
                chunks.append(chunk)
            start_idx = max(end_idx - self._chunk_overlap, end_idx - self._chunk_size)  # 重叠部分
        return chunks
