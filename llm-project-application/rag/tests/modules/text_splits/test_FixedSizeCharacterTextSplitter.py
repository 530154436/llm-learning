#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2025/5/28 15:27
# @function:
import unittest

from modules.text_splits import merge_small_chunks
from modules.text_splits.fix_size_text_spliter import FixedSizeCharacterTextSplitter


class TestFixedSizeCharacterTextSplitter(unittest.TestCase):

    def setUp(self):
        self.chunk_size = 20
        self.text_splitter = FixedSizeCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=0,  # 不允许重叠
            length_function=len
        )

    def test_basic_splitting(self):
        text = "This is a simple sentence.\n\nThis is another paragraph."
        chunks = merge_small_chunks(self.text_splitter.split_text(text), chunk_size=self.chunk_size)

        expected = ['This is a simple sen', 'tence.\n\nThis is anot', 'her paragraph.']
        print(chunks)
        self.assertEqual(chunks, expected)

    def test_recursive_fallback(self):
        text = "Longwordwithoutspaces"
        chunks = merge_small_chunks(self.text_splitter.split_text(text), chunk_size=self.chunk_size)
        print(chunks)
        expected = ['Longwordwithoutspace', 's']
        self.assertEqual(chunks, expected)

    def test_keep_separator(self):
        text = "Hello\n\nWorld"
        chunks = merge_small_chunks(self.text_splitter.split_text(text), chunk_size=self.chunk_size)
        print(chunks)
        expected = ['Hello\n\nWorld']
        self.assertEqual(chunks, expected)


if __name__ == "__main__":
    unittest.main()
