#!/usr/bin/env python3
# -*- coding:utf-8 -*--
from pathlib import Path
from typing import Union, List
import pdfplumber
from pdfplumber.table import Table


class PdfParserUsingPdfPlumber(object):
    """
    Pdfplumber： 识别PDF页面中的table并从中提取信息。
    参考：https://github.com/jsvine/pdfplumber
    """

    def __init__(self, chunk_size: int = 1000):
        self.chunk_size = chunk_size

    def table_to_markdown(self, table: Table):
        """
        将表格转换为Markdown格式的字符串。
        """
        headers = table[0]  # 假设第一行是表头
        markdown_table = []

        # 处理表头，替换 None 值为 ""
        headers = [str(cell) if cell is not None else "" for cell in headers]
        markdown_table.append("| " + " | ".join(headers) + " |")
        markdown_table.append("|" + "-----|" * len(headers))  # 添加分隔行

        # 添加表格内容，处理 None 值
        for row in table[1:]:
            row = [str(cell) if cell is not None else "" for cell in row]
            markdown_table.append("| " + " | ".join(row) + " |")

        return "\n".join(markdown_table)


    def parse(self, fnm: Union[str, Path], verbose: bool = False, clip: int = None) -> List[str]:
        """
        提取页面中的文本和表格内容，保持顺序。
        """
        # 读取PDF文档
        file = pdfplumber.open(fnm)

        final_texts = []
        final_tables = []

        # 遍历每一页
        for num, page in enumerate(file.pages, start=1):

            # 设置去掉页眉和页脚的 bbox（x0, top, x1, bottom）
            if isinstance(clip, int):
                bbox = (0, clip, page.width, page.height - clip)
                page = page.within_bbox(bbox)

            # 获取页面表格内容
            tables = page.find_tables()

            # 提取表格
            for i, table in enumerate(tables):
                markdown_table = self.table_to_markdown(table.extract())
                final_tables.append(markdown_table)

            # 获取页面文本内容， TODO: 文本中会包含表格的数据
            if page.extract_text():
                final_texts.append(page.extract_text())
        return final_texts



if __name__ == "__main__":
    from src.conf.config import BASE_DIR
    _file = BASE_DIR.joinpath("data", "temp", "AY01.pdf")
    # _file = BASE_DIR.joinpath("data", "temp", "AZ01.pdf")

    pdf_parser = PdfParserUsingPdfPlumber()
    contents = pdf_parser.parse(_file, clip=80)
    for _page in contents:
        print(_page)
