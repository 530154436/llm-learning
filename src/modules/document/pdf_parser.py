#!/usr/bin/env python3
# -*- coding:utf-8 -*--
import fitz
import pymupdf4llm
from pathlib import Path
from typing import Union, List
from io import BytesIO
from tqdm import tqdm


class PdfParserUsingPyMuPDF(object):
    """
    PyMuPDF
    https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/index.html
    """
    supported_file_extensions = [".pdf"]

    def __init__(self, chunk_size: int = 1000):
        self.chunk_size = chunk_size

    def parse_table(self, page) -> List[str]:
        """
        解析表格
        :param page: pymupdf.Page
        :return: MarkDown形式的表格文本
        """
        tables = []
        table = page.find_tables()
        for ix, tab in enumerate(list(table)):
            tab = tab.extract()
            tab = list(map(lambda x: [str(t) for t in x], tab))

            # 转换为 Markdown 格式
            header = tab[0]  # 假设第一行是表头
            separator = ["---" for _ in header]  # 创建分隔行
            rows = tab[1:]  # 其余为数据行

            # 创建 Markdown 表格字符串
            markdown_table = "| " + " | ".join(header) + " |\n"
            markdown_table += "| " + " | ".join(separator) + " |\n"
            for row in rows:
                markdown_table += "| " + " | ".join(row) + " |\n"
            tables.append(markdown_table)
        return tables

    def parse(self, fnm: Union[str, Path],
              verbose: bool = False, clip: int = 0) -> List[str]:
        """
        解析pdf文件
        1、表格格式：
        ```
        股票种类||股票上市交易所||股票简称||股票代码\n
        A股||上海证券交易所||中国联通||600050
        ```
        :param fnm:
        :param verbose:
        :param clip: 页眉、页脚高度，默认为0
        :return:
        """
        final_texts = []
        final_tables = []
        # Open the PDF file using pdfplumber
        doc = fitz.open(fnm) if isinstance(fnm, (str, Path)) else fitz.open(BytesIO(fnm))
        if verbose:
            doc = tqdm(doc, total=len(doc), desc="get pages")
        for i, page in enumerate(doc):

            # 删除页眉和页脚（假设它们在特定区域）
            crop = None
            if clip > 0:
                crop = fitz.Rect(0, clip, page.rect.width, page.rect.height - clip)
            text = page.get_text(clip=crop)
            final_texts.extend(text)

            # 提取表格
            final_tables.extend(self.parse_table(page))

        doc.close()
        return final_texts + final_tables


class PdfParserUsingPymupdf4llm(object):
    """
    PyMuPDF, LLM & RAG
    https://pymupdf.readthedocs.io/en/latest/rag.html
    """
    def __init__(self, chunk_size: int = 1000):
        self.chunk_size = chunk_size

    def parse(self, fnm: Union[str, Path],
              verbose: bool = False, clip: int = 0) -> str:
        return pymupdf4llm.to_markdown(doc=file)


if __name__ == "__main__":
    from src.conf.config import BASE_DIR
    file = BASE_DIR.joinpath("data", "temp", "AY01.pdf")

    # pdf_parser = PdfParserUsingPyMuPDF()
    # contents = pdf_parser.parse(file, clip=80)
    # print(contents)

    result = PdfParserUsingPymupdf4llm().parse(file)
    print(result)
