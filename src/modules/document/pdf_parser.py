#!/usr/bin/env python3
# -*- coding:utf-8 -*--
import fitz
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

    def parse(self, fnm: Union[str, Path], verbose: bool = False) -> List[str]:
        """
        Asynchronously extracts text from a PDF file and returns it in chunks.
        表格格式：
        ```
        股票种类||股票上市交易所||股票简称||股票代码\n
        A股||上海证券交易所||中国联通||600050
        ```
        """
        final_texts = []
        final_tables = []
        # Open the PDF file using pdfplumber
        doc = fitz.open(fnm) if isinstance(fnm, (str, Path)) else fitz.open(BytesIO(fnm))
        if verbose:
            doc = tqdm(doc, total=len(doc), desc="get pages")
        for page in doc:
            table = page.find_tables()
            table = list(table)
            for ix, tab in enumerate(table):
                tab = tab.extract()
                tab = list(map(lambda x: [str(t) for t in x], tab))
                tab = list(map("||".join, tab))
                tab = "\n".join(tab)
                final_tables.append(tab)

            text = page.get_text()
            # clean up text for any problematic characters
            # text = re.sub("\n", " ", text).strip()
            # text = text.encode("ascii", errors="ignore").decode("ascii")
            # text = re.sub(r"([^\w\s])\1{4,}", " ", text)
            # text = re.sub(" +", " ", text).strip()
            final_texts.append(text)
        doc.close()
        return final_texts + final_tables


if __name__ == "__main__":
    from src.conf.config import BASE_DIR
    pdf_parser = PdfParserUsingPyMuPDF()
    contents = pdf_parser.parse(BASE_DIR.joinpath("data", "temp", "AY11.pdf"))
    print(contents)
