#!/usr/bin/env python3
# -*- coding:utf-8 -*--
from tools import PRODUCTS_BY_CATEGORY


def load_prompt_find_category_and_product(delimiter="####") -> str:
    """
    抽取产品和类别的提示
    """
    with open("data/0_find_category_and_product.tmpl", encoding="utf8") as f:
        system_message = "".join(f.readlines())

    product_strs = []
    for k, v in PRODUCTS_BY_CATEGORY.items():
        category = [f"{k}类别："]
        category.extend(v)
        product_strs.append("\n".join(category))
    system_message = system_message.format(delimiter=delimiter,
                                           product_info="\n\n".join(product_strs))
    return system_message


def load_prompt(file: str) -> str:
    """
    加载提示词
    """
    with open(file, encoding="utf8") as f:
        system_message = "\n".join(f.readlines())
        return system_message
