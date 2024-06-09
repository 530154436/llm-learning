#!/usr/bin/env python3
# -*- coding:utf-8 -*--
import json
import traceback
from typing import Dict, List, Union

# 获取商品和目录
PRODUCTS: dict = json.load(open("data/products_zh.json", encoding='utf-8'))
PRODUCTS_BY_CATEGORY: Dict[str, list] = dict()
for _product_name, _product_info in PRODUCTS.items():
    _category = _product_info.get('category')
    if _category:
        PRODUCTS_BY_CATEGORY.setdefault(_category, []).append(_product_info.get('name'))
# print(PRODUCTS)
# print(PRODUCTS_BY_CATEGORY)


def get_product_by_name(name: str):
    """
    根据产品名称获取产品

    参数:
    name: 产品名称
    """
    return PRODUCTS.get(name, None)


def get_products_by_category(category: str):
    """
    根据类别获取产品

    参数:
    category: 产品类别
    """
    return [product for product in PRODUCTS.values() if product["category"] == category]


def generate_output_string(data_list: List[dict]):
    """
    根据输入的数据列表生成包含产品或类别信息的字符串。

    参数:
    data_list: 包含字典的列表，每个字典都应包含 "products" 或 "category" 的键。

    返回:
    output_string: 包含产品或类别信息的字符串。
    """
    output_string = ""
    if data_list is None:
        return output_string

    for data in data_list:
        try:
            if "products" in data:
                products_list = data["products"]
                for product_name in products_list:
                    product = get_product_by_name(product_name)
                    if product:
                        output_string += json.dumps(product, indent=4, ensure_ascii=False) + "\n"
                    else:
                        print(f"Error: Product '{product_name}' not found")
            elif "category" in data:
                category_name = data["category"]
                category_products = get_products_by_category(category_name)
                for product in category_products:
                    output_string += json.dumps(product, indent=4, ensure_ascii=False) + "\n"
            else:
                print("Error: Invalid object format")
        except Exception as e:
            traceback.print_exc(limit=5)
            print(f"Error: {e}")

    return output_string


def read_string_to_object(input_string) -> Union[List[dict], dict]:
    """
    字符串转为list对象
    """
    if input_string is None:
        return None
    try:
        input_string = input_string.replace("'", "\"")  # Replace single quotes with double quotes for valid JSON
        data = json.loads(input_string)
        return data
    except json.JSONDecodeError:
        print(input_string)
        print("错误：非法的 Json 格式")
        return None
