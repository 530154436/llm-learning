#!/usr/bin/env python3
# -*- coding:utf-8 -*--
import re

document=r"""
2023年12月版权声明\n本白皮书版权属于中国信息通信研究院，并受法律保\n护。转载、摘编或利用其它方式使用本白皮书文字或者观\n点的，应注明“来源：中国信息通信研究院”。违反上述声明\n者，本院将追究其相关法律责任。前 言\n今年我国经济呈现波浪式发展、曲
"""

del_pattern = '版权声明.*?追究其相关法律责任。'
text = re.sub(del_pattern, '', document, re.MULTILINE | re.DOTALL)
print(text)