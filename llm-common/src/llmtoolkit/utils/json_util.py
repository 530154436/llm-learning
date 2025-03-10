#!/usr/bin/env python3
# -*- coding:utf-8 -*--
import os
from langchain.output_parsers import OutputFixingParser
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser, JsonOutputKeyToolsParser, PydanticOutputParser
from pydantic import BaseModel
api_key = os.environ.get("CHAT_ANYWHERE_API_KEY")  # ~/.zshrc
base_url = "https://api.chatanywhere.com.cn/v1"


class Patent(BaseModel):
    patent_name: str
    patent_brief: str| None = None


parser = PydanticOutputParser(pydantic_object=Patent)
patent_json = r'''{"patent_name": "阀门布置", "patent_brief": "在柱塞处设置有密封材料的密封部分（19）。”}'''
new_parser = OutputFixingParser.from_llm(
    parser=parser,
    llm=ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=api_key, base_url=base_url)
)
result = new_parser.parse(patent_json)
print(result)