#!/usr/bin/env python3
# -*- coding:utf-8 -*--
import panel as pn
from typing import List
from panel.widgets import TextInput
from message_processor import process_message


def collect_messages(inp: TextInput, context: List[dict],
                     panels: List[pn.Row], debug=False):
    """
    用于收集用户的输入并生成助手的回答
    """
    user_input = inp.value_input
    if debug:
        print(f"User Input = {user_input}")
    if user_input == "":
        return
    inp.value = ''
    # 调用 process_user_message 函数
    response, context = process_message(user_input, context, debug=debug)
    context.append({'role': 'assistant', 'content': f"{response}"})
    panels.append(pn.Row('User:', pn.pane.Markdown(user_input, width=600)))
    panels.append(pn.Row('Assistant:', pn.pane.Markdown(response, width=600)))

    return pn.Column(*panels)  # 包含了所有的对话信息
