#!/usr/bin/env python3
# -*- coding:utf-8 -*--
# Panel 是一个开源的 Python 库，让你能够轻松地完全使用 Python 构建强大的工具、仪表盘和复杂的应用程序。
# GitHub：https://github.com/holoviz/panel
# 文档：https://panel.holoviz.org/getting_started/build_app.html
import functools
import panel as pn
from history import collect_messages
pn.extension()

# 全局变量
panels = []
context = [{'role': 'system', 'content': "You are Service Assistant"}]
inp = pn.widgets.TextInput(placeholder='Enter text here…')

# 页面
button_conversation = pn.widgets.Button(name="Service Assistant")
interactive_conversation = pn.bind(functools.partial(collect_messages, inp, context, panels),
                                   button_conversation)
dashboard = pn.Column(
    inp,
    pn.Row(button_conversation),
    pn.panel(interactive_conversation, loading_indicator=True, height=400, width=600),
)


if __name__ == "__main__":
    pn.serve(dashboard, port=5000)
