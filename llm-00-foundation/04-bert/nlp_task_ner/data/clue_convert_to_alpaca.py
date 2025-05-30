#!/usr/bin/env python
# -*- coding:utf-8 -*-
import json

INSTRUCTION_1 = \
    """你是一个文本实体识别领域的专家，请从给定的句子中识别并提取出以下指定类别的实体。

<实体类别集合>
name, organization, scene, company, movie, book, government, position, address, game

<任务说明>
1. 仅提取属于上述类别的实体，忽略其他类型的实体。
2. 以json格式输出，对于每个识别出的实体，请提供：
   - label: 实体类型，必须严格使用原始类型标识（不可更改）
   - text: 实体在原文中的中文内容

<输出格式要求>
```json
[{{"label": "实体类别", "text": "实体名称"}}]
```

<输入>
"""


def convert_clue_ner_to_prompt1(file: str):
    with open(file, 'r', encoding='utf-8') as f:
        lines = [json.loads(line.strip()) for line in f]

    alpaca_data = []
    for item in lines:
        text = item['text']
        labels = item['label']

        entities = []
        for ent_type, ents in labels.items():
            for name in ents.keys():
                entities.append({"label": ent_type, "text": name})

        alpaca_item = {
            "instruction": INSTRUCTION_1,
            "input": text,
            "output": json.dumps(entities, ensure_ascii=False)
        }
        alpaca_data.append(alpaca_item)

    # 写出结果文件
    print("number of train: ", len(alpaca_data))
    save_name = "dataset/" + "alpaca_" + '_'.join(file.split(".")[:-1]) + ".json"
    with open(save_name, 'w', encoding='utf-8') as f:
        json.dump(alpaca_data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    convert_clue_ner_to_prompt1('clue.train.jsonl')
