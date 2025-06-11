#!/usr/bin/env python
# -*- coding:utf-8 -*-
import json


def convert_clue_ner_to_prompt1(file: str) -> list:
    prompt_template = "".join(open("prompts/clue_prompt.txt", encoding="utf-8").readlines())

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
            "instruction": "你是一个文本实体识别领域的专家，擅长从自然语言中提取不同类别的实体名称。",
            "input": prompt_template.format(text=text),
            "output": json.dumps(entities, ensure_ascii=False)
        }
        alpaca_data.append(alpaca_item)
    return alpaca_data


if __name__ == "__main__":
    _alpaca_data = convert_clue_ner_to_prompt1('dataset/clue.train.jsonl')
    # 写出结果文件
    print("number of data: ", len(_alpaca_data))
    save_name = "dataset/alpaca/alpaca_clue_train.json"
    with open(save_name, 'w', encoding='utf-8') as f:
        json.dump(_alpaca_data, f, indent=2, ensure_ascii=False)
