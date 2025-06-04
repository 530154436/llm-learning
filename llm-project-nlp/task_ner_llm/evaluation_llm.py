#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2025/5/24 9:45
# @function:
import json
from pathlib import Path
import pandas as pd
from llm_util.llm_api import ChatClient
from llm_util.resolver import convert_json_array_in_text_to_list
from task_ner_llm.data.clue_convert_to_alpaca import convert_clue_ner_to_prompt1
from collections import defaultdict


def batch_evaluate(y_true_list, y_pred_list):
    # 初始化全局统计
    global_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    # 遍历每个样本
    for y_true, y_pred in zip(y_true_list, y_pred_list):
        truth = {(e["label"], e["text"]) for e in y_true}
        preds = {(e["label"], e["text"]) for e in y_pred}

        tp = preds & truth
        fp = preds - truth
        fn = truth - preds

        # 更新每个 label 的 TP/FP/FN
        for label, text in tp:
            global_stats[label]["tp"] += 1
        for label, text in fp:
            global_stats[label]["fp"] += 1
        for label, text in fn:
            global_stats[label]["fn"] += 1

    # 计算每个 label 的 P/R/F1
    results = {}

    for label in global_stats:
        tp_n = global_stats[label]["tp"]
        fp_n = global_stats[label]["fp"]
        fn_n = global_stats[label]["fn"]

        precision = tp_n / (tp_n + fp_n) if tp_n + fp_n else 0
        recall = tp_n / (tp_n + fn_n) if tp_n + fn_n else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

        results[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4)
        }

    # 宏平均 Macro-F1
    macro_f1 = round(sum(results[l]["f1"] for l in results) / len(results), 4) if results else 0
    results["macro"] = {"f1": macro_f1}

    # 微平均 Micro-F1
    total_tp = sum(global_stats[label]["tp"] for label in global_stats)
    total_fp = sum(global_stats[label]["fp"] for label in global_stats)
    total_fn = sum(global_stats[label]["fn"] for label in global_stats)

    micro_precision = round(total_tp / (total_tp + total_fp), 4) if (total_tp + total_fp) > 0 else 0
    micro_recall = round(total_tp / (total_tp + total_fn), 4) if (total_tp + total_fn) > 0 else 0
    micro_f1 = round(2 * micro_precision * micro_recall / (micro_precision + micro_recall), 4) \
        if (micro_precision + micro_recall) > 0 else 0

    results["micro"] = {
        "precision": micro_precision,
        "recall": micro_recall,
        "f1": micro_f1
    }

    return results


def metrics_to_dataframe(metrics) -> pd.DataFrame:
    rows = []
    for label in metrics:
        if label in ["macro", "micro"]:
            continue
        row = {
            "Label": label,
            "Precision": metrics[label]["precision"],
            "Recall": metrics[label]["recall"],
            "F1": metrics[label]["f1"]
        }
        rows.append(row)

    # 添加宏平均和微平均
    rows.append({
        "Label": "Macro Avg.",
        "Precision": "-",
        "Recall": "-",
        "F1": metrics["macro"]["f1"]
    })
    rows.append({
        "Label": "Micro Avg.",
        "Precision": metrics["micro"]["precision"],
        "Recall": metrics["micro"]["recall"],
        "F1": metrics["micro"]["f1"]
    })

    df = pd.DataFrame(rows)
    return df


def evaluate_llm(base_url: str, api_key: str, model: str):
    save_file = Path(f'data/outputs/eval_{model}.jsonl')
    data = []
    if save_file.exists():
        with open(save_file, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
    else:
        chat_client = ChatClient(stream=False, base_url=base_url, api_key=api_key, model=model,
                                 timeout=60, temperature=0.01, max_tokens=512, top_p=None,
                                 frequency_penalty=None, presence_penalty=None)
        writer = save_file.open('w', encoding='utf-8')
        for i, line in enumerate(convert_clue_ner_to_prompt1('data/dataset/clue.dev.jsonl'), start=1):
            prompt = line['instruction'] + "\n\n<输入>\n" + line['input']
            y_true = json.loads(line['output'])
            y_pred = convert_json_array_in_text_to_list(chat_client.completion(prompt).get("content"))
            item = {'input': line['input'], "y_true": y_true, "y_pred": y_pred}
            print(i)
            print(line['input'])
            print(y_true)
            print(y_pred)
            print()
            writer.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 评测
    y_trues = [i.get("y_true") for i in data]
    y_preds = [i.get("y_pred") for i in data]
    metrics = batch_evaluate(y_trues, y_preds)
    df = metrics_to_dataframe(metrics)
    print(df.to_string(index=False))


if __name__ == '__main__':
    # Qwen2.5-7B-Instruct-ner-lora-sft
    evaluate_llm(base_url="http://172.19.190.6:31833/v1", api_key="<KEY>", model="clue-ner-lora-sft")
