#!/usr/bin/env python3
"""Compute ACC, avg turns, ECE, and Brier from benchmark JSONL outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compute_metrics(rows: list[dict]) -> dict:
    if not rows:
        return {"acc": 0.0, "turns": 0.0, "ece": None, "brier": None, "n": 0}

    correct = []
    turns = []
    confidences = []
    labels = []

    for row in rows:
        system = row["interactive_system"]
        info = row["info"]
        is_correct = system.get("closest_option") == info.get("correct_answer_idx")
        correct.append(1.0 if is_correct else 0.0)
        turns.append(system.get("num_questions", 0))

        conf = None
        for extra in row["interactive_system"].get("temp_additional_info", []):
            if isinstance(extra, dict) and "confidence" in extra:
                conf = extra["confidence"]
                break
        if conf is not None:
            if conf > 1.0:
                conf = conf / 5.0
            confidences.append(float(conf))
            labels.append(1.0 if is_correct else 0.0)

    acc = float(np.mean(correct))
    avg_turns = float(np.mean(turns))

    ece = None
    brier = None
    if confidences:
        conf_arr = np.array(confidences)
        label_arr = np.array(labels)
        brier = float(np.mean((conf_arr - label_arr) ** 2))
        bins = np.linspace(0.0, 1.0, 11)
        ece_val = 0.0
        for i in range(len(bins) - 1):
            mask = (conf_arr >= bins[i]) & (conf_arr < bins[i + 1])
            if mask.any():
                bin_acc = label_arr[mask].mean()
                bin_conf = conf_arr[mask].mean()
                ece_val += mask.mean() * abs(bin_acc - bin_conf)
        ece = float(ece_val)

    return {
        "acc": acc,
        "turns": avg_turns,
        "ece": ece,
        "brier": brier,
        "n": len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_jsonl")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    rows = load_jsonl(args.results_jsonl)
    metrics = compute_metrics(rows)
    metrics["file"] = args.results_jsonl

    print(json.dumps(metrics, indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
