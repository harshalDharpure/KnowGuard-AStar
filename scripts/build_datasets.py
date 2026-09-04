#!/usr/bin/env python3
"""Build ioMEDQA-style interactive JSONL from public MedQA data."""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any


def split_atomic_facts(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    facts = [p.strip() for p in parts if p.strip()]
    return facts if facts else [text]


def infer_initial_info(context: str) -> str:
    sentences = split_atomic_facts(context)
    if not sentences:
        return context[:200]
    return sentences[0]


def medqa_row_to_sample(row: dict[str, Any], idx: int) -> dict[str, Any]:
    context = row.get("sent1") or row.get("context") or row.get("question") or ""
    question = row.get("sent2") or "What is the most likely diagnosis or correct answer?"
    options = row.get("options") or {}

    if not options and row.get("ending0") is not None:
        options = {
            "A": row.get("ending0", ""),
            "B": row.get("ending1", ""),
            "C": row.get("ending2", ""),
            "D": row.get("ending3", ""),
        }

    answer_idx = row.get("answer_idx") or row.get("cop")
    if answer_idx is None and row.get("label") is not None:
        answer_idx = chr(ord("A") + int(row["label"]))

    answer = row.get("answer") or (options.get(answer_idx) if answer_idx else "")

    if isinstance(options, list):
        letters = "ABCDE"
        options = {letters[i]: opt for i, opt in enumerate(options) if i < len(letters)}

    facts = split_atomic_facts(context)
    initial_info = infer_initial_info(context)

    return {
        "id": str(row.get("id", idx)),
        "question": question,
        "options": options,
        "answer": answer,
        "answer_idx": answer_idx,
        "context": context,
        "initial_info": initial_info,
        "atomic_facts": facts,
        "question_type": "mcq",
        "answer_rationale": answer,
    }


def load_medqa(limit: int | None = None) -> list[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset("GBaker/MedQA-USMLE-4-options-hf", split="validation")
    rows = []
    for i, row in enumerate(ds):
        rows.append(medqa_row_to_sample(dict(row), i))
        if limit and len(rows) >= limit:
            break
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/interactive")
    parser.add_argument("--full_size", type=int, default=0, help="0 = all validation rows")
    parser.add_argument("--smoke_size", type=int, default=10)
    args = parser.parse_args()

    limit = args.full_size if args.full_size > 0 else None
    rows = load_medqa(limit=limit)
    write_jsonl(rows, os.path.join(args.output_dir, "ioMEDQA.jsonl"))

    smoke = rows[: args.smoke_size]
    write_jsonl(smoke, os.path.join(args.output_dir, "ioMEDQA_smoke10.jsonl"))
    write_jsonl(smoke[:5], os.path.join(args.output_dir, "ioMEDQA_smoke5.jsonl"))
    write_jsonl(smoke[:1], os.path.join(args.output_dir, "ioMEDQA_smoke1.jsonl"))

    print(f"Wrote {len(rows)} ioMEDQA rows and smoke splits to {args.output_dir}")


if __name__ == "__main__":
    main()
