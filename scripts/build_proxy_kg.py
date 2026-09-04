#!/usr/bin/env python3
"""Build a free proxy medical knowledge graph from MedQA contexts."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict

import pandas as pd


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def extract_entities(sentence: str) -> tuple[str, str, str]:
    lower = sentence.lower()
    disease_candidates = re.findall(r"\b([a-z][a-z\- ]{2,40})\b", lower)
    if len(disease_candidates) >= 2:
        source = disease_candidates[0][:60]
        target = disease_candidates[-1][:60]
    else:
        source = "patient symptom"
        target = "clinical finding"
    relation = "associated_with"
    if "history" in lower:
        relation = "has_history_of"
    elif "pain" in lower:
        relation = "presents_with"
    elif "treat" in lower or "therapy" in lower:
        relation = "treated_with"
    return source, relation, target


def build_from_medqa(max_rows: int = 500) -> pd.DataFrame:
    from datasets import load_dataset

    ds = load_dataset("GBaker/MedQA-USMLE-4-options-hf", split="validation")
    rows = []
    seen = set()

    for i, sample in enumerate(ds):
        if i >= max_rows:
            break
        context = sample.get("sent1") or sample.get("context") or ""
        answer = sample.get("answer") or ""
        for sentence in split_sentences(context):
            if len(sentence) < 20:
                continue
            src, rel, tgt = extract_entities(sentence)
            key = (sentence, src, rel, tgt)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "relevant_description": sentence,
                    "x_name": src,
                    "y_name": rel,
                    "relationship": tgt,
                    "image_path": "",
                }
            )
        if answer:
            rows.append(
                {
                    "relevant_description": f"{answer} is a possible diagnosis in this clinical scenario.",
                    "x_name": "clinical scenario",
                    "y_name": "may_indicate",
                    "relationship": answer[:80],
                    "image_path": "",
                }
            )

    return pd.DataFrame(rows)


def build_disease2demo(df: pd.DataFrame) -> pd.DataFrame:
    diseases = sorted({str(x) for x in df["relationship"].tolist() if x})[:50]
    demographics = ["Adults", "Elderly", "Pregnant woman", "people with HIV"]
    rows = []
    for disease in diseases:
        for demo in demographics[:2]:
            rows.append(
                {
                    "demographic": demo,
                    "disease": disease,
                    "pdf_name": "{proxy_guideline.pdf}",
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/kg")
    parser.add_argument("--max_rows", type=int, default=500)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "baseline_dataset"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "WHO"), exist_ok=True)

    df = build_from_medqa(max_rows=args.max_rows)
    csv_path = os.path.join(args.output_dir, "filtered_data_v1.csv")
    df.to_csv(csv_path, index=False)

    demo_df = build_disease2demo(df)
    demo_path = os.path.join(args.output_dir, "baseline_dataset", "Disease2demo.csv")
    demo_df.to_csv(demo_path, index=False)

    overview = [{"name": "proxy_guideline.pdf", "png_dir_path": "WHO/proxy/"}]
    with open(os.path.join(args.output_dir, "WHO", "overview.json"), "w", encoding="utf-8") as f:
        json.dump(overview, f, indent=2)

    print(f"Wrote {len(df)} triplets to {csv_path}")
    print(f"Wrote demographics map to {demo_path}")


if __name__ == "__main__":
    main()
