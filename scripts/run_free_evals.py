#!/usr/bin/env python3
"""Run KnowGuard and baseline smoke evaluations locally for free."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)


def run_cmd(cmd: list[str], env: dict | None = None) -> None:
    print("$", " ".join(cmd))
    merged = os.environ.copy()
    if env:
        merged.update(env)
    subprocess.run(cmd, check=True, env=merged)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--data", default="data/interactive/ioMEDQA_smoke5.jsonl")
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--methods", nargs="+", default=["KnowGuardExpert", "OpenEndedScaleExpert", "OpenEndedNumericalCutOffExpert"])
    args = parser.parse_args()

    py = str(ROOT / ".venv" / "bin" / "python")
    env = {"CUDA_VISIBLE_DEVICES": args.gpu}

    run_cmd([py, "scripts/build_datasets.py", "--smoke_size", "5"], env)
    run_cmd([py, "scripts/build_proxy_kg.py", "--max_rows", "200"], env)

    run_cmd([py, "scripts/init_faiss.py"], env)

    summary = {}
    for expert in args.methods:
        out = f"results/{expert}_smoke.jsonl"
        if Path(out).exists():
            Path(out).unlink()

        cmd = [
            py,
            "Open_benchmark.py",
            "--expert_class",
            expert,
            "--patient_class",
            "FactSelectPatient",
            "--data_dir",
            "data/interactive",
            "--dev_filename",
            Path(args.data).name,
            "--output_filename",
            out,
            "--question_type",
            "open-ended",
            "--expert_model",
            args.model,
            "--patient_model",
            args.model,
            "--judge_model",
            args.model,
            "--max_questions",
            "3",
            "--max_tokens",
            "256",
            "--self_consistency",
            "1",
            "--abstain_threshold",
            "3.5",
            "--kg_threshold",
            "3.5",
            "--know_mode",
            "text_only",
            "--max_queue_size",
            "6",
            "--kg_csv",
            "data/kg/filtered_data_v1.csv",
            "--disease2demo_csv",
            "data/kg/baseline_dataset/Disease2demo.csv",
            "--faiss_dir",
            "data/kg/faiss_db_minilm",
            "--embedding_model",
            "sentence-transformers/all-MiniLM-L6-v2",
            "--who_overview_json",
            "data/kg/WHO/overview.json",
            "--log_filename",
            f"logs/{expert}_smoke.log",
        ]
        run_cmd(cmd, env)

        metrics_path = f"results/{expert}_metrics.json"
        run_cmd([py, "scripts/compute_metrics.py", out, "--output", metrics_path], env)
        with open(metrics_path, "r", encoding="utf-8") as f:
            summary[expert] = json.load(f)

    paper_ref = {
        "KnowGuardExpert": {"acc": 0.7098, "turns": 5.41, "dataset": "ioMEDQA (paper GPT-4)"},
        "OpenEndedScaleExpert": {"acc": 0.6423, "turns": 5.15, "dataset": "ioMEDQA (paper GPT-4)"},
        "OpenEndedNumericalCutOffExpert": {"acc": 0.6174, "turns": 2.51, "dataset": "ioMEDQA (paper GPT-4)"},
    }

    report = {
        "free_stack": {
            "model": args.model,
            "data": args.data,
            "gpu": args.gpu,
            "note": "Local free replication; not comparable to paper GPT-4 numbers.",
        },
        "local_results": summary,
        "paper_reference": paper_ref,
    }

    report_path = ROOT / "results" / "FREE_REPLICATION_REPORT.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    md_path = ROOT / "results" / "FREE_REPLICATION_REPORT.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Free KnowGuard Replication Report\n\n")
        f.write("This run uses local open-source models and a proxy KG built from MedQA contexts.\n")
        f.write("It does **not** reproduce paper GPT-4 numbers exactly.\n\n")
        f.write("| Method | Local ACC | Local Turns | Paper ACC (ref) | Paper Turns (ref) |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for expert, metrics in summary.items():
            ref = paper_ref.get(expert, {})
            f.write(
                f"| {expert} | {metrics.get('acc', 0):.3f} | {metrics.get('turns', 0):.2f} | "
                f"{ref.get('acc', 'NA')} | {ref.get('turns', 'NA')} |\n"
            )

    print(f"Wrote {report_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
