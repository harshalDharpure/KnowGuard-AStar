#!/usr/bin/env python3
"""Aggregate all experiment metrics into a final markdown report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PAPER_REF = {
    "KnowGuardExpert": {"acc": 0.7098, "turns": 5.41},
    "OpenEndedScaleExpert": {"acc": 0.6423, "turns": 5.15},
    "OpenEndedNumericalCutOffExpert": {"acc": 0.6174, "turns": 2.51},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--output", default="results/FREE_REPLICATION_REPORT.md")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    rows = []
    for path in sorted(results_dir.glob("*_metrics.json")):
        with open(path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        stem = path.name.replace("_metrics.json", "")
        method = stem.rsplit("_", 1)[0] if "_" in stem else stem
        dataset = stem.rsplit("_", 1)[-1] if "_" in stem else "unknown"
        ref = PAPER_REF.get(method, {})
        rows.append(
            {
                "method": method,
                "dataset": dataset,
                "acc": metrics.get("acc"),
                "turns": metrics.get("turns"),
                "ece": metrics.get("ece"),
                "brier": metrics.get("brier"),
                "n": metrics.get("n"),
                "paper_acc": ref.get("acc"),
                "paper_turns": ref.get("turns"),
            }
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("# Free KnowGuard Replication Report\n\n")
        f.write("Local open-source stack on GPU. Paper numbers are GPT-4 reference only.\n\n")
        f.write("| Method | Dataset | Local ACC | Local Turns | Paper ACC | Paper Turns | N |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|\n")
        for row in rows:
            acc = f"{row['acc']:.3f}" if row["acc"] is not None else "NA"
            turns = f"{row['turns']:.2f}" if row["turns"] is not None else "NA"
            f.write(
                f"| {row['method']} | {row['dataset']} | "
                f"{acc} | {turns} | "
                f"{row['paper_acc'] or 'NA'} | {row['paper_turns'] or 'NA'} | {row['n']} |\n"
            )

    json_out = out.with_suffix(".json")
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print(f"Wrote {out}")
    print(f"Wrote {json_out}")


if __name__ == "__main__":
    main()
