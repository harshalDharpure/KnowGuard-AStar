#!/usr/bin/env python3
"""Initialize FAISS knowledge store from CLI args or defaults."""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--kg_csv", default="data/kg/filtered_data_v1.csv")
    parser.add_argument("--disease2demo_csv", default="data/kg/baseline_dataset/Disease2demo.csv")
    parser.add_argument("--faiss_dir", default="data/kg/faiss_db_minilm")
    parser.add_argument("--who_overview_json", default="data/kg/WHO/overview.json")
    args = parser.parse_args()

    sys.path.insert(0, ".")

    class InitArgs:
        use_api = None
        embedding_model = args.embedding_model
        expert_model = "Qwen/Qwen2.5-1.5B-Instruct"
        kg_csv = args.kg_csv
        disease2demo_csv = args.disease2demo_csv
        faiss_dir = args.faiss_dir
        who_overview_json = args.who_overview_json

    from know_storage import initialize_db

    initialize_db(InitArgs())
    print("FAISS ready")


if __name__ == "__main__":
    main()
