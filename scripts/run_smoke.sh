#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
DATA="${DATA:-data/interactive/ioMEDQA_smoke5.jsonl}"
EXPERT="${EXPERT:-KnowGuardExpert}"

python scripts/build_datasets.py --smoke_size 10
python scripts/build_proxy_kg.py --max_rows 300

python - <<'PY'
import sys
sys.path.insert(0, ".")
from args import get_args
from know_storage import initialize_db
import argparse

class A:
    pass

a = A()
a.use_api = None
a.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
a.expert_model = "Qwen/Qwen2.5-1.5B-Instruct"
a.kg_csv = "data/kg/filtered_data_v1.csv"
a.disease2demo_csv = "data/kg/baseline_dataset/Disease2demo.csv"
a.faiss_dir = "data/kg/faiss_db_minilm"
initialize_db(a)
print("FAISS index ready")
PY

OUT="results/${EXPERT}_smoke.jsonl"
rm -f "$OUT"

python Open_benchmark.py \
  --expert_class "$EXPERT" \
  --patient_class FactSelectPatient \
  --data_dir data/interactive \
  --dev_filename "$(basename "$DATA")" \
  --output_filename "$OUT" \
  --question_type open-ended \
  --expert_model "$MODEL" \
  --patient_model "$MODEL" \
  --judge_model "$MODEL" \
  --max_questions 12 \
  --max_tokens 256 \
  --self_consistency 1 \
  --abstain_threshold 3.5 \
  --kg_threshold 3.5 \
  --know_mode text_only \
  --max_queue_size 6 \
  --kg_csv data/kg/filtered_data_v1.csv \
  --disease2demo_csv data/kg/baseline_dataset/Disease2demo.csv \
  --faiss_dir data/kg/faiss_db_minilm \
  --embedding_model sentence-transformers/all-MiniLM-L6-v2 \
  --log_filename logs/smoke_run.log

python scripts/compute_metrics.py "$OUT" --output results/smoke_metrics.json
