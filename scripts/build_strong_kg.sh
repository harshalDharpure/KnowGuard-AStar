#!/usr/bin/env bash
# Build-only: merge PrimeKG+Hetionet and (re)build FAISS. Does NOT run experiments.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source "$ROOT/.venv/bin/activate"
export PYTHONUNBUFFERED=1

MAX_EDGES="${MAX_EDGES:-400000}"
PER_CAP="${PER_SOURCE_CAP:-250000}"
LOG="${LOG:-logs/build_strong_kg.log}"
mkdir -p logs data/kg/raw data/kg/faiss_db_combined

echo "[$(date)] Building combined PrimeKG+Hetionet CSV..."
python scripts/build_primekg_hetionet.py --max_edges "$MAX_EDGES" --per_source_cap "$PER_CAP"

cp -f data/kg/baseline_dataset/Disease2demo_combined.csv data/kg/baseline_dataset/Disease2demo.csv
cp -f data/kg/WHO/overview_combined.json data/kg/WHO/overview.json

echo "[$(date)] Building FAISS at data/kg/faiss_db_combined (this can take hours for 400k edges)..."
python scripts/init_faiss.py \
  --kg_csv data/kg/combined_primekg_hetionet.csv \
  --faiss_dir data/kg/faiss_db_combined \
  --disease2demo_csv data/kg/baseline_dataset/Disease2demo.csv \
  --who_overview_json data/kg/WHO/overview.json

echo "[$(date)] BUILD_COMPLETE — KG+FAISS ready. Experiments NOT started."
echo "To run evals later: GPU=3 nohup bash scripts/run_strong_kg_evals.sh >> logs/nohup_strong_kg.log 2>&1 &"
