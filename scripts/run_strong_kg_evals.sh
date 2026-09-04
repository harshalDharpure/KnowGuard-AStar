#!/usr/bin/env bash
# Full ioMEDQA evals with combined PrimeKG+Hetionet KG.
# DOES NOT auto-start unless you invoke this script.
# Usage:
#   GPU=3 nohup bash scripts/run_strong_kg_evals.sh >> logs/nohup_strong_kg.log 2>&1 &
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source "$ROOT/.venv/bin/activate"

GPU="${GPU:-}"
MIN_FREE_MB="${MIN_FREE_MB:-20000}"
PY="$ROOT/.venv/bin/python"
LOG_DIR="$ROOT/logs"
RESULT_DIR="$ROOT/results"
TS="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="$LOG_DIR/strong_kg_${TS}.log"
mkdir -p "$LOG_DIR" "$RESULT_DIR"

pick_gpu() {
  if [[ -n "$GPU" ]]; then
    echo "$GPU"
    return
  fi
  "$PY" - <<'PY'
import subprocess
min_free = int(__import__("os").environ.get("MIN_FREE_MB", "20000"))
out = subprocess.check_output([
    "nvidia-smi",
    "--query-gpu=index,memory.free",
    "--format=csv,noheader,nounits",
], text=True)
best = None
best_free = -1
for line in out.strip().splitlines():
    idx, free = [x.strip() for x in line.split(",")]
    free = int(free)
    if free > best_free:
        best_free = free
        best = idx
if best is None or best_free < min_free:
    raise SystemExit(f"No GPU with >= {min_free} MiB free (best={best_free})")
print(best)
PY
}

KG_CSV="data/kg/combined_primekg_hetionet.csv"
FAISS_DIR="data/kg/faiss_db_combined"
DEMO_CSV="data/kg/baseline_dataset/Disease2demo.csv"
WHO_JSON="data/kg/WHO/overview.json"

if [[ ! -f "$KG_CSV" ]]; then
  echo "Missing $KG_CSV — run scripts/build_primekg_hetionet.py first"
  exit 1
fi
if [[ ! -f "$FAISS_DIR/index.faiss" ]]; then
  echo "Missing FAISS at $FAISS_DIR — run scripts/init_faiss.py first"
  exit 1
fi

METHODS=("KnowGuardExpert" "OpenEndedScaleExpert")
# Short tag -> HF id
MODELS=(
  "mistral7b|mistralai/Mistral-7B-Instruct-v0.3"
  "medgemma4b|google/medgemma-1.5-4b-it"
  "llama31_8b|meta-llama/Llama-3.1-8B-Instruct"
  "llavamed7b|microsoft/llava-med-v1.5-mistral-7b"
)

exec > >(tee -a "$MASTER_LOG") 2>&1
echo "=========================================="
echo "Strong KG evals start: $(date)"
echo "Master log: $MASTER_LOG"
echo "=========================================="

GPU_ID="$(pick_gpu)"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
echo "Using GPU $GPU_ID (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"

COMMON_ARGS=(
  --question_type open-ended
  --patient_class FactSelectPatient
  --data_dir data/interactive
  --dev_filename ioMEDQA.jsonl
  --max_questions 12
  --max_tokens 256
  --self_consistency 1
  --abstain_threshold 3.5
  --kg_threshold 3.5
  --know_mode text_only
  --max_queue_size 6
  --initial_triplets 2
  --max_hop_depth 2
  --beam_size 3
  --llm_relevance_threshold 0.1
  --kg_csv "$KG_CSV"
  --disease2demo_csv "$DEMO_CSV"
  --faiss_dir "$FAISS_DIR"
  --embedding_model sentence-transformers/all-MiniLM-L6-v2
  --who_overview_json "$WHO_JSON"
)

for entry in "${MODELS[@]}"; do
  tag="${entry%%|*}"
  model="${entry#*|}"
  echo ""
  echo "[$(date)] === Probing model $model ==="
  if ! "$PY" - <<PY
from transformers import AutoTokenizer
try:
    AutoTokenizer.from_pretrained("$model")
    print("OK")
except Exception as e:
    print("SKIP:", e)
    raise SystemExit(2)
PY
  then
    echo "[$(date)] SKIP model $model (download/auth failed)"
    continue
  fi

  for method in "${METHODS[@]}"; do
    out="$RESULT_DIR/${method}_${tag}_ioMEDQA_full.jsonl"
    run_log="$LOG_DIR/${method}_${tag}_ioMEDQA_full.log"
    echo "[$(date)] === $method | $tag | $model ==="
    if [[ "$method" == "KnowGuardExpert" ]]; then
      "$PY" scripts/init_faiss.py \
        --kg_csv "$KG_CSV" \
        --faiss_dir "$FAISS_DIR" \
        --disease2demo_csv "$DEMO_CSV" \
        --who_overview_json "$WHO_JSON" >> "$run_log" 2>&1 || true
    fi
    "$PY" Open_benchmark.py \
      --expert_class "$method" \
      --output_filename "$out" \
      --log_filename "$run_log" \
      --expert_model "$model" \
      --patient_model "$model" \
      --judge_model "$model" \
      "${COMMON_ARGS[@]}" 2>&1 | tee -a "$run_log"
    "$PY" scripts/compute_metrics.py "$out" \
      --output "$RESULT_DIR/${method}_${tag}_ioMEDQA_full_metrics.json"
  done
done

"$PY" scripts/build_final_report.py \
  --results_dir "$RESULT_DIR" \
  --output "$RESULT_DIR/STRONG_KG_REPORT.md"

echo "=========================================="
echo "Strong KG evals done: $(date)"
echo "Report: $RESULT_DIR/STRONG_KG_REPORT.md"
echo "=========================================="
