#!/usr/bin/env bash
# Full free pipeline on actual ioMEDQA (1272 MedQA validation cases).
# Resumable: Open_benchmark skips already-processed IDs in output JSONL.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

GPU="${GPU:-1}"
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
PY="$ROOT/.venv/bin/python"
LOG_DIR="$ROOT/logs"
RESULT_DIR="$ROOT/results"
TS="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="$LOG_DIR/full_ioMEDQA_${TS}.log"

mkdir -p "$LOG_DIR" "$RESULT_DIR"
exec > >(tee -a "$MASTER_LOG") 2>&1

echo "=========================================="
echo "Full ioMEDQA free pipeline: $(date)"
echo "GPU=$GPU MODEL=$MODEL DATA=1272 cases"
echo "Master log: $MASTER_LOG"
echo "=========================================="

source "$ROOT/.venv/bin/activate"
export CUDA_VISIBLE_DEVICES="$GPU"

"$PY" scripts/build_datasets.py --smoke_size 10
"$PY" scripts/build_proxy_kg.py --max_rows 500
"$PY" scripts/init_faiss.py

METHODS=(
  "KnowGuardExpert"
  "OpenEndedScaleExpert"
  "OpenEndedNumericalCutOffExpert"
)

COMMON_ARGS=(
  --question_type open-ended
  --patient_class FactSelectPatient
  --data_dir data/interactive
  --dev_filename ioMEDQA.jsonl
  --expert_model "$MODEL"
  --patient_model "$MODEL"
  --judge_model "$MODEL"
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
  --kg_csv data/kg/filtered_data_v1.csv
  --disease2demo_csv data/kg/baseline_dataset/Disease2demo.csv
  --faiss_dir data/kg/faiss_db_minilm
  --embedding_model sentence-transformers/all-MiniLM-L6-v2
  --who_overview_json data/kg/WHO/overview.json
)

for method in "${METHODS[@]}"; do
  out="$RESULT_DIR/${method}_ioMEDQA_full.jsonl"
  run_log="$LOG_DIR/${method}_ioMEDQA_full.log"

  echo ""
  echo "[$(date)] === $method on full ioMEDQA (1272 cases) ==="

  if [[ "$method" == "KnowGuardExpert" ]]; then
    "$PY" scripts/init_faiss.py >> "$run_log" 2>&1 || true
  fi

  "$PY" Open_benchmark.py \
    --expert_class "$method" \
    --output_filename "$out" \
    --log_filename "$run_log" \
    "${COMMON_ARGS[@]}" 2>&1 | tee -a "$run_log"

  "$PY" scripts/compute_metrics.py "$out" \
    --output "$RESULT_DIR/${method}_ioMEDQA_full_metrics.json"

  echo "[$(date)] Finished $method"
  cat "$RESULT_DIR/${method}_ioMEDQA_full_metrics.json"
done

"$PY" scripts/build_final_report.py \
  --results_dir "$RESULT_DIR" \
  --output "$RESULT_DIR/FULL_ioMEDQA_REPORT.md"

echo "=========================================="
echo "Full ioMEDQA pipeline done: $(date)"
echo "Report: $RESULT_DIR/FULL_ioMEDQA_REPORT.md"
echo "=========================================="
