#!/usr/bin/env bash
# Run all free KnowGuard experiments on GPU in background.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

GPU="${GPU:-1}"
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
PY="$ROOT/.venv/bin/python"
LOG_DIR="$ROOT/logs"
RESULT_DIR="$ROOT/results"
TS="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="$LOG_DIR/all_experiments_${TS}.log"

mkdir -p "$LOG_DIR" "$RESULT_DIR"

exec > >(tee -a "$MASTER_LOG") 2>&1

echo "=========================================="
echo "KnowGuard free experiments started: $(date)"
echo "GPU=$GPU MODEL=$MODEL"
echo "Master log: $MASTER_LOG"
echo "=========================================="

source "$ROOT/.venv/bin/activate"
export CUDA_VISIBLE_DEVICES="$GPU"

echo "[$(date)] Building datasets..."
"$PY" scripts/build_datasets.py --smoke_size 10

echo "[$(date)] Building proxy KG..."
"$PY" scripts/build_proxy_kg.py --max_rows 300

echo "[$(date)] Initializing FAISS..."
"$PY" scripts/init_faiss.py

METHODS=(
  "KnowGuardExpert"
  "OpenEndedScaleExpert"
  "OpenEndedNumericalCutOffExpert"
)

DATASETS=(
  "ioMEDQA_smoke5.jsonl"
  "ioMEDQA_smoke10.jsonl"
)

COMMON_ARGS=(
  --question_type open-ended
  --patient_class FactSelectPatient
  --data_dir data/interactive
  --expert_model "$MODEL"
  --patient_model "$MODEL"
  --judge_model "$MODEL"
  --max_questions 3
  --max_tokens 256
  --self_consistency 1
  --abstain_threshold 3.5
  --kg_threshold 3.5
  --know_mode text_only
  --max_queue_size 6
  --initial_triplets 1
  --max_hop_depth 1
  --beam_size 2
  --llm_relevance_threshold 0.1
  --kg_csv data/kg/filtered_data_v1.csv
  --disease2demo_csv data/kg/baseline_dataset/Disease2demo.csv
  --faiss_dir data/kg/faiss_db_minilm
  --embedding_model sentence-transformers/all-MiniLM-L6-v2
  --who_overview_json data/kg/WHO/overview.json
)

SUMMARY_JSON="$RESULT_DIR/ALL_EXPERIMENTS_SUMMARY.json"
echo "{}" > "$SUMMARY_JSON"

for dataset in "${DATASETS[@]}"; do
  tag="${dataset%.jsonl}"
  for method in "${METHODS[@]}"; do
    out="$RESULT_DIR/${method}_${tag}.jsonl"
    metrics="$RESULT_DIR/${method}_${tag}_metrics.json"
    run_log="$LOG_DIR/${method}_${tag}.log"

    echo ""
    echo "[$(date)] Running $method on $dataset ..."
    rm -f "$out"

    if [[ "$method" == "KnowGuardExpert" ]]; then
      "$PY" scripts/init_faiss.py >> "$run_log" 2>&1 || true
    fi

    "$PY" Open_benchmark.py \
      --expert_class "$method" \
      --dev_filename "$dataset" \
      --output_filename "$out" \
      --log_filename "$run_log" \
      "${COMMON_ARGS[@]}" >> "$run_log" 2>&1

    "$PY" scripts/compute_metrics.py "$out" --output "$metrics" >> "$run_log" 2>&1
    echo "[$(date)] Finished $method on $dataset -> $(cat "$metrics")"
  done
done

echo "[$(date)] Building combined report..."
"$PY" scripts/build_final_report.py --results_dir "$RESULT_DIR" --output "$RESULT_DIR/FREE_REPLICATION_REPORT.md"

echo "=========================================="
echo "All experiments completed: $(date)"
echo "Report: $RESULT_DIR/FREE_REPLICATION_REPORT.md"
echo "=========================================="
