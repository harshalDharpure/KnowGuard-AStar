#!/usr/bin/env bash
# Free + fast: NIM lite A* stack, 4-GPU parallel shards for full ioMEDQA.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source "$ROOT/.venv/bin/activate"
export PYTHONUNBUFFERED=1
if [[ -f "$ROOT/.env" ]]; then set -a; source "$ROOT/.env"; set +a; fi

PY="$ROOT/.venv/bin/python"
TAG="${TAG:-astar_lite_nim}"
NUM_SHARDS="${NUM_SHARDS:-4}"
CAMPAIGN_LOG="logs/astar_parallel_campaign.log"
SHARD_DIR="data/interactive/shards"

pick_gpus() {
  # Return N freest GPU indices (comma-separated)
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | sort -t',' -k2 -nr | head -n "$NUM_SHARDS" | cut -d',' -f1 | tr -d ' ' | paste -sd, -
}

IFS=',' read -ra GPUS <<< "$(pick_gpus)"
echo "[$(date)] Parallel campaign GPUs: ${GPUS[*]}" | tee -a "$CAMPAIGN_LOG"

if [[ -n "${NVIDIA_API_KEY:-}${NGC_API_KEY:-}" ]]; then
  MODEL="${MODEL:-nvidia/nemotron-3-ultra-550b-a55b}"
  USE_API=nvidia
elif [[ -n "${OPENROUTER_API_KEY:-}" ]]; then
  MODEL="${MODEL:-nvidia/nemotron-3-super-120b-a12b:free}"
  USE_API=openrouter
else
  echo "No API key; set NVIDIA_API_KEY or OPENROUTER_API_KEY" >&2
  exit 1
fi

build_common_args() {
  COMMON=(
    --expert_class KnowGuardExpert
    --question_type open-ended
    --patient_class FactSelectPatient
    --data_dir data/interactive
    --expert_model "$MODEL"
    --patient_model "$MODEL"
    --judge_model "$MODEL"
    --max_questions 12
    --max_tokens 512
    --self_consistency 1
    --min_questions 3
    --kg_threshold 4.0
    --abstain_threshold 4.0
    --rationale_generation
    --know_mode text_only
    --max_queue_size 10
    --initial_triplets 4
    --max_hop_depth 2
    --beam_size 3
    --use_question_query
    --llm_relevance_threshold 0.1
    --kg_csv data/kg/combined_primekg_hetionet.csv
    --disease2demo_csv data/kg/baseline_dataset/Disease2demo.csv
    --faiss_dir data/kg/faiss_db_combined
    --embedding_model sentence-transformers/all-MiniLM-L6-v2
    --who_overview_json data/kg/WHO/overview.json
    --use_clinical_rag
    --clinical_corpus_dir data/kg/clinical_corpus
    --clinical_rag_top_k 5
    --clinical_rag_rerank
    --use_adjudicator
    --use_discriminative_questions
    --use_entropy_gate
    --entropy_commit_threshold 0.9
    --no_dual_process
    --no_council
    --phase_routing
  )
  [[ -n "$USE_API" ]] && COMMON+=(--use_api "$USE_API")
}

finish_smoke50() {
  local OUT="results/KnowGuardExpert_${TAG}_ioMEDQA_smoke50.jsonl"
  local TARGET=50
  local DONE=0
  [[ -f "$OUT" ]] && DONE=$("$PY" -c "print(sum(1 for _ in open('$OUT')))")
  if [[ "$DONE" -ge "$TARGET" ]]; then
    echo "[$(date)] Smoke-50 complete ($DONE/$TARGET)" | tee -a "$CAMPAIGN_LOG"
    return 0
  fi
  echo "[$(date)] Finishing smoke-50 ($DONE/$TARGET) on GPU=${GPUS[0]}" | tee -a "$CAMPAIGN_LOG"
  build_common_args
  CUDA_VISIBLE_DEVICES="${GPUS[0]}" "$PY" Open_benchmark.py \
    --dev_filename ioMEDQA_smoke50.jsonl \
    --output_filename "$OUT" \
    --log_filename "logs/KnowGuardExpert_${TAG}_ioMEDQA_smoke50.log" \
    "${COMMON[@]}"
  "$PY" scripts/compute_metrics.py "$OUT" \
    --output "results/KnowGuardExpert_${TAG}_ioMEDQA_smoke50_metrics.json"
  cat "results/KnowGuardExpert_${TAG}_ioMEDQA_smoke50_metrics.json" | tee -a "$CAMPAIGN_LOG"
}

split_shards() {
  mkdir -p "$SHARD_DIR"
  "$PY" << 'PY'
import json
from pathlib import Path
rows = [json.loads(l) for l in Path("data/interactive/ioMEDQA.jsonl").open()]
n = len(rows)
shards = 4
for i in range(shards):
    part = rows[i * n // shards : (i + 1) * n // shards]
    p = Path(f"data/interactive/shards/ioMEDQA_shard{i}.jsonl")
    p.write_text("".join(json.dumps(r) + "\n" for r in part))
    print(f"shard{i}: {len(part)} cases -> {p}")
PY
}

run_shard() {
  local IDX="$1"
  local GPU="$2"
  build_common_args
  local DEV="shards/ioMEDQA_shard${IDX}.jsonl"
  local OUT="results/KnowGuardExpert_${TAG}_ioMEDQA_shard${IDX}.jsonl"
  local LOG="logs/KnowGuardExpert_${TAG}_ioMEDQA_shard${IDX}.log"
  echo "[$(date)] Shard $IDX on GPU=$GPU" | tee -a "$CAMPAIGN_LOG"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" Open_benchmark.py \
    --dev_filename "$DEV" \
    --output_filename "$OUT" \
    --log_filename "$LOG" \
    "${COMMON[@]}"
}

merge_shards() {
  local MERGED="results/KnowGuardExpert_${TAG}_ioMEDQA_merged.jsonl"
  local MET="results/KnowGuardExpert_${TAG}_ioMEDQA_merged_metrics.json"
  "$PY" << PY
import json
from pathlib import Path
out = Path("$MERGED")
parts = sorted(Path("results").glob("KnowGuardExpert_${TAG}_ioMEDQA_shard*.jsonl"))
rows = []
for p in parts:
    for line in p.open():
        if line.strip():
            rows.append(json.loads(line))
out.write_text("".join(json.dumps(r) + "\n" for r in rows))
print(f"Merged {len(rows)} cases -> {out}")
PY
  "$PY" scripts/compute_metrics.py "$MERGED" --output "$MET"
  echo "[$(date)] === MERGED METRICS ===" | tee -a "$CAMPAIGN_LOG"
  cat "$MET" | tee -a "$CAMPAIGN_LOG"
}

finish_smoke50
split_shards

PIDS=()
for i in "${!GPUS[@]}"; do
  run_shard "$i" "${GPUS[$i]}" &
  PIDS+=($!)
done

echo "[$(date)] Waiting for ${#PIDS[@]} shards: ${PIDS[*]}" | tee -a "$CAMPAIGN_LOG"
FAIL=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || FAIL=1
done
if [[ "$FAIL" -ne 0 ]]; then
  echo "[$(date)] WARNING: some shards failed; merging completed shards anyway" | tee -a "$CAMPAIGN_LOG"
fi
merge_shards
echo "[$(date)] === PARALLEL CAMPAIGN COMPLETE ===" | tee -a "$CAMPAIGN_LOG"
