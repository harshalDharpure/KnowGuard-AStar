#!/usr/bin/env bash
# Resume smoke-50 then run full ioMEDQA N=1272 (lite stack, auto-pick freest GPU).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source "$ROOT/.venv/bin/activate"
export PYTHONUNBUFFERED=1
if [[ -f "$ROOT/.env" ]]; then set -a; source "$ROOT/.env"; set +a; fi

pick_freest_gpu() {
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | sort -t',' -k2 -nr | head -1 | cut -d',' -f1 | tr -d ' '
}

GPU="${GPU:-$(pick_freest_gpu)}"
export CUDA_VISIBLE_DEVICES="$GPU"
PY="$ROOT/.venv/bin/python"
TAG="${TAG:-astar_lite_nim}"
CAMPAIGN_LOG="logs/astar_lite_resume_full.log"

run_split() {
  local N="$1"
  local DEV="ioMEDQA.jsonl"
  local SUFFIX=""
  if [[ "$N" != "1272" ]]; then
    DEV="ioMEDQA_smoke${N}.jsonl"
    SUFFIX="_smoke${N}"
    [[ -f "data/interactive/$DEV" ]] || "$PY" -c "
import json; from pathlib import Path
rows=[json.loads(l) for l in Path('data/interactive/ioMEDQA.jsonl').open()][:int('$N')]
Path('data/interactive/$DEV').write_text(''.join(json.dumps(r)+'\n' for r in rows))
"
  fi
  local OUT="results/KnowGuardExpert_${TAG}_ioMEDQA${SUFFIX}.jsonl"
  local LOG="logs/KnowGuardExpert_${TAG}_ioMEDQA${SUFFIX}.log"
  local MET="results/KnowGuardExpert_${TAG}_ioMEDQA${SUFFIX}_metrics.json"
  local TARGET="$N"
  [[ "$N" == "1272" ]] && TARGET=1272

  # Skip if already complete
  if [[ -f "$OUT" ]]; then
    DONE=$("$PY" -c "print(sum(1 for _ in open('$OUT')))")
    if [[ "$DONE" -ge "$TARGET" ]]; then
      echo "[$(date)] === SKIP N=$N already $DONE/$TARGET on GPU=$GPU ===" | tee -a "$CAMPAIGN_LOG"
      return 0
    fi
    echo "[$(date)] === RESUME N=$N at $DONE/$TARGET GPU=$GPU ===" | tee -a "$CAMPAIGN_LOG"
  else
    echo "[$(date)] === START N=$N GPU=$GPU dev=$DEV ===" | tee -a "$CAMPAIGN_LOG"
  fi

  if [[ -n "${NVIDIA_API_KEY:-}${NGC_API_KEY:-}" ]]; then
    MODEL="${MODEL:-nvidia/nemotron-3-ultra-550b-a55b}"
    USE_API=nvidia
  elif [[ -n "${OPENROUTER_API_KEY:-}" ]]; then
    MODEL="${MODEL:-nvidia/nemotron-3-super-120b-a12b:free}"
    USE_API=openrouter
  else
    MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
    USE_API=""
  fi

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

  "$PY" Open_benchmark.py \
    --dev_filename "$DEV" \
    --output_filename "$OUT" \
    --log_filename "$LOG" \
    "${COMMON[@]}"

  "$PY" scripts/compute_metrics.py "$OUT" --output "$MET"
  echo "[$(date)] === DONE N=$N GPU=$GPU ===" | tee -a "$CAMPAIGN_LOG"
  cat "$MET" | tee -a "$CAMPAIGN_LOG"
}

echo "[$(date)] Campaign using GPU=$GPU (freest)" | tee -a "$CAMPAIGN_LOG"

# Resume smoke-50 then full benchmark
run_split 50
run_split 1272

echo "[$(date)] === RESUME+FULL CAMPAIGN COMPLETE ===" | tee -a "$CAMPAIGN_LOG"
