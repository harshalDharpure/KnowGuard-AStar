#!/usr/bin/env bash
# Workaround NIM 429: run at most 1 Ultra (NIM) + 1 Gemini shard in parallel.
# Same JSONL outputs → resume-safe. Mixed providers → document in results.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source "$ROOT/.venv/bin/activate"
export PYTHONUNBUFFERED=1
if [[ -f "$ROOT/.env" ]]; then set -a; source "$ROOT/.env"; set +a; fi
unset OPENROUTER_API_KEY OPENAI_API_KEY
export KNOWGUARD_NIM_MIN_INTERVAL="${KNOWGUARD_NIM_MIN_INTERVAL:-3.0}"

PY="$ROOT/.venv/bin/python"
TAG="astar_lite_nim"
LOG="logs/hybrid_nim_gemini_resume.log"
mkdir -p logs results
echo "[$(date)] hybrid NIM+Gemini supervisor start nim_gap=$KNOWGUARD_NIM_MIN_INTERVAL" | tee -a "$LOG"

NIM_MODEL="nvidia/nemotron-3-ultra-550b-a55b"
GEM_MODEL="${GEMINI_EVAL_MODEL:-gemini-flash-latest}"

COMMON_BASE=(
  --expert_class KnowGuardExpert
  --question_type open-ended
  --patient_class FactSelectPatient
  --data_dir data/interactive
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

pick_gpu() {
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | sort -t',' -k2 -nr | head -n 1 | cut -d',' -f1 | tr -d ' '
}

shard_count() {
  local out="results/KnowGuardExpert_${TAG}_ioMEDQA_shard$1.jsonl"
  if [[ -f "$out" ]]; then wc -l < "$out"; else echo 0; fi
}

shard_done() {
  [[ "$(shard_count "$1")" -ge 318 ]]
}

shard_running() {
  pgrep -f "Open_benchmark.py --dev_filename shards/ioMEDQA_shard$1.jsonl" >/dev/null
}

provider_running() {
  # $1 = nvidia|google
  pgrep -af "Open_benchmark.py --dev_filename shards/ioMEDQA_shard" \
    | grep -v grep | grep -q -- "--use_api $1"
}

start_shard() {
  local i=$1 provider=$2
  local gpu model use_api
  gpu=$(pick_gpu)
  if [[ "$provider" == "nvidia" ]]; then
    model="$NIM_MODEL"; use_api=nvidia
  else
    model="$GEM_MODEL"; use_api=google
  fi
  local out="results/KnowGuardExpert_${TAG}_ioMEDQA_shard${i}.jsonl"
  local slog="logs/KnowGuardExpert_${TAG}_ioMEDQA_shard${i}.log"
  local count
  count=$(shard_count "$i")
  echo "[$(date)] start shard$i provider=$provider model=$model gpu=$gpu resume_from=$count" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" Open_benchmark.py \
    --dev_filename "shards/ioMEDQA_shard${i}.jsonl" \
    --output_filename "$out" \
    --log_filename "$slog" \
    --expert_model "$model" \
    --patient_model "$model" \
    --judge_model "$model" \
    --use_api "$use_api" \
    "${COMMON_BASE[@]}" \
    >> "$LOG" 2>&1 &
  sleep 20
}

# Prefer densest unfinished first.
ORDER=(2 3 1 0)

if [[ -z "${NVIDIA_API_KEY:-}${NGC_API_KEY:-}" ]]; then
  echo "[$(date)] ERROR: NVIDIA key missing" | tee -a "$LOG"; exit 1
fi
if [[ -z "${GOOGLE_API_KEY:-}${GEMINI_API_KEY:-}" ]]; then
  echo "[$(date)] ERROR: Gemini key missing" | tee -a "$LOG"; exit 1
fi

while true; do
  unfinished=()
  for i in "${ORDER[@]}"; do
    shard_done "$i" || unfinished+=("$i")
  done
  if [[ ${#unfinished[@]} -eq 0 ]]; then
    echo "[$(date)] all shards complete" | tee -a "$LOG"
    break
  fi

  # Ensure one NIM slot if any unfinished and NIM not running
  if ! provider_running nvidia; then
    for i in "${unfinished[@]}"; do
      if ! shard_running "$i"; then
        start_shard "$i" nvidia
        break
      fi
    done
  fi

  # Ensure one Gemini slot on a different unfinished shard
  if ! provider_running google; then
    for i in "${unfinished[@]}"; do
      if ! shard_running "$i"; then
        start_shard "$i" google
        break
      fi
    done
  fi

  sleep 60
done

"$PY" - <<'PY'
import json
from pathlib import Path
parts=sorted(Path('results').glob('KnowGuardExpert_astar_lite_nim_ioMEDQA_shard*.jsonl'))
rows=[]
for p in parts:
    rows.extend(json.loads(l) for l in p.open() if l.strip())
Path('results/KnowGuardExpert_astar_lite_nim_ioMEDQA_merged.jsonl').write_text(
    ''.join(json.dumps(r)+'\n' for r in rows)
)
print('merged_n', len(rows))
PY
"$PY" scripts/compute_metrics.py results/KnowGuardExpert_astar_lite_nim_ioMEDQA_merged.jsonl \
  --output results/KnowGuardExpert_astar_lite_nim_ioMEDQA_merged_metrics.json
echo "[$(date)] complete" | tee -a "$LOG"
cat results/KnowGuardExpert_astar_lite_nim_ioMEDQA_merged_metrics.json | tee -a "$LOG"
