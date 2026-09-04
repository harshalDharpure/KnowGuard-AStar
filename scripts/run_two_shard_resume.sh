#!/usr/bin/env bash
# Keep at most 2 NIM shards alive to reduce 429s. Resume-safe via JSONL skip.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source "$ROOT/.venv/bin/activate"
export PYTHONUNBUFFERED=1
if [[ -f "$ROOT/.env" ]]; then set -a; source "$ROOT/.env"; set +a; fi
unset OPENROUTER_API_KEY OPENAI_API_KEY
export USE_API=nvidia

PY="$ROOT/.venv/bin/python"
TAG="astar_lite_nim"
LOG="logs/two_shard_resume.log"
mkdir -p logs results
echo "[$(date)] two-shard supervisor start" | tee -a "$LOG"

COMMON=(
  --expert_class KnowGuardExpert
  --question_type open-ended
  --patient_class FactSelectPatient
  --data_dir data/interactive
  --expert_model nvidia/nemotron-3-ultra-550b-a55b
  --patient_model nvidia/nemotron-3-ultra-550b-a55b
  --judge_model nvidia/nemotron-3-ultra-550b-a55b
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
  --use_api nvidia
)

pick_gpu() {
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | sort -t',' -k2 -nr | head -n 1 | cut -d',' -f1 | tr -d ' '
}

shard_done() {
  local i=$1 out="results/KnowGuardExpert_${TAG}_ioMEDQA_shard${i}.jsonl"
  local n=0
  [[ -f "$out" ]] && n=$(wc -l < "$out")
  [[ "$n" -ge 318 ]]
}

shard_running() {
  pgrep -f "Open_benchmark.py --dev_filename shards/ioMEDQA_shard$1.jsonl" >/dev/null
}

start_shard() {
  local i=$1
  local gpu
  gpu=$(pick_gpu)
  local out="results/KnowGuardExpert_${TAG}_ioMEDQA_shard${i}.jsonl"
  local slog="logs/KnowGuardExpert_${TAG}_ioMEDQA_shard${i}.log"
  local count=0
  [[ -f "$out" ]] && count=$(wc -l < "$out")
  echo "[$(date)] start shard$i gpu=$gpu resume_from=$count" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" Open_benchmark.py \
    --dev_filename "shards/ioMEDQA_shard${i}.jsonl" \
    --output_filename "$out" \
    --log_filename "$slog" \
    "${COMMON[@]}" \
    >> "$LOG" 2>&1 &
  # Stagger API burst after spawn
  sleep 30
}

# Prefer finishing denser shards first, then remaining.
ORDER=(2 3 1 0)

while true; do
  unfinished=()
  for i in "${ORDER[@]}"; do
    if ! shard_done "$i"; then
      unfinished+=("$i")
    fi
  done
  if [[ ${#unfinished[@]} -eq 0 ]]; then
    echo "[$(date)] all shards complete" | tee -a "$LOG"
    break
  fi

  # Cap concurrency at 2
  running=0
  for i in 0 1 2 3; do
    if shard_running "$i"; then
      running=$((running + 1))
    fi
  done

  for i in "${unfinished[@]}"; do
    if [[ $running -ge 2 ]]; then
      break
    fi
    if ! shard_running "$i" && ! shard_done "$i"; then
      start_shard "$i"
      running=$((running + 1))
    fi
  done

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
