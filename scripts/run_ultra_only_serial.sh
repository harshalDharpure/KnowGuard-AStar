#!/usr/bin/env bash
# Nemotron Ultra ONLY — one shard worker at a time (avoids NIM 429).
# Fresh output tag: astar_ultra_only (does not mix Gemini/OpenRouter cases).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source "$ROOT/.venv/bin/activate"
export PYTHONUNBUFFERED=1
if [[ -f "$ROOT/.env" ]]; then set -a; source "$ROOT/.env"; set +a; fi
# Keep keys present-but-empty so helper._load_dotenv_if_present cannot re-inject them.
export OPENROUTER_API_KEY=
export OPENAI_API_KEY=
export GOOGLE_API_KEY=
export GEMINI_API_KEY=
export USE_API=nvidia
export KNOWGUARD_NIM_MIN_INTERVAL="${KNOWGUARD_NIM_MIN_INTERVAL:-2.5}"

PY="$ROOT/.venv/bin/python"
TAG="astar_ultra_only"
MODEL="nvidia/nemotron-3-ultra-550b-a55b"
LOG="logs/ultra_only_serial.log"
mkdir -p logs results data/interactive/shards
echo "[$(date)] Ultra-only serial supervisor start model=$MODEL" | tee -a "$LOG"

if [[ -z "${NVIDIA_API_KEY:-}${NGC_API_KEY:-}" ]]; then
  echo "[$(date)] ERROR: NVIDIA_API_KEY/NGC_API_KEY missing" | tee -a "$LOG"
  exit 1
fi

if [[ ! -f data/interactive/shards/ioMEDQA_shard0.jsonl ]]; then
  "$PY" - <<'PY'
import json
from pathlib import Path
rows=[json.loads(l) for l in Path('data/interactive/ioMEDQA.jsonl').open()]
n=len(rows)
for i in range(4):
    part=rows[i*n//4:(i+1)*n//4]
    p=Path(f'data/interactive/shards/ioMEDQA_shard{i}.jsonl')
    p.write_text(''.join(json.dumps(r)+'\n' for r in part))
    print(i, len(part), p)
PY
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
  --use_api nvidia
)

pick_gpu() {
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | sort -t',' -k2 -nr | head -n 1 | cut -d',' -f1 | tr -d ' '
}

shard_count() {
  local out="results/KnowGuardExpert_${TAG}_ioMEDQA_shard$1.jsonl"
  if [[ -f "$out" ]]; then wc -l < "$out"; else echo 0; fi
}
shard_done() { [[ "$(shard_count "$1")" -ge 318 ]]; }
shard_running() {
  pgrep -f "Open_benchmark.py --dev_filename shards/ioMEDQA_shard$1.jsonl" >/dev/null
}

start_shard() {
  local i=$1
  local gpu out slog count
  gpu=$(pick_gpu)
  out="results/KnowGuardExpert_${TAG}_ioMEDQA_shard${i}.jsonl"
  slog="logs/KnowGuardExpert_${TAG}_ioMEDQA_shard${i}.log"
  count=$(shard_count "$i")
  echo "[$(date)] start shard$i Ultra-only gpu=$gpu resume_from=$count" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" Open_benchmark.py \
    --dev_filename "shards/ioMEDQA_shard${i}.jsonl" \
    --output_filename "$out" \
    --log_filename "$slog" \
    "${COMMON[@]}" \
    >> "$LOG" 2>&1 &
}

ORDER=(2 0 1 3)  # finish densest Ultra history first if any overlap; else sequential

while true; do
  unfinished=()
  for i in "${ORDER[@]}"; do
    shard_done "$i" || unfinished+=("$i")
  done
  if [[ ${#unfinished[@]} -eq 0 ]]; then
    echo "[$(date)] all Ultra-only shards complete" | tee -a "$LOG"
    break
  fi

  # Exactly ONE worker at a time
  any=0
  for i in 0 1 2 3; do
    if shard_running "$i"; then any=1; break; fi
  done
  if [[ "$any" -eq 0 ]]; then
    start_shard "${unfinished[0]}"
  fi
  sleep 60
done

"$PY" - <<PY
import json
from pathlib import Path
parts=sorted(Path('results').glob('KnowGuardExpert_${TAG}_ioMEDQA_shard*.jsonl'))
rows=[]
for p in parts:
    rows.extend(json.loads(l) for l in p.open() if l.strip())
out=Path('results/KnowGuardExpert_${TAG}_ioMEDQA_merged.jsonl')
out.write_text(''.join(json.dumps(r)+'\n' for r in rows))
print('merged_n', len(rows))
PY

"$PY" scripts/compute_metrics.py "results/KnowGuardExpert_${TAG}_ioMEDQA_merged.jsonl" \
  --output "results/KnowGuardExpert_${TAG}_ioMEDQA_merged_metrics.json"
echo "[$(date)] complete" | tee -a "$LOG"
cat "results/KnowGuardExpert_${TAG}_ioMEDQA_merged_metrics.json" | tee -a "$LOG"
