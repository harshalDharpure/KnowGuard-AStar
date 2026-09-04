#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source "$ROOT/.venv/bin/activate"
export PYTHONUNBUFFERED=1
if [[ -f "$ROOT/.env" ]]; then set -a; source "$ROOT/.env"; set +a; fi

# Strict provider: NIM only for this campaign.
unset OPENROUTER_API_KEY
unset OPENAI_API_KEY
export USE_API=nvidia

PY="$ROOT/.venv/bin/python"
TAG="astar_lite_nim"
LOG="logs/full_resume_nim_only.log"
mkdir -p logs results data/interactive/shards

pick_gpus() {
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | sort -t',' -k2 -nr | head -n 4 | cut -d',' -f1 | tr -d ' '
}

mapfile -t GPUS < <(pick_gpus)
echo "[$(date)] GPUs: ${GPUS[*]}" | tee -a "$LOG"

if [[ -z "${NVIDIA_API_KEY:-}${NGC_API_KEY:-}" ]]; then
  echo "[$(date)] ERROR: NVIDIA_API_KEY/NGC_API_KEY missing" | tee -a "$LOG"
  exit 1
fi

# Build shards if missing
if [[ ! -f data/interactive/shards/ioMEDQA_shard0.jsonl ]]; then
  "$PY" - << 'PY'
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

pids=()
for i in 0 1 2 3; do
  gpu="${GPUS[$i]}"
  out="results/KnowGuardExpert_${TAG}_ioMEDQA_shard${i}.jsonl"
  slog="logs/KnowGuardExpert_${TAG}_ioMEDQA_shard${i}.log"
  count=0
  [[ -f "$out" ]] && count=$(wc -l < "$out")
  echo "[$(date)] shard$i gpu=$gpu resume_from=$count" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" Open_benchmark.py \
    --dev_filename "shards/ioMEDQA_shard${i}.jsonl" \
    --output_filename "$out" \
    --log_filename "$slog" \
    "${COMMON[@]}" \
    >> "$LOG" 2>&1 &
  pids+=($!)
done

echo "[$(date)] spawned pids: ${pids[*]}" | tee -a "$LOG"

fail=0
for p in "${pids[@]}"; do
  wait "$p" || fail=1
done

"$PY" - << 'PY'
import json
from pathlib import Path
parts=sorted(Path('results').glob('KnowGuardExpert_astar_lite_nim_ioMEDQA_shard*.jsonl'))
rows=[]
for p in parts:
    if p.exists():
        rows.extend(json.loads(l) for l in p.open() if l.strip())
out=Path('results/KnowGuardExpert_astar_lite_nim_ioMEDQA_merged.jsonl')
out.write_text(''.join(json.dumps(r)+'\n' for r in rows))
print('merged_n', len(rows))
PY

"$PY" scripts/compute_metrics.py results/KnowGuardExpert_astar_lite_nim_ioMEDQA_merged.jsonl \
  --output results/KnowGuardExpert_astar_lite_nim_ioMEDQA_merged_metrics.json

echo "[$(date)] complete fail=$fail" | tee -a "$LOG"
cat results/KnowGuardExpert_astar_lite_nim_ioMEDQA_merged_metrics.json | tee -a "$LOG"
exit $fail
