#!/usr/bin/env bash
# KnowGuard eval stack: caching, simulator fix, discriminative Q, entropy gate,
# dual-process, phase routing, multi-model council.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source "$ROOT/.venv/bin/activate"
export PYTHONUNBUFFERED=1
if [[ -f "$ROOT/.env" ]]; then set -a; source "$ROOT/.env"; set +a; fi

GPU="${GPU:-1}"
SMOKE_N="${SMOKE_N:-20}"
PY="$ROOT/.venv/bin/python"

if [[ -n "${NVIDIA_API_KEY:-}${NGC_API_KEY:-}" ]]; then
  MODEL="${MODEL:-nvidia/nemotron-3-ultra-550b-a55b}"
  USE_API=nvidia
  TAG="${TAG:-astar_nim}"
elif [[ -n "${OPENROUTER_API_KEY:-}" ]]; then
  MODEL="${MODEL:-nvidia/nemotron-3-super-120b-a12b:free}"
  USE_API=openrouter
  TAG="${TAG:-astar_or}"
else
  MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
  USE_API=""
  TAG="${TAG:-astar_local}"
fi

export CUDA_VISIBLE_DEVICES="$GPU"
SMOKE_FILE="data/interactive/ioMEDQA_smoke${SMOKE_N}.jsonl"
[[ -f "$SMOKE_FILE" ]] || "$PY" -c "
import json; from pathlib import Path
rows=[json.loads(l) for l in Path('data/interactive/ioMEDQA.jsonl').open()][:int('$SMOKE_N')]
Path('$SMOKE_FILE').write_text(''.join(json.dumps(r)+'\n' for r in rows))
"

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
  --self_consistency 3
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
  --use_dual_process
  --use_council
  --phase_routing
)
[[ -n "$USE_API" ]] && COMMON+=(--use_api "$USE_API")

OUT="results/KnowGuardExpert_${TAG}_ioMEDQA_smoke${SMOKE_N}.jsonl"
LOG="logs/KnowGuardExpert_${TAG}_ioMEDQA_smoke${SMOKE_N}.log"
echo "[$(date)] Eval stack: MODEL=$MODEL USE_API=${USE_API:-local} OUT=$OUT"
"$PY" Open_benchmark.py \
  --dev_filename "$(basename "$SMOKE_FILE")" \
  --output_filename "$OUT" \
  --log_filename "$LOG" \
  "${COMMON[@]}"
"$PY" scripts/compute_metrics.py "$OUT" --output "results/KnowGuardExpert_${TAG}_ioMEDQA_smoke${SMOKE_N}_metrics.json"
cat "results/KnowGuardExpert_${TAG}_ioMEDQA_smoke${SMOKE_N}_metrics.json"
