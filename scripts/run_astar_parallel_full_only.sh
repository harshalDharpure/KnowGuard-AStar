#!/usr/bin/env bash
# Full ioMEDQA N=1272 — 4-GPU parallel NIM lite A* (no smoke gate)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source "$ROOT/.venv/bin/activate"
export PYTHONUNBUFFERED=1
if [[ -f "$ROOT/.env" ]]; then set -a; source "$ROOT/.env"; set +a; fi

PY="$ROOT/.venv/bin/python"
TAG="${TAG:-astar_lite_nim}"
NUM_SHARDS="${NUM_SHARDS:-4}"
CAMPAIGN_LOG="logs/astar_parallel_full_only.log"

pick_gpus() {
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | sort -t',' -k2 -nr | head -n "$NUM_SHARDS" | cut -d',' -f1 | tr -d ' ' | paste -sd, -
}

IFS=',' read -ra GPUS <<< "$(pick_gpus)"
echo "[$(date)] Full-only parallel GPUs: ${GPUS[*]}" | tee -a "$CAMPAIGN_LOG"

if [[ -n "${NVIDIA_API_KEY:-}${NGC_API_KEY:-}" ]]; then
  MODEL="${MODEL:-nvidia/nemotron-3-ultra-550b-a55b}"
  USE_API=nvidia
elif [[ -n "${OPENROUTER_API_KEY:-}" ]]; then
  MODEL="${MODEL:-nvidia/nemotron-3-super-120b-a12b:free}"
  USE_API=openrouter
else
  echo "No API key" >&2; exit 1
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

mkdir -p data/interactive/shards
"$PY" - <<'PY'
import json
from pathlib import Path
rows = [json.loads(l) for l in Path("data/interactive/ioMEDQA.jsonl").open()]
n = len(rows)
for i in range(4):
    part = rows[i * n // 4 : (i + 1) * n // 4]
    p = Path(f"data/interactive/shards/ioMEDQA_shard{i}.jsonl")
    p.write_text("".join(json.dumps(r) + "\n" for r in part))
    print(f"shard{i}: {len(part)} cases -> {p}")
PY

PIDS=()
for i in "${!GPUS[@]}"; do
  OUT="results/KnowGuardExpert_${TAG}_ioMEDQA_shard${i}.jsonl"
  LOG="logs/KnowGuardExpert_${TAG}_ioMEDQA_shard${i}.log"
  echo "[$(date)] Shard $i on GPU=${GPUS[$i]} START" | tee -a "$CAMPAIGN_LOG"
  CUDA_VISIBLE_DEVICES="${GPUS[$i]}" "$PY" Open_benchmark.py \
    --dev_filename "shards/ioMEDQA_shard${i}.jsonl" \
    --output_filename "$OUT" \
    --log_filename "$LOG" \
    "${COMMON[@]}" &
  PIDS+=($!)
done

echo "[$(date)] Waiting for shards: ${PIDS[*]}" | tee -a "$CAMPAIGN_LOG"
FAIL=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || FAIL=1
done

MERGED="results/KnowGuardExpert_${TAG}_ioMEDQA_merged.jsonl"
MET="results/KnowGuardExpert_${TAG}_ioMEDQA_merged_metrics.json"
"$PY" - <<PY
import json
from pathlib import Path
parts = sorted(Path("results").glob("KnowGuardExpert_${TAG}_ioMEDQA_shard*.jsonl"))
rows = []
for p in parts:
    for line in p.open():
        if line.strip():
            rows.append(json.loads(line))
Path("$MERGED").write_text("".join(json.dumps(r)+"\n" for r in rows))
print(f"Merged {len(rows)} cases")
PY
"$PY" scripts/compute_metrics.py "$MERGED" --output "$MET"
echo "[$(date)] FAIL=$FAIL MERGED METRICS:" | tee -a "$CAMPAIGN_LOG"
cat "$MET" | tee -a "$CAMPAIGN_LOG"
echo "[$(date)] === FULL N=1272 PARALLEL COMPLETE ===" | tee -a "$CAMPAIGN_LOG"
