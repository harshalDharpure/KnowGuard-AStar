#!/usr/bin/env bash
# Path-to-80%: prefer free OpenAI-compatible frontier APIs
# (Kilo / OpenRouter / NVIDIA NIM), then paid OpenAI, else local Llama.
# Provider catalog: https://github.com/mnfst/awesome-free-llm-apis
# Note: that repo lists signup links — it does NOT embed API keys.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
export PYTHONUNBUFFERED=1

# Optional local secrets (gitignored)
if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

GPU="${GPU:-1}"
SMOKE_N="${SMOKE_N:-50}"
SMOKE_THRESHOLD="${SMOKE_THRESHOLD:-0.65}"
PY="$ROOT/.venv/bin/python"
LOG_DIR="$ROOT/logs"
RESULT_DIR="$ROOT/results"
mkdir -p "$LOG_DIR" "$RESULT_DIR" data/interactive

TS="$(date +%Y%m%d_%H%M%S)"
MASTER="$LOG_DIR/path_to_80_${TS}.log"
exec > >(tee -a "$MASTER") 2>&1

echo "=========================================="
echo "Path-to-80% KnowGuard: $(date)"
echo "GPU=$GPU (RAG/embeddings only when local models used)"
echo "=========================================="

# Ensure clinical corpus
if [[ ! -f data/kg/clinical_corpus/index.faiss ]]; then
  echo "[$(date)] Building clinical MedQA corpus..."
  "$PY" scripts/build_clinical_corpus.py --max_rows 4000
fi

# Ensure combined KG FAISS exists (fallback to minilm proxy)
KG_CSV="data/kg/combined_primekg_hetionet.csv"
FAISS_DIR="data/kg/faiss_db_combined"
if [[ ! -f "$FAISS_DIR/index.faiss" ]]; then
  if [[ -f data/kg/faiss_db_minilm/index.faiss ]]; then
    FAISS_DIR="data/kg/faiss_db_minilm"
    KG_CSV="data/kg/filtered_data_v1.csv"
    echo "WARN: combined FAISS missing; using proxy KG"
  else
    echo "ERROR: no FAISS KG available"
    exit 1
  fi
fi

# Smoke split
SMOKE_FILE="data/interactive/ioMEDQA_smoke${SMOKE_N}.jsonl"
if [[ ! -f "$SMOKE_FILE" ]]; then
  "$PY" - <<PY
import json
from pathlib import Path
src=Path("data/interactive/ioMEDQA.jsonl")
rows=[json.loads(l) for l in src.open()][:int("$SMOKE_N")]
out=Path("$SMOKE_FILE")
with out.open("w") as f:
    for r in rows:
        f.write(json.dumps(r)+"\n")
print("wrote", out, "n=", len(rows))
PY
fi

# Provider priority (free first):
# 1) Explicit MODEL/USE_API env
# 2) OpenRouter key → free :free models
# 3) NVIDIA NIM key → nemotron / llama-3.3-70b
# 4) Kilo Code free pool (NO KEY) → nemotron-3-super-120b:free
# 5) Paid OpenAI gpt-4o
# 6) Local Llama
USE_API_FLAG=()
API_LABEL="none"
if [[ -n "${MODEL:-}" && -n "${USE_API:-}" ]]; then
  MODEL="$MODEL"
  USE_API_FLAG=(--use_api "$USE_API")
  TAG="${TAG:-${USE_API}}"
  API_LABEL="$USE_API"
  echo "Using explicit MODEL=$MODEL USE_API=$USE_API"
elif [[ -n "${NVIDIA_API_KEY:-}${NGC_API_KEY:-}" ]]; then
  # Prefer NIM: ~40 RPM / 10k RPD vs OpenRouter :free ~50 RPD
  MODEL="${MODEL:-nvidia/nemotron-3-super-120b-a12b}"
  USE_API_FLAG=(--use_api nvidia)
  TAG="nvidia_nim_nemotron120"
  API_LABEL="nvidia"
  echo "Using NVIDIA NIM model: $MODEL"
elif [[ -n "${OPENROUTER_API_KEY:-}" ]]; then
  MODEL="${MODEL:-nvidia/nemotron-3-super-120b-a12b:free}"
  USE_API_FLAG=(--use_api openrouter)
  TAG="openrouter_nemotron120"
  API_LABEL="openrouter"
  echo "Using OpenRouter free model: $MODEL"
elif [[ "${FORCE_LOCAL:-0}" != "1" ]]; then
  # Default free path — no signup key required
  MODEL="${MODEL:-nvidia/nemotron-3-super-120b-a12b:free}"
  USE_API_FLAG=(--use_api kilo)
  TAG="${TAG:-kilo_nemotron120}"
  API_LABEL="kilo"
  echo "Using Kilo free pool (no API key): $MODEL"
elif [[ -n "${OPENAI_API_KEY:-}" ]]; then
  MODEL="${MODEL:-gpt-4o}"
  USE_API_FLAG=(--use_api openai)
  TAG="gpt4o"
  API_LABEL="openai"
  echo "Using OpenAI model: $MODEL"
else
  MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
  USE_API_FLAG=()
  TAG="llama31_8b_local80stack"
  API_LABEL="local"
  echo "WARN: no free/paid API — local $MODEL (local model only)"
fi

export CUDA_VISIBLE_DEVICES="$GPU"

COMMON=(
  --question_type open-ended
  --patient_class FactSelectPatient
  --data_dir data/interactive
  --expert_model "$MODEL"
  --patient_model "$MODEL"
  --judge_model "$MODEL"
  --max_questions 12
  --max_tokens 512
  --self_consistency 2
  --min_questions "${MIN_QUESTIONS:-3}"
  --abstain_threshold 4.0
  --kg_threshold "${KG_THRESHOLD:-4.0}"
  --rationale_generation
  --know_mode text_only
  --max_queue_size 6
  --initial_triplets 2
  --max_hop_depth 2
  --beam_size 3
  --llm_relevance_threshold 0.1
  --kg_csv "$KG_CSV"
  --disease2demo_csv data/kg/baseline_dataset/Disease2demo.csv
  --faiss_dir "$FAISS_DIR"
  --embedding_model sentence-transformers/all-MiniLM-L6-v2
  --who_overview_json data/kg/WHO/overview.json
  --use_clinical_rag
  --clinical_corpus_dir data/kg/clinical_corpus
  --clinical_rag_top_k 5
  --clinical_rag_rerank
  --use_adjudicator
  "${USE_API_FLAG[@]}"
)

run_one() {
  local method="$1"
  local split="$2"   # smoke50 | full
  local devfile="$3"
  local out="$RESULT_DIR/${method}_${TAG}_ioMEDQA_${split}.jsonl"
  local runlog="$LOG_DIR/${method}_${TAG}_ioMEDQA_${split}.log"
  echo "[$(date)] === $method | $TAG | $split ==="
  "$PY" Open_benchmark.py \
    --expert_class "$method" \
    --dev_filename "$devfile" \
    --output_filename "$out" \
    --log_filename "$runlog" \
    "${COMMON[@]}" 2>&1 | tee -a "$runlog"
  "$PY" scripts/compute_metrics.py "$out" \
    --output "$RESULT_DIR/${method}_${TAG}_ioMEDQA_${split}_metrics.json"
  cat "$RESULT_DIR/${method}_${TAG}_ioMEDQA_${split}_metrics.json"
}

# Milestone A: smoke
run_one KnowGuardExpert "smoke${SMOKE_N}" "$(basename "$SMOKE_FILE")"

SMOKE_METRICS="$RESULT_DIR/KnowGuardExpert_${TAG}_ioMEDQA_smoke${SMOKE_N}_metrics.json"
SMOKE_ACC=$("$PY" -c "import json; print(json.load(open('$SMOKE_METRICS'))['acc'])")
echo "Smoke ACC=$SMOKE_ACC threshold=$SMOKE_THRESHOLD"

"$PY" - <<PY
import json
from pathlib import Path
acc=float("$SMOKE_ACC")
thr=float("$SMOKE_THRESHOLD")
report={
  "smoke_acc": acc,
  "threshold": thr,
  "model": "$MODEL",
  "tag": "$TAG",
  "api": "$API_LABEL",
  "paper_knowguard_acc": 0.7098,
  "target_acc": 0.80,
  "milestone_A_pass": acc >= thr,
  "note": "Free path uses Kilo/OpenRouter/NIM (mnfst/awesome-free-llm-apis). That repo has signup links, not embedded keys.",
}
Path("results/PATH_TO_80_SMOKE_REPORT.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
if acc < thr:
    raise SystemExit(2)
PY

# Milestone B/C: full only if smoke passes
echo "[$(date)] Smoke passed — starting FULL ioMEDQA"
run_one KnowGuardExpert full ioMEDQA.jsonl

"$PY" - <<'PY'
import json
from pathlib import Path
files=list(Path("results").glob("KnowGuardExpert_*_ioMEDQA_full_metrics.json"))
cands=sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)
rows=[]
for p in cands[:6]:
    d=json.loads(p.read_text()); d["file"]=str(p); rows.append(d)
md=["# Path to 80% Report\n",
    "| Run | ACC | Turns | N | vs paper 70.98% | vs target 80% |\n|---|---:|---:|---:|---:|---:|\n"]
for d in rows:
    acc=d.get("acc",0)
    md.append(f"| `{Path(d['file']).name}` | {acc:.3f} | {d.get('turns',0):.2f} | {d.get('n',0)} | {acc-0.7098:+.3f} | {acc-0.80:+.3f} |\n")
md.append(
    "\n## Notes\n"
    "- Paper KnowGuard (GPT-4 + WHO KG): **70.98%**\n"
    "- Free frontier path: Kilo / OpenRouter `:free` / NVIDIA NIM (see awesome-free-llm-apis).\n"
    "- Goal: **≥80%**.\n"
)
Path("results/PATH_TO_80_REPORT.md").write_text("".join(md))
print("Wrote results/PATH_TO_80_REPORT.md")
PY

echo "=========================================="
echo "Path-to-80% done: $(date)"
echo "=========================================="
