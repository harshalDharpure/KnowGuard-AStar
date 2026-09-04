#!/usr/bin/env bash
# Wait for combined FAISS, then start strong-KG full evals on a free GPU.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source "$ROOT/.venv/bin/activate"
export PYTHONUNBUFFERED=1

FAISS_INDEX="data/kg/faiss_db_combined/index.faiss"
MIN_FREE_MB="${MIN_FREE_MB:-18000}"
LOG="logs/wait_and_run_strong_kg.log"
mkdir -p logs results

echo "[$(date)] Waiting for FAISS at $FAISS_INDEX ..." | tee -a "$LOG"

while true; do
  if [[ -f "$FAISS_INDEX" ]] && ! pgrep -f "scripts/init_faiss.py" >/dev/null; then
    echo "[$(date)] FAISS ready." | tee -a "$LOG"
    break
  fi
  # progress tip
  if [[ -f logs/build_combined_kg_faiss.log ]]; then
    prog=$(python3 - <<'PY'
from pathlib import Path
t=Path("logs/build_combined_kg_faiss.log").read_text(errors="ignore").replace("\r","\n")
lines=[ln for ln in t.splitlines() if "Processing data" in ln]
print(lines[-1][-80:] if lines else "building...")
PY
)
    echo "[$(date)] still waiting: $prog" >> "$LOG"
  fi
  sleep 120
done

# Pick freest GPU with enough memory
GPU_ID=$(MIN_FREE_MB="$MIN_FREE_MB" python3 - <<'PY'
import os, subprocess
min_free = int(os.environ.get("MIN_FREE_MB", "18000"))
out = subprocess.check_output([
    "nvidia-smi", "--query-gpu=index,memory.free",
    "--format=csv,noheader,nounits",
], text=True)
best, best_free = None, -1
for line in out.strip().splitlines():
    idx, free = [x.strip() for x in line.split(",")]
    free = int(free)
    if free >= min_free and free > best_free:
        best, best_free = idx, free
if best is None:
    # fallback: absolute freest
    for line in out.strip().splitlines():
        idx, free = [x.strip() for x in line.split(",")]
        free = int(free)
        if free > best_free:
            best, best_free = idx, free
print(best)
print(f"Selected GPU {best} with {best_free} MiB free", file=__import__("sys").stderr)
PY
)

echo "[$(date)] Starting strong KG evals on GPU=$GPU_ID" | tee -a "$LOG"
export GPU="$GPU_ID"
export MIN_FREE_MB
exec bash scripts/run_strong_kg_evals.sh
