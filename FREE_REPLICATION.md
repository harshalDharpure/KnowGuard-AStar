# Free KnowGuard Replication

This folder contains a **free, local** replication of [KnowGuard](https://github.com/IcecreamArtist/KnowGuard) using open-source models and rebuilt public datasets.

## Important caveat

Paper Tables 1–2 use **GPT-4 + authors' WHO knowledge graph**. This setup uses:

- Local model: `Qwen/Qwen2.5-1.5B-Instruct` (configurable)
- Proxy KG built from MedQA contexts + MiniLM FAISS
- Rebuilt `ioMEDQA` from public MedQA validation split

Numbers are **not** expected to match paper GPT-4 results exactly.

## Quick start

```bash
cd KnowGuard
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Build data + proxy KG
python scripts/build_datasets.py --smoke_size 5
python scripts/build_proxy_kg.py --max_rows 300

# Run smoke eval (5 cases, 3 methods)
CUDA_VISIBLE_DEVICES=1 python scripts/run_free_evals.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --data data/interactive/ioMEDQA_smoke5.jsonl \
  --gpu 1
```

Results are written to `results/FREE_REPLICATION_REPORT.md`.

## What was fixed in the upstream repo

- Added `args.py` + runnable `Open_benchmark.py` entrypoint
- Fixed `KnowGuardExpert` init / KG update / abstention path
- Added missing MediQ prompts (`scale`, `numcutoff`)
- Added FAISS wrapper methods + LLM scorer aliases
- Local HuggingFace generation + embedding support
- Dataset / KG builder scripts for free assets

## Paper reference (GPT-4, ioMEDQA)

| Method | Paper ACC | Paper Turns |
|---|---:|---:|
| KnowGuard | 70.98% | 5.41 |
| Scale Rating | 64.23% | 5.15 |
| Numerical Score | 61.74% | 2.51 |
