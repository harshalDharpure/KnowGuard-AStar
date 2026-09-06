# KnowGuard

Interactive clinical QA (KnowGuard / ICLR 2026).  
Paper: [arXiv:2509.24816](https://arxiv.org/abs/2509.24816)

## What this repo has

- Runnable KnowGuard eval (`Open_benchmark.py`)
- Interactive **ioMEDQA** (1272 cases) built from public MedQA
- Public KG (PrimeKG + Hetionet) + clinical RAG
- Local and NVIDIA NIM evaluation scripts

## Dataset

Source: [GBaker/MedQA-USMLE-4-options-hf](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options-hf) (validation)  
Built with: `scripts/build_datasets.py` → `data/interactive/ioMEDQA.jsonl`

## Methods (in `expert.py`)

| Class | Paper idea |
|-------|------------|
| `KnowGuardExpert` | Abstain using KG / evidence score |
| `OpenEndedScaleExpert` | Abstain using Likert confidence |
| `OpenEndedNumericalCutOffExpert` | Abstain using numeric confidence |

Paper (GPT-4 + WHO KG): KnowGuard **70.98%**, Scale ~64%, Numerical ~62%.

## Local full results (N=1272)

| Method | Model | ACC | Avg turns |
|--------|-------|----:|----------:|
| KnowGuardExpert | Qwen2.5-1.5B | 27.5% | 0.45 |
| KnowGuardExpert | Mistral-7B | 32.5% | 0.19 |
| KnowGuardExpert | Llama-3.1-8B | 34.9% | 0.92 |
| OpenEndedScaleExpert | Mistral-7B | 33.6% | 2.08 |
| OpenEndedScaleExpert | Llama-3.1-8B | 37.0% | 4.04 |
| OpenEndedNumericalCutOff | local | 28.3% | 12.0 |

Metrics files: `results/*_full_metrics.json`

## Ultra-only run

Nemotron Ultra (`scripts/run_ultra_only_serial.sh`).  
Partial output: `results/KnowGuardExpert_astar_ultra_only_ioMEDQA_shard2.jsonl`

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # set NVIDIA_API_KEY if using NIM
```

Rebuild KG FAISS if needed: `bash scripts/build_strong_kg.sh`

## Layout

```
data/interactive/   # ioMEDQA
data/kg/            # KG + clinical RAG
expert.py           # methods
Open_benchmark.py   # eval loop
results/            # metrics + predictions
scripts/            # build + run helpers
```
