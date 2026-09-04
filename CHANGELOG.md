# Changelog

All notable research milestones for **KnowGuard-AStar** are recorded here.
Format inspired by [Keep a Changelog](https://keepachangelog.com/).
Versioning follows research milestones (not strict semver API stability).

## [Unreleased]

- Finish full interactive ioMEDQA **N=1272**
- Ultra-only clean ablation table + bootstrap CI
- Error taxonomy + paper-ready LaTeX tables
- Optional secondary datasets (Interactive PubMedQA / ioCRAFT-MD)

## [0.2.0] — 2026-09-04 — A* lite multi-provider campaign (partial full run)

### Headline results
- **Merged ACC ≈ 71.2% on N≈597 / 1272** interactive ioMEDQA (live mid-campaign)
- Above paper **KnowGuard basic 70.98%** (GPT-4 + WHO KG); full N still running
- Smoke-20 (Ultra + A* lite): **70.0%** ACC, 6.45 avg turns

### Added
- A* modules: discriminative questions, entropy gate, phase routing, dual-process, council
- Clinical RAG (`clinical_rag.py`) + adjudicator (`adjudicator.py`)
- FAISS fast-path / singleton cache in `know_storage.py`
- Progressive `FactSelectPatient` disclosure
- Parallel shard runners + NIM throttle / long backoff in `helper.py`
- Multi-provider resume: NIM Ultra + OpenRouter Super:free + Gemini lite
- Presentation README with full result tables and architecture diagrams

### Artifacts
- `results/KnowGuardExpert_astar_lite_nim_ioMEDQA_shard{0..3}.jsonl`
- `results/KnowGuardExpert_astar_lite_nim_ioMEDQA_merged_metrics.json`
- `data/interactive/shards/` · `data/kg/combined_primekg_hetionet.csv`

### Known caveats
- Mid/late cases may mix Ultra / Super:free / Gemini (throughput workarounds)
- Do **not** claim final SOTA until N=1272 completes

## [0.1.0] — 2026-08 — Free local replication baseline

### Headline results (full N=1272)
| Method | Model | ACC | Turns |
|--------|-------|----:|------:|
| KnowGuardExpert | Qwen2.5-1.5B | 27.5% | 0.45 |
| KnowGuardExpert | Mistral-7B | 32.5% | 0.19 |
| KnowGuardExpert | Llama-3.1-8B | 34.9% | 0.92 |
| OpenEndedScaleExpert | Llama-3.1-8B | 37.0% | 4.04 |

### Added
- Runnable free/local fork of KnowGuard
- ioMEDQA interactive dataset build
- Proxy / Strong KG (PrimeKG + Hetionet) build scripts
- Protocol fixes: `--min_questions`, higher `--kg_threshold`

### Research note
Local 8B stacks under-ask historically; interactive 70%+ needs GPT-4-class APIs + protocol + KG/RAG.

## Paper reference targets

| Paper method | ACC |
|--------------|----:|
| KnowGuard basic (GPT-4 + WHO KG) | **70.98%** |
| KnowGuard enhanced | **74.12%** |

Paper: [arXiv:2509.24816](https://arxiv.org/abs/2509.24816)

[Unreleased]: https://github.com/harshalDharpure/KnowGuard-AStar/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/harshalDharpure/KnowGuard-AStar/releases/tag/v0.2.0
[0.1.0]: https://github.com/harshalDharpure/KnowGuard-AStar/releases/tag/v0.1.0
