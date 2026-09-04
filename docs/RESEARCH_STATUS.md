# Research status board

**Last updated:** 2026-09-04  
**Repo:** https://github.com/harshalDharpure/KnowGuard-AStar  
**Release:** `v0.2.0`

## Current claim (honest)

| Item | Value |
|------|------:|
| Task | Interactive **ioMEDQA** |
| Completed cases | **~597 / 1272** (~47%) |
| Live merged ACC | **~71.2%** |
| Paper basic to beat | **70.98%** |
| Paper enhanced target | **74.12%** |
| Final claim allowed? | **No** — wait for full N=1272 |

## Active campaign

| Shard | Typical provider (late campaign) | Role |
|------:|----------------------------------|------|
| 0 | Gemini lite / idle slots | Throughput |
| 1 | OpenRouter Super-120B `:free` | Throughput |
| 2 | NVIDIA NIM **Nemotron Ultra** | Quality anchor |
| 3 | OpenRouter Super-120B `:free` | Throughput |

**Lite A\* flags:** discriminative + entropy + clinical RAG + adjudicator + phase routing · no council · no dual-process · SC×1 · `min_questions=3` · `kg_threshold=4.0`

## Milestone tags

| Tag | Meaning |
|-----|---------|
| `v0.1.0` | Free local replication (Llama/Mistral/Qwen full N=1272) |
| `v0.2.0` | A\* lite + Ultra multi-provider partial full run |
| `v0.3.0` *(planned)* | Complete N=1272 + frozen metrics + CI tables |

## Artifact map

| Kind | Location |
|------|----------|
| Predictions | `results/*.jsonl` |
| Metrics | `results/*_metrics.json` |
| Research notes | `results/RESEARCH_PATH_TO_80.md`, `CHANGELOG.md` |
| Dataset | `data/interactive/ioMEDQA.jsonl` (+ `shards/`) |
| KG | `data/kg/combined_primekg_hetionet.csv` (FAISS local-only) |
| Citation | `CITATION.cff` |

## Open research risks

1. Provider mixing after ~case 576 — report Ultra-only subset for clean paper table if needed  
2. NIM 429 / Gemini free RPD / OpenRouter free limits  
3. Authors’ WHO multimodal KG not public  

## Next actions

1. Let `scripts/run_fast_multi_provider_resume.sh` finish or Ultra-only resume  
2. Merge shards → recompute metrics → cut **`v0.3.0`**  
3. Error taxonomy + ablations  
