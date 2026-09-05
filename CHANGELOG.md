# Changelog

## 2026-09-05

- README rewritten in plain language (removed hype / “beat the paper” framing).
- Added `scripts/run_ultra_only_serial.sh` for Nemotron Ultra–only eval (one worker).
- Archived mixed-provider shard outputs under `results/archive_mixed_*`.

## 2026-09-04

- Partial full ioMEDQA shard run (NIM + temporary other providers).
- Smoke-20 with Nemotron Ultra: 70.0% ACC (N=20).
- Added discriminative questions, entropy gate, phase routing, dual-process, council (optional).
- Multi-provider resume scripts for rate-limit workarounds.

## 2026-09-01 and earlier

- Free local replication (Qwen / Mistral / Llama) on full ioMEDQA.
- PrimeKG + Hetionet build and clinical RAG.
- Protocol defaults: `min_questions`, higher `kg_threshold`.
