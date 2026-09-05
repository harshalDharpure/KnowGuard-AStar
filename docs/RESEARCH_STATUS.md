# Research status

Last updated: 2026-09-05

| Item | Value |
|------|-------|
| Task | Interactive ioMEDQA |
| Target size | N=1272 |
| Paper (basic KnowGuard) | 70.98% ACC (GPT-4 + WHO KG) |
| Paper (enhanced) | 74.12% ACC |
| Current clean run | Nemotron Ultra only via `run_ultra_only_serial.sh` |
| Older partial merge | ~71% on ~587 cases (some later cases used other APIs) |

Notes:

- Prefer Ultra-only result files for reporting.
- Mixed-provider outputs are archived under `results/archive_mixed_*`.
- Rebuild FAISS locally if `data/kg/faiss_db_combined/` is missing.
