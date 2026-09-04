# Contributing (research repository)

This is an **active research codebase**, not a production product. Contributions should preserve experiment reproducibility.

## Principles

1. **Do not break resume-safe JSONL outputs** — cases are keyed by `id`; append-only.
2. **Record every eval** under `results/` with a clear filename + `*_metrics.json`.
3. **Update `CHANGELOG.md`** for any result-changing change.
4. **Never commit secrets** (`.env`, API keys). Use `.env.example`.
5. **Do not claim SOTA** until full **N=1272** ioMEDQA finishes cleanly.

## Experiment checklist

- [ ] Document model + `--use_api` + key flags in the run script / log header
- [ ] Note dataset path (`ioMEDQA.jsonl` or shard)
- [ ] Save metrics via `scripts/compute_metrics.py`
- [ ] Note whether council / dual-process / SC×N were on
- [ ] If mixing providers mid-shard, note it in CHANGELOG

## Code style

- Prefer additive flags in `args.py` over hard-coding behavior
- Keep high-level security: no payload/exploit code
- Large FAISS indexes stay local (`data/kg/faiss_db_combined/` is gitignored)

## Upstream

Please also cite and respect [IcecreamArtist/KnowGuard](https://github.com/IcecreamArtist/KnowGuard) and [arXiv:2509.24816](https://arxiv.org/abs/2509.24816).
