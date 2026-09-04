# Path to ≥80% — Live Run Status

## Keys / providers
- Stored in gitignored `KnowGuard/.env` (not committed).
- **Active:** NVIDIA NIM `nvidia/nemotron-3-super-120b-a12b` (preferred: ~40 RPM / 10k RPD).
- OpenRouter key also saved as fallback (`:free` models ~50 RPD — too tight for full multi-turn).
- [mnfst/awesome-free-llm-apis](https://github.com/mnfst/awesome-free-llm-apis) lists **signup links only** — no embedded keys.

## Current job
- Script: `scripts/run_path_to_80.sh`
- Tag: `nvidia_nim_nemotron120`
- Phase: **smoke-50** (Milestone A ≥65%), then full 1272 if pass
- Stack: KnowGuard + clinical RAG + adjudicator + SC×2
- Log: `logs/nohup_path_to_80_nim.log`
- Results: `results/KnowGuardExpert_nvidia_nim_nemotron120_ioMEDQA_smoke50.jsonl`

## Security
Rotate both keys after the experiment — they were pasted in chat.
