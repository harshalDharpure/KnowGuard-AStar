# KnowGuard-AStar release checklist

- [ ] `scripts/compute_metrics.py` on merged JSONL
- [ ] Update `CHANGELOG.md` + `docs/RESEARCH_STATUS.md` + `CITATION.cff` version
- [ ] Update README headline table
- [ ] `git tag -a vX.Y.Z -m "..."` and `gh release create`
- [ ] Confirm `.env` not in tree; FAISS not committed
- [ ] If claiming vs paper, state N and whether models were mixed
