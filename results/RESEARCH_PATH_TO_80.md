# Research Path to ≥80% on ioMEDQA (Literature Synthesis)

Last updated: 2026-09-01

## Honest ceiling by setup

| Setting | Realistic ACC on **interactive ioMEDQA** | Evidence |
|---------|------------------------------------------|----------|
| 8B local + biology KG only | ~35% | Our Llama full: 34.9% |
| 8B + RAG + adjudicator (no protocol fix) | ~36% | Smoke-50 |
| **Free API + full protocol fixes** | **45–60%** | Extrapolate from paper + i-MedRAG |
| GPT-4 + WHO KG (paper) | **71%** | KnowGuard arXiv:2509.24816 |
| Closed-book MedQA GPT-4o | ~81–92% | Not interactive; easier task |
| Llama 3.3 70B + QLoRA + textbook RAG (closed MedQA) | **~81%** | medqa-llm GitHub; not multi-turn |

**80% on interactive ioMEDQA is harder than 80% on closed MedQA.** The paper itself stops at 71% with GPT-4.

---

## What research says actually moves accuracy (ranked)

### Tier 1 — Dominant (without these, 80% is unrealistic)

1. **Strong backbone LLM** (GPT-4 class, or Llama-3.3-70B + MedQA fine-tune)
   - GPT-4 closed MedQA ~81%; GPT-4o higher ([Bhatti et al.](https://www.emergentmind.com/topics/medqa-usmle))
   - Open: Llama 3.3 70B + QLoRA + RAG → **80.99% closed MedQA** ([medqa-llm](https://github.com/MartinLiarte/medqa-llm))

2. **Clinical evidence corpus** (textbooks / MedQA contexts), not biology-only KGs
   - i-MedRAG: iterative follow-up retrieval → **69.7%** zero-shot GPT-3.5 on closed MedQA ([Xiong et al., 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11997844/))
   - MedMobile: **RAG hurt** small models (−12.6 pp); CoT + ensemble helped (+7.4 pp)

3. **Authors’ WHO guideline KG** (KnowGuard paper)
   - ~22k nodes, multimodal; **not public** — biggest KG gap vs our PrimeKG/Hetionet stack

### Tier 2 — Protocol / algorithm (free; we under-used these)

4. **Multi-turn investigate-before-abstain** (~5.7 turns in paper)
   - Our bug: **86% answered on turn 0** with kg_threshold=3.5
   - Fix: `--min_questions 3`, `--kg_threshold 4.0` (implemented 2026-09-01)

5. **Self-consistency ×2+** (+several pp on MedQA ensembles)
   - Paper uses SC×2; many of our runs used SC×1

6. **Chain-of-thought before abstention** (`--rationale_generation`)
   - MedMobile: CoT +2.4 pp; KnowGuard uses scale+RG prompts

7. **Evidence adjudication before commit** (SEMA-style)
   - Implemented in `adjudicator.py`; option letter now wired to scoring

8. **Information-gain questions** ([MedKGI](https://arxiv.org/abs/2512.24181))
   - Not yet implemented — ask discriminative questions, not generic

9. **Iterative RAG (i-MedRAG)** — multi-hop follow-up queries per case
   - Partially present via KG expansion; not full i-MedRAG loop

### Tier 3 — Marginal alone

10. More PrimeKG / Hetionet / DisGeNET edges (+1–3 pp typical)
11. Vanilla RAG on 8B can **hurt** ([BioNLP 2026](https://aclanthology.org/2026.bionlp-1.72.pdf))
12. Self-reflection loops — inconsistent on MedQA ([arXiv:2604.00261](https://arxiv.org/html/2604.00261v2))

---

## What we use vs what paper used

| Component | Paper (71%) | Our stack |
|-----------|-------------|-----------|
| LLM | GPT-4 | Free NIM/OpenRouter or Llama-8B |
| KG | WHO multimodal guidelines | PrimeKG + Hetionet + MedQA passages |
| Avg turns | **5.74** | **0.14–0.92** (before fixes) |
| SC | ×2 | ×2 in path-to-80; ×1 in early runs |
| min_questions | implicit in abstention | **NEW: default 3** |
| kg_threshold | tuned for GPT-4 | **NEW: default 4.0** |
| Adjudicator | N/A (paper method) | Added (SEMA-inspired) |
| Clinical RAG | N/A | MedQA 4001 passages |

---

## Free resources that can help (actionable)

| Resource | Cost | Expected gain | Status |
|----------|------|---------------|--------|
| NVIDIA NIM Nemotron Ultra/Super | Free tier | Strongest free API | Tested |
| OpenRouter `:free` models | Free (50 RPD) | Variable; Super-120 OK | Tested |
| Kilo Code free pool | No key | Backup API | Tested |
| **Protocol fixes** (min_q, kg, adjudicator) | Free | **+5–15 pp?** | **Just implemented** |
| Llama 3.3 70B local (if GPU fits) | Free compute | +20–30 pp vs 8B | Not run yet |
| MedQA QLoRA on 70B | Free (train) | Up to ~81% **closed** | Not implemented |
| WHO KG from authors | Free if granted | Unknown; paper-critical | Not available |
| i-MedRAG iterative retrieve | Free engineering | +3–10 pp (GPT-3.5) | Not implemented |
| MedKGI info-gain questions | Free engineering | Efficiency + accuracy | Not implemented |

---

## Realistic paths to 80%

### Path A — Paid API (fastest to test)
OpenRouter `gpt-4o` / `gemini-2.5-pro` + our fixed protocol + clinical RAG  
**Estimate: 65–78%** interactive; 80% possible but not guaranteed without WHO KG

### Path B — Free API + protocol (what we should run next)
NIM Nemotron Ultra + `--min_questions 3` + adjudicator + SC×2 + RAG  
**Estimate: 45–60%** — honest ceiling for free frontier API on interactive task

### Path C — Open 70B + fine-tune (heavy but free)
Llama 3.3 70B QLoRA on MedQA + textbook RAG + KnowGuard loop  
**Estimate: 55–70%** interactive (closed MedQA 81% doesn't transfer 1:1)

### Path D — Cannot reach 80% without
- GPT-4-class model **or** 70B MedQA-finetuned model
- Fixing turn-0 early answer bug (**done**)
- Clinical corpus (**done**)
- Ideally WHO KG (**missing**)

---

## Run the improved free stack

```bash
cd KnowGuard
# .env with OPENROUTER_API_KEY or NVIDIA_API_KEY
GPU=1 SMOKE_N=50 MIN_QUESTIONS=3 KG_THRESHOLD=4.0 bash scripts/run_research_free_stack.sh
```

## Code changes (2026-09-01)

- `args.py`: `--min_questions` (default 3), `--kg_threshold` (default 4.0)
- `expert_functions.py`: enforce min questions before answer
- `expert.py`: adjudicated option → scoring
- `Open_benchmark.py` + `LLM_judge.py`: judge uses same API as expert
- `scripts/run_research_free_stack.sh`: all literature tweaks in one runner

---

## References

- KnowGuard: [arXiv:2509.24816](https://arxiv.org/abs/2509.24816) — 70.98% ioMEDQA, GPT-4, WHO KG, 5.74 turns
- i-MedRAG: [PMC11997844](https://pmc.ncbi.nlm.nih.gov/articles/PMC11997844/) — iterative clinical RAG
- MedMobile: [arXiv:2410.09019](https://arxiv.org/abs/2410.09019) — CoT + ensemble > RAG for small models
- medqa-llm: [GitHub](https://github.com/MartinLiarte/medqa-llm) — Llama 3.3 70B + QLoRA + RAG = 80.99% closed MedQA
- MedKGI: [arXiv:2512.24181](https://arxiv.org/abs/2512.24181) — info-gain questioning + KG
- Self-correction MedQA: [arXiv:2604.00261](https://arxiv.org/html/2604.00261v2) — reflection inconsistent
