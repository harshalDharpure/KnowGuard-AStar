#!/usr/bin/env python3
"""Multi-model test-time council for final diagnostic commit."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from helper import get_response

COUNCIL_VOTE_PROMPT = """Choose the single best supported clinical answer.

Inquiry: {inquiry}
Patient info and conversation:
{context}
Evidence:
{evidence}
Options:
{options}

Reply EXACTLY:
OPTION: <letter A-D>
CONFIDENCE: <1-5>
"""

DEFAULT_COUNCIL = [
    {"model": "nvidia/nemotron-3-ultra-550b-a55b", "use_api": "nvidia"},
    {"model": "google/gemini-2.5-flash", "use_api": "google"},
    {"model": "meta-llama/llama-3.3-70b-versatile", "use_api": "groq"},
]


def _parse_vote(text: str) -> tuple[str | None, float]:
    letter = None
    m = re.search(r"OPTION:\s*([A-Da-d])", text or "", re.I)
    if m:
        letter = m.group(1).upper()
    conf = 3.0
    m2 = re.search(r"CONFIDENCE:\s*([0-9]+(?:\.[0-9]+)?)", text or "", re.I)
    if m2:
        conf = float(m2.group(1))
    return letter, conf


def council_vote(
    patient_state: dict,
    inquiry: str,
    evidence: str,
    options: dict | None,
    council_config: list[dict] | None = None,
    temperature: float = 0.3,
    max_tokens: int = 128,
) -> dict[str, Any]:
    conv = "\n".join(
        f"Q: {qa.get('question', '')}\nA: {qa.get('answer', '')}"
        for qa in patient_state.get("interaction_history", [])
    )
    context = f"{patient_state.get('initial_info', '')}\n{conv}".strip()
    opt_txt = "\n".join(f"{k}. {v}" for k, v in (options or {}).items())
    prompt = COUNCIL_VOTE_PROMPT.format(
        inquiry=inquiry, context=context, evidence=evidence or "None", options=opt_txt
    )
    messages = [
        {"role": "system", "content": "You are a clinical council member."},
        {"role": "user", "content": prompt},
    ]

    votes: list[dict] = []
    usage = {"input_tokens": 0, "output_tokens": 0}
    config = council_config or DEFAULT_COUNCIL

    for member in config:
        model = member["model"]
        api = member.get("use_api")
        try:
            text, _, u = get_response(
                messages,
                model_name=model,
                use_api=api,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            letter, conf = _parse_vote(text or "")
            if letter:
                votes.append({"model": model, "option": letter, "confidence": conf, "raw": text})
            if u:
                usage["input_tokens"] += u.get("input_tokens", 0)
                usage["output_tokens"] += u.get("output_tokens", 0)
        except Exception as exc:
            votes.append({"model": model, "error": str(exc)})

    valid = [v for v in votes if v.get("option")]
    if not valid:
        return {"option": None, "votes": votes, "usage": usage, "consensus": 0.0}

    weighted: Counter[str] = Counter()
    for v in valid:
        weighted[v["option"]] += v.get("confidence", 1.0)
    winner = weighted.most_common(1)[0][0]
    consensus = weighted[winner] / sum(weighted.values()) if weighted else 0.0
    return {
        "option": winner,
        "votes": votes,
        "usage": usage,
        "consensus": consensus,
        "weighted_counts": dict(weighted),
    }
