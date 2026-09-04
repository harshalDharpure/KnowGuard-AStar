#!/usr/bin/env python3
"""System-1 / System-2 dual-process verification at commit time."""

from __future__ import annotations

import re
from typing import Any

from helper import get_response

SYSTEM2_PROMPT = """You are an adversarial clinical critic.
A colleague proposed diagnosis/answer: {candidate}

Assume this answer is COMPLETELY WRONG.
Given the patient vignette and dialogue, which alternative option (A-D) explains the findings BETTER?

Patient info: {patient_info}
Conversation:
{conversation}
Inquiry: {inquiry}
Options:
{options}

Reply EXACTLY:
ALTERNATIVE: <letter A-D>
CONFIDENCE: <1-5>
REASON: <one sentence>
"""


def dual_process_verify(
    patient_state: dict,
    inquiry: str,
    candidate_option: str,
    options: dict | None,
    model_name: str,
    use_api: str | None = None,
    **kwargs,
) -> dict[str, Any]:
    conv = "\n".join(
        f"Q: {qa.get('question', '')}\nA: {qa.get('answer', '')}"
        for qa in patient_state.get("interaction_history", [])
    ) or "None"
    opt_txt = "\n".join(f"{k}. {v}" for k, v in (options or {}).items()) or "(open-ended)"
    prompt = SYSTEM2_PROMPT.format(
        candidate=candidate_option,
        patient_info=patient_state.get("initial_info", ""),
        conversation=conv,
        inquiry=inquiry,
        options=opt_txt,
    )
    messages = [
        {"role": "system", "content": "You are a skeptical clinical verifier."},
        {"role": "user", "content": prompt},
    ]
    text, _, usage = get_response(
        messages, model_name=model_name, use_api=use_api, max_tokens=128, **kwargs
    )
    alt = None
    m = re.search(r"ALTERNATIVE:\s*([A-Da-d])", text or "", re.I)
    if m:
        alt = m.group(1).upper()
    conf = 3.0
    m2 = re.search(r"CONFIDENCE:\s*([0-9]+(?:\.[0-9]+)?)", text or "", re.I)
    if m2:
        conf = float(m2.group(1))
    reason = ""
    m3 = re.search(r"REASON:\s*(.+)", text or "", re.I)
    if m3:
        reason = m3.group(1).strip()
    cand = (candidate_option or "").upper()[:1]
    disagree = alt is not None and cand and alt != cand
    return {
        "system1_option": candidate_option,
        "system2_option": alt,
        "disagreement": bool(disagree and conf >= 3.0),
        "confidence": conf,
        "reason": reason,
        "raw": text,
        "usage": usage or {"input_tokens": 0, "output_tokens": 0},
    }
