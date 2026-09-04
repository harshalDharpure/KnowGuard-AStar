#!/usr/bin/env python3
"""SEMA-inspired evidence adjudication before KnowGuard final answer."""

from __future__ import annotations

import re
from typing import Any

from helper import get_response


SUFFICIENCY_PROMPT = """You are a clinical evidence adjudicator.
Given the patient dialogue so far and retrieved medical evidence, decide whether there is ENOUGH evidence to commit to a diagnosis/answer.

Patient info:
{patient_info}

Conversation:
{conversation}

Inquiry: {inquiry}

Retrieved evidence:
{evidence}

Reply with EXACTLY one line:
SUFFICIENT: YES|NO
REASON: <short reason>
CONFIDENCE: <number 1-5>
"""

OPTION_PROMPT = """You are a clinical adjudicator. Choose the single best answer supported by the evidence.

Inquiry: {inquiry}
Patient info and conversation:
{context}

Retrieved evidence:
{evidence}

Options:
{options}

Reply with EXACTLY:
OPTION: <letter or short answer text>
CONFIDENCE: <number 1-5>
"""


def _parse_sufficient(text: str) -> tuple[bool, float, str]:
    yes = bool(re.search(r"SUFFICIENT:\s*YES", text, re.I))
    no = bool(re.search(r"SUFFICIENT:\s*NO", text, re.I))
    conf = 3.0
    m = re.search(r"CONFIDENCE:\s*([0-9]+(?:\.[0-9]+)?)", text, re.I)
    if m:
        conf = float(m.group(1))
    reason = ""
    m2 = re.search(r"REASON:\s*(.+)", text, re.I)
    if m2:
        reason = m2.group(1).strip()
    if yes and not no:
        return True, conf, reason
    if no:
        return False, conf, reason
    # fallback: treat medium/high conf as sufficient only if YES-ish
    return ("yes" in text.lower() and "no" not in text.lower()[:80]), conf, reason


def adjudicate_sufficiency(
    patient_state: dict,
    inquiry: str,
    evidence: str,
    model_name: str,
    use_api: str | None = None,
    **kwargs,
) -> dict[str, Any]:
    conv = "\n".join(
        f"Q: {qa.get('question','')}\nA: {qa.get('answer','')}"
        for qa in patient_state.get("interaction_history", [])
    ) or "None"
    prompt = SUFFICIENCY_PROMPT.format(
        patient_info=patient_state.get("initial_info", ""),
        conversation=conv,
        inquiry=inquiry,
        evidence=evidence or "None",
    )
    messages = [
        {"role": "system", "content": "You adjudicate clinical evidence sufficiency."},
        {"role": "user", "content": prompt},
    ]
    text, _, usage = get_response(
        messages, model_name=model_name, use_api=use_api, max_tokens=128, **kwargs
    )
    sufficient, conf, reason = _parse_sufficient(text or "")
    return {
        "sufficient": sufficient,
        "confidence": conf,
        "reason": reason,
        "raw": text,
        "usage": usage or {"input_tokens": 0, "output_tokens": 0},
    }


def adjudicate_option(
    patient_state: dict,
    inquiry: str,
    evidence: str,
    options: dict | None,
    model_name: str,
    use_api: str | None = None,
    **kwargs,
) -> dict[str, Any]:
    conv = "\n".join(
        f"Q: {qa.get('question','')}\nA: {qa.get('answer','')}"
        for qa in patient_state.get("interaction_history", [])
    )
    context = f"{patient_state.get('initial_info','')}\n{conv}".strip()
    if options:
        opt_txt = "\n".join(f"{k}. {v}" for k, v in options.items())
    else:
        opt_txt = "(open-ended — give best short answer)"
    prompt = OPTION_PROMPT.format(
        inquiry=inquiry,
        context=context,
        evidence=evidence or "None",
        options=opt_txt,
    )
    messages = [
        {"role": "system", "content": "You select the best-supported clinical answer."},
        {"role": "user", "content": prompt},
    ]
    text, _, usage = get_response(
        messages, model_name=model_name, use_api=use_api, max_tokens=128, **kwargs
    )
    letter = None
    m = re.search(r"OPTION:\s*([A-Da-d]|[^\n]+)", text or "")
    if m:
        letter = m.group(1).strip()
        if len(letter) == 1:
            letter = letter.upper()
    conf = 3.0
    m2 = re.search(r"CONFIDENCE:\s*([0-9]+(?:\.[0-9]+)?)", text or "", re.I)
    if m2:
        conf = float(m2.group(1))
    return {
        "option": letter,
        "confidence": conf,
        "raw": text,
        "usage": usage or {"input_tokens": 0, "output_tokens": 0},
    }


import math

OPTION_DIST_PROMPT = """Estimate probability each option is correct given the clinical evidence.
Reply EXACTLY:
P_A: <0-1>
P_B: <0-1>
P_C: <0-1>
P_D: <0-1>
"""


def _parse_probs(text: str, options: dict | None) -> dict[str, float]:
    probs = {}
    letters = list(options.keys()) if options else ["A", "B", "C", "D"]
    for letter in letters:
        m = re.search(rf"P_{letter}\s*:\s*([0-9]*\.?[0-9]+)", text or "", re.I)
        if m:
            probs[letter] = float(m.group(1))
    if not probs:
        return {l: 1.0 / len(letters) for l in letters}
    s = sum(probs.values()) or 1.0
    return {k: v / s for k, v in probs.items()}


def estimate_option_distribution(
    patient_state: dict,
    inquiry: str,
    evidence: str,
    options: dict | None,
    model_name: str,
    use_api: str | None = None,
    **kwargs,
) -> dict[str, float]:
    conv = "\n".join(
        f"Q: {qa.get('question','')}\nA: {qa.get('answer','')}"
        for qa in patient_state.get("interaction_history", [])
    )
    opt_txt = "\n".join(f"{k}. {v}" for k, v in (options or {}).items()) or "A-D unknown"
    prompt = f"""{OPTION_DIST_PROMPT}

Inquiry: {inquiry}
Patient info: {patient_state.get('initial_info','')}
Conversation:
{conv or 'None'}
Evidence:
{evidence or 'None'}
Options:
{opt_txt}
"""
    messages = [
        {"role": "system", "content": "You estimate diagnostic option probabilities."},
        {"role": "user", "content": prompt},
    ]
    text, _, _ = get_response(messages, model_name=model_name, use_api=use_api, max_tokens=128, **kwargs)
    return _parse_probs(text, options)


def shannon_entropy(probs: dict[str, float]) -> float:
    h = 0.0
    for p in probs.values():
        if p > 0:
            h -= p * math.log(p)
    return h


def entropy_commit_decision(
    entropy: float,
    tau_commit: float,
    num_turns: int,
    max_questions: int,
    max_round: bool = False,
    min_questions: int = 0,
) -> dict:
    if max_round:
        return {"commit": True, "reason": "max_round"}
    if min_questions > 0 and num_turns < min_questions:
        return {"commit": False, "reason": f"min_questions={min_questions}"}
    if entropy <= tau_commit:
        return {"commit": True, "reason": f"entropy={entropy:.3f}<={tau_commit}"}
    if num_turns >= max_questions:
        return {"commit": True, "reason": "max_questions"}
    return {"commit": False, "reason": f"entropy={entropy:.3f}>{tau_commit}"}
