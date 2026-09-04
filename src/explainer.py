"""LLM explanation layer (PS outcome h) — Ollama-backed with a deterministic fallback.

Deterministic scoring is shared with `finetune_data.py` so the app, the finetuning
labels and the runtime explanations all use the SAME numbers:
  - build_feature_record(row, is_time) -> dict of risk features
  - explain(feat) -> tries Ollama (open-source local model), else template fallback.

The template fallback guarantees the demo never breaks if Ollama / the local model
is absent. When a finetuned MoSPI model exists in Ollama (`mospi-llm`), set
MOSPI_LLM_MODEL env var (or default) to use it.
"""
from __future__ import annotations

import json
import os
import urllib.request
from typing import Optional

OLLAMA_URL = os.environ.get("MOSPI_OLLAMA_URL", "http://localhost:11434")
MODEL_NAME = os.environ.get("MOSPI_LLM_MODEL", "qwen2.5:3b-instruct")

SYSTEM = (
    "You are MoSPI's infrastructure project risk analyst. Given a JSON of a project's "
    "risk features, explain concisely why it is at risk and what to do, in 2-3 sentences."
)


# ---- shared scoring (kept identical to finetune_data.rule_based_scores) ----
def build_feature_record(row, is_time: bool) -> dict:
    orig = float(row.get("original_cost_cr") or 0)
    exp = float(row.get("expenditure_cum_cr") or 0)
    ant = float(row.get("anticipated_cost_cr") or 0)
    cor = float(row.get("cost_overrun_pct") or 0)
    exp_ratio = exp / orig if orig > 0 else 0.0
    cost_risk = min(1.0, max(0.0, cor / 200.0))
    progress_risk = min(1.0, max(0.0, (exp_ratio - 0.5) / 2.0))
    is_time_ds = "tor_months" in row
    rec = {
        "sector": str(row.get("sector") or "Unknown"),
        "project_name": str(row.get("project_name") or "Unknown"),
        "cost_overrun_pct": float(cor),
        "expenditure_ratio": round(exp_ratio, 3),
        "cost_risk": round(0.7 * cost_risk + 0.3 * progress_risk, 2),
        # budget / cost detail (for richer explanations)
        "original_cost_cr": float(orig),
        "anticipated_cost_cr": float(ant) if ant else None,
        "expenditure_cum_cr": float(exp),
        "approval_year": float(row.get("approval_year")) if row.get("approval_year") is not None else None,
        "tor_months": float(tor) if (tor := float(row.get("tor_months") or 0)) else None,
        "reasons": str(row.get("reasons") or "n/a"),
    }
    # dataset may or may not be the time one; recompute delay risk defensively
    tor = float(row.get("tor_months") or 0)
    delayed = int(row.get("delayed") or 0)
    rec["delay_risk"] = round(
        min(1.0, (tor if delayed else 0.0) / 60.0) if is_time_ds else 0.05, 2)
    return rec


def _tier(score: float) -> str:
    if score >= 0.7:
        return "Critical"
    if score >= 0.45:
        return "High"
    if score >= 0.25:
        return "Medium"
    return "Low"


def _fmt_cr(v):
    try:
        return f"₹{float(v):,.1f} Cr"
    except Exception:
        return "n/a"


def _template(feat: dict) -> str:
    sect = feat["sector"]; cor = feat["cost_overrun_pct"]
    cr, dr = feat["cost_risk"], feat["delay_risk"]
    pct = lambda v: f"{v*100:.1f}%" if v else "n/a"
    head = f"**{feat.get('project_name', '')}** ({sect})."
    budget = (f"Original cost {_fmt_cr(feat['original_cost_cr'])}; "
              f"estimated (anticipated) cost {_fmt_cr(feat.get('anticipated_cost_cr'))}; "
              f"expenditure so far {_fmt_cr(feat['expenditure_cum_cr'])} "
              f"({pct(feat['expenditure_ratio'])} of original).")
    parts = [budget]
    if dr > 0.05:
        parts.append(f"Rated {_tier(dr)} **delay risk** (TOR ~{feat.get('tor_months')} months).")
    if cr > 0.25 or cor:
        parts.append(f"{_tier(cr)} **cost risk**, driven by a cost overrun of "
                     f"**{cor:.1f}%** over the original budget.")
    else:
        parts.append("Currently **low cost risk** with no material cost overrun.")
    if feat.get("reasons") and str(feat["reasons"]).strip() and str(feat["reasons"]) != "nan":
        parts.append(f"Noted causes: {feat['reasons']}.")
    parts.append("Recommended action: re-baseline the flagged milestone and escalate if the "
                 "exposure persists beyond one review cycle.")
    return head + " " + " ".join(parts)


def _ollama(messages: list[dict], tools: Optional[list] = None, timeout: int = 60):
    """Call Ollama chat. Returns text content when no tools; when tools are given,
    returns the full response dict (so tool_calls can be read). None on any failure."""
    try:
        body: dict = {"model": MODEL_NAME, "messages": messages,
                      "stream": False, "options": {"temperature": 0.2}}
        if tools:
            body["tools"] = tools
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/chat", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp_obj = json.loads(resp.read().decode())
        if tools:
            return resp_obj
        return resp_obj["message"]["content"]
    except Exception:
        return None


def explain(feat: dict, prefer_llm: bool = True) -> tuple[str, str]:
    """Return (explanation, source) where source in {'ollama','template'}."""
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": feat["instruction"] + "\n" + feat["input"]}
        if "instruction" in feat else
        {"role": "user", "content": "Explain the risk of this infrastructure project:\n" + json.dumps(feat, sort_keys=True)},
    ]
    if prefer_llm:
        out = _ollama(messages)
        if out:
            return out.strip(), "ollama"
    return _template(feat), "template"


def enrich(row, is_time: bool, prefer_llm: bool = False) -> dict:
    """Build a full enriched record for one project row (scores + explanation)."""
    feat = build_feature_record(row, is_time)
    feat["instruction"] = "Explain the risk of the following infrastructure project in 2-3 sentences."
    feat["input"] = json.dumps({k: v for k, v in feat.items() if k not in ("instruction", "input")}, sort_keys=True)
    feat["cost_tier"] = _tier(feat["cost_risk"])
    feat["delay_tier"] = _tier(feat["delay_risk"])
    text, src = explain(feat, prefer_llm=prefer_llm)
    feat["explanation"] = text
    feat["explanation_source"] = src
    return feat