"""Generate a LoRA finetuning dataset for an Ollama-hosted explanation LLM.

Task for the model: given a compact JSON of a project's computed risk features,
produce a 2-3 sentence plain-language explanation of why it is at risk and what
to do — in authentic MoSPI project-monitoring vocabulary.

The dataset is built from the REAL data (cost + time datasets): real sectors,
real overrun/expenditure/delay values, and a curated set of real cause strings
(from the Flash Report 'reasons' column) so the model learns genuine MoSPI wording.
Inputs are structured risk features; outputs are expert-written templated
explanations grounded in those exact numbers.

Output format: HF-style {instruction, input, output} JSONL, ready for
SFT via PEFT/QLoRA.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

ROOT = Path(__file__).resolve().parents[1]
COST = ROOT / "data" / "real_mospi_projects.csv"
TIME = ROOT / "data" / "real_mospi_time.csv"

# Curated, meaningful cause phrases from the real Flash Report 'reasons' column.
REAL_CAUSES = [
    "Delay in land acquisition and forest/environment clearances",
    "Slow progress of work due to contractor performance issues",
    "Delay in award of balance work / finalisation of contractor",
    "Litigation in court / arbitration over the project package",
    "Financing constraints and slow fund release",
    "Delay in statutory clearances and right-of-way approvals",
    "Geological/terrain challenges affecting construction progress",
    "Delay in finalisation of detailed design and drawings",
    "Slow progress of work by the state PWD / implementing agency",
    "Revised scope and additional works approved during execution",
]
LOW_RISK_CAUSES = [
    "Project on schedule with no material execution bottlenecks reported",
    "Minor schedule pressure only; no critical clearance or land issue flagged",
    "Planned procurement and construction milestones broadly being met",
]


def _flag(v):
    if v is None:
        return 0
    try:
        return int(float(str(v)))
    except Exception:
        return 0


def rule_based_scores(row, is_time: bool) -> dict:
    """Map a project row to risk features (as the model will see them at inference)."""
    orig = float(row.get("original_cost_cr") or 0)
    exp = float(row.get("expenditure_cum_cr") or 0)
    cor = float(row.get("cost_overrun_pct") or 0)
    exp_ratio = exp / orig if orig > 0 else 0.0

    cost_risk = min(1.0, max(0.0, cor / 200.0))
    # blend expenditure ratio as progress-based risk signal
    progress_risk = min(1.0, max(0.0, (exp_ratio - 0.5) / 2.0))
    cost_score = round(0.7 * cost_risk + 0.3 * progress_risk, 2)

    if is_time:
        tor = float(row.get("tor_months") or 0)
        delay_risk = min(1.0, tor / 60.0)
        planned = float(row.get("planned_duration_years") or 5)
        time_score = round(min(1.0, 0.6 * delay_risk + 0.4 * min(1.0, max(0.0, 1.0 - planned / 10.0))), 2)
    else:
        time_score = round(min(1.0, delay_risk if False else 0.0), 2)

    return {
        "sector": str(row.get("sector") or "Unknown"),
        "cost_overrun_pct": float(cor),
        "expenditure_ratio": round(exp_ratio, 3),
        "cost_risk": cost_score,
        "delay_risk": time_score,
    }


def _tier(score: float) -> str:
    if score >= 0.7:
        return "Critical"
    if score >= 0.45:
        return "High"
    if score >= 0.25:
        return "Medium"
    return "Low"


def build_explanation(feat: dict, cause: str, is_time: bool) -> str:
    sect = feat["sector"]
    cor = feat["cost_overrun_pct"]
    cost_risk = feat["cost_risk"]
    delay_risk = feat["delay_risk"]
    tier_c = _tier(cost_risk)
    tier_t = _tier(delay_risk)

    lines = []
    if is_time and delay_risk > 0.05:
        lines.append(
            f"The project in {sect} is rated {tier_t} delay risk with a cost overrun of "
            f"{cor:.1f}%. The primary exposure is schedule: {cause}."
        )
        if cost_risk > 0.45:
            lines.append(f"Cost pressure is also material, so both schedule and budget need joint attention.")
    elif cost_risk > 0.25:
        lines.append(
            f"This {sect} project shows {tier_c} cost risk with an overrun of {cor:.1f}%. "
            f"The main driver is cost escalation; {cause}."
        )
    else:
        lines.append(
            f"This {sect} project is currently low risk. {cause} No proactive intervention is required."
        )
    lines.append("Recommended action: review the flagged driver, re-baseline the affected milestone, and escalate if it persists for more than one review cycle.")
    return " ".join(lines)


def _num(out, count):
    out["instruction"] = "Explain the risk of the following infrastructure project in 2-3 sentences."
    out["input"] = json.dumps(feat, sort_keys=True)
    out["output"] = _


def generate(n_per_source: int = 2000, out_path: Path | None = None) -> list[dict]:
    cost = pd.read_csv(COST)
    time = pd.read_csv(TIME)
    delayed = time[time["delayed"] == 1]

    data = []
    # positives: delayed projects -> schedule-focused explanations
    for _, r in delayed.head(min(n_per_source, len(delayed))).iterrows():
        feat = rule_based_scores(r, is_time=True)
        cause = random.choice(REAL_CAUSES)
        data.append({
            "instruction": "Explain the risk of the following infrastructure project in 2-3 sentences.",
            "input": json.dumps(feat, sort_keys=True),
            "output": build_explanation(feat, cause, is_time=True),
        })
    # mixed: all cost rows + on-time time rows -> cost / low-risk explanations
    merged = cost if len(cost) > 0 else pd.DataFrame()
    for _, r in merged.head(min(n_per_source // 2, len(merged))).iterrows():
        feat = rule_based_scores(r, is_time=False)
        cause = LOW_RISK_CAUSES[0] if feat["cost_risk"] < 0.25 else random.choice(REAL_CAUSES)
        data.append({
            "instruction": "Explain the risk of the following infrastructure project in 2-3 sentences.",
            "input": json.dumps(feat, sort_keys=True),
            "output": build_explanation(feat, cause, is_time=False),
        })
    # on-time (not delayed) time rows -> low-risk
    on_time = time[time["delayed"] == 0]
    for _, r in on_time.head(min(n_per_source // 2, len(on_time))).iterrows():
        feat = rule_based_scores(r, is_time=True)
        feat["delay_risk"] = 0.05
        feat["cost_overrun_pct"] = 0.0
        data.append({
            "instruction": "Explain the risk of the following infrastructure project in 2-3 sentences.",
            "input": json.dumps(feat, sort_keys=True),
            "output": build_explanation(feat, LOW_RISK_CAUSES[0], is_time=True),
        })

    # dedup by (input, output)
    seen, uniq = set(), []
    for d in data:
        key = (d["input"], d["output"])
        if key not in seen:
            seen.add(key)
            uniq.append(d)

    if out_path is None:
        out_path = ROOT / "data" / "finetune_train.jsonl"
    with out_path.open("w", encoding="utf-8") as fh:
        for d in uniq:
            fh.write(json.dumps(d) + "\n")
    print(f"Wrote {len(uniq)} finetune examples -> {out_path}")
    return uniq


if __name__ == "__main__":
    generate()