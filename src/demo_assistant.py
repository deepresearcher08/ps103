"""LLM Assistant demo — runs the demo questions through the live assistant.

Purpose: practise / record the value proposition in minutes. Each question goes
through the REAL pipeline: Ollama LLM selects a tool (function-calling), the tool
runs against the real MoSPI data, and the answer is returned. Nothing is canned.

Usage:
    python -m src.demo_assistant            # default questions
    python -m src.demo_assistant --all      # also show raw tool call + timing

Requires Ollama running with llama3.1:8b pulled (see README).
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from src.assistant import _llm_choose_tool, answer, build_assistant_view
from src.dashboard import build_view, load_time

DEMO_QUESTIONS = [
    "Give me a portfolio overview — how many projects, and what is the average cost overrun?",
    "Which sector has the worst cost overrun?",
    "Show me the top 5 projects at risk right now",
    "Tell me about a railway project and why it is delayed",
    "How many projects are in critical delay risk?",
]


def main(show_raw: bool = False):
    view = build_view()
    time_df = load_time()
    ctx = build_assistant_view(view, time_df)
    print("=" * 74)
    print(" MoSPI LLM project-intelligence demo (live tool-calling on real data)")
    print("=" * 74)
    for q in DEMO_QUESTIONS:
        print(f"\nQ: {q}")
        t0 = time.time()
        tc = _llm_choose_tool(q)
        text, table = answer(q, ctx)
        dt = time.time() - t0
        print(f"  [LLM tool call] {tc}   ({dt:.1f}s)")
        if show_raw:
            print(f"  [answer]\n{text}")
        elif table is not None:
            print(f"  [answer] {text[:160].strip()}...  (+ table {len(table)} rows)")
        else:
            print(f"  [answer] {text[:160].strip()}...")
        print("-" * 74)
    print("\nDone. If any tool call printed 'None', Ollama may be down — "
          "the deterministic fallback answered instead (still real data, no LLM routing).")


if __name__ == "__main__":
    main(show_raw="--all" in sys.argv)