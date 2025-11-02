# tests/test_planner_live.py
# Run with:  OPENAI_API_KEY=... python tests/test_planner_live.py
# Optional:  set PLANNER_PROVIDER=anthropic and ANTHROPIC_API_KEY=... to test Claude.

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from planner.planner_agent import PlannerConfig, plan_segment

class DummyLowRetriever:
    """Always returns a single low/mid score to force LLM escalation."""
    def __init__(self, score: float = 0.60) -> None:
        self.score = score
    def search(self, query_text: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        return [{"text": "dummy", "metadata": {}, "score": self.score}]

def main():
    provider = (os.getenv("PLANNER_PROVIDER") or "openai").lower()
    using_openai = provider == "openai" and bool(os.getenv("OPENAI_API_KEY"))
    using_anthropic = provider == "anthropic" and bool(os.getenv("ANTHROPIC_API_KEY"))

    print(f"Provider = {provider} | OpenAI? {using_openai} | Anthropic? {using_anthropic}")
    if not (using_openai or using_anthropic):
        print("No API key detected; please set OPENAI_API_KEY or ANTHROPIC_API_KEY.")
        print("You can still run unit tests that mock the LLM (see test_planner_unit.py).")
        return

    cfg = PlannerConfig(
        index_name="demo-index",
        top_k=5,
        tau_accept=0.83,
        tau_refine=0.68,
        tau_fail=0.45,
        adaptive_delta=0.05,
        max_refines=1,
        max_steps=3,
        llm_enabled=True,
        provider=provider,  # openai by default
        prompt_path="prompts/planner_prompt.txt"
    )

    # Diverse segments to probe NORMAL / REFINE / EXTERNAL tendencies
    segments = [
        "Definition of Gauss's Law and simple closed-surface flux example.",
        "Derive Prandtl–Meyer expansion fan relations and compute nu(M) for M>1.",
        "Compare attention head visualization libraries and summarize trade-offs.",
        "Explain the area–Mach relation for quasi-1D compressible nozzle flow.",
        "State the Rankine–Hugoniot conditions and derive jump in total enthalpy.",
        "Transformer MHA interpretability benchmarks across toolkits (AAN/Neel Nanda).",
    ]

    # Set scores to induce different branches:
    #   0.60 (grey zone → LLM), 0.30 (fail → LLM), 0.75 (grey zone → LLM)
    scores = [0.60, 0.30, 0.75, 0.60, 0.30, 0.75]
    retrievers = [DummyLowRetriever(s) for s in scores]

    print("\n=== LIVE LLM PLANNER RUN ===")
    for seg, r in zip(segments, retrievers):
        res = plan_segment(seg, r, cfg)
        print(f"- SEG: {seg[:60]}...")
        print(f"  decision={res['decision']}  trace={res['trace']}  best={res['best_score']}")
        if res.get("query_text"):
            print(f"  query_text={res['query_text']}")
        if res.get("suggested_book"):
            print(f"  suggested_book={res['suggested_book']}")
        print()

if __name__ == "__main__":
    main()
