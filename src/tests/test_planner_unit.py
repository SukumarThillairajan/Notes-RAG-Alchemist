# tests/test_planner_unit.py
# Run with:  python -m pytest -q tests/test_planner_unit.py

import os
import sys
from typing import Any, Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import planner.planner_agent as PA
from planner.planner_agent import PlannerConfig, plan_segment

class DummyRetriever:
    def __init__(self, scores: List[float]):
        self.scores = list(scores)
    def search(self, query_text: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        score = self.scores.pop(0) if self.scores else 0.0
        return [{"text": "x", "metadata": {}, "score": score}]

def test_normal_no_llm():
    cfg = PlannerConfig(index_name="idx", llm_enabled=False, tau_accept=0.7)
    r = DummyRetriever([0.71])  # >= tau_accept
    res = plan_segment("Basic Gauss law statement.", r, cfg)
    assert res["decision"] == "NORMAL"
    assert res["trace"] in ("Normal search", "Too short")

def test_refine_path_with_llm_stub(monkeypatch):
    cfg = PlannerConfig(index_name="idx", llm_enabled=True, tau_accept=0.83, tau_refine=0.68, tau_fail=0.45)
    r = DummyRetriever([0.60, 0.72])  # grey zone, then success after refine
    # Stub provider and availability
    monkeypatch.setenv(PA.PROVIDER_ENV, "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    # Stub the dispatcher to return a REFINE result
    monkeypatch.setattr(PA, "_call_llm_planner", lambda provider, sys, usr, model: "REFINE: gauss law electric flux derivation")
    res = plan_segment("Derive Gauss law relation between flux and enclosed charge.", r, cfg)
    assert res["decision"] == "REFINE"
    assert res["trace"] in ("Refine & retry", "Lowered tau")
    assert "gauss law" in (res["query_text"] or "").lower()

def test_external_path_with_llm_stub(monkeypatch):
    cfg = PlannerConfig(index_name="idx", llm_enabled=True, tau_accept=0.83, tau_refine=0.68, tau_fail=0.45)
    r = DummyRetriever([0.30])  # immediate LLM
    monkeypatch.setenv(PA.PROVIDER_ENV, "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setattr(PA, "_call_llm_planner", lambda provider, sys, usr, model: "EXTERNAL: Attention Is All You Need")
    res = plan_segment("Transformer attention head visualization tooling comparison", r, cfg)
    assert res["decision"] == "EXTERNAL"
    assert "Attention Is All You Need" in (res["suggested_book"] or "")

def test_malformed_llm_output_defaults_to_normal(monkeypatch):
    cfg = PlannerConfig(index_name="idx", llm_enabled=True)
    r = DummyRetriever([0.50])  # grey zone
    monkeypatch.setenv(PA.PROVIDER_ENV, "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    # Bad output (extra text) → parser should default to NORMAL
    monkeypatch.setattr(PA, "_call_llm_planner", lambda provider, sys, usr, model: "I think NORMAL is best because ...")
    res = plan_segment("Basic definition content", r, cfg)
    assert res["decision"] == "NORMAL"

def test_code_fence_guard(monkeypatch):
    cfg = PlannerConfig(index_name="idx", llm_enabled=True)
    r = DummyRetriever([0.50])
    monkeypatch.setenv(PA.PROVIDER_ENV, "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setattr(PA, "_call_llm_planner", lambda provider, sys, usr, model: "```\nREFINE: gauss law surface integral\n```")
    res = plan_segment("Flux and enclosed charge relation", r, cfg)
    assert res["decision"] in ("REFINE", "NORMAL")  # refine if second score passes; otherwise defaults
