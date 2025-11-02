"""Planner agent that decides retrieval strategies per lecture segment."""

from __future__ import annotations

import os
import hashlib
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from utils import pdf_utils

try:
    from workers.retriever import RetrieverWorker  # type: ignore
except Exception:  # pragma: no cover - fallback stub
    class RetrieverWorker:  # type: ignore
        """Fallback retriever stub for environments without the worker package."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise NotImplementedError("RetrieverWorker not available in this environment.")

        def search(self, query_text: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
            raise NotImplementedError("RetrieverWorker search unavailable.")


PROVIDER_ENV = "PLANNER_PROVIDER"
LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "planner_decisions.log"
MIN_SEGMENT_LENGTH = 32


DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    # use a dated Anthropic slug instead of "-latest"
    "anthropic": "claude-3-5-sonnet-20240620",
}

# Reasonable fallback candidates; order by preference
ANTHROPIC_FALLBACKS = [
    "claude-3-5-sonnet-20240620",
    "claude-3-5-haiku-20241022",
    "claude-3-opus-20240229",
    "claude-3-haiku-20240307",
]


def _resolve_model(provider: str, explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    if provider == "anthropic":
        return os.getenv("ANTHROPIC_MODEL") or DEFAULT_MODELS["anthropic"]
    if provider == "openai":
        return os.getenv("OPENAI_MODEL") or DEFAULT_MODELS["openai"]
    return DEFAULT_MODELS.get(provider, DEFAULT_MODELS["openai"])


@dataclass
class PlannerConfig:
    index_name: str
    top_k: int = 5
    tau_accept: float = 0.83
    tau_refine: float = 0.68
    tau_fail: float = 0.45
    adaptive_delta: float = 0.05
    max_refines: int = 6
    max_steps: int = 15
    llm_enabled: bool = True
    #provider: str = "openai"
    provider: str = "anthropic"
    model: Optional[str] = None
    prompt_path: str = "prompts/planner_prompt.txt"
    include_top_snippets: bool = True
    top_snippet_limit: int = 3
    noise_ratio: float = 0.0
    noisy_bias: bool = False


def _sha1_short(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:5]


def _ensure_log_dir() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def append_log(log_file: Path, content: str) -> None:
    """Append a content string to the specified log file."""
    _ensure_log_dir()
    with log_file.open("a", encoding="utf-8") as fh:
        fh.write(content + "\n")


def _append_log(segment_text: str, decision: str, trace: str, best_score: Optional[float]) -> None:
    _ensure_log_dir()
    seg_hash = _sha1_short(segment_text)
    score_str = "None" if best_score is None else f"{best_score:.4f}"
    line = f"[seg={seg_hash}] decision={decision} trace='{trace}' score={score_str}\n"
    with LOG_FILE.open("a", encoding="utf-8") as fh:
        fh.write(line)


def _load_prompt(path: str) -> Tuple[str, str]:
    default_system = "You are a planning assistant for academic retrieval decisions."
    default_user = (
        "Segment:\n\"\"\"{segment_text}\"\"\"\n"
        "Decide and output EXACTLY one of:\n"
        "  NORMAL\n"
        "  REFINE: <rewritten query>\n"
        "  EXTERNAL: <textbook title>"
    )

    prompt_file = Path(path)
    if not prompt_file.is_file():
        alt_path = Path(__file__).resolve().parent.parent / path
        if alt_path.is_file():
            prompt_file = alt_path
        else:
            raise FileNotFoundError(f"Planner prompt file not found at: {Path(path).resolve()} (also tried {alt_path})")

    content = prompt_file.read_text(encoding="utf-8").strip()
    if not content:
        return default_system, default_user

    system_prompt = default_system
    user_prompt = default_user

    upper = content.upper()
    if "SYSTEM:" in upper and "USER:" in upper:
        parts = content.split("USER:", 1)
        system_part = parts[0].split("SYSTEM:", 1)[-1].strip()
        user_part = parts[1].strip()
        if system_part:
            system_prompt = system_part
        if user_part:
            user_prompt = user_part
    else:
        user_prompt = content

    return system_prompt, user_prompt


def _llm_available(provider: str) -> bool:
    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            return False
        try:
            __import__("openai")
        except Exception:
            return False
        return True
    if provider == "anthropic":
        if not os.getenv("ANTHROPIC_API_KEY"):
            return False
        try:
            __import__("anthropic")
        except Exception:
            return False
        return True
    return False


def _call_llm_planner_openai(system_prompt: str, user_prompt: str, model: Optional[str]) -> str:
    try:
        from openai import OpenAI  # type: ignore
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError("OpenAI provider not available") from exc

    chosen_model = _resolve_model("openai", model)
    client = OpenAI()
    response = client.chat.completions.create(
        model=chosen_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )
    content = response.choices[0].message.content
    return content.strip() if content is not None else ""


def _call_llm_planner_anthropic(system_prompt: str, user_prompt: str, model: Optional[str]) -> str:
    try:
        import anthropic  # type: ignore
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError("Anthropic provider not available") from exc

    # Robust extractor to handle different response shapes from the Anthropic client
    def _extract_text(resp: Any) -> str:
        if resp is None:
            return ""
        # direct string
        if isinstance(resp, str):
            return resp.strip()
        # dict-like responses
        if isinstance(resp, dict):
            # common keys to check
            for key in ("completion", "text", "output", "content", "message", "response"):
                val = resp.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()
                if isinstance(val, list) and val:
                    first = val[0]
                    if isinstance(first, str) and first.strip():
                        return first.strip()
                    if isinstance(first, dict):
                        for k in ("text", "content", "message"):
                            if k in first and isinstance(first[k], str) and first[k].strip():
                                return first[k].strip()
            # support choices-like shape
            choices = resp.get("choices")
            if isinstance(choices, list) and choices:
                first = choices[0]
                if isinstance(first, dict):
                    for k in ("text", "message", "content", "completion"):
                        if k in first and isinstance(first[k], str) and first[k].strip():
                            return first[k].strip()
        # object with attributes
        for attr in ("content", "completion", "text", "output"):
            if hasattr(resp, attr):
                attrv = getattr(resp, attr)
                if isinstance(attrv, str) and attrv.strip():
                    return attrv.strip()
                if isinstance(attrv, list) and attrv:
                    first = attrv[0]
                    if isinstance(first, str) and first.strip():
                        return first.strip()
                    if not isinstance(first, str) and hasattr(first, "text"):
                        try:
                            return first.text.strip()
                        except Exception:
                            pass
                    if isinstance(first, dict):
                        for k in ("text", "content", "message"):
                            if k in first and isinstance(first[k], str) and first[k].strip():
                                return first[k].strip()
        return ""

    client = anthropic.Anthropic()
    # try the resolved model plus fallbacks
    candidates = []
    first = _resolve_model("anthropic", model)
    if first:
        candidates.append(first)
    for m in ANTHROPIC_FALLBACKS:
        if m not in candidates:
            candidates.append(m)

    last_err = None
    for m in candidates:
        try:
            response = client.messages.create(
                model=m,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                max_tokens=256,
                temperature=0,
            )
            return _extract_text(response)
        except anthropic.NotFoundError as e:
            last_err = e
            continue
    raise RuntimeError(
        f"Anthropic model not found. Tried: {candidates}. Set PlannerConfig.model or ANTHROPIC_MODEL."
    ) from last_err


PROVIDERS: Dict[str, Callable[[str, str, Optional[str]], str]] = {
    "openai": _call_llm_planner_openai,
    "anthropic": _call_llm_planner_anthropic,
}


def _call_llm_planner(provider: str, system_prompt: str, user_prompt: str, model: Optional[str]) -> str:
    func = PROVIDERS.get(provider)
    if not func:
        raise RuntimeError(f"Unknown provider: {provider}")
    return func(system_prompt, user_prompt, model)


'''def _parse_llm_output(raw: str) -> Tuple[str, Optional[str]]:
    if not raw:
        return "NORMAL", None
    text = raw.strip()
    upper = text.upper()

    if upper == "NORMAL":
        return "NORMAL", None

    if upper.startswith("REFINE"):
        payload = ""
        if ":" in text:
            payload = text.split(":", 1)[1].strip()
        elif " " in text:
            payload = text.split(" ", 1)[1].strip()
        return "REFINE", payload or None

    if upper.startswith("EXTERNAL"):
        payload = ""
        if ":" in text:
            payload = text.split(":", 1)[1].strip()
        elif " " in text:
            payload = text.split(" ", 1)[1].strip()
        return "EXTERNAL", payload or None

    return "NORMAL", None'''

def _parse_llm_output(raw: str) -> Tuple[str, Optional[str]]:
    # Guard: first non-empty line only, strip code-fences/quotes
    if not raw:
        return "NORMAL", None
    # keep just the first non-empty line
    line = next((ln for ln in raw.splitlines() if ln.strip()), "").strip()
    # strip common wrappers
    for fence in ("```", "'''", "“", "”", '"', "'"):
        if line.startswith(fence) and line.endswith(fence):
            line = line.strip(fence).strip()
    text = line
    upper = text.upper()

    if upper == "NORMAL":
        return "NORMAL", None

    if upper.startswith("REFINE"):
        payload = ""
        if ":" in text:
            payload = text.split(":", 1)[1].strip()
        elif " " in text:
            payload = text.split(" ", 1)[1].strip()
        return "REFINE", payload or None

    if upper.startswith("EXTERNAL"):
        payload = ""
        if ":" in text:
            payload = text.split(":", 1)[1].strip()
        elif " " in text:
            payload = text.split(" ", 1)[1].strip()
        return "EXTERNAL", payload or None

    return "NORMAL", None


def _resolve_provider(cfg: PlannerConfig) -> str:
    env_provider = os.getenv(PROVIDER_ENV)
    base = cfg.provider or env_provider or "openai"
    if env_provider:
        base = env_provider
    return base.lower()


def plan_segment(segment_text: str, retriever: Any, cfg: PlannerConfig) -> Dict[str, Any]:
    segment_text = segment_text or ""
    stripped = segment_text.strip()
    result: Dict[str, Any] = {
        "segment_text": segment_text,
        "decision": "NORMAL",
        "query_text": None,
        "suggested_book": None,
        "trace": "Normal search",
        "best_score": None,
        "top_snippets": "",
    }

    try:
        if len(stripped) < MIN_SEGMENT_LENGTH:
            result.update({"query_text": stripped or segment_text, "trace": "Too short"})
            _append_log(segment_text, result["decision"], result["trace"], None)
            return result

        provider = _resolve_provider(cfg)
        llm_usable = cfg.llm_enabled and _llm_available(provider)
        system_prompt: Optional[str] = None
        user_template: Optional[str] = None

        def ensure_prompt() -> Tuple[str, str]:
            nonlocal system_prompt, user_template
            if system_prompt is None or user_template is None:
                system_prompt, user_template = _load_prompt(cfg.prompt_path)
            return system_prompt, user_template  # type: ignore

        query_text = segment_text
        # Bias retrieval if segment mentions a figure
        if "fig" in query_text.lower():
            query_text += " figure"

        best_score: Optional[float] = None
        adaptive_used = False
        adjusted_tau_refine = cfg.tau_refine
        steps_taken = 0
        refine_attempts = 0

        try:
            initial_results = retriever.search(query_text, top_k=cfg.top_k)
            steps_taken += 1
        except Exception:
            result.update({"query_text": query_text, "trace": "Retriever error"})
            _append_log(segment_text, result["decision"], result["trace"], None)
            return result

        # Build evidence block if enabled
        top_snippets_block = ""
        if getattr(cfg, "include_top_snippets", True):
            try:
                limit = getattr(cfg, "top_snippet_limit", 3)
                top_snippets_block = retriever.format_top_snippets(initial_results, limit=limit)
            except Exception:
                top_snippets_block = ""
        result["top_snippets"] = top_snippets_block

        top_score = 0.0
        if initial_results:
            raw_score = initial_results[0].get("score", 0.0)
            try:
                top_score = float(raw_score)
            except (TypeError, ValueError):
                top_score = 0.0
        best_score = top_score
        result["best_score"] = best_score
        result["query_text"] = query_text

        if best_score >= cfg.tau_accept:
            result["trace"] = "Normal search"
            _append_log(segment_text, result["decision"], result["trace"], best_score)
            return result

        def call_llm(current_segment: str) -> Tuple[str, Optional[str]]:
            nonlocal llm_usable
            if not llm_usable:
                raise RuntimeError("LLM unavailable")
            sys_prompt, usr_template = ensure_prompt()
            user_prompt = usr_template.format(segment_text=current_segment, top_snippets=top_snippets_block or "")
            try:
                raw = _call_llm_planner(provider, sys_prompt, user_prompt, cfg.model)
            except Exception:
                llm_usable = False
                raise
            return _parse_llm_output(raw)

        def run_refine(new_query: str) -> Tuple[Optional[float], List[Dict[str, Any]]]:
            nonlocal steps_taken
            if steps_taken >= cfg.max_steps:
                return None, []
            try:
                results = retriever.search(new_query, top_k=cfg.top_k)
                steps_taken += 1
            except Exception:
                return None, []
            if not results:
                return 0.0, results
            top = results[0].get("score", 0.0)
            try:
                return float(top), results
            except (TypeError, ValueError):
                return 0.0, results

        if best_score <= cfg.tau_fail:
            if llm_usable:
                action, payload = call_llm(segment_text)
                if action == "REFINE" and payload:
                    new_score, _ = run_refine(payload)
                    if new_score is not None:
                        result["query_text"] = payload
                        result["best_score"] = new_score
                        if new_score >= adjusted_tau_refine:
                            result["decision"] = "REFINE"
                            result["trace"] = "Refine & retry"
                            _append_log(segment_text, result["decision"], result["trace"], new_score)
                            return result
                        if not adaptive_used:
                            adjusted_tau_refine = max(0.0, adjusted_tau_refine - cfg.adaptive_delta)
                            adaptive_used = True
                            if new_score >= adjusted_tau_refine:
                                result["decision"] = "REFINE"
                                result["trace"] = "Lowered tau"
                                _append_log(segment_text, result["decision"], result["trace"], new_score)
                                return result
                        best_score = new_score
                        result["trace"] = "Stop: budget reached"
                        _append_log(segment_text, result["decision"], result["trace"], new_score)
                        return result
                    result["trace"] = "Stop: budget reached"
                    _append_log(segment_text, result["decision"], result["trace"], best_score)
                    return result
                elif action == "EXTERNAL" and payload:
                    result.update({"decision": "EXTERNAL", "suggested_book": payload, "trace": "External book suggested"})
                    _append_log(segment_text, result["decision"], result["trace"], best_score)
                    return result
                elif action == "NORMAL":
                    result["trace"] = "LLM: normal"
                    _append_log(segment_text, result["decision"], result["trace"], best_score)
                    return result
                result["trace"] = "Stop: budget reached"
                _append_log(segment_text, result["decision"], result["trace"], best_score)
                return result
            result["trace"] = "LLM unavailable" if cfg.llm_enabled else "LLM disabled"
            _append_log(segment_text, result["decision"], result["trace"], best_score)
            return result

        if best_score < cfg.tau_accept and best_score > cfg.tau_fail and not llm_usable:
            if not adaptive_used:
                adjusted_tau_refine = max(0.0, adjusted_tau_refine - cfg.adaptive_delta)
                adaptive_used = True
                if best_score >= adjusted_tau_refine:
                    result["trace"] = "Lowered tau"
                    result["best_score"] = best_score
                    _append_log(segment_text, result["decision"], result["trace"], best_score)
                    return result
            result["trace"] = "LLM disabled" if not cfg.llm_enabled else "LLM unavailable"
            _append_log(segment_text, result["decision"], result["trace"], best_score)
            return result

        while best_score < cfg.tau_accept and best_score > cfg.tau_fail and steps_taken < cfg.max_steps and refine_attempts < cfg.max_refines:
            refine_attempts += 1

            if llm_usable:
                action, payload = call_llm(segment_text)
            else:
                action, payload = "NORMAL", None

            if action == "NORMAL":
                result["trace"] = "LLM: normal" if llm_usable else "LLM unavailable"
                _append_log(segment_text, result["decision"], result["trace"], best_score)
                return result

            if action == "EXTERNAL" and payload:
                result.update({"decision": "EXTERNAL", "suggested_book": payload, "trace": "External book suggested"})
                _append_log(segment_text, result["decision"], result["trace"], best_score)
                return result

            if action == "REFINE" and payload:
                new_score, _ = run_refine(payload)
                if new_score is None:
                    break
                result["query_text"] = payload
                result["best_score"] = new_score
                if new_score >= adjusted_tau_refine:
                    result["decision"] = "REFINE"
                    result["trace"] = "Refine & retry"
                    _append_log(segment_text, result["decision"], result["trace"], new_score)
                    return result
                if not adaptive_used:
                    adjusted_tau_refine = max(0.0, adjusted_tau_refine - cfg.adaptive_delta)
                    adaptive_used = True
                    if new_score >= adjusted_tau_refine:
                        result["decision"] = "REFINE"
                        result["trace"] = "Lowered tau"
                        _append_log(segment_text, result["decision"], result["trace"], new_score)
                        return result
                best_score = new_score
                continue

            result["trace"] = "LLM: normal" if llm_usable else "LLM unavailable"
            _append_log(segment_text, result["decision"], result["trace"], best_score)
            return result

        result["trace"] = "Stop: budget reached"
        _append_log(segment_text, result["decision"], result["trace"], best_score)
        return result

    except Exception as exc:
        trace_text = traceback.format_exc().strip()
        last_line = trace_text.splitlines()[-1] if trace_text else f"{exc.__class__.__name__}: {exc}"
        try:
            _append_log(segment_text, "ERROR", last_line, result.get("best_score"))
        except Exception:
            pass
        raise


def process_lecture(lecture_text: str, cfg: PlannerConfig) -> List[Dict[str, Any]]:
    if not lecture_text:
        return []
    segments = pdf_utils.split_text(
        text=lecture_text,
        max_length=480,
        overlap=40,
        min_chunk_len=200,
    )
    try:
        retriever = RetrieverWorker(cfg.index_name, top_k=cfg.top_k)
    except Exception as exc:
        raise RuntimeError("Failed to initialize RetrieverWorker") from exc

    results: List[Dict[str, Any]] = []
    for segment in segments:
        decision = plan_segment(segment, retriever, cfg)
        results.append(decision)
    return results


if __name__ == "__main__":
    class DummyRetriever:
        def __init__(self, responses: Optional[List[Tuple[str, float]]] = None) -> None:
            self.responses = responses or []

        def search(self, query_text: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
            if not self.responses:
                return []
            text, score = self.responses.pop(0)
            return [{"text": text, "metadata": {}, "score": score}]

    mock_segments = [
        "Quick summary of lecture on data structures.",
        "Short.",
    ]
    dummy_retriever = DummyRetriever(responses=[("Match content", 0.81), ("Match short", 0.3)])
    config = PlannerConfig(index_name="demo-index", llm_enabled=False)
    for seg in mock_segments:
        decision = plan_segment(seg, dummy_retriever, config)
        print(f"Segment decision: {decision['decision']} | Trace: {decision['trace']}")
