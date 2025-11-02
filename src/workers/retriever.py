"""Retriever worker that wraps vector store search for planner/compiler use."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple
import importlib
from utils.vector_store import search_vector_store


def _norm_text(text: str) -> str:
    """Normalize text for deduplication by collapsing whitespace."""
    return " ".join((text or "").split())


def _safe_score(value: Any) -> float:
    """Convert a score value to float, falling back to 0.0 on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _extract_fields(metadata: Dict[str, Any]) -> Tuple[str, Optional[Any], Optional[List[Any]], Optional[List[Any]], Optional[str]]:
    """Extract citation-related fields from metadata."""
    meta = metadata or {}
    book = meta.get("book") or meta.get("title") or "Unknown Book"

    raw_pages = meta.get("pages")
    pages_list: Optional[List[Any]] = None
    if raw_pages is not None:
        if isinstance(raw_pages, (list, tuple)):
            pages_list = list(raw_pages)
        else:
            pages_list = [raw_pages]

    page = meta.get("page")
    if page is None:
        page = meta.get("page_number")
    if page is None and pages_list:
        page = pages_list[0]
    if page is not None and pages_list is None:
        pages_list = [page]

    figures = meta.get("figure_nums") or meta.get("figures") or meta.get("figure")
    figure_list: Optional[List[Any]] = None
    if figures is not None:
        if isinstance(figures, (list, tuple)):
            figure_list = list(figures)
        else:
            figure_list = [figures]

    chunk_id = meta.get("chunk_id") or meta.get("id")

    return book, page, pages_list, figure_list, chunk_id


def _dedupe_by_text(excerpts: list[dict], min_unique: int = 3) -> list[dict]:
    """Deduplicate excerpts based on book, page, and the start of the text."""
    seen = set()
    out = []
    for ex in excerpts:
        key = (ex.get("book"), ex.get("page"), (ex.get("text") or "")[:200])
        if key in seen:
            continue
        seen.add(key)
        out.append(ex)
        if len(out) >= min_unique and len(out) >= 5:  # cap if long
            break
    return out


def _stitch_consecutive(excerpts: list[dict]) -> list[dict]:
    """Stitch together excerpts from the same book and page."""
    if not excerpts:
        return []
    stitched = []
    prev = excerpts[0].copy()
    for cur in excerpts[1:]:
        if cur.get("book") == prev.get("book") and cur.get("page") == prev.get("page"):
            prev["text"] = (prev.get("text") or "").rstrip() + " " + (cur.get("text") or "").lstrip()
            prev["score"] = max(prev.get("score", 0), cur.get("score", 0))
        else:
            stitched.append(prev)
            prev = cur.copy()
    stitched.append(prev)
    return stitched


class RetrieverWorker:
    def __init__(
        self,
        index_name: str,
        namespace: str | None = None,
        *,
        top_k: int = 5,
        min_score: float = 0.55,
        dedup: bool = True,
    ) -> None:
        self.index_name = index_name
        self.namespace = namespace
        self.top_k = int(top_k)
        self.min_score = float(min_score)
        self.dedup = dedup

    def search(self, query_text: str, top_k: Optional[int] = None) -> List[Dict]:
        """Search the vector store and normalize results."""
        if not query_text:
            return []
        k = top_k if top_k is not None else self.top_k
        if k <= 0:
            return []

        kwargs: Dict[str, Any] = {
            "query_text": query_text,
            "index_name": self.index_name,
            "top_k": k,
        }
        if self.namespace is not None:
            kwargs["namespace"] = self.namespace

        try:
            raw_results = search_vector_store(**kwargs)
        except TypeError as exc:
            if "namespace" in kwargs:
                kwargs.pop("namespace")
                self.namespace = None
                raw_results = search_vector_store(**kwargs)
            else:
                raise RuntimeError(f"Pinecone query failed: {exc}") from exc
        except Exception as exc:
            raise RuntimeError(f"Pinecone query failed: {exc}") from exc

        normalized_results = self._normalize_results(raw_results)
        return normalized_results[:k]

    def retrieve_excerpts(self, query_text: str, top_k: int = 7) -> List[Dict]:
        """Retrieve excerpts formatted for presentation."""
        if top_k <= 0:
            return []
        # Fetch more candidates (top_k=7), then stitch and dedupe.
        matches = self.search(query_text, top_k=top_k)
        if not matches:
            return []
        stitched = _stitch_consecutive(matches)
        return _dedupe_by_text(stitched)

    def retrieve_for_segments(self, segment_plans: List[Dict], *, attempt_on_external: bool = False, top_k: int = 3) -> List[Dict]:
        """Populate plans with excerpts based on segment decisions."""
        enriched_plans: List[Dict] = []
        for plan in segment_plans:
            decision = plan.get("decision")
            query_text = plan.get("query_text") or plan.get("segment_text") or ""

            if decision == "EXTERNAL" and not attempt_on_external:
                plan["excerpts"] = []
                plan["no_match"] = True
                plan["top_snippets"] = ""
                enriched_plans.append(plan)
                continue

            # Handle cases where planner errors result in a None score
            best_score = plan.get("best_score")
            if best_score is None or best_score < self.min_score:
                plan["decision"] = "EXTERNAL"
                plan["excerpts"] = []
                plan["no_match"] = True
                plan["top_snippets"] = ""
                enriched_plans.append(plan)
                continue

            matches = self.retrieve_excerpts(query_text or "", top_k=top_k)
            if not matches:
                plan["excerpts"] = []
                plan["no_match"] = True
                plan["top_snippets"] = ""
                if plan.get("decision") == "NORMAL" and plan.get("noisy_bias"):
                    plan["decision"] = "REFINE"
                    trace = plan.get("trace") or ""
                    suffix = "no-match→refine"
                    plan["trace"] = f"{trace} | {suffix}" if trace else suffix
                enriched_plans.append(plan)
                continue

            plan["excerpts"] = matches
            plan["no_match"] = False
            plan["top_snippets"] = self.format_top_snippets(matches, limit=top_k)
            enriched_plans.append(plan)
        return enriched_plans

    def format_top_snippets(self, matches: List[Dict], limit: int = 3) -> str:
        """Format a short multi-line snippet summary for LLM prompts."""
        lines = []
        for match in matches[:limit]:
            score = match.get("score", 0.0)
            book = match.get("book", "Unknown Book")
            snippet = match.get("text", "")
            snippet = snippet[:140].replace("\n", " ")
            lines.append(f"- score={score:.2f} | title={book} | snippet={snippet}")
        return "\n".join(lines)

    def _normalize_results(self, results: List[Dict]) -> List[Dict]:
        """Normalize and deduplicate search results."""
        normalized: List[Dict] = []
        seen_texts = set()

        for result in sorted(results, key=lambda x: _safe_score(x.get("score")), reverse=True):
            score = _safe_score(result.get("score"))

            if score is None or score < self.min_score:
                continue

            text = result.get("text") or ""
            norm_text = _norm_text(text)
            if self.dedup and norm_text in seen_texts:
                continue
            if self.dedup:
                seen_texts.add(norm_text)

            metadata = result.get("metadata") or {}
            book, page, pages, figures, chunk_id = _extract_fields(metadata)

            excerpt = {
                "text": text,
                "book": book,
                "page": page,
                "pages": list(pages) if pages else None,
                "figure_nums": list(figures) if figures else None,
                "chunk_id": chunk_id,
                "score": score,
                "metadata": metadata,
            }
            normalized.append(excerpt)

        if len(normalized) <= 1:
            return normalized

        return _stitch_consecutive(normalized)

    def _stitch_excerpts(self, excerpts: List[Dict]) -> List[Dict]:
        """Merge adjacent excerpts from the same book and consecutive pages."""
        stitched: List[Dict] = []
        previous: Optional[Dict] = None

        for current in excerpts:
            if previous is None:
                previous = current
                continue

            same_book = previous.get("book") == current.get("book")
            prev_page = previous.get("page")
            curr_page = current.get("page")

            if (
                same_book
                and isinstance(prev_page, int)
                and isinstance(curr_page, int)
                and curr_page - prev_page == 1
            ):
                combined_text = f"{previous.get('text', '').rstrip()}\n\n{current.get('text', '').lstrip()}"
                previous["text"] = combined_text.strip()
                previous["score"] = max(previous.get("score", 0.0), current.get("score", 0.0))

                prev_pages_list = list(previous.get("pages") or ([] if prev_page is None else [prev_page]))
                curr_pages_list = list(current.get("pages") or ([] if curr_page is None else [curr_page]))
                united_pages: List[Any] = []
                for val in prev_pages_list + curr_pages_list:
                    if val not in united_pages:
                        united_pages.append(val)
                previous["pages"] = united_pages or None
                if isinstance(prev_page, int) and isinstance(curr_page, int):
                    previous["page"] = min(prev_page, curr_page)
                elif prev_page is None and curr_page is not None:
                    previous["page"] = curr_page

                prev_figs_list = list(previous.get("figure_nums") or [])
                curr_figs_list = list(current.get("figure_nums") or [])
                combined_figs: List[Any] = []
                for val in prev_figs_list + curr_figs_list:
                    if val not in combined_figs:
                        combined_figs.append(val)
                previous["figure_nums"] = combined_figs or None
                continue

            stitched.append(previous)
            previous = current

        if previous is not None:
            stitched.append(previous)

        return stitched


if __name__ == "__main__":
    if not os.getenv("PINECONE_API_KEY"):
        print("PINECONE_API_KEY not set. RetrieverWorker dry-run skipped.")
    else:
        worker = RetrieverWorker(index_name="demo-index", namespace="default")
        try:
            sample_results = worker.search("test query")
            print(f"Retrieved {len(sample_results)} matches.")
        except Exception as e:
            print(f"RetrieverWorker encountered an error: {e}")
