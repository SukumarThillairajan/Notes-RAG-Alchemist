# tests/test_retriever_unit.py
import sys
import types
from pathlib import Path

# Ensure we can import retriever.py sitting beside this test file
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# --- 1) Fake vector_store module that retriever.py expects -------------------
fake_vs = types.ModuleType("vector_store")

def search_vector_store(query_text: str, *, index_name: str, top_k: int = 5):
    # Return deterministic, mixed-quality matches; include dup text to test dedup;
    # include consecutive pages to test stitching.
    return [
        {
            "id": "A-10",
            "text": "Oblique shock: theta-beta-M relation derived here.",
            "metadata": {"book": "Shapiro Vol.1", "page": 10, "chunk_id": "A-10"},
            "score": 0.86,
        },
        {
            "id": "A-11",
            "text": "…continuation of theta-beta-M derivation with examples.",
            "metadata": {"book": "Shapiro Vol.1", "page": 11, "chunk_id": "A-11"},
            "score": 0.84,
        },
        {
            "id": "B-42",
            "text": "A duplicate line to check deduplication    ",
            "metadata": {"book": "Anderson", "page": 42, "chunk_id": "B-42"},
            "score": 0.83,
        },
        {
            "id": "B-43",
            "text": "A duplicate line to   check   deduplication",  # same text, extra spaces
            "metadata": {"book": "Anderson", "page": 43, "chunk_id": "B-43"},
            "score": 0.82,
        },
        {
            "id": "C-99",
            "text": "Very low relevance filler.",
            "metadata": {"book": "Random Book", "page": 99, "chunk_id": "C-99"},
            "score": 0.07,
        },
    ]

setattr(fake_vs, "search_vector_store", search_vector_store)
sys.modules["vector_store"] = fake_vs  # retriever.py imports this name

# --- 2) Now import the real retriever worker ---------------------------------
from workers.retriever import RetrieverWorker

def test_retriever_normalize_and_stitch():
    worker = RetrieverWorker(index_name="dummy-index", top_k=5, min_score=0.30, dedup=True)

    # Run the high-level call
    results = worker.retrieve_excerpts("oblique shock", top_k=5)

    # Should drop the 0.07 score, dedup identical texts, and stitch consecutive 10->11
    assert len(results) == 3 or len(results) == 2  # stitch may reduce count further

    # Find Shapiro merged entry
    shapiro = [r for r in results if r["book"] == "Shapiro Vol.1"][0]
    # It should include combined text and set page to min page (10)
    assert shapiro["page"] == 10
    assert "theta-beta-M relation" in shapiro["text"]
    assert "continuation" in shapiro["text"]
    # Pages list should include both 10 and 11
    assert set(shapiro.get("pages") or []) >= {10, 11}

    # Dedup check: the two Anderson entries are the same text after whitespace norm → only one remains
    anderson = [r for r in results if r["book"] == "Anderson"]
    assert len(anderson) == 1

    # Scores are floats
    for r in results:
        assert isinstance(r["score"], float)

    print("Unit test passed with stitched + deduped results:")
    for r in results:
        print(f"- {r['book']} p.{r['page']} score={r['score']:.2f} :: {r['text'][:70]}...")

if __name__ == "__main__":
    test_retriever_normalize_and_stitch()
