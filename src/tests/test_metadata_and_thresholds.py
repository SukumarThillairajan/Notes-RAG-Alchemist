import sys
from pathlib import Path

# Ensure the project root is on the Python path to allow importing from utils/workers
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from workers.compiler import format_citation, _normalize_page_fields
    from workers.retriever import RetrieverWorker
except ImportError as e:
    print(f"Could not import required modules: {e}", file=sys.stderr)
    print("Please ensure the 'workers' package is available.", file=sys.stderr)
    sys.exit(1)


def test_page_metadata_normalization_and_citation():
    """
    Tests that page number metadata is correctly normalized from strings to integers
    and that the citation formatting reflects this.
    """
    # 1. Test with a single string page number
    page_int_1, pages_list_1 = _normalize_page_fields(page="10", pages=None)
    assert page_int_1 == 10
    assert pages_list_1 == []
    citation_1 = format_citation(book="Book A", page="10")
    assert citation_1 == "(Source: Book A, p. 10)"

    # 2. Test with a list of string page numbers
    page_int_2, pages_list_2 = _normalize_page_fields(page=None, pages=["11", "12", "invalid"])
    assert page_int_2 is None
    assert pages_list_2 == [11, 12]  # Should be sorted list of ints, invalid dropped
    citation_2 = format_citation(book="Book B", pages=["12", "11"]) # Unordered
    assert citation_2 == "(Source: Book B, pp. 11–12)"

    # 3. Test with None for page number
    page_int_3, pages_list_3 = _normalize_page_fields(page=None, pages=None)
    assert page_int_3 is None
    assert pages_list_3 == []
    citation_3 = format_citation(book="Book C", page=None)
    assert citation_3 == "(Source: Book C)"


def test_retriever_min_score_threshold():
    """
    Tests that the RetrieverWorker correctly filters out excerpts that
    are below its `min_score` threshold.
    """
    # This is a mock of the raw results from the vector store
    mock_search_results = [
        {"text": "a", "metadata": {"book": "B"}, "score": 0.60},
        {"text": "b", "metadata": {"book": "B"}, "score": 0.59},
        {"text": "c", "metadata": {"book": "B"}, "score": 0.44},
    ]

    # Case 1: min_score is 0.55, so the 0.44 result should be dropped.
    worker_high_threshold = RetrieverWorker(index_name="dummy", min_score=0.55)
    normalized_results_1 = worker_high_threshold._normalize_results(mock_search_results)

    assert len(normalized_results_1) == 2
    assert all(r["score"] >= 0.55 for r in normalized_results_1)
    assert not any(r["text"] == "c" for r in normalized_results_1)

    # Case 2: min_score is 0.40, so all results should be included.
    worker_low_threshold = RetrieverWorker(index_name="dummy", min_score=0.40)
    normalized_results_2 = worker_low_threshold._normalize_results(mock_search_results)

    assert len(normalized_results_2) == 3
    assert any(r["text"] == "c" for r in normalized_results_2)
