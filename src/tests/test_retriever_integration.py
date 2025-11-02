# tests/test_retriever_integration.py
import os, sys
from pathlib import Path

# at the very top
from dotenv import load_dotenv
load_dotenv()  # reads .env from repo root


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.vector_store import init_vector_store, store_documents_in_index as index_documents
from workers.retriever import RetrieverWorker

INDEX = "demo-retriever-index"

def main():
    # 1) Tiny “book” corpus
    docs = [
        "Oblique shock derivation: theta-beta-M relation is derived here.",
        "Continuation: examples for compression corners and attached shocks.",
        "Irrelevant note about gardening.",
    ]
    metas = [
        {"book": "Shapiro Vol.1", "page": 10, "chunk_id": "S-10"},
        {"book": "Shapiro Vol.1", "page": 11, "chunk_id": "S-11"},
        {"book": "Random Book", "page": 99, "chunk_id": "R-99"},
    ]

    # 2) Build index
    init_vector_store(INDEX)  # uses your default EMBED_DIM under the hood
    index_documents(docs, metas, index_name=INDEX)

    # 3) Query via RetrieverWorker
    worker = RetrieverWorker(index_name=INDEX, top_k=5, min_score=0.30, dedup=True)
    matches = worker.retrieve_excerpts("theta beta M oblique shock", top_k=5)

    print(f"Got {len(matches)} matches")
    for m in matches:
        pages = m.get("pages") or [m.get("page")]
        print(f"- {m['book']} pages={pages} score={m['score']:.2f} :: {m['text'][:80]}")

    # Basic asserts
    assert any(m["book"] == "Shapiro Vol.1" for m in matches)
    assert all(m["score"] >= 0.30 for m in matches)

if __name__ == "__main__":
    # Requires env vars; will error cleanly if missing
    need = ["PINECONE_API_KEY"]
    missing = [k for k in need if not os.getenv(k)]
    if missing:
        print(f"Skipping integration test; missing env: {missing}")
    else:
        main()
