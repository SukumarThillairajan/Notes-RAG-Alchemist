from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from utils.embeddings import EMBED_DIM, batch_get_embeddings, get_embedding
from utils.pinecone_compat import ensure_index, get_client, query as pc_query, upsert as pc_upsert

_client: Any = None
_cached_index_name: Optional[str] = None
_cached_index_handle: Any = None

def _normalize_page_fields(meta: dict) -> dict:
    m = dict(meta or {})
    # page
    if "page" in m:
        try:
            m["page"] = int(m["page"])
        except Exception:
            m["page"] = None
    # pages
    if "pages" in m and m["pages"] is not None:
        vals = []
        for v in (m["pages"] if isinstance(m["pages"], (list, tuple)) else [m["pages"]]):
            try:
                vals.append(int(v))
            except Exception:
                continue
        m["pages"] = [str(v) for v in sorted(set(vals))]
    return m


def init_vector_store(index_name: str, embedding_dim: int = EMBED_DIM) -> Any:
    """
    Ensure Pinecone index exists (v2/v3) and cache the handle for reuse.
    Embedding dim defaults to OpenAI text-embedding-3-large (3072).
    """
    global _client, _cached_index_name, _cached_index_handle

    if _cached_index_handle is not None and _cached_index_name == index_name:
        return _cached_index_handle

    _client = _client or get_client()
    _cached_index_handle = ensure_index(_client, index_name, dim=embedding_dim, metric="cosine")
    _cached_index_name = index_name
    return _cached_index_handle


def store_documents_in_index(
    docs: Sequence[str],
    metas: Sequence[Optional[Dict]],
    *,
    index_name: str,
) -> None:
    if len(docs) != len(metas):
        raise ValueError("docs and metas must have identical lengths.")
    if not docs:
        return

    index = init_vector_store(index_name)
    embeddings = batch_get_embeddings(list(docs))

    vectors: List[Tuple[str, List[float], Dict]] = []
    for i, (embedding, doc, meta) in enumerate(zip(embeddings, docs, metas)):
        md = _normalize_page_fields(dict(meta or {}))
        md.setdefault("text", doc)
        vectors.append((str(i), embedding, md))  # TODO: stable IDs across batches.

    pc_upsert(index, vectors)


def search_vector_store(
    query_text: str,
    *,
    index_name: str,
    namespace: str | None = None,
    top_k: int = 5,
) -> List[Dict]:
    if not query_text:
        return []

    index = init_vector_store(index_name)
    query_vector = get_embedding(query_text)
    effective_namespace = namespace or "default"
    result = pc_query(
        index,
        query_vector,
        top_k=top_k,
        namespace=effective_namespace,
        include_metadata=True,
        include_values=False,
    )

    output: List[Dict] = []
    for match in result.get("matches", []):
        metadata = match.get("metadata") or {}
        output.append(
            {
                "id": match.get("id"),
                "text": metadata.get("text"),
                "metadata": metadata,
                "score": match.get("score"),
            }
        )
    return output

# Put this just below the real search_vector_store implementation
def query_vector_store(
    query_text: str,
    *,
    index_name: str,
    namespace: str | None = None,
    top_k: int = 5,
) -> List[Dict]:
    """Alias for backward compatibility."""
    return search_vector_store(query_text, index_name=index_name, namespace=namespace, top_k=top_k)
