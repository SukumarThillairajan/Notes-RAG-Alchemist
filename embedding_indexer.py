"""Utilities for chunking text, generating embeddings, and indexing in Pinecone."""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple

from utils import pdf_utils, pinecone_compat
from utils.embeddings import EMBED_DIM, batch_get_embeddings

_cached_index_handle: Optional[Any] = None
_cached_index_name: Optional[str] = None
_client: Optional[Any] = None


def _normalize_text(text: str) -> str:
    """Collapse excessive newlines and trim surrounding whitespace."""
    if not text:
        return ""
    collapsed = re.sub(r"\n{3,}", "\n\n", text)
    return collapsed.strip()


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


def chunk_text(
    text: str,
    *,
    char_mode: str = "auto",
    min_chars: int = 200,
    chunk_size_tokens: int = 500,
    overlap_tokens: int = 50,
    tokenizer_name: str = "cl100k_base",
) -> List[str]:
    """Chunk text via delegated char-based splitting or optional token windows."""
    if not text:
        return []

    if char_mode != "tokens":
        segments = pdf_utils.split_text(
            text=text,
            #mode=char_mode,
            max_length=480,
            min_chunk_len=min_chars,
            overlap=overlap_tokens,
            #tokenizer=None,
            #join_short=True,
        )
        return [segment for segment in segments if segment.strip()]

    # Token-based path
    try:
        import tiktoken  # type: ignore
    except Exception:
        print("tiktoken not available; falling back to delegated splitting.")
        return chunk_text(
            text,
            char_mode="auto",
            min_chars=min_chars,
            chunk_size_tokens=chunk_size_tokens,
            overlap_tokens=overlap_tokens,
            tokenizer_name=tokenizer_name,
        )

    if chunk_size_tokens <= 0:
        raise ValueError("chunk_size_tokens must be positive.")

    effective_overlap = max(0, min(overlap_tokens, chunk_size_tokens - 1))

    encoding = tiktoken.get_encoding(tokenizer_name)
    normalized = _normalize_text(text)
    if not normalized:
        return []

    token_ids = encoding.encode(normalized)
    if not token_ids:
        return []

    chunks: List[str] = []
    start = 0
    total_tokens = len(token_ids)

    while start < total_tokens:
        end = min(start + chunk_size_tokens, total_tokens)
        window_tokens = token_ids[start:end]
        chunk = encoding.decode(window_tokens).strip()
        if chunk:
            chunks.append(chunk)
        if end >= total_tokens:
            break
        start = end - effective_overlap

    return chunks


def embed_chunks(
    chunks: List[str],
    *,
    batch_size: int = 64,
) -> List[List[float]]:
    """Embed chunks in batches using the embeddings helper."""
    if not chunks:
        return []
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    vectors: List[List[float]] = []
    total = len(chunks)
    total_batches = math.ceil(total / batch_size)

    for batch_index in range(total_batches):
        start = batch_index * batch_size
        end = min(start + batch_size, total)
        batch_chunks = chunks[start:end]
        print(f"Embedding batch {batch_index + 1}/{total_batches} ({len(batch_chunks)} chunks)...")
        batch_vectors = batch_get_embeddings(batch_chunks)
        vectors.extend(batch_vectors)

    if len(vectors) != len(chunks):
        raise RuntimeError("Mismatch between chunks and embedding results.")

    return vectors


def init_pinecone_index(
    index_name: str,
    embedding_dim: int = EMBED_DIM,
    metric: str = "cosine",
) -> Any:
    """Initialize or reuse a Pinecone index handle."""
    if not index_name:
        raise ValueError("index_name must be provided.")

    global _client, _cached_index_handle, _cached_index_name

    if _cached_index_handle is not None and _cached_index_name == index_name:
        return _cached_index_handle

    if _client is None:
        _client = pinecone_compat.get_client()

    _cached_index_handle = pinecone_compat.ensure_index(
        _client,
        name=index_name,
        dim=embedding_dim,
        metric=metric,
    )
    _cached_index_name = index_name
    return _cached_index_handle


def index_chunks(
    chunks: List[str],
    metadata_list: List[Dict],
    *,
    index_name: str,
    namespace: str,
    id_prefix: str = "chunk",
    batch_size: int = 100,
) -> None:
    """Embed and upsert chunks with metadata into Pinecone."""
    if len(chunks) != len(metadata_list):
        raise ValueError("chunks and metadata_list must have the same length.")
    if not chunks:
        print("No chunks provided; skipping indexing.")
        return
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    vectors = embed_chunks(chunks)
    index = init_pinecone_index(index_name)

    vector_payloads: List[Tuple[str, List[float], Dict]] = []
    for idx, (chunk, metadata, vector) in enumerate(zip(chunks, metadata_list, vectors)):
        md = _normalize_page_fields(metadata)
        md.setdefault("text", chunk)
        vector_id = f"{id_prefix}-{idx}"
        vector_payloads.append((vector_id, vector, md))

    total = len(vector_payloads)
    total_batches = math.ceil(total / batch_size)

    for batch_index in range(total_batches):
        start = batch_index * batch_size
        end = min(start + batch_size, total)
        batch = vector_payloads[start:end]
        print(f"Upserting batch {batch_index + 1}/{total_batches} ({len(batch)} vectors) to namespace '{namespace}'...")
        pinecone_compat.upsert(index, batch, namespace=namespace) # type: ignore


if __name__ == "__main__":
    sample_text = "Para1...\n\nPara2...\n"
    sample_chunks = chunk_text(sample_text, char_mode="auto")
    print(f"Generated {len(sample_chunks)} chunk(s) from sample text.")
    if sample_chunks:
        preview = sample_chunks[0][:100]
        print(f"First chunk preview: {preview!r}")
