"""Utilities package for the Lecture Notes Assistant."""

from . import embeddings
from . import embedding_indexer
from . import pdf_utils
from . import pinecone_compat
from . import vector_store


__all__ = [
    "pdf_utils",
    "embeddings",
    "embedding_indexer",
    "pinecone_compat",
    "vector_store",
]

# Optional convenience imports (avoid heavy imports at package init if they are slow):
# from .pdf_utils import extract_text_from_pdf, split_text
# from .embedding_indexer import index_chunks, init_pinecone_index