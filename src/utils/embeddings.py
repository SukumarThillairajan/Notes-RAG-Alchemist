import os
from typing import List, Optional, Sequence

from openai import OpenAI

EMBED_MODEL = "text-embedding-3-large"
EMBED_DIM = 3072

_client: Optional[OpenAI] = None


def _client_instance() -> OpenAI:
    """Lazily instantiate the OpenAI client and ensure the API key exists."""
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
        _client = OpenAI(api_key=api_key)
    return _client


def get_embedding(text: str, *, model: str = EMBED_MODEL) -> List[float]:
    """
    Obtain the embedding vector for the given text using OpenAI's embedding API.

    Args:
        text: The text to embed.
        model: Embedding model identifier.
    """
    client = _client_instance()
    response = client.embeddings.create(input=text, model=model)
    return list(response.data[0].embedding)


def batch_get_embeddings(
    texts: Sequence[str],
    *,
    model: str = EMBED_MODEL,
) -> List[List[float]]:
    """
    Get embeddings for a sequence of texts in a single API call.

    Args:
        texts: The strings to embed.
        model: Embedding model identifier.
    """
    if not texts:
        return []

    client = _client_instance()
    response = client.embeddings.create(input=list(texts), model=model)
    return [list(item.embedding) for item in response.data]
