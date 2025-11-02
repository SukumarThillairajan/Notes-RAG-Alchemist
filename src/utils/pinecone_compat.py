import os
from typing import Any, Dict, List, Sequence, Tuple

import pinecone

_IS_V3 = hasattr(pinecone, "Pinecone")
_SERVERLESS_SPEC = getattr(pinecone, "ServerlessSpec", None)


def _require_env(var_name: str) -> str:
    value = os.getenv(var_name)
    if not value:
        raise RuntimeError(f"Missing environment variable: {var_name}")
    return value


def get_client() -> Any:
    """
    Return a Pinecone client compatible with both v2 and v3 SDKs.
    """
    api_key = _require_env("PINECONE_API_KEY")

    if _IS_V3:
        PineconeCls = getattr(pinecone, "Pinecone")
        return PineconeCls(api_key=api_key)

    environment = _require_env("PINECONE_ENVIRONMENT")
    # Use getattr to avoid static attribute errors and to handle missing init gracefully.
    init_func = getattr(pinecone, "init", None)
    if callable(init_func):
        init_func(api_key=api_key, environment=environment)
    else:
        raise RuntimeError(
            "Pinecone v2 SDK 'init' function not found; please install pinecone-client (v2) "
            "or use the v3 Pinecone class."
        )
    return pinecone


def list_indexes(client: Any) -> List[str]:
    """
    List available Pinecone indexes for the provided client.
    """
    response = client.list_indexes()
    if isinstance(response, dict):
        items = response.get("indexes", []) or []
    else:
        items = getattr(response, "indexes", response)

    names: List[str] = []
    for item in items or []:
        if isinstance(item, str):
            names.append(item)
            continue
        name = getattr(item, "name", None)
        if name is None and isinstance(item, dict):
            name = item.get("name")
        if name:
            names.append(name)
    return names


def ensure_index(client: Any, name: str, dim: int, metric: str = "cosine") -> Any:
    """
    Ensure an index exists and return a handle to it.
    """
    if name not in list_indexes(client):
        if _IS_V3:
            if _SERVERLESS_SPEC is None:
                raise RuntimeError("Pinecone v3 detected but ServerlessSpec is unavailable.")
            spec = _SERVERLESS_SPEC(
                cloud=os.getenv("PINECONE_CLOUD", "aws"),
                region=os.getenv("PINECONE_REGION", "us-east-1"),
            )
            client.create_index(
                name=name,
                dimension=dim,
                metric=metric,
                spec=spec,
            )
        else:
            client.create_index(name=name, dimension=dim, metric=metric)

    return client.Index(name)


def upsert(
    index: Any,
    vectors: Sequence[Tuple[str, List[float], Dict]],
    *,
    namespace: str | None = None,
) -> None:
    """
    Upsert vectors into the provided index, adapting payload format per SDK.
    """
    if not vectors:
        return

    if _IS_V3:
        payload: List[Dict[str, Any]] = []
        for vec_id, values, metadata in vectors:
            payload.append(
                {
                    "id": vec_id,
                    "values": values,
                    "metadata": _normalize_metadata(metadata),
                }
            )
        kwargs: Dict[str, Any] = {"vectors": payload}
        if namespace is not None:
            kwargs["namespace"] = namespace
        index.upsert(**kwargs)
    else:
        normalized = [
            (
                vec_id,
                values,
                _normalize_metadata(metadata),
            )
            for vec_id, values, metadata in vectors
        ]
        kwargs = {"vectors": normalized}
        if namespace is not None:
            kwargs["namespace"] = namespace
        index.upsert(**kwargs)


def query(
    index: Any,
    vector: List[float],
    top_k: int,
    include_metadata: bool,
    include_values: bool,
    *,
    namespace: str | None = None,
) -> Dict[str, Any]:
    """
    Query the index and normalize results across Pinecone versions.
    """
    kwargs: Dict[str, Any] = {
        "vector": vector,
        "top_k": top_k,
        "include_metadata": include_metadata,
        "include_values": include_values,
    }
    if namespace is not None:
        kwargs["namespace"] = namespace

    response = index.query(**kwargs)

    if isinstance(response, dict):
        raw_matches = response.get("matches", [])
    else:
        raw_matches = getattr(response, "matches", None)
        if raw_matches is None and hasattr(response, "to_dict"):
            raw_matches = response.to_dict().get("matches", [])

    matches: List[Dict[str, Any]] = []
    for match in raw_matches or []:
        if isinstance(match, dict):
            match_id = match.get("id")
            score = match.get("score")
            metadata = match.get("metadata")
        else:
            match_id = getattr(match, "id", None)
            score = getattr(match, "score", None)
            metadata = getattr(match, "metadata", None)
        metadata = _normalize_metadata(metadata)
        matches.append(
            {
                "id": match_id,
                "score": score,
                "metadata": metadata,
            }
        )

    return {"matches": matches}


def _normalize_metadata(metadata: Any) -> Dict[str, Any]:
    if metadata is None:
        return {}
    if isinstance(metadata, dict):
        return dict(metadata)
    if hasattr(metadata, "to_dict"):
        return metadata.to_dict()
    if hasattr(metadata, "model_dump"):
        return metadata.model_dump()
    try:
        return dict(metadata)
    except Exception:
        return {"value": metadata}
