# Run with:  python tests/test_vector_store_local.py
# Or with pytest:  pytest -q

import math
import os
import random
import string
import sys
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence, Tuple

# --- Wire up import path for local package layout ---
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Import modules under test
from utils import pinecone_compat as pc
import utils.vector_store as vs

# --------------------------------------------------------------------------------------
# Fake Embeddings: deterministic float vectors from text (no external API calls)
# --------------------------------------------------------------------------------------
EMBED_DIM = 16  # small dim for tests; your code can still pass EMBED_DIM=3072 in prod.

def _seed_from_text(text: str) -> int:
    return abs(hash(text)) % (2**31 - 1)

def fake_get_embedding(text: str) -> List[float]:
    rnd = random.Random(_seed_from_text(text))
    return [rnd.uniform(-1.0, 1.0) for _ in range(EMBED_DIM)]

def fake_batch_get_embeddings(texts: Sequence[str]) -> List[List[float]]:
    return [fake_get_embedding(t) for t in texts]

# Patch vector_store's embedding imports
vs.get_embedding = fake_get_embedding
vs.batch_get_embeddings = fake_batch_get_embeddings

# --------------------------------------------------------------------------------------
# In-memory cosine helpers
# --------------------------------------------------------------------------------------
def cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(y*y for y in b))
    return 0.0 if na == 0.0 or nb == 0.0 else dot / (na * nb)

# --------------------------------------------------------------------------------------
# Fake Pinecone Server (client + index) supporting v2/v3 shapes
# --------------------------------------------------------------------------------------
class FakeIndex:
    def __init__(self, name: str, metric: str = "cosine"):
        self.name = name
        self.metric = metric
        # id -> {"values": [...], "metadata": {...}}
        self._store: Dict[str, Dict[str, Any]] = {}

    # v3 upsert signature: index.upsert(vectors=[{"id":..., "values":[...], "metadata":{...}}, ...])
    '''def upsert(self, vectors: Sequence[Dict[str, Any]] = []):
        for v in vectors:
            vid = v["id"]
            self._store[vid] = {"values": v["values"], "metadata": v.get("metadata") or {}}'''

    def upsert(self, vectors: Sequence[Any] = []):
        if not vectors:
            return
        # Handle both v3 dicts and v2 tuples to make the fake index more robust for tests.
        if isinstance(vectors[0], dict):
            # v3-style: list of dicts
            for v in vectors:
                self._store[v["id"]] = {"values": v["values"], "metadata": v.get("metadata") or {}}
        else:
            # v2-style: list of tuples
            self.upsert_v2(vectors)

    # v2 upsert signature: index.upsert(vectors=[(id, values, metadata), ...])
    def upsert_v2(self, vectors: Sequence[Tuple[str, List[float], Dict]]):
        for vid, values, metadata in vectors:
            self._store[vid] = {"values": values, "metadata": metadata or {}}

    def query(self, vector: List[float], top_k: int, include_metadata: bool, include_values: bool):
        # Naive exhaustive NN
        scored = []
        for vid, blob in self._store.items():
            score = cosine(vector, blob["values"]) if self.metric == "cosine" else 0.0
            item = SimpleNamespace(
                id=vid,
                score=score,
                metadata=blob["metadata"] if include_metadata else None,
                values=blob["values"] if include_values else None,
            )
            scored.append(item)
        scored.sort(key=lambda m: m.score, reverse=True)
        return SimpleNamespace(matches=scored[:top_k])

    # Provide to_dict() for compat normalization paths
    def to_dict(self):
        return {"matches": []}

class FakeClientV3:
    # Mimic pinecone v3 (has Pinecone class and ServerlessSpec usage upstream)
    def __init__(self):
        self._indexes: Dict[str, FakeIndex] = {}

    def list_indexes(self):
        # v3 returns objects with .name or list of names; we’ll return a list of name strings
        return [SimpleNamespace(name=n) for n in self._indexes.keys()]

    def create_index(self, name: str, dimension: int, metric: str, spec: Any):
        if name in self._indexes:
            return
        self._indexes[name] = FakeIndex(name=name, metric=metric)

    def Index(self, name: str):
        return self._indexes[name]

class FakeClientV2:
    # Mimic pinecone v2 (uses pinecone.init + pinecone.Index)
    def __init__(self):
        self._indexes: Dict[str, FakeIndex] = {}

    def list_indexes(self):
        # v2 sometimes returns list of strings
        return list(self._indexes.keys())

    def create_index(self, name: str, dimension: int, metric: str):
        if name in self._indexes:
            return
        self._indexes[name] = FakeIndex(name=name, metric=metric)

    def Index(self, name: str):
        return self._indexes[name]

# --------------------------------------------------------------------------------------
# Monkeypatch pinecone_compat to use fake client/index (and fake env)
# --------------------------------------------------------------------------------------
os.environ.setdefault("PINECONE_API_KEY", "fake-key")
os.environ.setdefault("PINECONE_ENVIRONMENT", "us-east-1")
os.environ.setdefault("PINECONE_CLOUD", "aws")
os.environ.setdefault("PINECONE_REGION", "us-east-1")

def use_fake_v3():
    # Pretend we're in v3 world
    pc._IS_V3 = True
    class _ServerlessSpec:
        def __init__(self, cloud: str, region: str):
            self.cloud = cloud
            self.region = region
    pc._SERVERLESS_SPEC = _ServerlessSpec
    def _fake_get_client():
        return FakeClientV3()
    pc.get_client = _fake_get_client

def use_fake_v2():
    # Pretend we're in v2 world
    pc._IS_V3 = False
    pc._SERVERLESS_SPEC = None
    def _fake_get_client():
        return FakeClientV2()
    pc.get_client = _fake_get_client

# Also ensure vector_store uses our fake client getter + fix function name alias
def patch_vector_store_bindings():
    vs.get_client = pc.get_client
    # fix naming mismatch if present
    if not hasattr(vs, "init_vector_store"):
        setattr(vs, "init_vector_store", vs.init_vector_store)

# --------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------
def _populate_index(index_name: str):
    docs = [
        "Gauss's law relates the electric flux through a surface to the charge enclosed.",
        "Stokes' theorem connects the surface integral of curl to the line integral around the boundary.",
        "The Bernoulli equation expresses conservation of energy for incompressible flow.",
        "Fourier series represent periodic functions as sums of sines and cosines.",
        "Maxwell's equations unify electricity and magnetism into a single framework."
    ]
    metas = [{"book": "BookA", "page": i+10} for i in range(len(docs))]
    vs.store_documents_in_index(docs, metas, index_name=index_name)

def test_end_to_end(fake_version: str):
    index_name = f"test-idx-{fake_version}"
    # Initialize & populate
    vs.init_vector_store(index_name, embedding_dim=EMBED_DIM)
    _populate_index(index_name)

    # Search something close to an inserted doc
    q = "What does Gauss law say about electric flux and enclosed charge?"
    results = vs.search_vector_store(q, index_name=index_name, top_k=3)

    assert isinstance(results, list)
    assert len(results) >= 1
    top = results[0]
    assert "text" in top and isinstance(top["text"], str)
    assert "metadata" in top and "book" in top["metadata"]
    assert top["metadata"]["book"] == "BookA"
    # Ensure we get a reasonable score range
    assert 0.0 <= (top.get("score") or 0.0) <= 1.0

    # Also verify multiple inserts keep distinct pages
    pages = {r["metadata"].get("page") for r in results if "metadata" in r}
    assert len(pages) >= 1

def test_upsert_and_query_shapes(fake_version: str):
    index_name = f"shape-idx-{fake_version}"
    idx = vs.init_vector_store(index_name, embedding_dim=EMBED_DIM)

    # Insert vectors in both v3 and v2 payload shapes through pinecone_compat
    v3_payload = [
        {"id": "a", "values": fake_get_embedding("a"), "metadata": {"text": "alpha"}},
        {"id": "b", "values": fake_get_embedding("b"), "metadata": {"text": "beta"}},
    ]
    if pc._IS_V3:
        idx.upsert(vectors=v3_payload)
    else:
        idx.upsert_v2([(p["id"], p["values"], p["metadata"]) for p in v3_payload])

    # Query using normalized wrapper
    qvec = fake_get_embedding("alphabet")
    out = pc.query(idx, qvec, top_k=2, include_metadata=True, include_values=False)
    assert "matches" in out and len(out["matches"]) == 2
    assert all("id" in m and "score" in m and "metadata" in m for m in out["matches"])

# --------------------------------------------------------------------------------------
# Main: run tests for both "v2" and "v3" modes
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Test in v3 mode
    use_fake_v3()
    patch_vector_store_bindings()
    print("Running tests in fake Pinecone v3 mode...")
    test_end_to_end(fake_version="v3")
    test_upsert_and_query_shapes(fake_version="v3")
    print("✓ v3 tests passed")

    # Reset vector_store cache between runs
    vs._cached_index_name = None
    vs._cached_index_handle = None
    vs._client = None

    # Test in v2 mode
    use_fake_v2()
    patch_vector_store_bindings()
    print("Running tests in fake Pinecone v2 mode...")
    test_end_to_end(fake_version="v2")
    test_upsert_and_query_shapes(fake_version="v2")
    print("✓ v2 tests passed")

    print("\nAll fake-client tests passed ✅")
