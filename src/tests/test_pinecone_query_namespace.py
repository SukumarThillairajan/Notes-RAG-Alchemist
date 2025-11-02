# tests/test_pinecone_query_namespace.py
import types

def test_pinecone_query_accepts_namespace(monkeypatch):
    from utils import pinecone_compat as pc

    # fake index with a 'query' that records kwargs
    called = {}
    class FakeIndex:
        def query(self, **kwargs):
            called.update(kwargs)
            return {"matches": []}

    idx = FakeIndex()
    pc.query(idx, [0.1, 0.2], top_k=3, include_metadata=True, include_values=False, namespace="ns123")
    assert called.get("namespace") == "ns123"
