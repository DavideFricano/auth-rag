from datetime import date

import numpy as np

from auth_rag.embedding.embedder import BaseEmbedder
from auth_rag.indexing.similarity.semantic_index import VolatileSemanticIndex
from auth_rag.storing.store import VolatileStore
from auth_rag.types import Chunk, Metadata, Origin, Source


class _FakeEmbedder(BaseEmbedder):
    """Maps exact text -> preset vector (zeros for unknown), so tests control geometry."""

    def __init__(self, table: dict[str, list[float]], dim: int) -> None:
        self._table = {t: np.asarray(v, dtype=np.float32) for t, v in table.items()}
        self._dim = dim

    def _vec(self, text: str) -> np.ndarray:
        return self._table.get(text, np.zeros(self._dim, dtype=np.float32))

    def embed_chunks(self, chunks):
        return np.array([self._vec(t) for t in chunks], dtype=np.float32)

    def embed_query(self, query):
        return np.array([self._vec(query)], dtype=np.float32)


def _chunk(id: str, text: str, source_id: str = "doc1") -> Chunk:
    s = Source(id=source_id, name="d.pdf", origin=Origin.LOCAL, time=date(2024, 1, 1))
    return Chunk(id=id, text=text, metadata=Metadata(source=s, title="S"))


def _wired(table: dict[str, list[float]], dim: int):
    store = VolatileStore()
    index = VolatileSemanticIndex(store, _FakeEmbedder(table, dim))
    return store, index


def _ingest(store, index, chunks):
    """Simulate the ingestion pipeline: record to the store once, then index."""
    store.add(chunks)
    index.insert(chunks)


def test_retrieve_returns_chunks_resolved_via_store():
    store, index = _wired({"t0": [1.0, 0.0], "t1": [0.0, 1.0], "q": [1.0, 0.0]}, dim=2)
    _ingest(store, index, [_chunk("c0", "t0"), _chunk("c1", "t1")])
    results = index.retrieve("q", top_i=1)
    assert results[0].chunk.id == "c0"
    assert results[0].chunk.text == "t0"


def test_search_returns_ids_and_raw_cosine():
    """Scores are Qdrant's cosine as-is, not rescaled into [0, 1]: identical vectors give
    exactly 1.0. Making dense and BM25 comparable is the fusion ranker's job."""
    store, index = _wired({"t0": [1.0, 0.0], "q": [1.0, 0.0]}, dim=2)
    _ingest(store, index, [_chunk("c0", "t0")])
    scored = index._search("q", top_i=1)
    assert scored[0][0] == "c0"
    assert abs(scored[0][1] - 1.0) < 1e-6


def test_delete_removes_only_that_source_from_index():
    store, index = _wired(
        {"t0": [1.0, 0.0, 0.0], "t1": [0.0, 1.0, 0.0], "t2": [0.0, 0.0, 1.0], "q": [0.0, 0.0, 1.0]},
        dim=3,
    )
    _ingest(store, index, [_chunk("a0", "t0", "doc1"), _chunk("a1", "t1", "doc1"), _chunk("b0", "t2", "doc2")])
    index.delete("doc1")  # index-side removal only (records handled by the pipeline)
    assert {id_ for id_, _ in index._search("q", top_i=10)} == {"b0"}


def test_search_on_empty_index_returns_empty():
    _, index = _wired({}, dim=4)
    assert index._search("x", top_i=1) == []


def test_insert_is_idempotent_by_id():
    store, index = _wired({"t0": [1.0, 0.0], "q": [1.0, 0.0]}, dim=2)
    _ingest(store, index, [_chunk("c0", "t0")])
    index.insert([_chunk("c0", "t0")])  # same id again
    assert len({id_ for id_, _ in index._search("q", top_i=10)}) == 1
