import os
import sqlite3
from datetime import date

import pytest

from auth_rag.storing.store import PersistentStore, RemoteStore, VolatileStore
from auth_rag.types import Chunk, Metadata, Origin, Source


def _chunk(id: str, text: str, source_id: str = "doc1") -> Chunk:
    s = Source(id=source_id, name="d.pdf", origin=Origin.LOCAL, time=date(2024, 1, 1))
    return Chunk(id=id, text=text, metadata=Metadata(source=s, title="S"))


@pytest.fixture(params=["memory", "persistent"])
def store(request):
    if request.param == "memory":
        return VolatileStore()
    return PersistentStore(connection=sqlite3.connect(":memory:"))


def test_add_get_roundtrip(store):
    store.add([_chunk("c0", "t0"), _chunk("c1", "t1")])
    assert {c.id: c.text for c in store.get(["c0", "c1"])} == {"c0": "t0", "c1": "t1"}


def test_get_skips_missing_ids(store):
    store.add([_chunk("c0", "t0")])
    assert [c.id for c in store.get(["c0", "nope"])] == ["c0"]


def test_add_is_idempotent_upsert_by_id(store):
    store.add([_chunk("c0", "old")])
    store.add([_chunk("c0", "new")])  # same id -> replaced
    assert [c.text for c in store.get(["c0"])] == ["new"]


def test_delete_by_source(store):
    store.add([_chunk("a0", "t", "doc1"), _chunk("b0", "t", "doc2")])
    store.delete("doc1")
    assert [c.id for c in store.get(["a0", "b0"])] == ["b0"]


def test_get_by_source(store):
    store.add([_chunk("a0", "t", "doc1"), _chunk("a1", "t", "doc1"), _chunk("b0", "t", "doc2")])
    assert {c.id for c in store.get_by_source("doc1")} == {"a0", "a1"}
    assert store.get_by_source("nope") == []


def test_persistent_indexes_the_source_id():
    """delete and get_by_source look chunks up by source: without the index, a scan each."""
    conn = sqlite3.connect(":memory:")
    PersistentStore(connection=conn)
    plan = conn.execute("EXPLAIN QUERY PLAN SELECT data FROM chunks WHERE source_id = 'x'")
    assert "chunks_source_id" in " ".join(str(row) for row in plan.fetchall())


def test_persistent_survives_reopening_same_db(tmp_path):
    db = str(tmp_path / "store.db")
    PersistentStore(connection=sqlite3.connect(db)).add([_chunk("c0", "t0")])
    reopened = PersistentStore(connection=sqlite3.connect(db))
    assert [c.id for c in reopened.get(["c0"])] == ["c0"]


@pytest.mark.skipif(not os.getenv("AUTH_RAG_TEST_PG"), reason="needs a Postgres (set AUTH_RAG_TEST_PG=<url>)")
def test_remote_store_roundtrip():
    import psycopg

    conn = psycopg.connect(os.environ["AUTH_RAG_TEST_PG"])
    conn.execute("DROP TABLE IF EXISTS chunks")
    store = RemoteStore(connection=conn)
    store.add([_chunk("a0", "t", "doc1"), _chunk("b0", "t", "doc2")])
    assert {c.id for c in store.get(["a0", "b0"])} == {"a0", "b0"}
    store.delete("doc1")
    assert [c.id for c in store.get(["a0", "b0"])] == ["b0"]
