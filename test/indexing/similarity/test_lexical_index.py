from datetime import date

from auth_rag.indexing.similarity.lexical_index import VolatileLexicalIndex
from auth_rag.storing.store import VolatileStore
from auth_rag.types import Chunk, Language, Metadata, Origin, Source


def _chunk(id: str, text: str, source_id: str = "doc1") -> Chunk:
    s = Source(id=source_id, name="d.pdf", origin=Origin.LOCAL, time=date(2024, 1, 1))
    return Chunk(id=id, text=text, metadata=Metadata(source=s, title="S"))


def _wired():
    store = VolatileStore()
    index = VolatileLexicalIndex(store, language=Language.ITALIAN)
    return store, index


def _ingest(store, index, chunks):
    store.add(chunks)
    index.insert(chunks)


def test_retrieve_resolves_fever_chunks_via_store():
    store, index = _wired()
    _ingest(store, index, [
        _chunk("c0", "Il paziente presenta febbre alta e tosse."),
        _chunk("c1", "La pressione arteriosa è nella norma."),
        _chunk("c2", "Febbre e brividi da tre giorni."),
    ])
    ids = {r.chunk.id for r in index.retrieve("febbre", top_i=2)}
    assert ids == {"c0", "c2"}


def test_search_returns_ids_and_scores():
    store, index = _wired()
    _ingest(store, index, [_chunk("c0", "febbre alta"), _chunk("c1", "pressione arteriosa")])
    scored = index._search("febbre", top_i=2)
    assert scored[0][0] == "c0"
    assert isinstance(scored[0][1], float)


def test_delete_removes_only_that_source():
    store, index = _wired()
    _ingest(store, index, [_chunk("a0", "febbre", "doc1"), _chunk("b0", "pressione arteriosa", "doc2")])
    index.delete("doc1")
    ids = {id_ for id_, _ in index._search("febbre pressione", top_i=10)}
    assert "a0" not in ids
    assert "b0" in ids


def test_search_on_empty_index_returns_empty():
    _, index = _wired()
    assert index._search("febbre", top_i=1) == []
