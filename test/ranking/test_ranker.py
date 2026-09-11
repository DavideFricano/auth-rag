from datetime import date

from auth_rag.ranking.ranker import BaseRanker
from auth_rag.types import Chunk, Metadata, Origin, ScoredChunk, Source


def _sc(chunk_id, score):
    s = Source(id="doc1", name="doc.pdf", origin=Origin.LOCAL, time=date(2024, 1, 1))
    c = Chunk(id=chunk_id, text="T", metadata=Metadata(source=s, title="S"))
    return ScoredChunk(chunk=c, score=score)


def test_extract_top_k_correct_order():
    result = BaseRanker.extract_top_k([_sc("a", 0.3), _sc("b", 0.9), _sc("c", 0.6)], top_k=3)
    assert [r.chunk.id for r in result] == ["b", "c", "a"]

def test_extract_top_k_returns_highest_scored():
    result = BaseRanker.extract_top_k([_sc("a", 0.3), _sc("b", 0.9), _sc("c", 0.6)], top_k=2)
    assert result[0].chunk.id == "b"
    assert result[1].chunk.id == "c"

def test_extract_top_k_none_returns_all():
    chunks = [_sc(f"c{i}", float(i)) for i in range(4)]
    result = BaseRanker.extract_top_k(chunks, top_k=None)
    assert [r.chunk.id for r in result] == ["c3", "c2", "c1", "c0"]
