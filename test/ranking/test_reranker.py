from datetime import date

from auth_rag.ranking.reranker import CrossReranker
from auth_rag.types import Chunk, Metadata, Origin, Source


def _chunks():
    s = Source(id="doc1", name="doc.pdf", origin=Origin.LOCAL, time=date(2024, 1, 1))
    return [
        Chunk(id="c0", text="Testo A", metadata=Metadata(source=s, title="S")),
        Chunk(id="c1", text="Testo B", metadata=Metadata(source=s, title="S")),
        Chunk(id="c2", text="Testo C", metadata=Metadata(source=s, title="S")),
    ]


class _FakeModel:
    def __init__(self, scores):
        self._scores = scores

    def predict(self, pairs):
        return self._scores


def test_reranker_orders_by_score():
    reranker = CrossReranker.__new__(CrossReranker)
    reranker.model = _FakeModel([0.9, 0.2, 0.7])
    ranked = reranker.rank(_chunks(), "query")
    assert [r.chunk.id for r in ranked] == ["c0", "c2", "c1"]

def test_reranker_top_n_returns_correct_chunks():
    reranker = CrossReranker.__new__(CrossReranker)
    reranker.model = _FakeModel([0.9, 0.2, 0.7])
    ranked = reranker.rank(_chunks(), "query", top_n=2)
    assert len(ranked) == 2
    assert ranked[0].chunk.id == "c0"
    assert ranked[1].chunk.id == "c2"

def test_reranker_scores_match_model_output():
    reranker = CrossReranker.__new__(CrossReranker)
    reranker.model = _FakeModel([0.9, 0.2, 0.7])
    ranked = reranker.rank(_chunks(), "query")
    assert abs(ranked[0].score - 0.9) < 1e-6
    assert abs(ranked[1].score - 0.7) < 1e-6
    assert abs(ranked[2].score - 0.2) < 1e-6
