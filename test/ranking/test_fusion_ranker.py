from datetime import date

import numpy as np
import pytest

from auth_rag.ranking.fusion_ranker import (
    DistributionScoreFusionRanker,
    ReciprocalRankFusionRanker,
    RelativeScoreFusionRanker,
)
from auth_rag.types import Chunk, Metadata, Origin, ScoredChunk, Source


def _sc(chunk_id, score):
    s = Source(id="doc1", name="doc.pdf", origin=Origin.LOCAL, time=date(2024, 1, 1))
    c = Chunk(id=chunk_id, text="T", metadata=Metadata(source=s, title="S"))
    return ScoredChunk(chunk=c, score=score)


def test_reciprocal_fusion_deduplicates():
    ranker = ReciprocalRankFusionRanker()
    shared = _sc("c0", 0.9)
    ranked = ranker.rank([[shared, _sc("c1", 0.5)], [shared, _sc("c2", 0.4)]])
    ids = [r.chunk.id for r in ranked]
    assert len(ids) == len(set(ids))


def test_reciprocal_fusion_ignores_absent_retrievers():
    """A doc gets a term only from retrievers that returned it — absence adds nothing.
    Under the old zero-fill matrix an absent retriever still granted ~1/(k+n) credit."""
    ranker = ReciprocalRankFusionRanker(k=60.0)
    a = [_sc("c0", 0.9), _sc("c1", 0.5)]  # c0 rank 1, c1 rank 2
    b = [_sc("c2", 0.9)]  # c2 rank 1; c0 and c1 are absent here
    scores = {r.chunk.id: r.score for r in ranker.rank([a, b])}
    assert scores["c0"] == pytest.approx(1 / 61)  # only A's rank-1 term, no phantom from B
    assert scores["c1"] == pytest.approx(1 / 62)  # only A's rank-2 term
    assert scores["c2"] == pytest.approx(1 / 61)  # only B's rank-1 term


def test_reciprocal_fusion_rewards_corroboration():
    """A doc found by both retrievers outranks docs found by only one."""
    ranker = ReciprocalRankFusionRanker(k=60.0)
    a = [_sc("only_a", 0.9), _sc("both", 0.5)]  # both at rank 2 here
    b = [_sc("both", 0.9), _sc("only_b", 0.5)]  # both at rank 1 here
    ranked = ranker.rank([a, b])
    assert ranked[0].chunk.id == "both"  # 1/62 + 1/61 beats either single-retriever term


def test_reciprocal_fusion_defaults_to_uniform_weights():
    """weights=None must reproduce plain RRF, i.e. every weight at 1."""
    a = [_sc("c0", 0.9), _sc("c1", 0.5)]
    b = [_sc("c0", 0.4)]
    plain = ReciprocalRankFusionRanker().rank([a, b])
    explicit = ReciprocalRankFusionRanker(weights=np.array([1.0, 1.0])).rank([a, b])
    assert {r.chunk.id: r.score for r in plain} == {r.chunk.id: r.score for r in explicit}


def test_reciprocal_fusion_weights_change_winner():
    """Two disjoint retrievers tie at 1/(k+1) unweighted; a weight breaks the tie."""
    a = [_sc("from_a", 0.9), _sc("a1", 0.5)]
    b = [_sc("from_b", 0.9), _sc("b1", 0.5)]
    tied = ReciprocalRankFusionRanker(k=60.0).rank([a, b])
    assert tied[0].score == pytest.approx(tied[1].score)  # both rank-1 docs at 1/61
    ranked = ReciprocalRankFusionRanker(k=60.0, weights=np.array([1.0, 5.0])).rank([a, b])
    assert ranked[0].chunk.id == "from_b"
    assert ranked[0].score == pytest.approx(5 / 61)


def test_reciprocal_fusion_rejects_mismatched_weights():
    """One weight per result list — a silent zip truncation would drop a retriever."""
    ranker = ReciprocalRankFusionRanker(weights=np.array([1.0, 1.0]))
    with pytest.raises(ValueError):
        ranker.rank([[_sc("c0", 0.9)]])


def test_rank_top_k():
    ranker = RelativeScoreFusionRanker(weights=np.array([1.0]))
    ranked = ranker.rank([[_sc(f"c{i}", float(i)) for i in range(5)]], top_k=2)
    assert len(ranked) == 2


def test_relative_fusion_weights_change_winner():
    ranker = RelativeScoreFusionRanker(weights=np.array([1.0, 2.0]))
    bm25     = [_sc("c0", 0.9), _sc("c1", 0.1)]
    semantic = [_sc("c0", 0.1), _sc("c1", 0.8)]
    ranked = ranker.rank([bm25, semantic])
    assert ranked[0].chunk.id == "c1"


def test_relative_fusion_equal_weights_other_winner():
    ranker = RelativeScoreFusionRanker(weights=np.array([1.0, 1.0]))
    bm25     = [_sc("c0", 0.9), _sc("c1", 0.1)]
    semantic = [_sc("c0", 0.1), _sc("c1", 0.8)]
    ranked = ranker.rank([bm25, semantic])
    assert ranked[0].chunk.id == "c0"


def test_relative_fusion_all_tied_counts_as_fully_relevant():
    """A retriever whose scores are all equal has no ordering to contribute, so each doc
    it returned gets the full weight — not 0, which would drop the retriever silently."""
    ranker = RelativeScoreFusionRanker(weights=np.array([1.0, 1.0]))
    tied     = [_sc("c0", 2.0), _sc("c1", 2.0)]  # no spread: min == max
    semantic = [_sc("c0", 0.9), _sc("c1", 0.1)]
    scores = {r.chunk.id: r.score for r in ranker.rank([tied, semantic])}
    assert scores["c0"] == pytest.approx(1.0)  # 0.5 tied + 0.5 semantic top
    assert scores["c1"] == pytest.approx(0.5)  # 0.5 tied + 0.0 semantic bottom


def test_score_fusion_weights_are_proportions():
    """Weights are rescaled to sum to 1, so only their ratio matters and the fused score
    cannot exceed 1."""
    bm25     = [_sc("c0", 12.0), _sc("c1", 3.0)]
    semantic = [_sc("c0", 0.2), _sc("c1", 0.9)]
    small = RelativeScoreFusionRanker(weights=np.array([1.0, 3.0])).rank([bm25, semantic])
    large = RelativeScoreFusionRanker(weights=np.array([10.0, 30.0])).rank([bm25, semantic])
    assert {r.chunk.id: r.score for r in small} == {r.chunk.id: r.score for r in large}
    assert all(r.score <= 1.0 for r in small)


def test_score_fusion_rejects_degenerate_weights():
    with pytest.raises(ValueError):
        RelativeScoreFusionRanker(weights=np.array([0.0, 0.0]))


def test_relative_fusion_normalizes_across_scales():
    ranker = RelativeScoreFusionRanker(weights=np.array([1.0, 1.0]))
    bm25     = [_sc("c0", 12.0), _sc("c1", 3.0)]  # unbounded BM25 scale
    semantic = [_sc("c0", 0.2), _sc("c1", 0.9)]   # bounded cosine scale
    ranked = ranker.rank([bm25, semantic])
    # After min-max both retrievers land in [0, 1], so neither scale dominates:
    # c0 tops bm25 (norm 1) and bottoms semantic (norm 0); c1 the reverse -> tie,
    # broken to c0 by stable ordering. Without normalization BM25 would decide it.
    assert {r.chunk.id for r in ranked} == {"c0", "c1"}
    assert ranked[0].score == ranked[1].score


def test_distribution_fusion_keeps_worst_doc_above_zero():
    """Unlike min-max, DBSF doesn't collapse the last doc of a list to exactly 0."""
    ranker = DistributionScoreFusionRanker(weights=np.array([1.0]))
    ranked = ranker.rank([[_sc("c0", 12.0), _sc("c1", 6.0), _sc("c2", 3.0)]])
    scores = {r.chunk.id: r.score for r in ranked}
    assert scores["c2"] > 0.0
    assert scores["c0"] > scores["c1"] > scores["c2"]


def test_distribution_fusion_clips_outliers_into_unit_range():
    """A doc past mean ± sigma*std would normalize outside [0, 1] without clipping."""
    ranker = DistributionScoreFusionRanker(weights=np.array([1.0]), sigma=0.5)
    ranked = ranker.rank([[_sc("hit", 100.0), _sc("c1", 1.0), _sc("c2", 0.9)]])
    assert all(0.0 <= r.score <= 1.0 for r in ranked)
    assert ranked[0].chunk.id == "hit"


def test_distribution_fusion_all_tied_counts_as_fully_relevant():
    """The std == 0 degenerate case mirrors min-max's span == 0."""
    ranker = DistributionScoreFusionRanker(weights=np.array([1.0]))
    ranked = ranker.rank([[_sc("c0", 2.0), _sc("c1", 2.0)]])
    assert [r.score for r in ranked] == [1.0, 1.0]


def test_distribution_fusion_is_less_sensitive_to_candidate_depth():
    """min-max reads off the observed extremes, so widening the candidate set moves a
    mid-list doc a lot; mean and std move less. Probing the top doc would hide it, since
    min-max pins that one to 1 regardless of depth."""
    head = [_sc("c0", 10.0), _sc("c1", 9.0)]
    deep = head + [_sc(f"t{i}", 1.0) for i in range(8)]

    def shift(ranker):
        before = {r.chunk.id: r.score for r in ranker.rank([head])}
        after = {r.chunk.id: r.score for r in ranker.rank([deep])}
        return abs(after["c1"] - before["c1"])

    weights = np.array([1.0])
    assert shift(DistributionScoreFusionRanker(weights)) < shift(RelativeScoreFusionRanker(weights))
