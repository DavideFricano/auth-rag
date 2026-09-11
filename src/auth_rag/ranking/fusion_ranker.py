from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np

from auth_rag.ranking.ranker import BaseRanker
from auth_rag.types import Chunk, ScoredChunk


class FusionRanker(BaseRanker, ABC):
    """Combines results from multiple retrievers into a single ranked list."""

    @abstractmethod
    def rank(self, results: list[list[ScoredChunk]], top_k: int | None = None) -> list[ScoredChunk]:
        """Fuse the per-retriever ranked lists into one ranked list with top_k length."""


class ReciprocalRankFusionRanker(FusionRanker):
    """Reciprocal Rank Fusion (RRF): a doc gets ``weight / (k + rank)`` from each retriever
    that returned it, and nothing from those that didn't. Only the ordering matters.
    """

    def __init__(self, k: float = 60.0, weights: np.ndarray | None = None) -> None:
        self.k = k
        self.weights = None if weights is None else np.asarray(weights, dtype=float)

    def rank(self, results: list[list[ScoredChunk]], top_k: int | None = None) -> list[ScoredChunk]:
        weights = np.ones(len(results)) if self.weights is None else self.weights
        scores: defaultdict[str, float] = defaultdict(float)
        chunks: dict[str, Chunk] = {}
        for weight, result in zip(weights, results, strict=True):
            ordered = sorted(result, key=lambda sc: sc.score, reverse=True)
            for rank, sc in enumerate(ordered, start=1):
                scores[sc.chunk.id] += float(weight) / (self.k + rank)
                chunks[sc.chunk.id] = sc.chunk
        scored_chunks = [ScoredChunk(chunk=chunks[id_], score=score) for id_, score in scores.items()]
        return self.extract_top_k(scored_chunks, top_k)


class ScoreFusionRanker(FusionRanker, ABC):
    """Normalizes each retriever's scores to [0, 1] and takes a weighted sum, so bounded
    cosine and unbounded BM25 contribute comparably. Subclasses choose the normalization.

    Weights are rescaled to sum to 1, so they read as proportions and the fused score
    stays in [0, 1].
    """

    def __init__(self, weights: np.ndarray) -> None:
        weights = np.asarray(weights, dtype=float)
        if weights.sum() <= 0:
            raise ValueError("weights must sum to a positive value")
        self.weights = weights / weights.sum()

    @abstractmethod
    def normalize(self, scores: list[float]) -> list[float]:
        """Map one retriever's scores to [0, 1], preserving their order."""

    def rank(self, results: list[list[ScoredChunk]], top_k: int | None = None) -> list[ScoredChunk]:
        scores: defaultdict[str, float] = defaultdict(float)
        chunks: dict[str, Chunk] = {}
        for weight, result in zip(self.weights, results, strict=True):
            if not result:
                continue
            normalized = self.normalize([sc.score for sc in result])
            for sc, norm in zip(result, normalized, strict=True):
                scores[sc.chunk.id] += float(weight) * norm
                chunks[sc.chunk.id] = sc.chunk
        scored_chunks = [ScoredChunk(chunk=chunks[id_], score=score) for id_, score in scores.items()]
        return self.extract_top_k(scored_chunks, top_k)


class RelativeScoreFusionRanker(ScoreFusionRanker):
    """Relative Score Fusion (RSF): min-max normalization, so the best doc of each list
    becomes 1 and the worst 0.
    """

    def normalize(self, scores: list[float]) -> list[float]:
        low, span = min(scores), max(scores) - min(scores)
        if span == 0:
            return [1.0] * len(scores)
        return [(score - low) / span for score in scores]


class DistributionScoreFusionRanker(ScoreFusionRanker):
    """Distribution-Based Score Fusion (DBSF): normalization over ``mean ± sigma * std``
    instead of the observed extremes, which shift less as the candidate set grows.
    """

    def __init__(self, weights: np.ndarray, sigma: float = 3.0) -> None:
        super().__init__(weights)
        self.sigma = sigma

    def normalize(self, scores: list[float]) -> list[float]:
        raw = np.asarray(scores, dtype=float)
        std = float(raw.std())
        if std == 0:
            return [1.0] * len(scores)
        low, span = float(raw.mean()) - self.sigma * std, 2 * self.sigma * std
        return [float(np.clip((score - low) / span, 0.0, 1.0)) for score in raw]
