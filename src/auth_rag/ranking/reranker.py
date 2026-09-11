from __future__ import annotations

from abc import ABC, abstractmethod

from sentence_transformers import CrossEncoder

from auth_rag.ranking.ranker import BaseRanker
from auth_rag.types import Chunk, ScoredChunk


class Reranker(BaseRanker, ABC):
    """Reranks a flat list of chunks against a query using a cross-encoder."""

    @abstractmethod
    def rank(self, chunks: list[Chunk], query: str, top_n: int | None = None) -> list[ScoredChunk]:
        pass


class CrossReranker(Reranker):
    """Cross-encoder reranker: scores each (query, chunk) pair jointly for higher precision."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3") -> None:
        self.model = CrossEncoder(model_name)

    def rank(self, chunks: list[Chunk], query: str, top_n: int | None = None) -> list[ScoredChunk]:
        pairs = [(query, chunk.text) for chunk in chunks]
        scores = self.model.predict(pairs)
        scored = [ScoredChunk(chunk=c, score=float(s)) for c, s in zip(chunks, scores, strict=True)]
        return self.extract_top_k(scored, top_n)
