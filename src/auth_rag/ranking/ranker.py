from __future__ import annotations

from abc import ABC, abstractmethod

from auth_rag.types import ScoredChunk


class BaseRanker(ABC):
    """Shared utility for ranking operations."""

    @abstractmethod
    def rank(self, *args, **kwargs) -> list[ScoredChunk]:
        pass

    @staticmethod
    def extract_top_k(scored_list: list[ScoredChunk], top_k: int | None) -> list[ScoredChunk]:
        """Returns the top_k highest-scored chunks, or all if top_k is None. Ties break on
        chunk id, so the ranking doesn't depend on the order the chunks arrived in."""
        ordered = sorted(scored_list, key=lambda x: (-x.score, x.chunk.id))
        return ordered if top_k is None else ordered[:top_k]
