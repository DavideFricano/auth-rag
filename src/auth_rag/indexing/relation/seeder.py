from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from auth_rag.indexing.relation.extractor import BaseExtractor

if TYPE_CHECKING:
    from auth_rag.indexing.relation.index import RelationIndex

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def terms(name: str) -> set[str]:
    """Words of a concept name, for matching one name against another.

    Digits and words under three letters are dropped: they match too much to carry
    signal, at the cost of short acronyms, which cannot be seeds.
    """
    return {word for word in _WORD.findall(name.lower()) if len(word) > 2}


class BaseSeeder(ABC):
    """Where a query enters the graph: which concepts it starts from, and how strongly.

    A seeder owns both sides of the match. ``represent`` says what a new node carries for
    it to be found by, and ``seeds`` reads a query against that; the tier stores the one
    and searches the other, so changing seeder changes the entry without touching the
    graph.
    """

    @abstractmethod
    def represent(self, name: str) -> dict[str, object]:
        """The properties a new node with this name carries."""

    @abstractmethod
    def seeds(self, query: str, graph: RelationIndex) -> dict[str, float]:
        """Concept name -> score, at most ``graph.max_seeds``, best first."""


class LexicalSeeder(BaseSeeder):
    """Matches words of the query against the words of the concept names.

    With an ``extractor`` the query is first read by it, so the words matched are those of
    the concepts it names rather than the user's — "how", "treated" — which no concept
    name contains anyway.
    """

    def __init__(self, extractor: BaseExtractor | None = None) -> None:
        self.extractor = extractor

    def represent(self, name: str) -> dict[str, object]:
        return {"tokens": sorted(terms(name))}

    def seeds(self, query: str, graph: RelationIndex) -> dict[str, float]:
        texts = self.extractor.entities(query) if self.extractor is not None else [query]
        return graph._seeds({term for text in texts for term in terms(text)})
