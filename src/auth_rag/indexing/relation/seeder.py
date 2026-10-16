from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from functools import cache
from typing import TYPE_CHECKING

import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

from auth_rag.embedding.embedder import BaseEmbedder
from auth_rag.indexing.relation.extractor import BaseExtractor
from auth_rag.types import Language

if TYPE_CHECKING:
    from auth_rag.indexing.relation.index import RelationIndex


@cache
def _analysis(language: Language) -> tuple[SnowballStemmer, frozenset[str]]:
    nltk.download("stopwords", quiet=True)
    return SnowballStemmer(str(language)), frozenset(stopwords.words(str(language)))


def roots(text: str, language: Language) -> list[str]:
    """The stems of a text, read exactly as the ``LexicalIndex`` reads its chunks.

    FastEmbed's BM25 does the same — punctuation out, lowercase, stopwords out, Snowball —
    but hands back hashes; this keeps the stems readable on the nodes. That the two agree
    is checked against FastEmbed in the tests, so neither can drift on its own.
    """
    stemmer, stop = _analysis(language)
    words = re.sub(r"[^\w]", " ", re.sub(r"[^\w\s]", " ", text).lower()).split()
    return [stemmer.stem(word) for word in words if word not in stop and len(word) <= 40]


def bm25(
    documents: dict[str, list[str]], query: list[str], k1: float = 1.2, b: float = 0.75
) -> dict[str, float]:
    """BM25 of ``query`` against each document, for the documents that share a root."""
    if not documents or not query:
        return {}
    average = sum(len(roots_) for roots_ in documents.values()) / len(documents)
    frequency: dict[str, int] = {}
    for roots_ in documents.values():
        for root in set(roots_):
            frequency[root] = frequency.get(root, 0) + 1
    scores: dict[str, float] = {}
    for name, roots_ in documents.items():
        score = 0.0
        for root in set(query):
            tf = roots_.count(root)
            if not tf:
                continue
            idf = math.log(1 + (len(documents) - frequency[root] + 0.5) / (frequency[root] + 0.5))
            score += idf * tf / (tf + k1 * (1 - b + b * len(roots_) / average))
        if score > 0:
            scores[name] = score
    return scores


def _best(
    found: list[dict[str, float]], limit: int, floor: float = 0.0
) -> dict[str, float]:
    """The best score each concept got, scaled so the strongest is 1, at most ``limit``."""
    best: dict[str, float] = {}
    for scores in found:
        for name, score in scores.items():
            if score > floor:
                best[name] = max(best.get(name, 0.0), score)
    if not best:
        return {}
    top = max(best.values())
    ranked = sorted(best.items(), key=lambda row: (-row[1], row[0]))
    return {name: score / top for name, score in ranked[:limit]}


class BaseSeeder(ABC):
    """Where a query enters the graph: which concepts it starts from, and how strongly.

    A seeder owns both sides of the match. ``represent`` says what a new node carries for
    it to be found by, and ``seeds`` reads a query against that; the tier stores the one
    and searches the other, so changing seeder changes the entry without touching the
    graph.
    """

    @abstractmethod
    def represent(self, names: list[str]) -> list[dict[str, object]]:
        """The properties each new node carries, in the order of ``names``.

        A list rather than one name, because a representation may cost a call whatever its
        size — an embedding does — and one call for every node an insert creates beats one
        per node.
        """

    @abstractmethod
    def seeds(self, query: str, graph: RelationIndex) -> dict[str, float]:
        """Concept name -> score, at most ``graph.max_seeds``, best first."""


class LexicalSeeder(BaseSeeder):
    """BM25 of the query's roots against the roots of the concept names.

    Stopwords and IDF take the question's own words out — "how", "is", a word every
    concept shares — so the query can be read as it is, with no call to make. With an
    ``extractor`` the query is read by it first, and each concept it names is matched on
    its own: joining them into one string would match a long text against short names
    again.
    """

    def __init__(
        self, extractor: BaseExtractor | None = None, language: Language = Language.ITALIAN
    ) -> None:
        self.extractor = extractor
        self.language = language

    def represent(self, names: list[str]) -> list[dict[str, object]]:
        return [{"roots": " ".join(roots(name, self.language))} for name in names]

    def seeds(self, query: str, graph: RelationIndex) -> dict[str, float]:
        texts = self.extractor.entities(query) if self.extractor is not None else [query]
        found = [graph._match(roots(text, self.language)) for text in texts]
        return _best(found, graph.max_seeds)


class SemanticSeeder(BaseSeeder):
    """Cosine between the query and the embeddings of the concept names.

    Finds what no word match can — "steroidi" near "corticosteroidi" — at the price of an
    embedding per node. With an ``extractor`` each concept it names is embedded on its own:
    short against short is where similarity of names works best, and a whole question is
    further from a two-word name than any two names are. Concepts at or below
    ``min_similarity`` are not entered, since a cosine is never zero and would otherwise
    open ``max_seeds`` doors for any query at all.
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        extractor: BaseExtractor | None = None,
        min_similarity: float = 0.0,
    ) -> None:
        self.embedder = embedder
        self.extractor = extractor
        self.min_similarity = min_similarity

    def represent(self, names: list[str]) -> list[dict[str, object]]:
        if not names:
            return []
        return [{"embedding": vector.tolist()} for vector in self.embedder.embed_chunks(names)]

    def seeds(self, query: str, graph: RelationIndex) -> dict[str, float]:
        if self.extractor is None:
            vectors = self.embedder.embed_query(query)
        else:
            named = self.extractor.entities(query)
            vectors = self.embedder.embed_chunks(named) if named else []
        found = [graph._nearest(vector.tolist()) for vector in vectors]
        return _best(found, graph.max_seeds, self.min_similarity)
