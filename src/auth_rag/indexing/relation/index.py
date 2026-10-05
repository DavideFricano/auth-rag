from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence

from pydantic import BaseModel, ConfigDict, Field

from auth_rag.authorization.schema import AccessSchema
from auth_rag.indexing.index import BaseIndex
from auth_rag.indexing.relation.extractor import BaseExtractor
from auth_rag.storing.store import BaseStore
from auth_rag.types import Chunk

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def normalize(name: str) -> str:
    """Canonical spelling of a concept name: nodes merge by name, and the merge is what
    builds the bridges, so casing and spacing must not decide whether one exists."""
    return " ".join(name.lower().split())


def terms(name: str) -> set[str]:
    """Words of a concept name, for matching one name against another.

    Both sides are concept names — the extractor produces them at ingestion and again
    from the query — so this is a matcher, never the way a query enters the graph.
    Digits and words under three letters are dropped: they match too much to carry
    signal, at the cost of short acronyms, which cannot be seeds.
    """
    return {word for word in _WORD.findall(name.lower()) if len(word) > 2}


class Relation(BaseModel):
    """A triple anchored to the chunk it was read in.

    A node is merged by name and belongs to many chunks, so it is a bad place for anything
    chunk-specific; an edge has one provenance, because the extractor reads one chunk at
    a time. That is also what makes it deletable by source.
    """

    model_config = ConfigDict(frozen=True)

    subject: str = Field(description="Normalized name of the concept the relation starts from")
    predicate: str = Field(description="What relates them; carried, never interpreted")
    object: str = Field(description="Normalized name of the concept the relation reaches")
    chunk_id: str = Field(description="Id of the chunk this relation was read in")
    source_id: str = Field(description="Id of that chunk's source document")


class RelationIndex(BaseIndex, ABC):
    """Relevance by links instead of by similarity to the query.

    The similarity family scores each chunk on its own; this one scores a chunk by what it
    is connected to, which pays exactly where two chunks are *not* similar and similarity
    would never have put them together.

    Holds what makes the tiers one family — how a query becomes seeds, how a traversal
    becomes a score. A tier owns storage and walking, and nothing else is shared: unlike
    ``SimilarityIndex``, which is one Qdrant client built three ways, an in-process walk
    and a Cypher traversal are different engines.

    **Both sides go through the extractor**: it reads chunks to build the graph and reads
    the query to find where to enter it, so the names being matched were produced the same
    way rather than by a second vocabulary invented for retrieval. The entry is where this
    family lives or dies — a graph nobody can enter expands nothing, however good it is.

    **It does no filtering, deliberately.** It expands freely and ``retrieve`` applies the
    predicate once, as for the other two. Attributes on the edges with a check per hop
    would give a provably closed expansion, at the price of a second enforcement point to
    keep aligned with the first. The accepted cost: a path through chunks the subject
    cannot see may lift an authorized chunk up the ranking — an inference channel, not a
    content leak, since ``_search`` returns only ids and scores. It is also the one place
    where "never post-filter" is relaxed, and ``over_fetch`` is what pays for it.
    """

    def __init__(
        self,
        store: BaseStore,
        extractor: BaseExtractor,
        schema: AccessSchema | None = None,
        hops: int = 2,
        decay: float = 0.5,
        over_fetch: int = 3,
        max_seeds: int = 10,
    ) -> None:
        super().__init__(store, schema)
        if hops < 1:
            raise ValueError("hops must be at least 1, or nothing is ever expanded")
        if not 0.0 < decay < 1.0:
            raise ValueError("decay must lie in (0, 1): at 1, distance stops costing anything")
        if over_fetch < 1:
            raise ValueError("over_fetch must be at least 1")
        self.extractor = extractor
        self.hops = hops  # how far from a seed the expansion may reach
        self.decay = decay  # what one hop costs, multiplicatively
        self.over_fetch = over_fetch  # candidates asked for, per unit of the caller's top_i
        self.max_seeds = max_seeds  # how many entry points one query may open

    def insert(self, chunks: list[Chunk]) -> None:
        """Extract each chunk's relations and hand them to the graph.

        Extraction lives here because a relation is this index's own representation, the
        way a vector is the semantic index's: ``IngestionPipeline`` does not change.
        """
        relations = []
        for chunk in chunks:
            for triple in self.extractor.extract(chunk):
                subject, object_ = normalize(triple.subject), normalize(triple.object)
                if not subject or not object_ or subject == object_:
                    continue  # a self-loop bridges nothing
                relations.append(
                    Relation(
                        subject=subject,
                        predicate=triple.predicate,
                        object=object_,
                        chunk_id=chunk.id,
                        source_id=chunk.metadata.source.id,
                    )
                )
        self._add(relations)

    @abstractmethod
    def _add(self, relations: Sequence[Relation]) -> None:
        """Record these relations; the same relation twice is one edge, or re-ingesting
        a corpus would grow the graph instead of leaving it as it was."""

    @abstractmethod
    def _seeds(self, query_terms: set[str]) -> dict[str, float]:
        """Entry points: concept name -> the share of its own name that was matched.

        Receives the words of the concepts the extractor named, not the words of the
        query. It stays a token overlap rather than an exact lookup because the two sides
        do not have to agree on granularity: ingestion may have produced the node
        ``colite immunocorrelata`` where the query only names ``colite``, and an exact
        match would find nothing.

        At most ``max_seeds``, best first: a common word would otherwise open half the
        graph at once.
        """

    @abstractmethod
    def _expand(self, seeds: dict[str, float]) -> Iterable[tuple[str, int, float]]:
        """``(chunk_id, hop, seed score)`` for each chunk within ``hops`` of a seed.
        Duplicates are expected — picking the best is ``_search``'s job."""

    def _search(self, query: str, top_i: int) -> list[tuple[str, float]]:
        """Entities, seeds, expansion, and the step that turns reachability into a score.

        **The query is read by the same extractor that built the graph**, which is what
        keeps the two sides speaking one vocabulary: the entry is not a second way of
        naming concepts invented for retrieval. Splitting the query on whitespace instead
        would hand the matcher the words the *user* chose — "how", "treated" — rather than
        the concepts a document would name.

        A query the extractor finds no concept in opens nothing, and that is the honest
        answer: this index has no entry point for it. The similarity indexes are still
        answering, so the pipeline is not left empty.

        A traversal then yields reachability, not similarity, so the score is invented:
        ``seed * decay ** hop``. Monotone and one parameter, at the price of ignoring how
        strong each relation is.

        **This constrains the fusion ranker to RRF.** The number is not commensurable with
        a cosine or with BM25; RRF reads only positions, while RSF and DBSF normalize over
        each list's extremes or moments and would hand the graph whatever weight the shape
        of the decay curve implied. Calibrating instead would need an evaluation harness.
        """
        named = self.extractor.entities(query)
        seeds = self._seeds({term for name in named for term in terms(name)})
        if not seeds:
            return []
        best: dict[str, float] = {}
        for chunk_id, hop, seed_score in self._expand(seeds):
            score = seed_score * self.decay**hop
            if score > best.get(chunk_id, 0.0):
                best[chunk_id] = score
        # ties broken by id, so the same graph always answers the same way
        ranked = sorted(best.items(), key=lambda row: (-row[1], row[0]))
        return ranked[: top_i * self.over_fetch]
