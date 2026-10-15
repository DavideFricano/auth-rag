from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence

from pydantic import BaseModel, ConfigDict, Field

from auth_rag.authorization.filter import Filter
from auth_rag.authorization.schema import AccessSchema
from auth_rag.indexing.index import BaseIndex
from auth_rag.indexing.relation.extractor import BaseExtractor
from auth_rag.indexing.relation.seeder import BaseSeeder, LexicalSeeder, _best
from auth_rag.storing.store import BaseStore
from auth_rag.types import Chunk, ScoredChunk


def normalize(name: str) -> str:
    """Canonical spelling of a concept name: nodes merge by name, and the merge is what
    builds the bridges, so casing and spacing must not decide whether one exists."""
    return " ".join(name.lower().split())


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

    **The entry is a list of seeders**, injected: each decides what a node carries and how a
    query is read against it, and the graph enters from the best each concept scored with
    any of them. By default one, lexical, reading the query as it is, which costs no call
    per search; reading it through the extractor first is a choice, made by giving a seeder
    one. An empty list leaves no entry from a query, for a graph used only as an expander.
    The entry is where this family lives or dies — a graph nobody can enter expands
    nothing, however good it is.

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
        seeders: list[BaseSeeder] | None = None,
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
        self.seeders = seeders if seeders is not None else [LexicalSeeder()]

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

    def _represent(self, name: str) -> dict[str, object]:
        """What a new node carries: what every seeder needs to find it by."""
        return {
            key: value for seeder in self.seeders for key, value in seeder.represent(name).items()
        }

    @abstractmethod
    def _add(self, relations: Sequence[Relation]) -> None:
        """Record these relations; the same relation twice is one edge, or re-ingesting
        a corpus would grow the graph instead of leaving it as it was."""

    @abstractmethod
    def _match(self, roots: list[str]) -> dict[str, float]:
        """BM25 of these roots against the roots each node carries, for the lexical seeder.

        Matched root by root and not as a whole name, because the two sides do not have to
        agree on granularity: ingestion may have produced the node ``colite
        immunocorrelata`` where the query only names ``colite``. At most ``max_seeds``,
        best first.
        """

    @abstractmethod
    def _nearest(self, vector: list[float]) -> dict[str, float]:
        """Cosine of this vector against the embedding each node carries, for the semantic
        seeder. At most ``max_seeds``, best first."""

    @abstractmethod
    def _concepts(self, chunk_ids: Sequence[str]) -> Iterable[tuple[str, str]]:
        """``(chunk_id, concept name)`` for each concept these chunks were read in."""

    @abstractmethod
    def _expand(self, seeds: dict[str, float]) -> Iterable[tuple[str, int, float]]:
        """``(chunk_id, hop, seed score)`` for each chunk within ``hops`` of a seed.
        Duplicates are expected — picking the best is ``_search``'s job."""

    def _search(self, query: str, top_i: int) -> list[tuple[str, float]]:
        """Seeds, expansion, and the step that turns reachability into a score.

        The seeders decide where the query enters. A query it finds no concept for opens
        nothing, and that is the honest answer: this index has no entry point for it. The
        similarity indexes are still answering, so the pipeline is not left empty.

        A traversal then yields reachability, not similarity, so the score is invented:
        ``seed * decay ** hop``. Monotone and one parameter, at the price of ignoring how
        strong each relation is.

        **This constrains the fusion ranker to RRF.** The number is not commensurable with
        a cosine or with BM25; RRF reads only positions, while RSF and DBSF normalize over
        each list's extremes or moments and would hand the graph whatever weight the shape
        of the decay curve implied. Calibrating instead would need an evaluation harness.
        """
        seeds = _best([seeder.seeds(query, self) for seeder in self.seeders], self.max_seeds)
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

    def expand(
        self, found: list[ScoredChunk], top_i: int, filter: Filter | None = None
    ) -> list[ScoredChunk]:
        """The chunks the found ones lead to, for an expansion stage after fusion.

        Enters through the concepts the found chunks were read in rather than through the
        query, so it follows the retrievers instead of competing with them. A chunk reached
        inherits the score of the one it was reached from, ``score * decay ** (hop + 1)``:
        at least one step below it, on the scale of the list it extends. What was already
        found is not handed back.
        """
        self._check(filter)
        scores = {sc.chunk.id: sc.score for sc in found}
        seeds: dict[str, float] = {}
        for chunk_id, name in self._concepts(list(scores)):
            seeds[name] = max(seeds.get(name, 0.0), scores[chunk_id])
        best: dict[str, float] = {}
        for chunk_id, hop, seed_score in self._expand(seeds):
            if chunk_id in scores:
                continue
            score = seed_score * self.decay ** (hop + 1)
            if score > best.get(chunk_id, 0.0):
                best[chunk_id] = score
        ranked = sorted(best.items(), key=lambda row: (-row[1], row[0]))
        return self._authorized(ranked[: top_i * self.over_fetch], filter)[:top_i]
