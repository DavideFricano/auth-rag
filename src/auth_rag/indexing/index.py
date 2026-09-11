from __future__ import annotations

from abc import ABC, abstractmethod

from auth_rag.authorization.filter import Filter, evaluate
from auth_rag.authorization.schema import AccessSchema
from auth_rag.errors import EnforcementError
from auth_rag.storing.store import BaseStore
from auth_rag.types import Chunk, ScoredChunk


class BaseIndex(ABC):
    """A retrieval index over chunks: insert, delete, retrieve.

    Keeps only ids plus its own representation (dense vectors, sparse/BM25 terms, …) and
    resolves chunk data back from a **shared** ``Store``, which it **never writes**: the
    ingestion pipeline adds and deletes the records once, while each index manages only
    its own ids. Carries no backend import.

    An optional ``schema`` says whether this deployment does ABAC at all — without one a
    filter is merely accepted if given; with one it is mandatory. See ``retrieve``.
    """

    def __init__(self, store: BaseStore, schema: AccessSchema | None = None) -> None:
        self.store = store
        self.schema = schema

    @abstractmethod
    def insert(self, chunks: list[Chunk]) -> None:
        """Index the chunks (id + vector/tokens + their source id); idempotent by chunk.id.
        Does not write the store — the pipeline adds the records once."""

    @abstractmethod
    def delete(self, source_id: str) -> None:
        """Remove from THIS index every id of the given source. Does not touch the shared
        store — the pipeline deletes the records once, after every index has removed its ids."""

    @abstractmethod
    def _search(self, query: str, top_i: int) -> list[tuple[str, float]]:
        """Backend primitive: top ``top_i`` ``(chunk_id, score)``, higher = more relevant.
        Ids are unique — a chunk appears at most once, or fusion would count it twice.
        Not called directly: the public entry point is ``retrieve``."""

    def retrieve(
        self, query: str, top_i: int, filter: Filter | None = None
    ) -> list[ScoredChunk]:
        """Retrieve the authorized top ``top_i``, resolving chunks through the shared store.

        The filter is applied here, before fusion, because a ranker reads positions and
        score spreads within each list. A subclass may also push it into its backend; this
        check stays regardless, so an index that mistranslates it returns less rather than
        leaking. A chunk missing the schema's required attributes is dropped before the
        predicate runs.

        Schema and filter go together, in both directions. With a schema, omitting the
        filter raises — ``Allow()`` is how a call states it has no restriction. Without
        one, passing a filter raises too: there is no vocabulary to validate it against
        and no notion of a required attribute, so an unlabelled chunk would satisfy a
        negation. And a schemaless index asked for chunks that turn out to be labelled
        raises as well, since someone enforced something the query side knows nothing of.
        """
        if self.schema is None:
            if filter is not None:
                raise EnforcementError(
                    "filtering needs the AccessSchema this index was not given: without a "
                    "declared vocabulary the predicate cannot be validated and the required "
                    "attributes are unknown, so an unlabelled chunk would satisfy a negation"
                )
        elif filter is None:
            raise EnforcementError(
                "this index declares an access schema, so retrieve requires a filter; "
                "pass Allow() to state explicitly that the call has no restriction"
            )
        else:
            self.schema.validate_filter(filter)
        scored = self._search(query, top_i)
        by_id = {chunk.id: chunk for chunk in self.store.get([id_ for id_, _ in scored])}
        results = [
            ScoredChunk(chunk=by_id[id_], score=score) for id_, score in scored if id_ in by_id
        ]
        if filter is None:
            if any(sc.chunk.metadata.source.access for sc in results):
                raise EnforcementError(
                    "these chunks carry access attributes but this index enforces nothing: "
                    "give it the AccessSchema they were labelled against"
                )
            return results
        return [
            sc
            for sc in results
            if self.schema.is_labeled(sc.chunk.metadata.source.access)
            and evaluate(filter, sc.chunk.metadata.source.access)
        ]
