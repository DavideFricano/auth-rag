from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from typing import Any

from qdrant_client import QdrantClient, models

from auth_rag.authorization.schema import AccessSchema
from auth_rag.indexing.index import BaseIndex
from auth_rag.storing.store import BaseStore
from auth_rag.types import Chunk

# Fixed namespace so a chunk id always maps to the same point id (idempotent upsert).
_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00cf4fc964ff")


class SimilarityIndex(BaseIndex, ABC):
    """Shared plumbing for the indexes that score each chunk on its own, by similarity to
    the query — as opposed to the relation family, which scores a chunk by its links.

    Subclasses own only the collection setup and the query value; point ids, payload,
    deletion and the search call are shared so the dense and the sparse index cannot
    drift apart. The engine never appears in the public signatures.
    """

    def __init__(
        self,
        store: BaseStore,
        db: QdrantClient,
        collection: str,
        schema: AccessSchema | None = None,
    ) -> None:
        super().__init__(store, schema)
        self.db = db
        self.collection = collection

    @abstractmethod
    def _query_vector(self, query: str) -> tuple[Any, str | None]:
        """The value to search with and the named vector to use (None = the unnamed one)."""

    def _point(self, chunk: Chunk, vector: Any) -> models.PointStruct:
        """One point per chunk: the id is a uuid5 of ``chunk.id`` so re-inserting upserts
        in place, and the payload carries only what this index must act on without reading
        the store — the id it resolves chunks by, the source it deletes by."""
        return models.PointStruct(
            id=str(uuid.uuid5(_NAMESPACE, chunk.id)),
            vector=vector,
            payload={"chunk_id": chunk.id, "source_id": chunk.metadata.source.id},
        )

    def delete(self, source_id: str) -> None:
        if not self.db.collection_exists(self.collection):
            return
        self.db.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source_id", match=models.MatchValue(value=source_id)
                        )
                    ]
                )
            ),
        )

    def _search(self, query: str, top_i: int) -> list[tuple[str, float]]:
        if not self.db.collection_exists(self.collection):
            return []
        value, using = self._query_vector(query)
        response = self.db.query_points(
            collection_name=self.collection,
            query=value,
            using=using,
            limit=top_i,
            with_payload=True,
        )
        return [(hit.payload["chunk_id"], float(hit.score)) for hit in response.points]
