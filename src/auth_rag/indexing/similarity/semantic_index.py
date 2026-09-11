from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from auth_rag.authorization.schema import AccessSchema
from auth_rag.embedding.embedder import BaseEmbedder
from auth_rag.indexing.similarity.index import SimilarityIndex
from auth_rag.storing.store import BaseStore
from auth_rag.types import Chunk


class SemanticIndex(SimilarityIndex):
    """Shared dense-retrieval logic backed by Qdrant (cosine).

    Not used directly: the deployment-role classes below subclass it and differ only
    in how the client (the hidden ``db``) is built — in-memory, on disk, or a server —
    so behaviour is identical across tiers. Holds only ids + dense vectors; the chunks
    live in the shared Store, so there is no duplication.
    """

    def __init__(
        self,
        store: BaseStore,
        embedder: BaseEmbedder,
        db: QdrantClient,
        collection: str,
        schema: AccessSchema | None = None,
    ) -> None:
        super().__init__(store, db, collection, schema)
        self.embedder = embedder
        self.dim: int | None = None

    def _ensure_collection(self, dim: int) -> None:
        if self.dim is not None:
            return
        self.dim = dim
        if not self.db.collection_exists(self.collection):
            self.db.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    def insert(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        embeddings = self.embedder.embed_chunks([chunk.text for chunk in chunks])
        self._ensure_collection(embeddings.shape[1])
        points = [
            self._point(chunk, row.tolist()) for chunk, row in zip(chunks, embeddings, strict=True)
        ]
        self.db.upsert(collection_name=self.collection, points=points)

    def _query_vector(self, query: str) -> tuple[Any, str | None]:
        return self.embedder.embed_query(query)[0].tolist(), None


class VolatileSemanticIndex(SemanticIndex):
    """Dense index in an in-memory Qdrant instance. Non-durable; zero setup."""

    def __init__(
        self,
        store: BaseStore,
        embedder: BaseEmbedder,
        collection: str = "semantic",
        schema: AccessSchema | None = None,
    ) -> None:
        super().__init__(store, embedder, QdrantClient(location=":memory:"), collection, schema)


class PersistentSemanticIndex(SemanticIndex):
    """Dense index in an embedded Qdrant persisted under `path` (single process, no server)."""

    def __init__(
        self,
        store: BaseStore,
        embedder: BaseEmbedder,
        path: str = "./data/db/semantic",
        collection: str = "semantic",
        schema: AccessSchema | None = None,
    ) -> None:
        super().__init__(store, embedder, QdrantClient(path=path), collection, schema)


class RemoteSemanticIndex(SemanticIndex):
    """Dense index on a Qdrant server at `url` (shared across processes)."""

    def __init__(
        self,
        store: BaseStore,
        embedder: BaseEmbedder,
        url: str,
        collection: str = "semantic",
        schema: AccessSchema | None = None,
        **client_kwargs,
    ) -> None:
        super().__init__(
            store, embedder, QdrantClient(url=url, **client_kwargs), collection, schema
        )
