from __future__ import annotations

from typing import Any

from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Modifier, SparseVector, SparseVectorParams

from auth_rag.authorization.schema import AccessSchema
from auth_rag.indexing.similarity.index import SimilarityIndex
from auth_rag.storing.store import BaseStore
from auth_rag.types import Chunk, Language


class LexicalIndex(SimilarityIndex):
    """Shared sparse (BM25) retrieval backed by Qdrant.

    Tokenizing, stemming and stopwords come from FastEmbed's ``Qdrant/bm25`` (language
    aware); the corpus IDF is applied server-side by Qdrant's IDF modifier — so it is
    real BM25, identical across tiers. Holds only ids + the sparse vector; chunks live
    in the shared Store. Role subclasses differ only in how the ``db`` client is built.
    """

    def __init__(
        self,
        store: BaseStore,
        db: QdrantClient,
        collection: str = "lexical",
        language: Language = Language.ENGLISH,
        schema: AccessSchema | None = None,
    ) -> None:
        super().__init__(store, db, collection, schema)
        self.model = SparseTextEmbedding("Qdrant/bm25", language=str(language))
        self._ready = False

    def _ensure_collection(self) -> None:
        if self._ready:
            return
        self._ready = True
        if not self.db.collection_exists(self.collection):
            self.db.create_collection(
                collection_name=self.collection,
                vectors_config={},
                sparse_vectors_config={"text": SparseVectorParams(modifier=Modifier.IDF)},
            )

    def insert(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        self._ensure_collection()
        embeddings = self.model.embed([chunk.text for chunk in chunks])
        points = [
            self._point(
                chunk,
                {"text": SparseVector(indices=emb.indices.tolist(), values=emb.values.tolist())},
            )
            for chunk, emb in zip(chunks, embeddings, strict=True)
        ]
        self.db.upsert(collection_name=self.collection, points=points)

    def _query_vector(self, query: str) -> tuple[Any, str | None]:
        emb = next(iter(self.model.query_embed(query)))
        return SparseVector(indices=emb.indices.tolist(), values=emb.values.tolist()), "text"


class VolatileLexicalIndex(LexicalIndex):
    """Sparse BM25 index in an in-memory Qdrant instance. Non-durable; zero setup."""

    def __init__(
        self,
        store: BaseStore,
        collection: str = "lexical",
        language: Language = Language.ENGLISH,
        schema: AccessSchema | None = None,
    ) -> None:
        super().__init__(store, QdrantClient(location=":memory:"), collection, language, schema)


class PersistentLexicalIndex(LexicalIndex):
    """Sparse BM25 index in an embedded Qdrant persisted under `path` (single process)."""

    def __init__(
        self,
        store: BaseStore,
        path: str = "./data/db/lexical",
        collection: str = "lexical",
        language: Language = Language.ENGLISH,
        schema: AccessSchema | None = None,
    ) -> None:
        super().__init__(store, QdrantClient(path=path), collection, language, schema)


class RemoteLexicalIndex(LexicalIndex):
    """Sparse BM25 index on a Qdrant server at `url` (shared across processes)."""

    def __init__(
        self,
        store: BaseStore,
        url: str,
        collection: str = "lexical",
        language: Language = Language.ENGLISH,
        schema: AccessSchema | None = None,
        **client_kwargs,
    ) -> None:
        super().__init__(
            store, QdrantClient(url=url, **client_kwargs), collection, language, schema
        )
