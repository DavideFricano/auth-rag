from __future__ import annotations

from collections.abc import Iterator

from auth_rag.augmentation.augmenter import BaseAugmenter
from auth_rag.authorization.filter import Filter
from auth_rag.generation.llm import BaseLLMClient
from auth_rag.indexing.index import BaseIndex
from auth_rag.ingestion.chunker import BaseChunker
from auth_rag.ingestion.labeler import BaseLabeler
from auth_rag.ingestion.loader import BaseLoader
from auth_rag.ranking.fusion_ranker import FusionRanker
from auth_rag.ranking.reranker import Reranker
from auth_rag.storing.store import BaseStore
from auth_rag.types import Chunk, ScoredChunk


class IngestionPipeline:
    """Offline pipeline: loads and chunks documents, then writes the store and indexes.

    It is the write-side coordinator between the shared store (the chunk records) and
    the indexes (ids + vectors/tokens): the store is written once, then every index is
    told to index the same chunks. ``remove`` fans out the same way so a source is
    dropped from every index and then from the store — no dangling ids, no orphans.

    The labeler, when there is one, runs between loading and chunking: it writes the
    attributes once on the document's ``Source``, which every chunk then carries.
    """

    def __init__(
        self,
        loader: BaseLoader,
        chunker: BaseChunker,
        store: BaseStore,
        indexes: list[BaseIndex],
        labeler: BaseLabeler | None = None,
    ) -> None:
        self.loader = loader
        self.chunker = chunker
        self.store = store
        self.indexes = indexes
        self.labeler = labeler

    def ingest(self) -> list[Chunk]:
        docs = self.loader.load()
        if self.labeler is not None:
            docs = (self.labeler.label(doc) for doc in docs)
        chunks = [chunk for doc in docs for chunk in self.chunker.chunk(doc)]
        self.store.add(chunks)
        for index in self.indexes:
            index.insert(chunks)
        return chunks

    def remove(self, source_id: str) -> None:
        for index in self.indexes:
            index.delete(source_id)
        self.store.delete(source_id)


class QueryPipeline:
    """Online pipeline: retrieves from the indexes, fuses, optionally reranks, and generates.

    Each index resolves its own hits to chunks through the shared store, so retrieval is
    a list of ``ScoredChunk`` lists that the fusion ranker merges (RRF/RSF).

    The authorization filter travels through it without being interpreted: the caller (a
    PEP, holding the identity this library never sees) decides it, the indexes apply it.
    """

    def __init__(
        self,
        indexes: list[BaseIndex],
        ranker: FusionRanker,
        augmenter: BaseAugmenter,
        llm: BaseLLMClient,
        top_i: int,
        top_k: int,
        top_n: int,
        reranker: Reranker | None = None,
    ) -> None:
        self.indexes = indexes
        self.ranker = ranker
        self.augmenter = augmenter
        self.reranker = reranker
        self.llm = llm
        self.top_i = top_i  # candidates pulled from each index
        self.top_k = top_k  # results kept after fusion (fed to the reranker)
        self.top_n = top_n  # results kept after reranking (final context)

    def retrieve(self, query: str, filter: Filter | None = None) -> list[ScoredChunk]:
        """The authorization predicate is handed to every index untouched.

        Whether it may be omitted is decided by the index, not here: duplicating that rule
        would only create a second place to get it wrong.
        """
        retrieved = [index.retrieve(query, self.top_i, filter) for index in self.indexes]
        fused = self.ranker.rank(retrieved, self.top_k)
        if self.reranker is not None:
            context = self.reranker.rank([sc.chunk for sc in fused], query, self.top_n)
        else:
            context = fused[:self.top_n]
        return context

    def query(self, query: str, filter: Filter | None = None) -> str:
        context = self.retrieve(query, filter)
        prompt = self.augmenter.build(query, context)
        return self.llm.answer(prompt)

    def stream(self, query: str, filter: Filter | None = None) -> Iterator[str]:
        context = self.retrieve(query, filter)
        prompt = self.augmenter.build(query, context)
        yield from self.llm.stream(prompt)


class RagPipeline:
    """Collects the ingestion and query pipelines behind one object (facade).

    Store and indexes are shared between the two: what the ingestion writes is exactly
    what the query reads.
    """

    def __init__(
        self,
        loader: BaseLoader,
        chunker: BaseChunker,
        store: BaseStore,
        indexes: list[BaseIndex],
        ranker: FusionRanker,
        augmenter: BaseAugmenter,
        llm: BaseLLMClient,
        top_i: int,
        top_k: int,
        top_n: int,
        reranker: Reranker | None = None,
        labeler: BaseLabeler | None = None,
    ) -> None:
        self.ingest_pipeline = IngestionPipeline(loader, chunker, store, indexes, labeler)
        self.query_pipeline = QueryPipeline(
            indexes, ranker, augmenter, llm, top_i, top_k, top_n, reranker
        )
