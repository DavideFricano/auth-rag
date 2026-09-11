"""End-to-end integration tests for the RAG pipeline wiring.

Unit tests cover each component in isolation; this file checks that the pieces the
recent refactor touched fit together: the shared store resolving ids back to chunks,
the ``top_i`` -> ``top_k`` -> ``top_n`` cascade (per-index depth, fused cut, final
context), the fusion over both indices, and the delete fan-out (every index, then the
store). Everything runs on the in-memory tiers with fakes for the embedder, loader,
chunker and LLM, so no Qdrant server, embedding model or Ollama endpoint is needed.

The observable seam is ``QueryPipeline.retrieve`` (the retrieval step behind
``query``/``stream``): it returns the fused, reranked, cut context as ``ScoredChunk``.
"""

from collections.abc import Iterator
from datetime import date

import numpy as np
import pytest

from auth_rag.augmentation.augmenter import PromptAugmenter
from auth_rag.authorization.filter import Match
from auth_rag.authorization.schema import AccessSchema, Attribute, AttributeType
from auth_rag.embedding.embedder import BaseEmbedder
from auth_rag.errors import EnforcementError
from auth_rag.generation.llm import BaseLLMClient
from auth_rag.indexing.similarity.lexical_index import VolatileLexicalIndex
from auth_rag.indexing.similarity.semantic_index import VolatileSemanticIndex
from auth_rag.ingestion.chunker import BaseChunker
from auth_rag.ingestion.labeler import StaticLabeler
from auth_rag.ingestion.loader import BaseLoader
from auth_rag.pipeline import RagPipeline
from auth_rag.ranking.fusion_ranker import ReciprocalRankFusionRanker
from auth_rag.storing.store import VolatileStore
from auth_rag.types import Chunk, Document, Language, Message, Metadata, Origin, Source


def _source(id: str) -> Source:
    return Source(id=id, name=f"{id}.pdf", origin=Origin.LOCAL, time=date(2024, 1, 1))


def _chunk(id: str, text: str, source: Source) -> Chunk:
    return Chunk(id=id, text=text, metadata=Metadata(source=source, title="S"))


# Two source documents, each split into two chunks. Only the "febbre" chunks (c0, c2)
# are relevant to the query below; c1/c3 are distractors from a different topic.
_SRC1, _SRC2 = _source("doc1"), _source("doc2")
_C0, _C1 = "febbre alta e tosse", "pressione arteriosa nella norma"
_C2, _C3 = "febbre e brividi", "glicemia e insulina"
_CHUNKS = {
    "doc1": [_chunk("c0", _C0, _SRC1), _chunk("c1", _C1, _SRC1)],
    "doc2": [_chunk("c2", _C2, _SRC2), _chunk("c3", _C3, _SRC2)],
}
_DOCS = [
    Document(text=f"{_C0}. {_C1}", source=_SRC1),
    Document(text=f"{_C2}. {_C3}", source=_SRC2),
]

# Fake dense geometry: the query aligns with the two "febbre" chunks, the distractors
# sit on an orthogonal axis (cosine 0). Dim 2 keeps it trivially readable.
_TABLE = {
    _C0: [1.0, 0.0],
    _C2: [0.8, 0.2],
    _C1: [0.0, 1.0],
    _C3: [0.0, 1.0],
    "febbre": [1.0, 0.0],
}


class _FakeEmbedder(BaseEmbedder):
    """Maps exact text -> preset embedding (zeros for unknown), so tests control geometry."""

    def __init__(self, table: dict[str, list[float]], dim: int) -> None:
        self._table = {t: np.asarray(v, dtype=np.float32) for t, v in table.items()}
        self._dim = dim

    def _vec(self, text: str) -> np.ndarray:
        return self._table.get(text, np.zeros(self._dim, dtype=np.float32))

    def embed_chunks(self, chunks):
        return np.array([self._vec(t) for t in chunks], dtype=np.float32)

    def embed_query(self, query):
        return np.array([self._vec(query)], dtype=np.float32)


class _FakeLoader(BaseLoader):
    """Yields preset Documents instead of reading the filesystem."""

    def __init__(self, docs: list[Document]) -> None:
        self._docs = docs

    def load(self) -> Iterator[Document]:
        yield from self._docs


class _FakeChunker(BaseChunker):
    """Returns preset chunks per source id, so ids and source ids are fully controlled."""

    def __init__(self, by_source: dict[str, list[Chunk]]) -> None:
        self._by_source = by_source

    def chunk(self, doc: Document) -> list[Chunk]:
        return self._by_source[doc.source.id]


class _CapturingLLM(BaseLLMClient):
    """Records the messages it was handed and returns a canned answer."""

    def __init__(self) -> None:
        self.last_messages: list[Message] = []

    def answer(self, messages: list[Message], **kwargs) -> str:
        self.last_messages = messages
        return "RISPOSTA"

    def stream(self, messages: list[Message], **kwargs) -> Iterator[str]:
        self.last_messages = messages
        yield from ("RIS", "POSTA")


def _build(top_i: int, top_k: int, top_n: int) -> tuple[RagPipeline, VolatileStore, _CapturingLLM]:
    store = VolatileStore()
    embedder = _FakeEmbedder(_TABLE, dim=2)
    llm = _CapturingLLM()
    rag = RagPipeline(
        loader=_FakeLoader(_DOCS),
        chunker=_FakeChunker(_CHUNKS),
        store=store,
        indexes=[
            VolatileSemanticIndex(store, embedder),
            VolatileLexicalIndex(store, language=Language.ITALIAN),
        ],
        ranker=ReciprocalRankFusionRanker(),
        augmenter=PromptAugmenter(),
        llm=llm,
        top_i=top_i,
        top_k=top_k,
        top_n=top_n,
    )
    return rag, store, llm


def _ids(context) -> set[str]:
    return {sc.chunk.id for sc in context}


def test_query_end_to_end_answers_from_shared_store():
    """ingest -> retrieve -> fuse -> rerank/cut -> augment -> generate, with chunk text
    coming back through the shared store (indices hold only ids)."""
    rag, _, llm = _build(top_i=10, top_k=10, top_n=5)
    rag.ingest_pipeline.ingest()

    out = rag.query_pipeline.query("febbre")

    assert out == "RISPOSTA"
    context = llm.last_messages[-1].content  # the user message carries the retrieved context
    assert _C0 in context  # resolved from ids -> full chunk text via the store


def test_retrieve_fuses_both_indices_and_resolves_chunks():
    rag, _, _ = _build(top_i=10, top_k=10, top_n=10)
    rag.ingest_pipeline.ingest()

    context = rag.query_pipeline.retrieve("febbre")

    assert {"c0", "c2"} <= _ids(context)  # both "febbre" chunks surface, fused across indices
    assert all(sc.chunk.text for sc in context)  # ids were resolved to full chunks via the store


def test_top_k_caps_the_fused_result():
    """top_k is the cut applied after fusion; top_n is kept wide so top_k is the binding limit."""
    rag, _, _ = _build(top_i=10, top_k=2, top_n=10)
    rag.ingest_pipeline.ingest()

    assert len(rag.query_pipeline.retrieve("febbre")) == 2


def test_top_i_caps_candidates_per_index():
    """top_i is the per-index depth: with 1, each index contributes at most its single
    best hit, so the fused set holds at most one chunk per index."""
    rag, _, _ = _build(top_i=1, top_k=10, top_n=10)
    rag.ingest_pipeline.ingest()

    context = rag.query_pipeline.retrieve("febbre")

    assert 1 <= len(context) <= 2  # two indices, one candidate each
    assert _ids(context) <= {"c0", "c2"}  # only the relevant top hits, never the distractors


def test_top_n_caps_the_final_context():
    """top_n is the final cut sent to the LLM, applied after fusion/reranking."""
    rag, _, _ = _build(top_i=10, top_k=10, top_n=1)
    rag.ingest_pipeline.ingest()

    assert len(rag.query_pipeline.retrieve("febbre")) == 1


def test_delete_fans_out_to_every_index_and_store():
    rag, store, _ = _build(top_i=10, top_k=10, top_n=10)
    rag.ingest_pipeline.ingest()

    rag.ingest_pipeline.remove("doc1")

    context = rag.query_pipeline.retrieve("febbre")
    assert all(sc.chunk.metadata.source.id != "doc1" for sc in context)  # gone from the indices
    assert store.get(["c0"]) == []  # and the record was dropped from the store
    assert store.get(["c2"])  # doc2 is untouched


# --- ABAC end to end: what the labeler writes at ingestion is what the filter reads -------

_ABAC_SCHEMA = AccessSchema(
    [Attribute(name="tenant", type=AttributeType.KEYWORD, required=True)]
)


class _SourceChunker(BaseChunker):
    """One chunk per document, carrying the document's ``Source`` — the way the real
    chunkers do, so whatever the labeler wrote reaches the chunk."""

    def chunk(self, doc: Document) -> list[Chunk]:
        metadata = Metadata(source=doc.source, title="S")
        return [Chunk(id=f"{doc.source.id}:0", text=doc.text, metadata=metadata)]


def _abac_pipeline(labeler: StaticLabeler | None) -> RagPipeline:
    store = VolatileStore()
    embedder = _FakeEmbedder(_TABLE, dim=2)
    return RagPipeline(
        loader=_FakeLoader(_DOCS),
        labeler=labeler,
        chunker=_SourceChunker(),
        store=store,
        indexes=[
            VolatileSemanticIndex(store, embedder, schema=_ABAC_SCHEMA),
            VolatileLexicalIndex(store, language=Language.ITALIAN, schema=_ABAC_SCHEMA),
        ],
        ranker=ReciprocalRankFusionRanker(),
        augmenter=PromptAugmenter(system="S"),
        llm=_CapturingLLM(),
        top_i=10,
        top_k=10,
        top_n=10,
    )


def test_what_the_labeler_wrote_is_what_the_filter_matches():
    """The whole round trip: the attributes are written once on the document's Source at
    ingestion, ride into every chunk, and are read back by the enforcement at query time."""
    rag = _abac_pipeline(StaticLabeler(_ABAC_SCHEMA, {"tenant": "acme"}))
    rag.ingest_pipeline.ingest()

    authorized = Match(attribute="tenant", values={"acme"})
    results = [
        index.retrieve("febbre", 10, filter=authorized)
        for index in rag.query_pipeline.indexes
    ]
    assert all(hits for hits in results)
    assert all(sc.chunk.metadata.source.access == {"tenant": "acme"} for hits in results for sc in hits)

    other_tenant = Match(attribute="tenant", values={"globex"})
    assert all(
        index.retrieve("febbre", 10, filter=other_tenant) == []
        for index in rag.query_pipeline.indexes
    )


def test_an_ingestion_without_a_labeler_produces_nothing_retrievable():
    """The schema is declared but nothing labels, so every chunk misses the required
    attribute and is denied — the corpus is unreachable rather than wide open."""
    rag = _abac_pipeline(labeler=None)
    assert rag.ingest_pipeline.ingest()  # the chunks are indexed...

    authorized = Match(attribute="tenant", values={"acme"})
    assert all(
        index.retrieve("febbre", 10, filter=authorized) == []
        for index in rag.query_pipeline.indexes
    )  # ...but none of them can be retrieved


def test_the_pipeline_carries_the_filter_down_to_every_index():
    """The propagation is the whole feature: what the caller passes reaches the indexes
    untouched, and the enforcement there is what shapes the result."""
    rag = _abac_pipeline(StaticLabeler(_ABAC_SCHEMA, {"tenant": "acme"}))
    rag.ingest_pipeline.ingest()

    authorized = rag.query_pipeline.retrieve("febbre", Match(attribute="tenant", values={"acme"}))
    denied = rag.query_pipeline.retrieve("febbre", Match(attribute="tenant", values={"globex"}))

    assert authorized
    assert denied == []


def test_querying_an_abac_pipeline_without_a_filter_is_refused():
    """QueryPipeline does not re-implement the rule — it lets the index raise, so there is
    a single place where 'a declared schema means a mandatory filter' is decided."""
    rag = _abac_pipeline(StaticLabeler(_ABAC_SCHEMA, {"tenant": "acme"}))
    rag.ingest_pipeline.ingest()

    with pytest.raises(EnforcementError, match="requires a filter"):
        rag.query_pipeline.retrieve("febbre")


def test_the_filter_reaches_generation_too():
    """query/stream retrieve through the same path, so an unauthorized chunk never gets
    near the prompt: the LLM only ever sees context that was already authorized."""
    rag = _abac_pipeline(StaticLabeler(_ABAC_SCHEMA, {"tenant": "acme"}))
    rag.ingest_pipeline.ingest()
    llm = rag.query_pipeline.llm

    # _C1 marks the retrieved text: unlike "febbre" it cannot come from the question itself
    rag.query_pipeline.query("febbre", Match(attribute="tenant", values={"globex"}))
    assert _C1 not in llm.last_messages[-1].content  # no chunk survived the filter

    rag.query_pipeline.query("febbre", Match(attribute="tenant", values={"acme"}))
    assert _C1 in llm.last_messages[-1].content


def test_without_a_schema_the_pipeline_keeps_working_unfiltered():
    """The simple deployment is untouched: no schema, no filter, everything retrievable."""
    rag, _, _ = _build(top_i=10, top_k=10, top_n=10)
    rag.ingest_pipeline.ingest()

    assert rag.query_pipeline.retrieve("febbre")
