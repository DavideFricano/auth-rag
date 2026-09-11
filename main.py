from __future__ import annotations

from auth_rag.augmentation.augmenter import PromptAugmenter
from auth_rag.authorization.schema import AccessSchema
from auth_rag.config import Settings
from auth_rag.embedding.embedder import LocalEmbedder
from auth_rag.generation.llm import OllamaClient
from auth_rag.indexing.similarity.lexical_index import VolatileLexicalIndex
from auth_rag.indexing.similarity.semantic_index import VolatileSemanticIndex
from auth_rag.ingestion.chunker import HierarchicalChunker
from auth_rag.ingestion.labeler import ManifestLabeler, PropagatingLabeler
from auth_rag.ingestion.loader import FileLoader
from auth_rag.pipeline import RagPipeline
from auth_rag.ranking.fusion_ranker import ReciprocalRankFusionRanker
from auth_rag.storing.store import VolatileStore

if __name__ == "__main__":
    settings = Settings()

    # One shared store for the chunk data; the indexes hold only ids + vectors/tokens
    # and resolve chunks back through it.
    store = VolatileStore()
    embedder = LocalEmbedder(settings.embed_model)

    # The access schema is the contract between the two pipelines: declared once here and
    # injected into everything that writes attributes or filters on them. Left unset it is
    # None, and this stays a deployment without access control.
    path = settings.access_schema_path
    schema = AccessSchema.from_file(path) if path is not None else None

    # No schema means no ABAC. With one, the attributes either arrive with the documents
    # (a gateway, a producer upstream) or are curated locally in a manifest.
    if schema is None:
        labeler = None
    elif settings.access_manifest_path is not None:
        labeler = ManifestLabeler(schema, settings.access_manifest_path)
    else:
        labeler = PropagatingLabeler(schema)

    rag = RagPipeline(
        loader=FileLoader(settings.in_dir, settings.out_dir, save_output=True),
        labeler=labeler,
        chunker=HierarchicalChunker(),
        store=store,
        indexes=[
            VolatileSemanticIndex(store, embedder, schema=schema),
            VolatileLexicalIndex(store, language=settings.language, schema=schema),
        ],
        ranker=ReciprocalRankFusionRanker(),
        augmenter=PromptAugmenter(system=settings.system_prompt_path.read_text(encoding="utf-8")),
        llm=OllamaClient(model=settings.llm_model, url=settings.llm_url),
        top_i=settings.top_i,
        top_k=settings.top_k,
        top_n=settings.top_n,
    )

    print("Ingestion...")
    chunks = rag.ingest_pipeline.ingest()
    print(f"Totale chunk: {len(chunks)}")

    print("Query...")
    while True:
        try:
            question = input("\nDomanda: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        print("\n--- RISPOSTA ---\n")
        for token in rag.query_pipeline.stream(question):
            print(token, end="", flush=True)
        print()
