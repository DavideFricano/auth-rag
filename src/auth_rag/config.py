from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from auth_rag.types import Language


class Settings(BaseSettings):
    """Runtime configuration, read from environment variables or a .env file.

    Field names map to upper-case env vars (e.g. llm_url -> LLM_URL).
    Everything has a sensible local default, so an empty environment still works.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # General (e.g. BM25 stemming and stopwords)
    language: Language = Language.ITALIAN

    # Generation (Ollama endpoint by default)
    llm_url: str = "http://localhost:11434/api/chat"
    llm_model: str = "qwen2.5:7b"
    system_prompt_path: Path = Path("data/prompts/system_prompt.md")

    # Embedding
    embed_model: str = "BAAI/bge-m3"

    # Retrieval
    top_i: int = 20  # per-index retrieval depth (candidate pool fed to fusion)
    top_k: int = 10  # results kept after fusion (candidates fed to the reranker)
    top_n: int = 5  # results kept after reranking (final context sent to the LLM)

    # Ingestion paths
    in_dir: Path = Path("data/raw")
    out_dir: Path = Path("data/out")

    # Authorization: the declared access schema. Unset means this deployment does no
    # ABAC — nothing labels, nothing filters, retrieval returns everything.
    access_schema_path: Path | None = None

    # Where the access attributes of local documents are curated, for a corpus with no
    # producer upstream. Unset means they are expected to arrive with the documents.
    access_manifest_path: Path | None = None
