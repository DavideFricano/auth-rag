from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray
from openai import OpenAI
from sentence_transformers import SentenceTransformer


class BaseEmbedder(ABC):
    """Common interface for embedding models."""

    @abstractmethod
    def embed_chunks(self, chunks: list[str]) -> NDArray[np.float32]:
        pass

    @abstractmethod
    def embed_query(self, query: str) -> NDArray[np.float32]:
        pass


class LocalEmbedder(BaseEmbedder):
    """Embedding model backed by a local SentenceTransformer."""

    def __init__(self, model_name: str) -> None:
        self.model_name = str(model_name)
        self.model = SentenceTransformer(model_name)

    def embed_chunks(self, chunks: list[str]) -> NDArray[np.float32]:
        embeddings = self.model.encode(
            chunks,
            convert_to_numpy=True,
            show_progress_bar=True,
            batch_size=16,
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> NDArray[np.float32]:
        embedding = self.model.encode([query], convert_to_numpy=True)
        return embedding.astype(np.float32)


class OpenAIEmbedder(BaseEmbedder):
    """Embedding model served by the OpenAI Embeddings API."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.client = OpenAI()

    def embed_chunks(self, chunks: list[str]) -> NDArray[np.float32]:
        response = self.client.embeddings.create(input=chunks, model=self.model_name)
        vectors = [item.embedding for item in response.data]
        return np.array(vectors, dtype=np.float32)

    def embed_query(self, query: str) -> NDArray[np.float32]:
        response = self.client.embeddings.create(input=[query], model=self.model_name)
        return np.array([response.data[0].embedding], dtype=np.float32)
