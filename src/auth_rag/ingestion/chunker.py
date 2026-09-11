from __future__ import annotations

import re
from abc import ABC, abstractmethod

import nltk
import numpy as np
from llama_index.core import Document as LlamaDocument
from llama_index.core.node_parser import MarkdownNodeParser
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer

from auth_rag.types import Chunk, Document, Language, Metadata, content_hash


class BaseChunker(ABC):
    """Splits a Document into a list of Chunks."""

    @abstractmethod
    def chunk(self, doc: Document) -> list[Chunk]:
        pass

    def _make_chunk(self, doc: Document, text: str, title: str = "") -> Chunk:
        metadata = Metadata(source=doc.source, title=title)
        chunk_id = f"{doc.source.id}:{content_hash(text)}"
        chunk = Chunk(id=chunk_id, text=text, metadata=metadata)
        return chunk


class FixedSizeChunker(BaseChunker):
    """Splits text into fixed-size character windows with optional overlap."""

    def __init__(self, chunk_size: int = 512, overlap: int = 64) -> None:
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, doc: Document) -> list[Chunk]:
        text = doc.text
        step = self.chunk_size - self.overlap
        chunks: list[Chunk] = []
        for start in range(0, len(text), step):
            window = text[start:start + self.chunk_size]
            if window.strip():
                chunks.append(self._make_chunk(doc, window))
        return chunks


class SentenceChunker(BaseChunker):
    """Groups sentences into chunks using NLTK tokenization."""

    def __init__(
        self,
        sentences_per_chunk: int = 5,
        overlap: int = 1,
        language: Language = Language.ITALIAN,
    ) -> None:
        if overlap >= sentences_per_chunk:
            raise ValueError("overlap must be smaller than sentences_per_chunk")
        self.sentences_per_chunk = sentences_per_chunk
        self.overlap = overlap
        self.language = language
        nltk.download("punkt_tab", quiet=True)

    def chunk(self, doc: Document) -> list[Chunk]:
        sentences = sent_tokenize(doc.text, language=self.language)
        step = self.sentences_per_chunk - self.overlap
        chunks: list[Chunk] = []
        for start in range(0, len(sentences), step):
            group = sentences[start:start + self.sentences_per_chunk]
            if group:
                chunks.append(self._make_chunk(doc, " ".join(group)))
        return chunks


class HierarchicalChunker(BaseChunker):
    """Splits markdown into chunks following the heading hierarchy as section path."""

    def __init__(self):
        self.parser = MarkdownNodeParser()

    def chunk(self, doc: Document) -> list[Chunk]:
        llama_doc = LlamaDocument(text=doc.text)
        nodes = self.parser.get_nodes_from_documents([llama_doc])

        chunks: list[Chunk] = []
        for node in nodes:
            title = self._build_section(node.metadata.get("header_path", "/"), node.text)
            text = self._strip_heading(node.text)
            if text:
                chunks.append(self._make_chunk(doc, text, title))

        return chunks

    def _build_section(self, header_path: str, text: str) -> str:
        """Combines parent path from header_path with the current node's heading."""
        parts = [p for p in header_path.split("/") if p]
        heading = self._extract_heading(text)
        if heading:
            parts.append(heading)
        return " > ".join(parts)

    def _extract_heading(self, text: str) -> str:
        match = re.match(r"^#{1,6}\s+(.+)", text)
        return match.group(1).strip() if match else ""

    def _strip_heading(self, text: str) -> str:
        """Removes the leading heading line from node text."""
        lines = text.split("\n")
        if lines and re.match(r"^#{1,6}\s+", lines[0]):
            return "\n".join(lines[1:]).strip()
        return text.strip()


class RecursiveCharacterChunker(BaseChunker):
    """Splits text by trying separators in order until chunks fit within chunk_size.

    Tries each separator in sequence; if a piece is still too large it recurses
    with the next separator, falling back to character-level splitting.
    """

    _DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 64,
        separators: list[str] | None = None,
    ) -> None:
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = separators if separators is not None else self._DEFAULT_SEPARATORS

    def chunk(self, doc: Document) -> list[Chunk]:
        pieces = self._split(doc.text, self.separators)
        merged = self._merge(pieces)
        return [self._make_chunk(doc, text) for text in merged if text.strip()]

    def _split(self, text: str, separators: list[str]) -> list[str]:
        if not separators or separators[0] == "":
            return [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]
        sep, *rest = separators
        parts = text.split(sep)
        result: list[str] = []
        for part in parts:
            if len(part) <= self.chunk_size:
                result.append(part)
            else:
                result.extend(self._split(part, rest))
        return result

    def _merge(self, pieces: list[str]) -> list[str]:
        chunks: list[str] = []
        current = ""
        for piece in pieces:
            candidate = (current + " " + piece).strip() if current else piece
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                current = (current[-self.overlap:].lstrip() + " " + piece).strip() if self.overlap and current else piece
        if current.strip():
            chunks.append(current)
        return chunks


class SemanticChunker(BaseChunker):
    """Groups sentences into semantically coherent chunks.

    Encodes every sentence with a SentenceTransformer and inserts a chunk
    boundary wherever the cosine similarity between adjacent sentences drops
    below breakpoint_threshold.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        breakpoint_threshold: float = 0.5,
        max_chunk_size: int = 2048,
        language: Language = Language.ITALIAN,
    ) -> None:
        self.model = SentenceTransformer(model_name)
        self.breakpoint_threshold = breakpoint_threshold
        self.max_chunk_size = max_chunk_size
        self.language = language
        nltk.download("punkt_tab", quiet=True)

    def chunk(self, doc: Document) -> list[Chunk]:
        sentences = sent_tokenize(doc.text, language=self.language)
        if not sentences:
            return []
        if len(sentences) == 1:
            return [self._make_chunk(doc, sentences[0])]

        embeddings = self.model.encode(sentences, convert_to_numpy=True, normalize_embeddings=True)

        groups: list[list[str]] = []
        current: list[str] = [sentences[0]]
        current_len = len(sentences[0])
        for idx in range(1, len(sentences)):
            similarity = float(np.dot(embeddings[idx - 1], embeddings[idx]))
            fits = current_len + 1 + len(sentences[idx]) <= self.max_chunk_size
            if similarity >= self.breakpoint_threshold and fits:
                current.append(sentences[idx])
                current_len += 1 + len(sentences[idx])
            else:
                groups.append(current)
                current = [sentences[idx]]
                current_len = len(sentences[idx])
        groups.append(current)

        return [
            self._make_chunk(doc, " ".join(group))
            for group in groups
            if " ".join(group).strip()
        ]