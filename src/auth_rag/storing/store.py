from __future__ import annotations

import sqlite3
from abc import ABC, abstractmethod

import psycopg

from auth_rag.types import Chunk


class BaseStore(ABC):
    """Holds the chunk data — the single source of truth for what a chunk *is*.

    An index keeps only ids (plus its vectors/tokens) and resolves the actual chunks
    through this store, so the store is shared across indices and holds each chunk once
    (no duplication). Keyed by ``chunk.id`` for idempotent upsert; ``delete`` drops every
    chunk of a source document. Tiers differ only in the backing tech — an in-memory
    dict, a local SQLite file, a shared Postgres — never in behaviour.
    """

    @abstractmethod
    def add(self, chunks: list[Chunk]) -> None:
        """Idempotent upsert keyed by chunk.id."""

    @abstractmethod
    def get(self, ids: list[str]) -> list[Chunk]:
        """Resolve ids to chunks; missing ids are skipped, not an error."""

    @abstractmethod
    def delete(self, source_id: str) -> None:
        """Remove every chunk whose ``metadata.source.id`` matches (idempotent)."""


class VolatileStore(BaseStore):
    """In-memory chunk store backed by a dict. Non-durable (lost when the process exits)."""

    def __init__(self) -> None:
        self._chunks: dict[str, Chunk] = {}

    def add(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            self._chunks[chunk.id] = chunk  # idempotent upsert by id

    def get(self, ids: list[str]) -> list[Chunk]:
        return [self._chunks[id_] for id_ in ids if id_ in self._chunks]

    def delete(self, source_id: str) -> None:
        self._chunks = {
            id_: chunk
            for id_, chunk in self._chunks.items()
            if chunk.metadata.source.id != source_id
        }


class PersistentStore(BaseStore):
    """Durable local chunk store backed by SQLite (single file, stdlib, single process).

    A ``source_id`` column makes ``delete`` a one-statement drop. Pass a custom connection
    to override storage (e.g. an in-memory ``:memory:`` one in tests).
    """

    def __init__(self, path: str = "./data/db/store.db", connection: sqlite3.Connection | None = None) -> None:
        self.conn = connection if connection is not None else sqlite3.connect(path)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS chunks "
            "(id TEXT PRIMARY KEY, source_id TEXT NOT NULL, data TEXT NOT NULL)"
        )
        self.conn.commit()

    def add(self, chunks: list[Chunk]) -> None:
        self.conn.executemany(
            "INSERT OR REPLACE INTO chunks (id, source_id, data) VALUES (?, ?, ?)",
            [(chunk.id, chunk.metadata.source.id, chunk.model_dump_json()) for chunk in chunks],
        )
        self.conn.commit()

    def get(self, ids: list[str]) -> list[Chunk]:
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        rows = self.conn.execute(f"SELECT data FROM chunks WHERE id IN ({placeholders})", ids).fetchall()
        return [Chunk.model_validate_json(data) for (data,) in rows]

    def delete(self, source_id: str) -> None:
        self.conn.execute("DELETE FROM chunks WHERE source_id = ?", (source_id,))
        self.conn.commit()


class RemoteStore(BaseStore):
    """Durable shared chunk store on Postgres (id -> chunk), reachable across processes.

    Same shape as the local tier, one server instead of a file — this is the tier that
    lets an ingestion worker and a query API (separate processes) share the chunks. Pass
    a custom connection to override (e.g. in tests).
    """

    def __init__(
        self,
        url: str = "postgresql://localhost/auth_rag",
        connection: psycopg.Connection | None = None,
    ) -> None:
        self.conn = connection if connection is not None else psycopg.connect(url)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS chunks "
            "(id TEXT PRIMARY KEY, source_id TEXT NOT NULL, data TEXT NOT NULL)"
        )
        self.conn.commit()

    def add(self, chunks: list[Chunk]) -> None:
        with self.conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO chunks (id, source_id, data) VALUES (%s, %s, %s) "
                "ON CONFLICT (id) DO UPDATE SET source_id = EXCLUDED.source_id, data = EXCLUDED.data",
                [(chunk.id, chunk.metadata.source.id, chunk.model_dump_json()) for chunk in chunks],
            )
        self.conn.commit()

    def get(self, ids: list[str]) -> list[Chunk]:
        if not ids:
            return []
        with self.conn.cursor() as cur:
            cur.execute("SELECT data FROM chunks WHERE id = ANY(%s)", (list(ids),))
            rows = cur.fetchall()
        return [Chunk.model_validate_json(data) for (data,) in rows]

    def delete(self, source_id: str) -> None:
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE source_id = %s", (source_id,))
        self.conn.commit()
