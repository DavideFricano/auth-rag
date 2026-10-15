from __future__ import annotations

from collections.abc import Iterable, Sequence

import networkx as nx
import numpy as np
from neo4j import Driver, GraphDatabase

from auth_rag.authorization.schema import AccessSchema
from auth_rag.indexing.relation.extractor import BaseExtractor
from auth_rag.indexing.relation.index import Relation, RelationIndex
from auth_rag.indexing.relation.seeder import BaseSeeder, bm25
from auth_rag.storing.store import BaseStore


class VolatileGraphIndex(RelationIndex):
    """Relation graph in an in-process networkx ``MultiGraph``. Non-durable; zero setup.

    Written first because a graph database has no equivalent of Qdrant's ``:memory:``:
    without an in-process engine, every test of the graph would need a container.

    Undirected — a relation is extracted with a direction, kept on the edge, and ignored
    by the walk, which asks what a chunk is connected to. A ``MultiGraph`` and not a
    ``Graph`` because two chunks may state the same pair of concepts for different
    reasons, and collapsing those would lose a provenance ``delete`` needs.
    """

    def __init__(
        self,
        store: BaseStore,
        extractor: BaseExtractor,
        schema: AccessSchema | None = None,
        hops: int = 2,
        decay: float = 0.5,
        over_fetch: int = 3,
        max_seeds: int = 10,
        seeders: list[BaseSeeder] | None = None,
    ) -> None:
        super().__init__(store, extractor, schema, hops, decay, over_fetch, max_seeds, seeders)
        self.graph = nx.MultiGraph()
        self.by_chunk: dict[str, set[str]] = {}  # the reverse of each node's "chunks"

    def _add(self, relations: Sequence[Relation]) -> None:
        for relation in relations:
            for name in (relation.subject, relation.object):
                if name not in self.graph:
                    self.graph.add_node(name, chunks={}, **self._represent(name))
                self.graph.nodes[name]["chunks"][relation.chunk_id] = relation.source_id
                self.by_chunk.setdefault(relation.chunk_id, set()).add(name)
            self.graph.add_edge(
                relation.subject,
                relation.object,
                key=(relation.predicate, relation.chunk_id),  # re-inserting upserts in place
                relation=relation,
            )

    def delete(self, source_id: str) -> None:
        """Drop every edge this source produced, and every node left with nothing.

        Dropping a whole source is also how re-ingestion stays neutral: the graph is
        rebuilt for that source rather than patched chunk by chunk, so a drifting
        extractor cannot leave stale edges behind.
        """
        for subject, object_, key, data in list(self.graph.edges(keys=True, data=True)):
            if data["relation"].source_id == source_id:
                self.graph.remove_edge(subject, object_, key=key)
        for name in list(self.graph.nodes):
            chunks = self.graph.nodes[name]["chunks"]
            for chunk_id in [c for c, source in chunks.items() if source == source_id]:
                del chunks[chunk_id]
                self.by_chunk.pop(chunk_id, None)
            if not chunks:
                self.graph.remove_node(name)

    def _match(self, roots: list[str]) -> dict[str, float]:
        documents = {
            name: data["roots"].split()
            for name, data in self.graph.nodes(data=True)
            if "roots" in data
        }
        ranked = sorted(bm25(documents, roots).items(), key=lambda row: (-row[1], row[0]))
        return dict(ranked[: self.max_seeds])

    def _nearest(self, vector: list[float]) -> dict[str, float]:
        names = [name for name, data in self.graph.nodes(data=True) if "embedding" in data]
        if not names:
            return {}
        matrix = np.array([self.graph.nodes[name]["embedding"] for name in names])
        norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(vector)
        similarity = matrix @ np.array(vector) / np.where(norms == 0, 1.0, norms)
        scored = zip(names, similarity.tolist(), strict=True)
        ranked = sorted(scored, key=lambda row: (-row[1], row[0]))
        return dict(ranked[: self.max_seeds])

    def _concepts(self, chunk_ids: Sequence[str]) -> Iterable[tuple[str, str]]:
        return [
            (chunk_id, name) for chunk_id in chunk_ids for name in self.by_chunk.get(chunk_id, ())
        ]

    def _expand(self, seeds: dict[str, float]) -> Iterable[tuple[str, int, float]]:
        """One breadth-first walk per seed, rather than one from all of them at once: a
        node far from a strong seed can beat the same node near a weak one, and a single
        frontier would have to commit to one before knowing which."""
        for seed, score in seeds.items():
            if seed not in self.graph:
                continue
            frontier, seen = [seed], {seed}
            for hop in range(self.hops + 1):
                for node in frontier:
                    for chunk_id in self.graph.nodes[node]["chunks"]:
                        yield chunk_id, hop, score
                if hop == self.hops:
                    break
                next_frontier = []
                for node in frontier:
                    for neighbour in self.graph.neighbors(node):
                        if neighbour not in seen:
                            seen.add(neighbour)
                            next_frontier.append(neighbour)
                if not next_frontier:
                    break
                frontier = next_frontier


class RemoteGraphIndex(RelationIndex):
    """Relation graph on a Neo4j server (shared across processes).

    Same family and same scoring; the walk happens in the database instead of in Python,
    which is the point of this tier — the volatile one pulls a frontier at a time, and at
    corpus size that is the wrong shape.

    The chunk is a node rather than a list property on the concept::

        (:Concept {name, roots})
        (:Concept)-[:FROM_CHUNK]->(:Chunk {id, source_id})
        (:Concept)-[:RELATED {predicate, chunk_id, source_id}]->(:Concept)

    ``source_id`` on the relationship makes ``delete`` a statement instead of a scan, as
    the ``source_id`` column does in the SQL stores. Pass ``driver`` to point this at an
    existing connection.
    """

    def __init__(
        self,
        store: BaseStore,
        extractor: BaseExtractor,
        url: str = "bolt://localhost:7687",
        auth: tuple[str, str] | None = None,
        driver: Driver | None = None,
        database: str | None = None,
        schema: AccessSchema | None = None,
        hops: int = 2,
        decay: float = 0.5,
        over_fetch: int = 3,
        max_seeds: int = 10,
        seeders: list[BaseSeeder] | None = None,
    ) -> None:
        super().__init__(store, extractor, schema, hops, decay, over_fetch, max_seeds, seeders)
        self.driver = driver if driver is not None else GraphDatabase.driver(url, auth=auth)
        self.database = database
        self._ready = False
        self._vectors_ready = False

    def _run(self, query: str, **parameters) -> list[dict]:
        with self.driver.session(database=self.database) as session:
            return [record.data() for record in session.run(query, **parameters)]

    def _ensure_constraints(self) -> None:
        if self._ready:
            return
        self._ready = True
        self._run(  # without these, every MERGE below is a scan
            "CREATE CONSTRAINT concept_name IF NOT EXISTS "
            "FOR (n:Concept) REQUIRE n.name IS UNIQUE"
        )
        self._run(
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE"
        )
        self._run(  # and without these, every delete
            "CREATE INDEX chunk_source_id IF NOT EXISTS FOR (c:Chunk) ON (c.source_id)"
        )
        self._run(
            "CREATE INDEX related_source_id IF NOT EXISTS "
            "FOR ()-[r:RELATED]-() ON (r.source_id)"
        )
        self._run(
            "CREATE FULLTEXT INDEX concept_roots IF NOT EXISTS FOR (n:Concept) ON EACH [n.roots] "
            "OPTIONS {indexConfig: {`fulltext.analyzer`: 'whitespace'}}"
        )
        self._run("CALL db.awaitIndexes()")

    def _ensure_vector_index(self, dimensions: int) -> None:
        """Created on first sight of a vector, because its size comes from the embedder."""
        if self._vectors_ready:
            return
        self._vectors_ready = True
        self._run(
            "CREATE VECTOR INDEX concept_embedding IF NOT EXISTS FOR (n:Concept) ON n.embedding "
            f"OPTIONS {{indexConfig: {{`vector.dimensions`: {int(dimensions)}, "
            "`vector.similarity_function`: 'cosine'}}"
        )
        self._run("CALL db.awaitIndexes()")

    def _add(self, relations: Sequence[Relation]) -> None:
        if not relations:
            return
        self._ensure_constraints()
        rows = [
            {
                "subject": relation.subject,
                "subject_props": self._represent(relation.subject),
                "predicate": relation.predicate,
                "object": relation.object,
                "object_props": self._represent(relation.object),
                "chunk_id": relation.chunk_id,
                "source_id": relation.source_id,
            }
            for relation in relations
        ]
        embedded = [row["subject_props"] for row in rows if "embedding" in row["subject_props"]]
        if embedded:
            self._ensure_vector_index(len(embedded[0]["embedding"]))
        self._run(  # every MERGE is on the identity of the thing, so this upserts
            """
            UNWIND $rows AS row
            MERGE (s:Concept {name: row.subject})
              ON CREATE SET s += row.subject_props
            MERGE (o:Concept {name: row.object})
              ON CREATE SET o += row.object_props
            MERGE (c:Chunk {id: row.chunk_id})
              SET c.source_id = row.source_id
            MERGE (s)-[:FROM_CHUNK]->(c)
            MERGE (o)-[:FROM_CHUNK]->(c)
            MERGE (s)-[r:RELATED {predicate: row.predicate, chunk_id: row.chunk_id}]->(o)
              ON CREATE SET r.source_id = row.source_id
            """,
            rows=rows,
        )

    def delete(self, source_id: str) -> None:
        """Orphans are looked for only among the concepts this source's chunks named:
        sweeping every concept on each delete would cost a scan of the graph per document."""
        self._ensure_constraints()
        self._run(
            "MATCH ()-[r:RELATED {source_id: $source_id}]->() DELETE r", source_id=source_id
        )
        self._run(
            """
            MATCH (c:Chunk {source_id: $source_id})
            OPTIONAL MATCH (n:Concept)-[:FROM_CHUNK]->(c)
            WITH collect(DISTINCT n) AS touched, collect(DISTINCT c) AS chunks
            FOREACH (c IN chunks | DETACH DELETE c)
            WITH touched
            UNWIND touched AS n
            WITH n WHERE NOT (n)-[:FROM_CHUNK]->(:Chunk)
            DETACH DELETE n
            """,
            source_id=source_id,
        )

    def _match(self, roots: list[str]) -> dict[str, float]:
        """Lucene does the BM25 and keeps the index; the roots are ours, already analysed,
        which the whitespace analyzer leaves as they are — so both tiers read text alike."""
        if not roots:
            return {}
        self._ensure_constraints()
        rows = self._run(
            """
            CALL db.index.fulltext.queryNodes('concept_roots', $roots) YIELD node, score
            RETURN node.name AS name, score
            ORDER BY score DESC, name ASC
            LIMIT $limit
            """,
            roots=" ".join(roots),
            limit=self.max_seeds,
        )
        return {row["name"]: float(row["score"]) for row in rows}

    def _nearest(self, vector: list[float]) -> dict[str, float]:
        """Neo4j scores a cosine as ``(1 + cosine) / 2``; it is turned back into a cosine so
        both tiers hand the seeder the same numbers."""
        self._ensure_vector_index(len(vector))
        rows = self._run(
            """
            CALL db.index.vector.queryNodes('concept_embedding', $limit, $vector)
            YIELD node, score
            RETURN node.name AS name, 2 * score - 1 AS similarity
            ORDER BY similarity DESC, name ASC
            """,
            vector=vector,
            limit=self.max_seeds,
        )
        return {row["name"]: float(row["similarity"]) for row in rows}

    def _concepts(self, chunk_ids: Sequence[str]) -> Iterable[tuple[str, str]]:
        rows = self._run(
            """
            UNWIND $ids AS id
            MATCH (n:Concept)-[:FROM_CHUNK]->(:Chunk {id: id})
            RETURN id AS chunk_id, n.name AS name
            """,
            ids=list(chunk_ids),
        )
        return [(row["chunk_id"], row["name"]) for row in rows]

    def _expand(self, seeds: dict[str, float]) -> Iterable[tuple[str, int, float]]:
        """One statement for the whole expansion, hop distance computed in the database.

        ``hops`` is interpolated because Cypher takes no parameter in a variable-length
        pattern. It is this object's own integer, validated in the base class — nothing a
        caller's query reaches.
        """
        rows = self._run(
            f"""
            UNWIND $seeds AS seed
            MATCH (n:Concept {{name: seed.name}})
            MATCH path = (n)-[:RELATED*0..{int(self.hops)}]-(m:Concept)
            WITH seed.score AS score, m, min(length(path)) AS hop
            MATCH (m)-[:FROM_CHUNK]->(c:Chunk)
            RETURN c.id AS chunk_id, hop, score
            """,
            seeds=[{"name": name, "score": score} for name, score in seeds.items()],
        )
        return [(row["chunk_id"], int(row["hop"]), float(row["score"])) for row in rows]
