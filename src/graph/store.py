"""Neo4j-backed knowledge graph linked to Qdrant chunks by stable chunk ids."""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any, LiteralString, cast

from neo4j import GraphDatabase

logger = logging.getLogger(__name__)

ALLOWED_RELATIONSHIP_TYPES = frozenset(
    {
        "OFFERED_IN",
        "PART_OF",
        "REQUIRES",
        "HAS_DEADLINE",
        "APPLIES_TO",
        "EVALUATED_BY",
        "PREREQUISITE_OF",
        "REQUIRED_FOR",
        "CONDITION_OF",
        "GOVERNS",
    }
)


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class GraphStore:
    """Small, bounded Neo4j data-access layer for the academic knowledge graph."""

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: str = "neo4j",
        timeout_seconds: float = 3.0,
    ) -> None:
        self.database = database
        self.timeout_seconds = timeout_seconds
        self._driver = GraphDatabase.driver(
            uri,
            auth=(user, password),
            connection_timeout=timeout_seconds,
        )

    def close(self) -> None:
        self._driver.close()

    def verify(self) -> None:
        self._driver.verify_connectivity()

    def ensure_schema(self) -> None:
        statements: list[LiteralString] = [
            "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS "
            "FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
            "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS "
            "FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE",
            "CREATE CONSTRAINT document_id_unique IF NOT EXISTS "
            "FOR (d:Document) REQUIRE d.document_id IS UNIQUE",
            "CREATE INDEX chunk_document_index IF NOT EXISTS FOR (c:Chunk) ON (c.document_id)",
            "CREATE INDEX entity_type_index IF NOT EXISTS FOR (e:Entity) ON (e.entity_type)",
            "CREATE INDEX entity_name_index IF NOT EXISTS FOR (e:Entity) ON (e.normalized_name)",
        ]
        with self._driver.session(database=self.database) as session:
            for statement in statements:
                session.run(statement).consume()

    def upsert_chunk(
        self,
        chunk: dict[str, Any],
        entities: list[dict[str, Any]],
        relationships: list[dict[str, Any]],
    ) -> None:
        """Replace one chunk and its extraction, so re-ingestion is idempotent."""
        chunk_id = chunk["chunk_id"]
        with (
            self._driver.session(database=self.database) as session,
            session.begin_transaction() as transaction,
        ):
            self._delete_chunk_relationships(transaction, [chunk_id])
            transaction.run(
                "MATCH (c:Chunk {chunk_id: $chunk_id}) DETACH DELETE c",
                chunk_id=chunk_id,
            ).consume()
            transaction.run(
                """
                MERGE (d:Document {document_id: $document_id})
                SET d.filename = $filename, d.title = $title
                MERGE (c:Chunk {chunk_id: $chunk_id})
                SET c.document_id = $document_id,
                    c.filename = $filename,
                    c.page = $page,
                    c.chunk_index = $chunk_index,
                    c.text = $text,
                    c.text_hash = $text_hash
                MERGE (d)-[:HAS_CHUNK]->(c)
                """,
                chunk_id=chunk_id,
                document_id=chunk["document_id"],
                filename=chunk.get("filename") or "",
                title=chunk.get("doc_title"),
                page=chunk.get("page"),
                chunk_index=chunk.get("chunk_index"),
                text=chunk.get("text") or "",
                text_hash=text_hash(chunk.get("text") or ""),
            ).consume()
            if entities:
                transaction.run(
                    """
                    UNWIND $entities AS row
                    MERGE (e:Entity {entity_id: row.entity_id})
                    SET e.name = row.name,
                        e.entity_type = row.entity_type,
                        e.normalized_name = row.normalized_name,
                        e.aliases = row.aliases
                    WITH e, row
                    MATCH (c:Chunk {chunk_id: $chunk_id})
                    MERGE (c)-[m:MENTIONS]->(e)
                    SET m.confidence = row.confidence, m.evidence = row.evidence
                    """,
                    entities=entities,
                    chunk_id=chunk_id,
                ).consume()
            for relationship_type, rows in _group_by_type(relationships).items():
                if relationship_type not in ALLOWED_RELATIONSHIP_TYPES:
                    logger.warning(
                        "Skipping unsupported graph relationship type: %s", relationship_type
                    )
                    continue
                # The only interpolated value was checked against the allow-list.
                transaction.run(
                    cast(
                        LiteralString,
                        f"""
                    UNWIND $rows AS row
                    MATCH (a:Entity {{entity_id: row.source}})
                    MATCH (b:Entity {{entity_id: row.target}})
                    MERGE (a)-[r:{relationship_type} {{source_chunk_id: $chunk_id}}]->(b)
                    SET r.evidence = row.evidence, r.confidence = row.confidence
                    """,
                    ),
                    rows=rows,
                    chunk_id=chunk_id,
                ).consume()

    def clear(self) -> None:
        """Remove every graph node; used only by an explicit rebuild."""
        with self._driver.session(database=self.database) as session:
            session.run("MATCH (n) DETACH DELETE n").consume()

    def delete_document(self, document_id: str) -> None:
        with (
            self._driver.session(database=self.database) as session,
            session.begin_transaction() as transaction,
        ):
            chunk_ids = [
                row["chunk_id"]
                for row in transaction.run(
                    "MATCH (c:Chunk {document_id: $document_id}) RETURN c.chunk_id AS chunk_id",
                    document_id=document_id,
                )
            ]
            self._delete_chunks(transaction, chunk_ids)
            transaction.run(
                "MATCH (d:Document {document_id: $document_id}) DETACH DELETE d",
                document_id=document_id,
            ).consume()
            self._delete_orphan_entities(transaction)

    def prune_document_chunks(self, document_id: str, keep_chunk_ids: list[str]) -> None:
        """Remove graph chunks from a document that no longer exist in Qdrant."""
        with (
            self._driver.session(database=self.database) as session,
            session.begin_transaction() as transaction,
        ):
            chunk_ids = [
                row["chunk_id"]
                for row in transaction.run(
                    """
                    MATCH (c:Chunk {document_id: $document_id})
                    WHERE NOT c.chunk_id IN $keep
                    RETURN c.chunk_id AS chunk_id
                    """,
                    document_id=document_id,
                    keep=keep_chunk_ids,
                )
            ]
            self._delete_chunks(transaction, chunk_ids)
            self._delete_orphan_entities(transaction)

    def delete_chunk(self, chunk_id: str) -> None:
        with (
            self._driver.session(database=self.database) as session,
            session.begin_transaction() as transaction,
        ):
            self._delete_chunks(transaction, [chunk_id])
            self._delete_orphan_entities(transaction)

    def document_ids(self) -> set[str]:
        """Include documents whose HAS_CHUNK links are missing after an old failure."""
        with self._driver.session(database=self.database) as session:
            return {
                row["document_id"]
                for row in session.run(
                    """
                    MATCH (n) WHERE n:Document OR n:Chunk
                    RETURN DISTINCT n.document_id AS document_id
                    """
                )
                if row["document_id"]
            }

    def document_chunks(self, document_id: str) -> dict[str, dict[str, Any]]:
        """Read source fields so unchanged chunks retain their optional LLM extraction."""
        with self._driver.session(database=self.database) as session:
            result = session.run(
                """
                MATCH (d:Document {document_id: $document_id})-[:HAS_CHUNK]->
                      (c:Chunk {document_id: $document_id})
                RETURN c.chunk_id AS chunk_id, c.document_id AS document_id,
                       c.filename AS filename, d.title AS doc_title, c.page AS page,
                       c.chunk_index AS chunk_index, c.text AS text
                """,
                document_id=document_id,
            )
            return {row["chunk_id"]: dict(row) for row in result}

    def seed_entities(
        self,
        seed_chunk_ids: list[str],
        query: str,
        max_entities: int,
        max_frequency: int,
        query_linking: bool = True,
    ) -> list[dict[str, Any]]:
        """Find entities mentioned by seed chunks and optionally in the query text."""
        if not seed_chunk_ids and not query:
            return []
        found: dict[str, dict[str, Any]] = {}
        with self._driver.session(database=self.database) as session:
            if seed_chunk_ids:
                result = session.run(
                    """
                    MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
                    WHERE c.chunk_id IN $seed_ids
                    OPTIONAL MATCH (e)<-[:MENTIONS]-(other:Chunk)
                    WITH e, count(other) AS df
                    WHERE df <= $max_frequency
                    RETURN e.entity_id AS entity_id, e.name AS name,
                           e.entity_type AS entity_type, df
                    ORDER BY df ASC
                    LIMIT $limit
                    """,
                    seed_ids=seed_chunk_ids,
                    max_frequency=max_frequency,
                    limit=max_entities,
                    timeout=self.timeout_seconds,
                )
                for row in result:
                    found[row["entity_id"]] = {**dict(row), "source": "chunk"}
            if query_linking and query:
                result = session.run(
                    """
                    MATCH (e:Entity)
                    WHERE size(e.normalized_name) >= 4
                      AND toLower($query_text) CONTAINS e.normalized_name
                    OPTIONAL MATCH (e)<-[:MENTIONS]-(other:Chunk)
                    WITH e, count(other) AS df
                    WHERE df <= $max_frequency
                    RETURN e.entity_id AS entity_id, e.name AS name,
                           e.entity_type AS entity_type, e.normalized_name AS normalized_name, df
                    ORDER BY df ASC
                    LIMIT $limit
                    """,
                    query_text=query.lower(),
                    max_frequency=max_frequency,
                    limit=max_entities * 3,
                    timeout=self.timeout_seconds,
                )
                for row in result:
                    name = row.get("normalized_name", "")
                    if name and not re.search(
                        rf"(?<![a-z0-9]){re.escape(name)}(?![a-z0-9])", query.lower()
                    ):
                        continue
                    found[row["entity_id"]] = {**dict(row), "source": "query"}
        ordered = sorted(found.values(), key=lambda item: item.get("source") != "query")
        return ordered[:max_entities]

    def expand_entities(
        self,
        entity_ids: list[str],
        allowed_relationships: list[str],
        max_hops: int,
        max_entities: int,
        max_frequency: int,
    ) -> list[dict[str, Any]]:
        """Traverse up to max_hops typed entity relationships with hard bounds."""
        visited = set(entity_ids)
        frontier = list(entity_ids)
        expanded: list[dict[str, Any]] = []
        with self._driver.session(database=self.database) as session:
            for hop in range(max_hops):
                if not frontier:
                    break
                result = session.run(
                    """
                    MATCH (e:Entity)-[r]-(n:Entity)
                    WHERE e.entity_id IN $frontier
                      AND type(r) IN $allowed
                      AND NOT n.entity_id IN $visited
                    OPTIONAL MATCH (n)<-[:MENTIONS]-(other:Chunk)
                    WITH n, count(other) AS df, type(r) AS rel, r.evidence AS evidence,
                         r.confidence AS confidence
                    WHERE df <= $max_frequency
                    RETURN DISTINCT n.entity_id AS entity_id, n.name AS name,
                           n.entity_type AS entity_type, df, rel, evidence, confidence
                    LIMIT $limit
                    """,
                    frontier=frontier,
                    allowed=list(allowed_relationships),
                    visited=list(visited),
                    max_frequency=max_frequency,
                    limit=max_entities,
                    timeout=self.timeout_seconds,
                )
                current: list[str] = []
                for row in result:
                    entity_id = row["entity_id"]
                    if entity_id in visited:
                        continue
                    visited.add(entity_id)
                    current.append(entity_id)
                    expanded.append({"hop": hop + 1, **dict(row)})
                frontier = current
        return expanded

    def chunks_for_entities(
        self,
        entity_ids: list[str],
        exclude_chunk_ids: list[str],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Return source chunks mentioning expanded entities, ranked by shared entities."""
        if not entity_ids or limit <= 0:
            return []
        with self._driver.session(database=self.database) as session:
            result = session.run(
                """
                MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
                WHERE e.entity_id IN $entity_ids
                  AND NOT c.chunk_id IN $exclude
                WITH c, count(DISTINCT e) AS shared,
                     collect(DISTINCT e.name)[..6] AS entities
                RETURN c.chunk_id AS chunk_id, c.text AS text, c.filename AS filename,
                       c.page AS page, c.document_id AS document_id,
                       shared, entities
                ORDER BY shared DESC, c.document_id, c.chunk_index
                LIMIT $limit
                """,
                entity_ids=entity_ids,
                exclude=exclude_chunk_ids,
                limit=limit,
                timeout=self.timeout_seconds,
            )
            return [dict(row) for row in result]

    def stats(self) -> dict[str, Any]:
        with self._driver.session(database=self.database) as session:
            counts = session.run(
                """
                MATCH (c:Chunk) WITH count(c) AS chunks
                MATCH (e:Entity) WITH chunks, count(e) AS entities
                MATCH ()-[r]->() WITH chunks, entities, count(r) AS relationships
                RETURN chunks, entities, relationships
                """
            ).single()
            types = session.run(
                "MATCH (e:Entity) RETURN e.entity_type AS type, count(*) AS count ORDER BY count DESC"
            ).data()
        return {
            "chunks": counts["chunks"] if counts else 0,
            "entities": counts["entities"] if counts else 0,
            "relationships": counts["relationships"] if counts else 0,
            "entity_types": {row["type"]: row["count"] for row in types},
        }

    def chunk_ids(self) -> set[str]:
        with self._driver.session(database=self.database) as session:
            return {
                row["chunk_id"]
                for row in session.run("MATCH (c:Chunk) RETURN c.chunk_id AS chunk_id")
            }

    def consistency_report(self, qdrant_chunk_ids: set[str]) -> dict[str, Any]:
        graph_ids = self.chunk_ids()
        with self._driver.session(database=self.database) as session:
            provenance = session.run(
                """
                MATCH (c:Chunk)-[m:MENTIONS]->(e:Entity)
                WHERE m.evidence IS NULL OR m.evidence = ''
                RETURN count(m) AS count
                """
            ).single()
        return {
            "qdrant_chunks": len(qdrant_chunk_ids),
            "graph_chunks": len(graph_ids),
            "missing_in_graph": sorted(qdrant_chunk_ids - graph_ids)[:20],
            "orphaned_in_graph": sorted(graph_ids - qdrant_chunk_ids)[:20],
            "mentions_without_provenance": provenance["count"] if provenance else 0,
        }

    def _delete_orphan_entities(self, session) -> None:
        # Entity-to-entity edges survive DETACH DELETE of a Chunk. Clean legacy
        # edges too, including those between entities still used by other chunks.
        session.run(
            """
            MATCH ()-[r]->()
            WHERE r.source_chunk_id IS NOT NULL
              AND NOT EXISTS { MATCH (c:Chunk) WHERE c.chunk_id = r.source_chunk_id }
            DELETE r
            """
        ).consume()
        session.run(
            """
            MATCH (e:Entity)
            WHERE NOT (e)<-[:MENTIONS]-(:Chunk)
            DETACH DELETE e
            """
        ).consume()

    def _delete_chunk_relationships(self, session, chunk_ids: list[str]) -> None:
        session.run(
            "MATCH ()-[r]->() WHERE r.source_chunk_id IN $chunk_ids DELETE r",
            chunk_ids=chunk_ids,
        ).consume()

    def _delete_chunks(self, session, chunk_ids: list[str]) -> None:
        self._delete_chunk_relationships(session, chunk_ids)
        session.run(
            "MATCH (c:Chunk) WHERE c.chunk_id IN $chunk_ids DETACH DELETE c",
            chunk_ids=chunk_ids,
        ).consume()


def _group_by_type(relationships: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for relationship in relationships:
        grouped.setdefault(relationship["type"].upper(), []).append(relationship)
    return grouped
