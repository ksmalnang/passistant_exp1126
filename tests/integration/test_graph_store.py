"""Neo4j integration tests for graph sync, pruning, and deletion.

Skipped unless NEO4J_PASSWORD is set and Neo4j is reachable.
"""

from __future__ import annotations

import uuid

import pytest

from src.config import get_settings
from src.graph.extraction import extract_deterministic
from src.graph.indexing import GraphChunkIndexer
from src.graph.store import GraphStore


def _store() -> GraphStore:
    settings = get_settings()
    if not settings.NEO4J_PASSWORD:
        pytest.skip("NEO4J_PASSWORD is not configured")
    store = GraphStore(
        uri=settings.NEO4J_URI,
        user=settings.NEO4J_USER,
        password=settings.NEO4J_PASSWORD,
        database=settings.NEO4J_DATABASE,
    )
    try:
        store.verify()
    except Exception:
        pytest.skip("Neo4j is not reachable")
    return store


def _chunk(document_id: str, chunk_id: str, text: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "filename": "test.pdf",
        "doc_title": "Test",
        "chunk_index": 0,
        "page": 1,
        "text": text,
    }


def test_upsert_is_idempotent_and_delete_prunes_chunks() -> None:
    store = _store()
    document_id = f"test-doc-{uuid.uuid4().hex[:8]}"
    chunk_id = f"{document_id}:chunk:0"
    chunk = _chunk(document_id, chunk_id, "| 3 | IF21W0803 | Keamanan Informasi | 2 |")
    try:
        store.ensure_schema()
        entities, relationships = extract_deterministic(chunk)
        store.upsert_chunk(chunk, entities, relationships)
        store.upsert_chunk(chunk, entities, relationships)
        assert chunk_id in store.chunk_ids()

        with store._driver.session(database=store.database) as session:
            count = session.run(
                "MATCH (c:Chunk {chunk_id: $chunk_id}) RETURN count(c) AS count",
                chunk_id=chunk_id,
            ).single()["count"]
            assert count == 1

        store.prune_document_chunks(document_id, [])
        assert chunk_id not in store.chunk_ids()

        store.upsert_chunk(chunk, entities, relationships)
        store.delete_document(document_id)
        assert chunk_id not in store.chunk_ids()
    finally:
        store.delete_document(document_id)
        store.close()


@pytest.mark.parametrize("operation", ["reindex", "prune", "delete_document", "delete_chunk"])
def test_removed_provenance_does_not_leave_edges_between_shared_entities(operation) -> None:
    """Edges must disappear even when another document still mentions both endpoints."""
    store = _store()
    prefix = f"test-sync-{uuid.uuid4().hex}"
    document_id, keeper_id = f"{prefix}-source", f"{prefix}-keeper"
    chunk = _chunk(document_id, f"{document_id}:chunk:0", "Program A belongs to Program B.")
    keeper = _chunk(keeper_id, f"{keeper_id}:chunk:0", chunk["text"])
    entities = [
        {
            "entity_id": f"{prefix}:{name}",
            "name": name,
            "entity_type": "Program",
            "normalized_name": name.lower(),
            "aliases": [name],
            "confidence": 1.0,
            "evidence": chunk["text"],
        }
        for name in ("Program A", "Program B")
    ]
    relationships = [
        {
            "source": entities[0]["entity_id"],
            "target": entities[1]["entity_id"],
            "type": "PART_OF",
            "evidence": chunk["text"],
            "confidence": 1.0,
        }
    ]

    def sourced_edge_count():
        with store._driver.session(database=store.database) as session:
            row = session.run(
                "MATCH ()-[r]->() WHERE r.source_chunk_id = $chunk_id RETURN count(r) AS count",
                chunk_id=chunk["chunk_id"],
            ).single()
            return row["count"]

    try:
        store.ensure_schema()
        store.upsert_chunk(chunk, entities, relationships)
        store.upsert_chunk(keeper, entities, [])
        indexer = GraphChunkIndexer(store)
        assert indexer.replace_document(document_id, [chunk]) == []
        assert sourced_edge_count() == 1  # Unchanged text keeps optional policy extraction.

        if operation == "reindex":
            assert indexer.replace_document(document_id, [{**chunk, "text": "Updated text."}]) == []
        elif operation == "prune":
            store.prune_document_chunks(document_id, [])
        elif operation == "delete_document":
            store.delete_document(document_id)
        else:
            store.delete_chunk(chunk["chunk_id"])

        assert sourced_edge_count() == 0
        assert keeper["chunk_id"] in store.document_chunks(keeper_id)
        with store._driver.session(database=store.database) as session:
            row = session.run(
                "MATCH (e:Entity) WHERE e.entity_id IN $ids RETURN count(e) AS count",
                ids=[entity["entity_id"] for entity in entities],
            ).single()
            assert row["count"] == 2
    finally:
        store.delete_document(document_id)
        store.delete_document(keeper_id)
        store.close()
