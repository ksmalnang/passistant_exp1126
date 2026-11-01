"""Neo4j integration tests for graph sync, pruning, and deletion.

Skipped unless NEO4J_PASSWORD is set and Neo4j is reachable.
"""

from __future__ import annotations

import uuid

import pytest

from src.config import get_settings
from src.graph.extraction import extract_deterministic
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
