"""Keep Neo4j chunk nodes in step with Qdrant ingestion and deletion."""

from __future__ import annotations

import logging
from typing import Any

from src.config import get_settings
from src.graph.extraction import extract_deterministic
from src.graph.store import GraphStore

logger = logging.getLogger(__name__)


class GraphChunkIndexer:
    """Deterministic graph sync for ingested chunks; policy extraction stays offline."""

    def __init__(self, store: GraphStore):
        self._store = store

    def index_chunks(self, chunks: list[dict[str, Any]]) -> list[str]:
        """Upsert chunk nodes and deterministic course/semester links."""
        issues: list[str] = []
        for chunk in chunks:
            try:
                entities, relationships = extract_deterministic(chunk)
                self._store.upsert_chunk(chunk, entities, relationships)
            except Exception as exc:
                issues.append(f"{chunk.get('chunk_id')}: {exc}")
                logger.warning("Graph sync failed for chunk %s: %s", chunk.get("chunk_id"), exc)
        return issues

    def replace_document(self, document_id: str, chunks: list[dict[str, Any]]) -> list[str]:
        """Upsert current chunks and prune graph chunks that no longer exist."""
        issues = self.index_chunks(chunks)
        self._store.prune_document_chunks(document_id, [chunk["chunk_id"] for chunk in chunks])
        return issues

    def delete_document(self, document_id: str) -> None:
        self._store.delete_document(document_id)

    def close(self) -> None:
        self._store.close()


def build_graph_chunk_indexer() -> GraphChunkIndexer | None:
    """Build the sync hook when graph retrieval is configured."""
    settings = get_settings()
    if not settings.GRAPH_RAG_ENABLED or not settings.NEO4J_PASSWORD:
        return None
    store = GraphStore(
        uri=settings.NEO4J_URI,
        user=settings.NEO4J_USER,
        password=settings.NEO4J_PASSWORD,
        database=settings.NEO4J_DATABASE,
        timeout_seconds=settings.GRAPH_RAG_TIMEOUT_SECONDS,
    )
    return GraphChunkIndexer(store)
