"""Repair the derived Neo4j graph from the authoritative Qdrant collection."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, suppress
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http.models import FieldCondition, Filter, MatchValue

from src.clients.qdrant import get_qdrant_client
from src.config import get_settings
from src.graph.indexing import GraphChunkIndexer, build_graph_chunk_indexer, chunk_from_payload
from src.services.document_locks import knowledge_base_lock

logger = logging.getLogger(__name__)


class GraphSyncService:
    """Reconcile additions, replacements and deletions, retrying on the next pass."""

    def __init__(self, client: QdrantClient, collection: str, indexer: GraphChunkIndexer):
        self._client = client
        self._collection = collection
        self._indexer = indexer

    def _payloads(self, document_id: str | None = None) -> Iterator[dict[str, Any]]:
        offset = None
        while True:
            points, offset = self._client.scroll(
                collection_name=self._collection,
                scroll_filter=(
                    Filter(
                        must=[
                            FieldCondition(key="document_id", match=MatchValue(value=document_id))
                        ]
                    )
                    if document_id is not None
                    else None
                ),
                limit=256,
                with_payload=True if document_id is not None else ["document_id"],
                with_vectors=False,
                offset=offset,
            )
            for point in points:
                yield point.payload or {}
            if offset is None:
                break

    def reconcile(self) -> list[str]:
        """Read every page successfully before considering graph-only documents.

        A failed Qdrant read must never be interpreted as an empty knowledge base.
        Re-read each document under the mutation lock rather than deleting from a
        potentially stale full-collection snapshot.
        """
        document_ids = {p["document_id"] for p in self._payloads() if p.get("document_id")}
        document_ids.update(self._indexer.document_ids())
        issues: list[str] = []
        for document_id in sorted(document_ids):
            try:
                with knowledge_base_lock:
                    payloads = list(self._payloads(document_id))
                    # Invalid payloads fail this document without pruning it.
                    chunks = [chunk_from_payload(payload) for payload in payloads]
                    if chunks:
                        issues.extend(self._indexer.replace_document(document_id, chunks))
                    else:
                        self._indexer.delete_document(document_id)
            except Exception as exc:
                issues.append(f"{document_id}: {exc}")
        if issues:
            logger.warning("Graph reconciliation incomplete; will retry: %s", issues)
        else:
            logger.info("Graph reconciliation finished for %d documents", len(document_ids))
        return issues

    def close(self) -> None:
        self._indexer.close()


async def _run_sync(service: GraphSyncService, stop: asyncio.Event, interval: float) -> None:
    while not stop.is_set():
        try:
            await asyncio.to_thread(service.reconcile)
        except Exception:
            logger.warning("Graph reconciliation failed; will retry", exc_info=True)
        with suppress(TimeoutError):
            await asyncio.wait_for(stop.wait(), timeout=interval)


@asynccontextmanager
async def graph_sync_lifespan() -> AsyncIterator[None]:
    """Run immediately at API startup and periodically, without blocking requests."""
    indexer = build_graph_chunk_indexer()
    if indexer is None:
        yield
        return
    settings = get_settings()
    service = GraphSyncService(get_qdrant_client(), settings.QDRANT_COLLECTION_NAME, indexer)
    stop = asyncio.Event()
    task = asyncio.create_task(
        _run_sync(service, stop, settings.GRAPH_SYNC_INTERVAL_SECONDS), name="graph-sync"
    )
    try:
        yield
    finally:
        stop.set()
        # Let an in-flight threaded pass finish before closing its driver and
        # the shared Qdrant client. Cancelling the task would leave that thread running.
        await task
        await asyncio.to_thread(service.close)
