"""Qdrant-backed synchronization tests; Neo4j and providers stay offline."""

from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

from src.graph.indexing import GraphChunkIndexer, build_graph_chunk_indexer, chunk_from_payload
from src.services.graph_sync import GraphSyncService, graph_sync_lifespan
from src.services.indexing import DocumentIndexingService
from src.utils.state import DocumentUpload
from src.utils.vector_store.collection import CollectionOperations

COLLECTION = "graph-sync-test"


class _Store:
    """In-memory graph boundary, preserving a record of actual extraction writes."""

    def __init__(self):
        self.chunks = {}
        self.written = []
        self.deleted = []
        self.pruned = []
        self.fail = False
        self.schemas = 0

    def ensure_schema(self):
        self.schemas += 1

    def document_ids(self):
        return {chunk["document_id"] for chunk in self.chunks.values()}

    def document_chunks(self, document_id):
        return {k: v for k, v in self.chunks.items() if v["document_id"] == document_id}

    def upsert_chunk(self, chunk, entities, relationships):
        if self.fail:
            raise ConnectionError("Neo4j unavailable")
        self.chunks[chunk["chunk_id"]] = dict(chunk)
        self.written.append(chunk["chunk_id"])

    def prune_document_chunks(self, document_id, keep_chunk_ids):
        self.pruned.append(document_id)
        for chunk_id in set(self.document_chunks(document_id)) - set(keep_chunk_ids):
            del self.chunks[chunk_id]

    def delete_document(self, document_id):
        self.deleted.append(document_id)
        for chunk_id in list(self.document_chunks(document_id)):
            del self.chunks[chunk_id]

    def close(self):
        pass


def _payload(document_id="doc", index=0, text="Teknik Informatika", page=2):
    return {
        "chunk_id": f"{document_id}:chunk:{index}",
        "document_id": document_id,
        "text": text,
        "filename": "handbook.pdf",
        "doc_title": "Handbook",
        "chunk_index": index,
        "source_locations": [{"page": page}],
    }


def _put(client, *payloads):
    client.upsert(
        COLLECTION,
        [PointStruct(id=str(uuid.uuid4()), vector=[0.1, 0.2], payload=p) for p in payloads],
        wait=True,
    )


@pytest.fixture
def backend(monkeypatch):
    client = QdrantClient(":memory:")
    client.create_collection(
        COLLECTION, vectors_config=VectorParams(size=2, distance=Distance.COSINE)
    )
    store = _Store()
    indexer = GraphChunkIndexer(store)
    monkeypatch.setattr("src.services.indexing.get_cache", lambda: Mock())
    yield client, store, indexer, GraphSyncService(client, COLLECTION, indexer)
    client.close()


def test_reconcile_adds_all_pages_and_preserves_unchanged_extraction(backend):
    client, store, _, service = backend
    payloads = [_payload(index=i) for i in range(257)]
    _put(client, *payloads)

    assert service.reconcile() == []
    assert set(store.chunks) == {p["chunk_id"] for p in payloads}
    assert store.chunks["doc:chunk:0"]["page"] == 2
    # No extraction rewrite: pre-existing LLM mentions/edges remain intact.
    written = list(store.written)
    assert service.reconcile() == []
    assert store.written == written
    assert store.schemas == 1


def test_reconcile_replaces_changed_chunks_and_removes_deleted_documents(backend):
    client, store, _, service = backend
    _put(client, _payload(), _payload(index=1), _payload("removed"))
    assert service.reconcile() == []
    client.delete_collection(COLLECTION)
    client.create_collection(
        COLLECTION, vectors_config=VectorParams(size=2, distance=Distance.COSINE)
    )
    updated = _payload(text="Persyaratan baru", page=5)
    _put(client, updated, _payload("new"))

    assert service.reconcile() == []
    assert set(store.chunks) == {"doc:chunk:0", "new:chunk:0"}
    assert store.chunks["doc:chunk:0"] == chunk_from_payload(updated)
    assert store.deleted == ["removed"]
    client.delete_collection(COLLECTION)
    client.create_collection(
        COLLECTION, vectors_config=VectorParams(size=2, distance=Distance.COSINE)
    )
    assert service.reconcile() == []
    assert store.chunks == {}


@pytest.mark.parametrize("fail_on_call", [1, 2])
def test_qdrant_read_failure_does_not_prune_graph(backend, monkeypatch, fail_on_call):
    client, store, _, service = backend
    payloads = [_payload(index=i) for i in range(257)]
    _put(client, *payloads)
    old = chunk_from_payload(_payload("old"))
    store.chunks[old["chunk_id"]] = old
    original_scroll = client.scroll
    calls = 0

    def broken_scroll(**kwargs):
        nonlocal calls
        calls += 1
        if calls == fail_on_call:
            raise ConnectionError("Qdrant unavailable")
        return original_scroll(**kwargs)

    monkeypatch.setattr(client, "scroll", broken_scroll)
    with pytest.raises(ConnectionError):
        service.reconcile()
    assert store.chunks == {old["chunk_id"]: old}
    assert store.deleted == store.pruned == store.written == []


def test_document_read_failure_or_invalid_payload_does_not_prune(backend, monkeypatch):
    client, store, _, service = backend
    _put(client, _payload())
    assert service.reconcile() == []
    store.pruned.clear()
    original_scroll = client.scroll

    def broken_scroll(**kwargs):
        if kwargs.get("scroll_filter"):
            raise ConnectionError("Qdrant document read failed")
        return original_scroll(**kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(client, "scroll", broken_scroll)
        assert service.reconcile()
    assert store.pruned == store.deleted == []
    _put(client, {"document_id": "doc", "text": "Malformed indexed data"})
    assert service.reconcile()
    assert store.pruned == store.deleted == []


def test_failed_graph_write_is_retried_without_pruning_previous_chunks(backend):
    client, store, _, service = backend
    old = chunk_from_payload(_payload(index=99))
    store.chunks[old["chunk_id"]] = old
    _put(client, _payload())
    store.fail = True

    assert service.reconcile()
    assert store.pruned == []
    assert old["chunk_id"] in store.chunks
    store.fail = False
    assert service.reconcile() == []
    assert set(store.chunks) == {"doc:chunk:0"}


def test_reconcile_reads_current_document_after_inventory(backend, monkeypatch):
    client, store, indexer, service = backend
    _put(client, _payload())
    assert service.reconcile() == []

    def change_after_inventory():
        # An API write may finish between inventory and document reconciliation.
        client.delete_collection(COLLECTION)
        client.create_collection(
            COLLECTION, vectors_config=VectorParams(size=2, distance=Distance.COSINE)
        )
        _put(client, _payload(index=1, text="Latest Qdrant contents"))
        return {"doc"}

    monkeypatch.setattr(indexer, "document_ids", change_after_inventory)
    assert service.reconcile() == []
    assert set(store.chunks) == {"doc:chunk:1"}


@pytest.mark.parametrize(
    "enabled,password,expected", [(True, "test", True), (False, "test", False), (True, None, False)]
)
def test_sync_factory_is_independent_of_retrieval(monkeypatch, enabled, password, expected):
    settings = SimpleNamespace(
        GRAPH_SYNC_ENABLED=enabled,
        GRAPH_RAG_ENABLED=False,
        NEO4J_PASSWORD=password,
        NEO4J_URI="bolt://localhost:7687",
        NEO4J_USER="neo4j",
        NEO4J_DATABASE="neo4j",
        GRAPH_RAG_TIMEOUT_SECONDS=3.0,
    )
    store_factory = Mock()
    monkeypatch.setattr("src.graph.indexing.get_settings", lambda: settings)
    monkeypatch.setattr("src.graph.indexing.GraphStore", store_factory)
    assert (build_graph_chunk_indexer() is not None) is expected
    assert store_factory.called is expected


async def test_lifespan_retries_failed_pass_and_closes_after_stop(monkeypatch):
    recovered = asyncio.Event()
    loop = asyncio.get_running_loop()
    calls = 0

    def reconcile():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ConnectionError("Temporary outage")
        loop.call_soon_threadsafe(recovered.set)
        return []

    service = Mock(reconcile=reconcile)
    monkeypatch.setattr("src.services.graph_sync.build_graph_chunk_indexer", lambda: Mock())
    monkeypatch.setattr("src.services.graph_sync.get_qdrant_client", lambda: Mock())
    monkeypatch.setattr("src.services.graph_sync.GraphSyncService", lambda *args: service)
    monkeypatch.setattr(
        "src.services.graph_sync.get_settings",
        lambda: SimpleNamespace(
            QDRANT_COLLECTION_NAME=COLLECTION,
            GRAPH_SYNC_INTERVAL_SECONDS=0.001,
        ),
    )

    async with graph_sync_lifespan():
        await asyncio.wait_for(recovered.wait(), timeout=5)
        service.close.assert_not_called()
    assert calls >= 2
    service.close.assert_called_once()


class _VectorStore(CollectionOperations):
    collection_name = COLLECTION

    def __init__(self, client):
        self.client = client

    def ensure_collection(self):
        pass

    def _ensure_collection_for_vector_size(self, size):
        pass

    def _get_embeddings(self):
        return SimpleNamespace(
            aembed_documents=AsyncMock(side_effect=lambda texts: [[0.1, 0.2] for _ in texts])
        )

    def _supports_bm25_vectors(self):
        return False


async def test_ingest_and_reindex_sync_immediately_and_recover_from_outage(backend, monkeypatch):
    client, store, indexer, sync = backend
    service = DocumentIndexingService(vector_store=_VectorStore(client), graph_indexer=indexer)
    document = DocumentUpload(
        document_id="doc",
        filename="test.pdf",
        file_path="test.pdf",
        file_size=1,
        mime_type="application/pdf",
        extracted_text="Teknik Informatika",
    )
    result = await service.index_document(document)
    assert result.issues == []
    assert set(store.chunks) == {"doc:chunk:0"}
    document.extracted_text = "Isi terbaru"
    store.fail = True
    result = await service.index_document(document, replace_existing=True)
    assert result.chunk_count == 1
    assert result.issues
    service._cache.delete_prefix.assert_any_call("search:")
    store.fail = False
    assert sync.reconcile() == []
    assert store.chunks["doc:chunk:0"]["text"] == "Isi terbaru"


async def test_graph_read_failure_keeps_successful_index_and_cache_invalidation(
    backend, monkeypatch
):
    client, _, indexer, _ = backend
    monkeypatch.setattr(
        indexer, "replace_document", Mock(side_effect=ConnectionError("Neo4j offline"))
    )
    service = DocumentIndexingService(vector_store=_VectorStore(client), graph_indexer=indexer)
    document = DocumentUpload(
        document_id="doc",
        filename="test.pdf",
        file_path="test.pdf",
        file_size=1,
        mime_type="application/pdf",
        extracted_text="Teknik Informatika",
    )
    result = await service.index_document(document)
    assert result.chunk_count == 1
    assert "Graph sync deferred" in result.issues[0]
    assert client.count(COLLECTION).count == 1
    service._cache.delete_prefix.assert_any_call("search:")
