"""Neighbor context from Qdrant after the final reranked seed selection."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

from src.config import get_settings
from src.utils.nodes.retrieval import RetrievalNode
from src.utils.state import AgentState
from src.utils.vector_store import VectorStoreTools


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch):
    client = QdrantClient(":memory:")
    client.create_collection(
        "neighbors", vectors_config=VectorParams(size=2, distance=Distance.COSINE)
    )
    points = []
    for document_id, count in (("a", 6), ("b", 2)):
        for index in range(count):
            points.append(
                PointStruct(
                    id=len(points),
                    vector=[1.0, 0.0],
                    payload={
                        "chunk_id": f"{document_id}:chunk:{index}",
                        "document_id": document_id,
                        "chunk_index": index,
                        "text": f"{document_id} excerpt {index}",
                        "filename": f"{document_id}.pdf",
                        "doc_title": document_id.upper(),
                        "document_type": "other",
                        "chunk_type": "chunk",
                        "source_locations": [{"page": index}],
                    },
                )
            )
    client.upsert("neighbors", points)
    embeddings = SimpleNamespace(
        aembed_query=AsyncMock(return_value=[1.0, 0.0]),
        aembed_documents=AsyncMock(return_value=[[1.0, 0.0]]),
    )
    monkeypatch.setattr("src.utils.vector_store.tools.get_embeddings", lambda: embeddings)
    monkeypatch.setattr(get_settings(), "GRAPH_RAG_ENABLED", False)
    cache = Mock()
    cache.get_json.return_value = None
    tools = VectorStoreTools(client=client, cache=cache)
    tools.collection_name = "neighbors"
    tools.retrieval_strategy = "reranker"
    tools.reranker_candidate_multiplier = 10
    yield tools
    client.close()


async def test_unit_neighbors_stay_in_document_and_deduplicate_overlapping_windows(
    store: VectorStoreTools,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scroll = store.client.scroll
    monkeypatch.setattr(store.client, "scroll", lambda **kwargs: scroll(**{**kwargs, "limit": 1}))
    seeds = [
        {"chunk_id": "a:chunk:1", "document_id": "a", "chunk_index": 1, "score": 0.9},
        {"chunk_id": "a:chunk:3", "document_id": "a", "chunk_index": 3, "score": 0.8},
        {"chunk_id": "b:chunk:0", "document_id": "b", "chunk_index": 0, "score": 0.7},
    ]

    expanded = await store.expand_neighbors(seeds)

    assert expanded[:3] == seeds
    neighbors = expanded[3:]
    assert [chunk["chunk_id"] for chunk in neighbors] == [
        "a:chunk:0",
        "a:chunk:2",
        "a:chunk:4",
        "b:chunk:1",
    ]
    assert neighbors[1]["neighbor_of"] == "a:chunk:1"
    assert neighbors[1]["source_locations"] == [{"page": 2}]
    assert neighbors[1]["filename"] == "a.pdf"
    assert neighbors[1]["doc_title"] == "A"
    assert neighbors[1]["score"] == neighbors[1]["final_score"] == 0.9
    assert all(chunk["origin"] == "neighbor" for chunk in neighbors)
    assert all("reranker_score" not in chunk for chunk in neighbors)


async def test_unit_neighbors_skip_empty_or_legacy_seeds_without_querying(
    store: VectorStoreTools,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scroll = Mock()
    monkeypatch.setattr(store.client, "scroll", scroll)
    seeds = [{"chunk_id": "legacy", "text": "old excerpt", "score": 0.9}]

    assert await store.expand_neighbors([]) == []
    assert await store.expand_neighbors(seeds) == seeds
    scroll.assert_not_called()


async def test_unit_neighbors_preserve_ranked_seeds_and_skip_unavailable_positions(
    store: VectorStoreTools,
) -> None:
    seeds = [
        {"chunk_id": "a:chunk:5", "document_id": "a", "chunk_index": 5, "score": 0.9},
        {"chunk_id": "a:chunk:4", "document_id": "a", "chunk_index": 4, "score": 0.8},
        {"chunk_id": "legacy", "document_id": "a", "score": 0.7},
    ]

    expanded = await store.expand_neighbors(seeds)

    assert expanded[:3] == seeds
    assert [chunk["chunk_id"] for chunk in expanded[3:]] == ["a:chunk:3"]
    assert expanded[1]["score"] == 0.8
    assert "origin" not in expanded[1]


async def test_unit_retrieval_adds_neighbors_after_final_reranked_top_k(
    store: VectorStoreTools,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reranker = Mock()
    reranker.rerank.side_effect = lambda *, query, documents: [
        {"a excerpt 1": 0.9, "a excerpt 4": 0.8}.get(text, -0.2) for text in documents
    ]
    monkeypatch.setattr("src.utils.vector_store.tools.get_reranker", lambda: reranker)
    node = RetrievalNode(retriever=store, llm_provider=lambda: None)
    node.top_k = 1
    node.max_query_variants = 1
    state = AgentState(
        current_intent="query_document", requires_retrieval=True, retrieval_query="pendaftaran"
    )

    update = await node.run(state)

    assert "error" not in update
    assert [chunk["chunk_id"] for chunk in update["retrieved_chunks"]] == [
        "a:chunk:1",
        "a:chunk:0",
        "a:chunk:2",
    ]
    assert update["retrieved_chunks"][0]["chunk_index"] == 1
    assert update["retrieved_chunks"][0]["reranker_score"] == 0.9
    reranker.rerank.assert_called_once()
    assert len(reranker.rerank.call_args.kwargs["documents"]) == 8


@pytest.mark.parametrize("strategy", ["similarity", "rrf"])
async def test_unit_retrieval_does_not_expand_other_strategies(
    store: VectorStoreTools,
    strategy: str,
) -> None:
    store.retrieval_strategy = strategy
    node = RetrievalNode(retriever=store, llm_provider=lambda: None)
    node.top_k = 1
    node.max_query_variants = 1

    update = await node.run(
        AgentState(
            current_intent="query_document", requires_retrieval=True, retrieval_query="pendaftaran"
        )
    )

    assert "error" not in update
    assert len(update["retrieved_chunks"]) == 1
    assert "origin" not in update["retrieved_chunks"][0]
