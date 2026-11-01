"""Unit tests for bounded graph expansion and evidence merging."""

from __future__ import annotations

from types import SimpleNamespace

from src.graph.expand import GraphExpander


class _Store:
    def verify(self) -> None:
        return None

    def seed_entities(self, **kwargs):
        return [
            {
                "entity_id": "regulation:cuti",
                "name": "Cuti",
                "entity_type": "Regulation",
                "df": 3,
                "source": "query",
            }
        ]

    def expand_entities(self, **kwargs):
        return [
            {
                "entity_id": "requirement:biaya",
                "name": "Biaya administrasi",
                "entity_type": "Requirement",
                "df": 2,
                "rel": "REQUIRES",
                "hop": 1,
            }
        ]

    def chunks_for_entities(self, **kwargs):
        return [
            {
                "chunk_id": "kb:chunk:2",
                "text": "Biaya administrasi cuti sebesar 5% dari DPP.",
                "filename": "wadek.pdf",
                "page": 86,
                "document_id": "kb",
                "shared": 2,
                "entities": ["Cuti", "Biaya administrasi"],
            }
        ]


def _settings(**overrides):
    values = {
        "GRAPH_RAG_ROUTING": "always",
        "GRAPH_RAG_MAX_ENTITIES": 10,
        "GRAPH_RAG_MAX_ENTITY_FREQUENCY": 40,
        "GRAPH_RAG_MAX_HOPS": 1,
        "GRAPH_RAG_MAX_EXTRA_CHUNKS": 1,
        "GRAPH_RAG_ALLOWED_RELATIONSHIPS": ["REQUIRES"],
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_expansion_reserves_slots_and_marks_graph_origin() -> None:
    expander = GraphExpander(_Store(), _settings())
    seeds = [
        {"chunk_id": "kb:chunk:1", "text": "Cuti akademik.", "score": 0.9},
        {"chunk_id": "kb:chunk:3", "text": "Lain.", "score": 0.8},
    ]

    results, trace = expander.expand("syarat cuti", seeds, top_k=2)

    assert trace["used"] is True
    assert len(results) == 2
    assert results[0]["chunk_id"] == "kb:chunk:1"
    assert results[1]["origin"] == "graph"
    assert results[1]["chunk_id"] == "kb:chunk:2"


def test_expansion_falls_back_to_seeds_when_store_fails() -> None:
    class Broken(_Store):
        def seed_entities(self, **kwargs):
            raise RuntimeError("neo4j down")

    expander = GraphExpander(Broken(), _settings())
    seeds = [{"chunk_id": "kb:chunk:1", "text": "Cuti.", "score": 0.9}]

    results, trace = expander.expand("syarat cuti", seeds, top_k=5)

    assert results == seeds
    assert trace["used"] is False
    assert "neo4j down" in trace["error"]


def test_routing_skips_simple_lookup_in_multi_hop_mode() -> None:
    expander = GraphExpander(_Store(), _settings(GRAPH_RAG_ROUTING="multi_hop"))
    seeds = [{"chunk_id": "kb:chunk:1", "text": "Cuti.", "score": 0.9}]

    results, trace = expander.expand("berapa lama cuti", seeds, top_k=5)

    assert results == seeds
    assert trace["skipped"] == "routing"
