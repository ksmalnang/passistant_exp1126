"""Bounded graph expansion and evidence consolidation for Qdrant seed results."""

from __future__ import annotations

import logging
import time
from typing import Any

from src.graph.store import GraphStore

logger = logging.getLogger(__name__)

MULTI_HOP_MARKERS = (
    " dan ",
    " serta ",
    " atau ",
    "beda",
    "perbedaan",
    "banding",
    "selain",
    " jika ",
    " kalau ",
    "syarat",
    "prosedur",
    "bagaimana",
    "gimana",
    "apa saja",
)


class GraphExpander:
    """Expand vector seeds through Neo4j entities without displacing strong direct evidence."""

    def __init__(self, store: GraphStore, settings: Any):
        self._store = store
        self._settings = settings

    def available(self) -> bool:
        try:
            self._store.verify()
            return True
        except Exception as exc:
            logger.warning("Neo4j is unavailable; using vector-only retrieval: %s", exc)
            return False

    def should_expand(self, query: str) -> bool:
        if self._settings.GRAPH_RAG_ROUTING == "always":
            return True
        lowered = f" {query.lower()} "
        return any(marker in lowered for marker in MULTI_HOP_MARKERS)

    def expand(
        self,
        query: str,
        results: list[dict[str, Any]],
        top_k: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        trace: dict[str, Any] = {
            "used": False,
            "seed_entities": 0,
            "expanded_entities": 0,
            "extra_chunks": 0,
            "elapsed_ms": 0.0,
        }
        if not results or top_k <= 0:
            return results, trace
        if not self.should_expand(query):
            trace["skipped"] = "routing"
            return results, trace

        started = time.perf_counter()
        seed_ids = [str(result["chunk_id"]) for result in results if result.get("chunk_id")]
        if not seed_ids:
            return results, trace
        try:
            seeds = self._store.seed_entities(
                seed_chunk_ids=seed_ids,
                query=query,
                max_entities=self._settings.GRAPH_RAG_MAX_ENTITIES,
                max_frequency=self._settings.GRAPH_RAG_MAX_ENTITY_FREQUENCY,
            )
            query_seeds = [
                seed
                for seed in seeds
                if seed.get("source") == "query"
                and seed.get("entity_type")
                in {
                    "Regulation",
                    "Deadline",
                    "Course",
                    "Scholarship",
                    "Evaluation",
                    "ElectiveGroup",
                }
            ]
            if query_seeds:
                chunk_seeds = [seed for seed in seeds if seed.get("source") == "chunk"][:5]
                seeds = query_seeds + chunk_seeds
            expanded = self._store.expand_entities(
                entity_ids=[seed["entity_id"] for seed in seeds],
                allowed_relationships=self._settings.GRAPH_RAG_ALLOWED_RELATIONSHIPS,
                max_hops=self._settings.GRAPH_RAG_MAX_HOPS,
                max_entities=self._settings.GRAPH_RAG_MAX_ENTITIES,
                max_frequency=self._settings.GRAPH_RAG_MAX_ENTITY_FREQUENCY,
            )
            candidate_ids = sorted(
                {entity["entity_id"] for entity in seeds}
                | {entity["entity_id"] for entity in expanded}
            )
            extra_limit = min(
                self._settings.GRAPH_RAG_MAX_EXTRA_CHUNKS,
                max(0, top_k - 1),
            )
            extra_chunks = self._store.chunks_for_entities(
                entity_ids=candidate_ids,
                exclude_chunk_ids=seed_ids,
                limit=extra_limit,
            )
        except Exception as exc:
            logger.warning("Graph expansion failed; using vector-only results: %s", exc)
            trace["error"] = str(exc)
            return results, trace

        trace.update(
            {
                "used": True,
                "seed_entities": len(seeds),
                "expanded_entities": len(expanded),
                "extra_chunks": len(extra_chunks),
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
            }
        )
        if not extra_chunks:
            return results, trace

        seeds_kept = max(1, top_k - len(extra_chunks))
        kept = results[:seeds_kept]
        lowest = min(float(result.get("score") or 0.0) for result in kept)
        graph_results: list[dict[str, Any]] = []
        for chunk in extra_chunks:
            if not chunk.get("text"):
                continue
            graph_results.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "filename": chunk.get("filename"),
                    "document_id": chunk.get("document_id"),
                    "source_locations": [{"page": chunk.get("page")}]
                    if chunk.get("page") is not None
                    else [],
                    "score": lowest * 0.5,
                    "final_score": lowest * 0.5,
                    "origin": "graph",
                    "graph_entities": chunk.get("entities") or [],
                    "graph_shared": chunk.get("shared") or 0,
                }
            )
        merged = kept + graph_results
        trace["graph_chunk_ids"] = [chunk["chunk_id"] for chunk in graph_results]
        returned = merged[:top_k]
        trace["returned"] = len(returned)
        return returned, trace


def build_graph_expander(settings: Any) -> GraphExpander | None:
    """Build the expander only when graph retrieval is enabled and configured."""
    if not settings.GRAPH_RAG_ENABLED:
        return None
    if not settings.NEO4J_PASSWORD:
        logger.warning("GRAPH_RAG_ENABLED is true but NEO4J_PASSWORD is unset")
        return None
    store = GraphStore(
        uri=settings.NEO4J_URI,
        user=settings.NEO4J_USER,
        password=settings.NEO4J_PASSWORD,
        database=settings.NEO4J_DATABASE,
        timeout_seconds=settings.GRAPH_RAG_TIMEOUT_SECONDS,
    )
    return GraphExpander(store, settings)
