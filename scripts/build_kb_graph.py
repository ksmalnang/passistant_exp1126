"""Build the Neo4j academic graph from the Qdrant chunk index.

Deterministic extraction covers courses, semesters, and programs. The optional
LLM pass adds policy entities and typed relationships for rule-bearing chunks,
with a per-chunk evidence quote and an on-disk cache.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from openai import OpenAI
from qdrant_client import QdrantClient

from src.config import get_settings
from src.graph.extraction import (
    extract_deterministic,
    extract_with_llm,
    has_policy_signal,
    merge_extractions,
    normalize_name,
)
from src.graph.store import GraphStore

CACHE_ROOT = Path("data/processed/graph_extraction")
MANIFEST_ROOT = Path("data/processed/graph_manifests")


def load_chunks(client: QdrantClient, collection: str) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=256,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        for point in points:
            payload = point.payload or {}
            locations = payload.get("source_locations") or [{}]
            chunks.append(
                {
                    "chunk_id": payload.get("chunk_id"),
                    "text": payload.get("text") or "",
                    "document_id": payload.get("document_id"),
                    "filename": payload.get("filename"),
                    "doc_title": payload.get("doc_title"),
                    "chunk_index": payload.get("chunk_index"),
                    "page": locations[0].get("page"),
                }
            )
        if offset is None:
            break
    return [chunk for chunk in chunks if chunk["chunk_id"] and chunk["text"]]


def course_titles(chunks: list[dict[str, Any]]) -> dict[str, str]:
    """Map normalized course title to course code using the deterministic parser."""
    titles: dict[str, str] = {}
    for chunk in chunks:
        entities, _ = extract_deterministic(chunk)
        for entity in entities:
            if entity["entity_type"] != "Course":
                continue
            code = entity["entity_id"].split(":", 1)[1]
            if code.startswith("IF21"):
                titles[normalize_name(entity["name"])] = code
    return titles


def resolve_courses(
    entities: list[dict[str, Any]],
    relationships: list[dict[str, Any]],
    titles: dict[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Point LLM-discovered course names at the canonical code-based node."""
    remap: dict[str, str] = {}
    by_id = {entity["entity_id"]: entity for entity in entities}
    for entity in list(entities):
        if entity["entity_type"] != "Course" or entity["entity_id"].startswith("course:IF21"):
            continue
        code = titles.get(entity["normalized_name"])
        if not code:
            continue
        canonical = f"course:{code}"
        remap[entity["entity_id"]] = canonical
        by_id.pop(entity["entity_id"], None)
        existing = by_id.get(canonical)
        if existing:
            existing["aliases"] = sorted(
                set(existing.get("aliases") or []) | {entity["name"]} | set(entity.get("aliases") or [])
            )
            if existing.get("normalized_name") in {"", None} or existing["normalized_name"].startswith("if21"):
                existing["normalized_name"] = entity["normalized_name"]
        else:
            entity["entity_id"] = canonical
            entity["aliases"] = sorted({entity["name"], code} | set(entity.get("aliases") or []))
            by_id[canonical] = entity
    resolved = []
    for relationship in relationships:
        source = remap.get(relationship["source"], relationship["source"])
        target = remap.get(relationship["target"], relationship["target"])
        if source not in by_id or target not in by_id or source == target:
            continue
        resolved.append({**relationship, "source": source, "target": target})
    return list(by_id.values()), resolved


def select_llm_chunks(chunks: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Round-robin rule-bearing chunks per document up to the requested limit."""
    candidates = [chunk for chunk in chunks if has_policy_signal(chunk["text"])]
    by_doc: dict[str, list[dict[str, Any]]] = {}
    for chunk in candidates:
        by_doc.setdefault(chunk["document_id"], []).append(chunk)
    selected: list[dict[str, Any]] = []
    index = 0
    while len(selected) < limit:
        added = False
        for document_chunks in by_doc.values():
            if index < len(document_chunks) and len(selected) < limit:
                selected.append(document_chunks[index])
                added = True
        if not added:
            break
        index += 1
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", default=None)
    parser.add_argument("--llm-extract", action="store_true")
    parser.add_argument("--max-llm-chunks", type=int, default=200)
    parser.add_argument("--reset", action="store_true", help="Delete existing graph nodes first")
    args = parser.parse_args()

    settings = get_settings()
    collection = args.collection or settings.QDRANT_COLLECTION_NAME
    if not settings.NEO4J_PASSWORD:
        raise SystemExit("NEO4J_PASSWORD is required")
    if not settings.OPENAI_API_KEY:
        raise SystemExit("OPENAI_API_KEY is required for LLM extraction")

    qdrant = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
    chunks = load_chunks(qdrant, collection)
    if not chunks:
        raise SystemExit(f"no chunks found in {collection}")

    store = GraphStore(
        uri=settings.NEO4J_URI,
        user=settings.NEO4J_USER,
        password=settings.NEO4J_PASSWORD,
        database=settings.NEO4J_DATABASE,
    )
    store.verify()
    store.ensure_schema()
    if args.reset:
        store.clear()

    titles = course_titles(chunks)
    llm_chunks = select_llm_chunks(chunks, args.max_llm_chunks) if args.llm_extract else []
    llm_ids = {chunk["chunk_id"] for chunk in llm_chunks}
    client = OpenAI(api_key=settings.OPENAI_API_KEY, base_url=settings.OPENAI_BASE_URL)
    cache_dir = CACHE_ROOT / collection
    started = time.perf_counter()
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "cached": 0, "llm_chunks": 0}
    entity_counts: dict[str, int] = {}
    relationship_counts: dict[str, int] = {}

    for position, chunk in enumerate(chunks, start=1):
        deterministic_entities, deterministic_relationships = extract_deterministic(chunk)
        entity_groups = [deterministic_entities]
        relationship_groups = [deterministic_relationships]
        if chunk["chunk_id"] in llm_ids:
            llm_entities, llm_relationships, token_usage = extract_with_llm(
                chunk=chunk,
                client=client,
                model=settings.LLM_MODEL,
                cache_dir=cache_dir,
                provider=settings.LLM_PROVIDER,
            )
            entity_groups.append(llm_entities)
            relationship_groups.append(llm_relationships)
            usage["prompt_tokens"] += token_usage.get("prompt_tokens", 0)
            usage["completion_tokens"] += token_usage.get("completion_tokens", 0)
            usage["cached"] += int(bool(token_usage.get("cached")))
            usage["llm_chunks"] += 1
        entities, relationships = merge_extractions(entity_groups, relationship_groups)
        entities, relationships = resolve_courses(entities, relationships, titles)
        store.upsert_chunk(chunk, entities, relationships)
        for entity in entities:
            entity_counts[entity["entity_type"]] = entity_counts.get(entity["entity_type"], 0) + 1
        for relationship in relationships:
            relationship_counts[relationship["type"]] = relationship_counts.get(relationship["type"], 0) + 1
        if position % 100 == 0:
            print(f"ingested {position}/{len(chunks)} chunks", flush=True)

    stats = store.stats()
    qdrant_ids = {chunk["chunk_id"] for chunk in chunks}
    consistency = store.consistency_report(qdrant_ids)
    manifest = {
        "collection": collection,
        "created_at": datetime.now(UTC).isoformat(),
        "chunks": len(chunks),
        "llm_extraction": {
            "enabled": args.llm_extract,
            "selected_chunks": len(llm_chunks),
            "max_chunks": args.max_llm_chunks,
            "cache_dir": str(cache_dir),
            "usage": usage,
        },
        "course_titles": len(titles),
        "stats": stats,
        "entity_mentions": entity_counts,
        "relationship_assertions": relationship_counts,
        "consistency": consistency,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }
    MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = MANIFEST_ROOT / f"{collection}.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"manifest: {manifest_path}")
    store.close()


if __name__ == "__main__":
    main()
