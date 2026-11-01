"""Verify Qdrant chunk ids and Neo4j Chunk nodes are in exact correspondence."""

from __future__ import annotations

import argparse
import json

from qdrant_client import QdrantClient

from src.config import get_settings
from src.graph.store import GraphStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", default=None)
    args = parser.parse_args()

    settings = get_settings()
    collection = args.collection or settings.QDRANT_COLLECTION_NAME
    if not settings.NEO4J_PASSWORD:
        raise SystemExit("NEO4J_PASSWORD is required")
    client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
    chunk_ids: set[str] = set()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=256,
            with_payload=["chunk_id"],
            with_vectors=False,
            offset=offset,
        )
        chunk_ids.update(str(point.payload["chunk_id"]) for point in points)
        if offset is None:
            break
    store = GraphStore(
        uri=settings.NEO4J_URI,
        user=settings.NEO4J_USER,
        password=settings.NEO4J_PASSWORD,
        database=settings.NEO4J_DATABASE,
    )
    report = store.consistency_report(chunk_ids)
    report["collection"] = collection
    report["stats"] = store.stats()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    store.close()
    if report["missing_in_graph"] or report["orphaned_in_graph"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
