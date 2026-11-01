"""Offline diagnostic for graph expansion using stored baseline contexts.

This does not replace the RAGAS judge. It measures how many source chunks graph
expansion adds and whether those chunks improve a lexical coverage proxy for the
reference answer.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from src.config import get_settings
from src.graph.expand import GraphExpander
from src.graph.store import GraphStore

TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "yang",
    "dan",
    "di",
    "ke",
    "dari",
    "untuk",
    "pada",
    "adalah",
    "dengan",
    "atau",
    "ini",
    "itu",
    "a",
    "the",
    "of",
    "to",
    "in",
    "is",
    "are",
}


def tokens(text: str) -> set[str]:
    return {token for token in TOKEN_RE.findall(text.lower()) if token not in STOPWORDS and len(token) > 2}


def coverage(reference: str, contexts: list[str]) -> float:
    reference_tokens = tokens(reference)
    if not reference_tokens:
        return 0.0
    context_tokens = tokens(" ".join(contexts))
    return len(reference_tokens & context_tokens) / len(reference_tokens)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-extra", type=int, default=3)
    parser.add_argument("--routing", choices=("always", "multi_hop"), default="always")
    parser.add_argument("--max-hops", type=int, default=1)
    args = parser.parse_args()

    settings = get_settings()
    if not settings.NEO4J_PASSWORD:
        raise SystemExit("NEO4J_PASSWORD is required")
    settings.GRAPH_RAG_ENABLED = True
    settings.GRAPH_RAG_ROUTING = args.routing
    settings.GRAPH_RAG_MAX_EXTRA_CHUNKS = args.max_extra
    settings.GRAPH_RAG_MAX_HOPS = args.max_hops
    store = GraphStore(
        uri=settings.NEO4J_URI,
        user=settings.NEO4J_USER,
        password=settings.NEO4J_PASSWORD,
        database=settings.NEO4J_DATABASE,
        timeout_seconds=settings.GRAPH_RAG_TIMEOUT_SECONDS,
    )
    expander = GraphExpander(store, settings)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for sample in report["samples"]:
        contexts = sample.get("retrieved_contexts") or []
        chunks = sample.get("retrieved_chunks") or []
        seeds = [
            {
                "chunk_id": chunk.get("chunk_id"),
                "text": contexts[index] if index < len(contexts) else "",
                "filename": chunk.get("filename"),
                "score": chunk.get("score") or 0.0,
                "source_locations": [{"page": chunk.get("page")}],
            }
            for index, chunk in enumerate(chunks)
            if chunk.get("chunk_id")
        ]
        expanded, trace = expander.expand(sample["user_input"], seeds, args.top_k)
        graph_contexts = [chunk["text"] for chunk in expanded if chunk.get("origin") == "graph"]
        rows.append(
            {
                "id": sample["id"],
                "baseline_contexts": len(contexts),
                "graph_added": len(graph_contexts),
                "baseline_coverage": round(coverage(sample["reference"], contexts), 4),
                "graph_coverage": round(coverage(sample["reference"], [c["text"] for c in expanded]), 4),
                "graph_trace": trace,
            }
        )
    store.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    changed = [row for row in rows if row["graph_added"]]
    improved = [row for row in changed if row["graph_coverage"] > row["baseline_coverage"] + 1e-9]
    print(
        json.dumps(
            {
                "samples": len(rows),
                "samples_with_graph_chunks": len(changed),
                "samples_improved_proxy": len(improved),
                "mean_baseline_coverage": round(sum(r["baseline_coverage"] for r in rows) / len(rows), 4),
                "mean_graph_coverage": round(sum(r["graph_coverage"] for r in rows) / len(rows), 4),
                "mean_added": round(sum(r["graph_added"] for r in rows) / len(rows), 2),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
