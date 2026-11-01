"""Print retrieval diagnostics for one question.

Shows the query variants the retrieval node would use, each variant's ranked
results, and the final context the generator receives.
"""

from __future__ import annotations

import argparse
import asyncio
import os


async def trace(question: str, collection: str, strategy: str, top_k: int, no_rewrite: bool) -> None:
    from langchain_core.messages import HumanMessage

    from src.clients.llm import get_rewrite_llm
    from src.utils.nodes.retrieval import RetrievalNode
    from src.utils.state import AgentState

    node = RetrievalNode(llm_provider=(lambda: None) if no_rewrite else get_rewrite_llm)
    node.top_k = top_k
    state = AgentState(
        messages=[HumanMessage(content=question)],
        requires_retrieval=True,
        retrieval_query=question,
        current_intent="query_document",
    )
    queries = await node._build_queries(state, "", follow_up=False)
    print(f"collection: {collection}  strategy: {strategy}  top_k: {top_k}")
    print(f"query: {question}")
    for index, query in enumerate(queries, start=1):
        print(f"variant {index}: {query}")

    results = await node.vector_tools.search_many(
        queries=queries, document_type=None, top_k=top_k, score_threshold=0.2
    )
    seen: dict[str, dict] = {}
    for query, found in zip(queries, results, strict=True):
        print(f"\n--- variant results: {query[:80]}")
        for chunk in found:
            print(
                f"  {chunk.get('score', 0):.3f}  rerank={chunk.get('reranker_score')} "
                f"vec={chunk.get('vector_score')}  {chunk.get('filename')} "
                f"p.{((chunk.get('source_locations') or [{}])[0]).get('page')} "
                f"{chunk.get('chunk_id')}"
            )
            print("      " + " ".join(str(chunk.get("text") or "")[:180].split()))
        seen.update({str(chunk.get("chunk_id")): chunk for chunk in found})

    update = await node.run(state)
    print(f"\n=== final context: {len(update.get('retrieved_chunks', []))} chunks, "
          f"confidence={update.get('response_confidence')}")
    for chunk in update.get("retrieved_chunks", []):
        page = ((chunk.get("source_locations") or [{}])[0]).get("page")
        print(
            f"  {chunk.get('score', 0):.3f}  {chunk.get('filename')} p.{page} "
            f"{chunk.get('chunk_id')}"
        )
        print("      " + " ".join(str(chunk.get("text") or "")[:300].split()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--strategy", default="reranker")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--no-rewrite", action="store_true")
    args = parser.parse_args()

    os.environ["QDRANT_COLLECTION_NAME"] = args.collection
    os.environ["RETRIEVAL_STRATEGY"] = args.strategy
    os.environ["RETRIEVAL_TOP_K"] = str(args.top_k)
    asyncio.run(trace(args.query, args.collection, args.strategy, args.top_k, args.no_rewrite))


if __name__ == "__main__":
    main()
