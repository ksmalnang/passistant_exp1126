"""Regenerate answers from a saved evaluation report's retrieved contexts.

Isolates generation changes (prompt, model) from retrieval: the contexts are
taken verbatim from a previous live report, and the resulting responses file can
be scored with `evaluate --mode fixture`.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage

from src.services.response_generation import ResponseGenerationService
from src.utils.state import AgentState


def chunks_from_sample(sample: dict[str, Any]) -> list[dict[str, Any]]:
    contexts = sample.get("retrieved_contexts") or []
    metas = sample.get("retrieved_chunks") or []
    chunks: list[dict[str, Any]] = []
    for index, text in enumerate(contexts):
        meta = metas[index] if index < len(metas) else {}
        chunks.append(
            {
                "chunk_id": meta.get("chunk_id") or f"{sample['id']}:{index}",
                "filename": meta.get("filename"),
                "text": text,
                "final_score": meta.get("score") or 0.0,
                "score": meta.get("score") or 0.0,
                "source_locations": [{"page": meta.get("page")}] if meta.get("page") else [],
            }
        )
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, default=None)
    parser.add_argument("--editor-prompt", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    prompt = args.prompt.read_text(encoding="utf-8") if args.prompt else None
    editor = None
    if args.editor_prompt:
        from src.services.response_generation import GroundedAnswerEditor

        editor = GroundedAnswerEditor(prompt=args.editor_prompt.read_text(encoding="utf-8"))
    service = ResponseGenerationService(system_prompt=prompt, grounding_editor=editor)
    responses: dict[str, dict[str, Any]] = {}
    started = time.perf_counter()
    for position, sample in enumerate(report["samples"], start=1):
        state = AgentState(
            messages=[HumanMessage(content=sample["user_input"])],
            current_intent="query_document",
            retrieved_chunks=chunks_from_sample(sample),
        )
        result = service.generate(state)
        responses[sample["id"]] = {
            "response": result.get("draft_response") or "",
            "retrieved_contexts": sample.get("retrieved_contexts") or [],
        }
        print(f"[{position}/{len(report['samples'])}] {sample['id']} {time.perf_counter() - started:.0f}s")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(responses, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(responses)} responses to {args.output}")


if __name__ == "__main__":
    main()
