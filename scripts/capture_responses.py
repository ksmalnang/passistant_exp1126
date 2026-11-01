"""Capture live pipeline responses without running the RAGAS judge.

The resulting responses file can be scored repeatedly with different fixed
judges via `evaluate --mode fixture`, which isolates generation/retrieval from
judge availability and cost.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from src.eval.dataset import load_dataset
from src.eval.pipeline import LivePipeline


async def capture(dataset: Path, output: Path) -> None:
    samples = load_dataset(dataset)
    results = await LivePipeline().run(samples)
    payload = {
        result.sample.id: {
            "response": result.response,
            "retrieved_contexts": result.retrieved_contexts,
            "retrieved_chunks": result.retrieved_chunks,
        }
        for result in results
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(payload)} responses to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(capture(args.dataset, args.output))


if __name__ == "__main__":
    main()
