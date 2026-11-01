"""Reproduce one ContextRecall score from a saved report and print the judge output."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path


async def main(report_path: Path, sample_id: str) -> None:
    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics._context_recall import (
        QCA,
        ContextRecall,
        ContextRecallClassificationPrompt,
    )
    from ragas.run_config import RunConfig

    from src.eval.evaluate import default_openrouter_settings
    from src.eval.openrouter import (
        OpenRouterClient,
        OpenRouterRagasLLM,
        Usage,
        ragas_extra_body,
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    sample = next(item for item in report["samples"] if item["id"] == sample_id)
    client = OpenRouterClient(default_openrouter_settings(), Usage())
    judge = OpenRouterRagasLLM(
        client, report["run"]["models"]["judge"], extra_body=ragas_extra_body("llm")
    )
    prompt = ContextRecallClassificationPrompt()
    data = QCA(
        question=sample["user_input"],
        context="\n".join(sample["retrieved_contexts"]),
        answer=sample["reference"],
    )
    print("=== prompt ===")
    print(prompt.to_string(data)[:6000])
    result = await prompt.generate(data=data, llm=judge, callbacks=None)
    print("=== classifications ===")
    print(result.model_dump_json(indent=2))

    metric = ContextRecall()
    metric.llm = judge
    metric.init(RunConfig())
    score = await metric.single_turn_ascore(
        SingleTurnSample(
            user_input=sample["user_input"],
            retrieved_contexts=sample["retrieved_contexts"],
            reference=sample["reference"],
            response=sample["response"],
            reference_contexts=[],
        ),
        None,
    )
    print("metric score:", score)


def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--id", required=True)
    args = parser.parse_args()
    asyncio.run(main(args.report, args.id))


if __name__ == "__main__":
    run()
