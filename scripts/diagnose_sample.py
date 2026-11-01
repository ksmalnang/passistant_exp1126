"""Run one evaluation sample and print per-metric judge output."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from src.eval.models import DatasetSample


async def diagnose(sample: DatasetSample) -> None:
    from ragas import EvaluationDataset, SingleTurnSample, aevaluate
    from ragas.run_config import RunConfig

    from src.eval.evaluate import build_metrics, default_openrouter_settings
    from src.eval.openrouter import (
        OpenRouterClient,
        OpenRouterRagasLLM,
        Usage,
        ragas_extra_body,
    )
    from src.eval.pipeline import LivePipeline, strip_citations

    usage = Usage()
    client = OpenRouterClient(default_openrouter_settings(), usage)
    result = (await LivePipeline(usage=usage).run([sample]))[0]
    print("=" * 80)
    print("Q:", sample.user_input)
    print("REF:", sample.reference)
    print("RESP:", result.response)
    print("chunks:")
    for chunk in result.retrieved_chunks:
        print("  ", chunk)
    print("contexts:")
    for context in result.retrieved_contexts:
        print("  ---", context[:600].replace("\n", " "))

    judge = OpenRouterRagasLLM(client, "deepseek/deepseek-v4-flash-0731", extra_body=ragas_extra_body("llm"))
    dataset = EvaluationDataset(
        samples=[
            SingleTurnSample(
                user_input=sample.user_input,
                retrieved_contexts=result.retrieved_contexts,
                response=strip_citations(result.response),
                reference=sample.reference,
                reference_contexts=list(sample.reference_contexts),
            )
        ]
    )
    for metric in build_metrics():
        metric.llm = judge
        metric.init(RunConfig())
        score = await metric.single_turn_ascore(dataset.samples[0], None)
        print(f"metric {metric.name}: {score}")

    outcome = await aevaluate(dataset=dataset, metrics=build_metrics(), llm=judge, show_progress=False)
    print("aevaluate:", outcome)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--id", required=True)
    args = parser.parse_args()

    from src.eval.dataset import load_dataset

    samples = {sample.id: sample for sample in load_dataset(args.dataset)}
    asyncio.run(diagnose(samples[args.id]))


if __name__ == "__main__":
    main()
