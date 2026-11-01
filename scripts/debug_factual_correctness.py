"""Print claim-level verdicts for FactualCorrectness on a saved sample."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path


async def main(dataset: Path, report: Path, sample_id: str) -> None:
    from ragas.metrics import FactualCorrectness
    from ragas.run_config import RunConfig

    from src.eval.evaluate import default_openrouter_settings
    from src.eval.openrouter import OpenRouterClient, OpenRouterRagasLLM, Usage, ragas_extra_body

    rows = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines() if line]
    sample = next(row for row in rows if row["id"] == sample_id)
    saved = json.loads(report.read_text(encoding="utf-8"))
    response = next(item for item in saved["samples"] if item["id"] == sample_id)["response"]

    client = OpenRouterClient(default_openrouter_settings(), Usage())
    judge = OpenRouterRagasLLM(
        client, saved["run"]["models"]["judge"], extra_body=ragas_extra_body("llm")
    )
    metric = FactualCorrectness(mode="f1")
    metric.llm = judge
    metric.init(RunConfig())

    print("REFERENCE:", sample["reference"])
    print("RESPONSE:", response)

    reference_claims = await metric.decompose_claims(sample["reference"], None)
    response_claims = await metric.decompose_claims(response, None)
    print("\nREFERENCE CLAIMS:")
    for claim in reference_claims:
        verdicts = await metric.verify_claims(response, [claim], None)
        print(f"  {bool(verdicts[0])!s:5} {claim}")
    print("\nRESPONSE CLAIMS:")
    for claim in response_claims:
        verdicts = await metric.verify_claims(sample["reference"], [claim], None)
        print(f"  {bool(verdicts[0])!s:5} {claim}")


def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--id", required=True)
    args = parser.parse_args()
    asyncio.run(main(args.dataset, args.report, args.id))


if __name__ == "__main__":
    run()
