"""Run a live evaluation in bounded batches and merge the results.

Provider calls occasionally stall. A single long process loses the whole run when
that happens; this runner isolates batches, retries a failed batch, and merges
the per-sample scores with the repository's own report writer.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from src.eval.models import DatasetSample, EvaluationConfig, RunContext, SampleResult
from src.eval.reporting import build_report, write_report


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def run_batch(
    dataset: Path,
    batch_path: Path,
    report_dir: Path,
    run_name: str,
    timeout: int,
    log_path: Path,
) -> bool:
    command = [
        sys.executable,
        "-m",
        "src.eval",
        "evaluate",
        "--dataset",
        str(batch_path),
        "--mode",
        "live",
        "--run-name",
        run_name,
        "--output-dir",
        str(report_dir),
    ]
    for attempt in range(2):
        started = time.perf_counter()
        with open(log_path, "a", encoding="utf-8") as log:
            log.write(f"\n=== {run_name} attempt {attempt + 1} ===\n")
            try:
                completed = subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                log.write(f"\nTIMEOUT after {timeout}s\n")
                print(f"{run_name}: timeout after {timeout}s (attempt {attempt + 1})", flush=True)
                continue
        elapsed = time.perf_counter() - started
        print(f"{run_name}: exit={completed.returncode} in {elapsed:.0f}s", flush=True)
        if completed.returncode == 0 and (report_dir / f"{run_name}.json").is_file():
            return True
    return False


def merge_reports(
    reports: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    dataset: Path,
    output_dir: Path,
    run_name: str,
) -> dict[str, Any]:
    sample_map = {sample["id"]: sample for report in reports for sample in report["samples"]}
    missing = [row["id"] for row in rows if row["id"] not in sample_map]
    results: list[SampleResult] = []
    for row in rows:
        record = sample_map.get(row["id"])
        if record is None:
            results.append(
                SampleResult(
                    sample=DatasetSample(
                        id=row["id"],
                        user_input=row["user_input"],
                        reference=row["reference"],
                        reference_contexts=tuple(row.get("reference_contexts") or ()),
                        metadata=row.get("metadata") or {},
                    ),
                    response="",
                    retrieved_contexts=[],
                    failures={"pipeline": "batch did not complete"},
                )
            )
            continue
        results.append(
            SampleResult(
                sample=DatasetSample(
                    id=record["id"],
                    user_input=record["user_input"],
                    reference=record["reference"],
                    reference_contexts=tuple(row.get("reference_contexts") or ()),
                    metadata=record.get("metadata") or {},
                ),
                response=record["response"],
                retrieved_contexts=record.get("retrieved_contexts") or [],
                retrieved_chunks=record.get("retrieved_chunks") or [],
                scores=record.get("scores") or {},
                failures=record.get("failures") or {},
            )
        )

    first = reports[0]["run"]
    usage: dict[str, Any] = {}
    for report in reports:
        for key, value in report.get("usage", {}).items():
            if isinstance(value, (int, float)):
                usage[key] = usage.get(key, 0) + value
    usage["cost_usd"] = round(usage.get("cost_usd", 0.0), 6)
    context = RunContext(
        strategy=first["retrieval"]["strategy"],
        top_k=first["retrieval"]["top_k"],
        judge_model=first["models"]["judge"],
        embedding_model=first["models"]["embedding"],
        pipeline_model=first["models"]["pipeline"],
        started_at=datetime.fromisoformat(first["started_at"]),
        duration_seconds=sum(report["run"]["duration_seconds"] for report in reports),
        collection=(first.get("index") or {}).get("collection", ""),
        index_points=(first.get("index") or {}).get("points"),
        index_manifest=(first.get("index") or {}).get("manifest"),
    )
    config = EvaluationConfig(dataset_path=dataset, mode="live", output_dir=output_dir, run_name=run_name)
    report = build_report(config, results, usage, context)
    report["batches"] = {
        "count": len(reports),
        "missing_samples": missing,
        "duration_seconds": round(time.perf_counter(), 1),
    }
    write_report(report, config)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("src/eval/reports"))
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--skip-batches", type=int, default=0)
    args = parser.parse_args()

    rows = read_rows(args.dataset)
    batch_dir = Path(".tmp_probe/batches")
    log_dir = Path(".tmp_probe/logs")
    batch_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []
    for index in range(args.skip_batches, len(rows), args.batch_size):
        batch_rows = rows[index : index + args.batch_size]
        batch_path = batch_dir / f"{args.run_name}_b{index // args.batch_size:02d}.jsonl"
        batch_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in batch_rows),
            encoding="utf-8",
        )
        batch_name = f"{args.run_name}_b{index // args.batch_size:02d}"
        ok = run_batch(
            dataset=args.dataset,
            batch_path=batch_path,
            report_dir=args.output_dir,
            run_name=batch_name,
            timeout=args.timeout,
            log_path=log_dir / f"{batch_name}.log",
        )
        report_path = args.output_dir / f"{batch_name}.json"
        if ok and report_path.is_file():
            reports.append(json.loads(report_path.read_text(encoding="utf-8")))
        else:
            print(f"{batch_name}: failed; continuing to next batch", flush=True)
    if not reports:
        raise SystemExit("no batch completed")
    report = merge_reports(reports, rows, args.dataset, args.output_dir, args.run_name)
    print(json.dumps(report["aggregate"], ensure_ascii=False, indent=2))
    print(f"merged report: {args.output_dir / (args.run_name + '.json')}")


if __name__ == "__main__":
    main()
