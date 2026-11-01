"""Convert the repository's legacy RAGAS fixtures into the current testset schema.

The legacy fixtures use id/question/contexts/answer/ground_truth. The current
evaluator reads id/user_input/reference/reference_contexts/metadata. This script
keeps every original question and ground truth verbatim; only field names and
sample ids change. The originals under tests/fixtures/ are left untouched.

Outputs:
  tests/fixtures/eval/d1_legacy30_v1.jsonl   from ragas_dataset_11051234.jsonl
  tests/fixtures/eval/d2_legacy35_v1.jsonl   from ragas_dataset_15.jsonl
  tests/fixtures/eval/public_beta_v1.jsonl   union of both (65 samples)
"""

from __future__ import annotations

import json
from pathlib import Path

FIXTURES = Path("tests/fixtures")
OUTPUT = FIXTURES / "eval"
DATASET_VERSION = "public_beta_v1"

SOURCES = {
    "d1": FIXTURES / "ragas_dataset_11051234.jsonl",
    "d2": FIXTURES / "ragas_dataset_15.jsonl",
}

REASONING_TO_QUESTION_TYPE = {
    "single-hop": "simple",
    "multi-hop": "multi_doc",
    "comparison": "multi_doc",
    "inference": "conditional",
}

TOPIC_KEYWORDS = {
    "cuti": ("cuti",),
    "sks": ("sks", "beban studi", "ambil matkul", "ambil mata kuliah"),
    "mbkm": ("mbkm", "magang"),
    "beasiswa": ("beasiswa", "kip"),
    "ujian": ("uts", "uas", "ujian", "remedial", "remed", "susulan"),
    "pelanggaran": ("nyontek", "plagiat", "joki", "sanksi", "pelanggaran", "skorsing"),
    "dpp": ("dpp", "spp", "pembayaran", "bayar", "tagihan"),
    "tugas_akhir": ("tugas akhir", "ta", "skripsi", "kerja praktek", "kp"),
    "yudisium": ("yudisium", "wisuda", "cumlaude", "ijazah", "legalisir"),
    "kurikulum": ("kurikulum", "mata kuliah", "matkul", "cpl", "semester"),
    "organisasi": ("bem", "hima", "ormawa", "kegiatan", "proposal", "pengurus"),
}


def topic_for(question: str) -> str:
    lowered = question.lower()
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return topic
    return "lainnya"


def convert(source: Path, prefix: str) -> list[dict]:
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    converted = []
    for position, row in enumerate(rows, start=1):
        original_id = row["id"]
        contexts = [
            item["text"]
            for item in row.get("contexts", [])
            if isinstance(item, dict) and item.get("is_relevant") and item.get("text")
        ]
        metadata = {
            **(row.get("metadata") or {}),
            "original_id": original_id,
            "original_dataset": source.name,
            "answerable": True,
            "dataset_version": DATASET_VERSION,
            "question_type": REASONING_TO_QUESTION_TYPE.get(
                (row.get("metadata") or {}).get("reasoning_type", ""), "simple"
            ),
            "topic": topic_for(row["question"]),
            "source_doc_ids": [],
        }
        converted.append(
            {
                "id": f"{prefix}-{position:02d}",
                "user_input": row["question"],
                "reference": row["ground_truth"],
                "reference_contexts": contexts,
                "metadata": metadata,
            }
        )
    return converted


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    written: list[dict] = []
    for prefix, source in SOURCES.items():
        rows = convert(source, prefix)
        path = OUTPUT / f"{prefix}_legacy{len(rows)}_v1.jsonl"
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        print(f"{path}: {len(rows)} samples")
        written.extend(rows)

    ids = [row["id"] for row in written]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate sample ids in converted datasets")

    union = OUTPUT / "public_beta_v1.jsonl"
    union.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in written),
        encoding="utf-8",
    )
    print(f"{union}: {len(written)} samples")


if __name__ == "__main__":
    main()
