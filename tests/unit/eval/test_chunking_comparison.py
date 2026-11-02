"""Experiment checkpoints must not reuse results for different inputs."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import compare_chunking_recall as comparison


@pytest.mark.parametrize("changed_file", ["baseline_config.json", "dataset.jsonl"])
def test_prepare_rejects_changed_inputs_before_first_completed_query(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed_file: str,
) -> None:
    text = "Mahasiswa wajib membayar administrasi sebelum ujian."
    document = SimpleNamespace(extracted_text=text, filename="rules.pdf")
    monkeypatch.setattr(comparison, "OCRCache", lambda: SimpleNamespace(load=lambda key: object()))
    monkeypatch.setattr(
        comparison,
        "OCRReindexService",
        lambda **kwargs: SimpleNamespace(_rebuild_document=lambda entry, rebuild_text: document),
    )
    comparison.write_json(
        tmp_path / "baseline_points.json",
        [
            {
                "payload": {
                    "document_id": "doc",
                    "chunk_id": "doc:chunk:0",
                    "chunk_index": 0,
                    "text": text,
                }
            }
        ],
    )
    comparison.write_json(
        tmp_path / "baseline_config.json",
        {
            "chunk_overlap_chars": 10,
            "candidate_multiplier": 6,
        },
    )
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "q1",
                "user_input": "Syarat ujian?",
                "reference_contexts": [text],
            }
        ),
        encoding="utf-8",
    )
    comparison.prepare(tmp_path, dataset)
    comparison.write_json(tmp_path / "queries.json", {"q1": ["Syarat ujian?"]})
    changed = tmp_path / changed_file
    payload = json.loads(changed.read_text(encoding="utf-8"))
    payload["candidate_multiplier" if changed_file.endswith(".json") else "user_input"] = 12
    comparison.write_json(changed, payload)

    with pytest.raises(ValueError, match="Checkpoint inputs changed"):
        comparison.prepare(tmp_path, dataset)
