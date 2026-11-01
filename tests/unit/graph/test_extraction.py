"""Unit tests for source-grounded graph extraction."""

from __future__ import annotations

from src.graph.extraction import (
    _semester_from_code,
    extract_deterministic,
    merge_extractions,
    normalize_name,
)


def test_deterministic_extraction_links_course_to_semester_and_program() -> None:
    chunk = {
        "chunk_id": "c1",
        "text": "| 3 | IF21W0803 | Keamanan Informasi | 2 |  |  | 2 |",
    }

    entities, relationships = extract_deterministic(chunk)

    courses = [entity for entity in entities if entity["entity_type"] == "Course"]
    semesters = [entity for entity in entities if entity["entity_type"] == "Semester"]
    assert courses[0]["entity_id"] == "course:IF21W0803"
    assert courses[0]["name"] == "Keamanan Informasi"
    assert courses[0]["normalized_name"] == normalize_name("Keamanan Informasi")
    assert semesters[0]["name"] == "Semester 8"
    assert {relationship["type"] for relationship in relationships} == {"OFFERED_IN", "PART_OF"}


def test_semester_from_code_handles_elective_codes() -> None:
    assert _semester_from_code("IF21W0503") == 5
    assert _semester_from_code("IF21P1502") == 5
    assert _semester_from_code("IF21P3702") == 7
    assert _semester_from_code("not-a-code") is None


def test_merge_extraction_deduplicates_entities_and_edges() -> None:
    first = [
        {
            "entity_id": "regulation:cuti",
            "name": "Cuti",
            "entity_type": "Regulation",
            "normalized_name": "cuti",
            "aliases": ["cuti akademik"],
            "confidence": 0.8,
        }
    ]
    second = [
        {
            "entity_id": "regulation:cuti",
            "name": "Cuti",
            "entity_type": "Regulation",
            "normalized_name": "cuti",
            "aliases": ["izin cuti"],
            "confidence": 0.9,
        }
    ]
    edge = {
        "source": "regulation:cuti",
        "type": "HAS_DEADLINE",
        "target": "deadline:30-september",
        "evidence": "30 September",
        "confidence": 1.0,
    }

    entities, relationships = merge_extractions([first, second], [[edge], [edge]])

    assert len(entities) == 1
    assert entities[0]["aliases"] == ["cuti akademik", "izin cuti"]
    assert entities[0]["confidence"] == 0.9
    assert len(relationships) == 1
