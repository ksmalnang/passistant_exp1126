"""Passage recall uses source evidence that stays fixed across chunking strategies."""

import pytest

from src.eval.retrieval_recall import find_reference_spans, passage_recall


@pytest.mark.parametrize(
    "quote, source",
    [
        ("semester 14", "semester 140"),
        ("IPK >= 3", "IPK >= 30"),
        ("IPK >= 3", "IPK >= 3.5"),
        ("wajib", "diwajibkan"),
    ],
)
def test_reference_matches_require_complete_words_and_numbers(quote: str, source: str) -> None:
    assert find_reference_spans(quote, {"doc": source}) == []


def test_recall_uses_union_of_source_spans_across_chunk_boundaries() -> None:
    sources = {"doc": "Mahasiswa wajib membayar administrasi sebelum perwalian."}
    references = [find_reference_spans("wajib membayar administrasi", sources)]
    chunks = [
        {"document_id": "doc", "start_char": 0, "end_char": 24},
        {"document_id": "doc", "start_char": 24, "end_char": len(sources["doc"])},
    ]

    assert passage_recall(references, chunks, sources) == 1.0
    assert passage_recall(references, chunks[:1], sources) == 0.0
    assert (
        passage_recall(references, [{**chunk, "document_id": "wrong"} for chunk in chunks], sources)
        == 0.0
    )


def test_reference_matching_normalizes_whitespace_and_unicode_without_fuzzy_matches() -> None:
    sources = {"doc": "Syarat:\n\nBayar\u00a0administrasi ５０% sebelum ujian."}
    spans = find_reference_spans("Bayar administrasi 50%", sources)

    assert len(spans) == 1
    span = spans[0]
    assert sources["doc"][span.start : span.end] == "Bayar\u00a0administrasi ５０%"
    assert find_reference_spans("Lunas seluruh administrasi", sources) == []
    assert passage_recall([[]], [], sources) is None


def test_reference_matching_ignores_markdown_but_preserves_comparison_operators() -> None:
    sources = {"doc": "**Nilai** minimal: >= ５０%."}
    assert find_reference_spans("nilai minimal >= 50%", sources)
    assert find_reference_spans("nilai minimal <= 50%", sources) == []


def test_recall_counts_duplicate_occurrences_as_one_reference_and_ignores_whitespace_gaps() -> None:
    sources = {"doc": "Aturan sama.\n\nAturan sama."}
    references = [find_reference_spans("Aturan sama.", sources)]
    chunks = [{"document_id": "doc", "start_char": 14, "end_char": 26}]
    assert passage_recall(references, chunks, sources) == 1.0
    assert passage_recall([], chunks, sources) is None

    reference = [find_reference_spans("sama. Aturan", sources)]
    split = [
        {"document_id": "doc", "start_char": 7, "end_char": 12},
        {"document_id": "doc", "start_char": 14, "end_char": 20},
    ]
    assert passage_recall(reference, split, sources) == 1.0
