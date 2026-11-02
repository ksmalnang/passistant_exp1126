"""Deterministic passage recall against fixed source spans, without an LLM judge."""

import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from typing import Any


@dataclass(frozen=True)
class ReferenceSpan:
    document_id: str
    start: int
    end: int


@lru_cache(maxsize=32)
def _normalize_offsets(text: str) -> tuple[str, list[int]]:
    chars: list[str] = []
    offsets = []
    for offset, char in enumerate(text):
        for normalized in unicodedata.normalize("NFKC", char).casefold():
            decimal_separator = (
                normalized in ".,"
                and 0 < offset < len(text) - 1
                and text[offset - 1].isdecimal()
                and text[offset + 1].isdecimal()
            )
            if normalized.isspace() or (
                normalized in ',.;:!?*#`"()[]{}|•' and not decimal_separator
            ):
                if chars and chars[-1] != " ":
                    chars.append(" ")
                    offsets.append(offset)
            else:
                chars.append(normalized)
                offsets.append(offset)
    if chars and chars[-1] == " ":
        chars.pop()
        offsets.pop()
    return "".join(chars), offsets


def find_reference_spans(quote: str, sources: dict[str, str]) -> list[ReferenceSpan]:
    """Match exact words/symbols after case, NFKC, and formatting normalization.

    Sentence/Markdown punctuation is ignored; numbers, percent signs, and
    comparison operators are retained. No substitutions or fuzzy matches.
    """
    normalized, _ = _normalize_offsets(quote)
    if not normalized:
        return []
    spans = []
    for document_id, text in sources.items():
        source, offsets = _normalize_offsets(text)
        start = source.find(normalized)
        while start >= 0:
            end = start + len(normalized)
            left_boundary = start == 0 or not (
                source[start - 1].isalnum() or source[start - 1] in "_.,"
            )
            right_boundary = end == len(source) or not (
                source[end].isalnum() or source[end] in "_.,"
            )
            if left_boundary and right_boundary:
                spans.append(ReferenceSpan(document_id, offsets[start], offsets[end - 1] + 1))
            start = source.find(normalized, start + 1)
    return spans


def passage_recall(
    references: list[list[ReferenceSpan]],
    chunks: list[dict[str, Any]],
    sources: dict[str, str],
) -> float | None:
    """Fraction of reference passages fully covered by retrieved source spans.

    Alternative occurrences satisfy the same passage once. Whitespace gaps are
    allowed. Missing ground truth makes a query unscorable, rather than reducing
    its denominator or treating an unverifiable quote as a retrieval miss.
    """
    if not references or any(not alternatives for alternatives in references):
        return None
    intervals: dict[str, list[tuple[int, int]]] = {}
    for chunk in chunks:
        intervals.setdefault(chunk["document_id"], []).append(
            (chunk["start_char"], chunk["end_char"])
        )
    for spans in intervals.values():
        spans.sort()

    recovered = 0
    for alternatives in references:
        for reference in alternatives:
            cursor = reference.start
            text = sources[reference.document_id]
            for start, end in intervals.get(reference.document_id, []):
                if end <= cursor:
                    continue
                if start > cursor and text[cursor : min(start, reference.end)].strip():
                    break
                cursor = max(cursor, end)
                if cursor >= reference.end:
                    break
            if cursor >= reference.end:
                recovered += 1
                break
    return recovered / len(references)
