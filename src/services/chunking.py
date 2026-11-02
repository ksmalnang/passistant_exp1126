"""Recursive token chunks that retain exact character spans in the source text."""

import re
from collections import deque
from itertools import pairwise

import tiktoken


def split_token_spans(
    text: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
    separators: list[str],
    encoding_name: str = "cl100k_base",
) -> list[tuple[int, int]]:
    """Return nonempty source spans bounded by tokens, preferring ordered separators.

    Overlap retains whole separator-delimited pieces up to the token target.
    Carrying spans through recursion avoids ambiguous searches in repeated text.
    """
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be nonnegative and smaller than chunk_size")
    encoder = tiktoken.get_encoding(encoding_name)

    def count(start: int, end: int) -> int:
        return len(encoder.encode_ordinary(text[start:end].strip()))

    def merge(parts: list[tuple[int, int]]) -> list[tuple[int, int]]:
        merged = []
        window: deque[tuple[int, int]] = deque()
        for start, end in parts:
            if window and count(window[0][0], end) > chunk_size:
                merged.append((window[0][0], window[-1][1]))
                while window and (
                    count(window[0][0], window[-1][1]) > chunk_overlap
                    or count(window[0][0], end) > chunk_size
                ):
                    window.popleft()
            window.append((start, end))
        if window:
            merged.append((window[0][0], window[-1][1]))
        return merged

    def split(start: int, end: int, remaining: list[str]) -> list[tuple[int, int]]:
        separator = ""
        following: list[str] = []
        for index, candidate in enumerate(remaining):
            if not candidate or text.find(candidate, start, end) >= 0:
                separator = candidate
                following = remaining[index + 1 :]
                break
        if separator:
            starts = (
                [start]
                + [
                    match.start()
                    for match in re.compile(re.escape(separator)).finditer(text, start, end)
                    if match.start() > start
                ]
                + [end]
            )
        else:
            starts = list(range(start, end + 1))
        spans = []
        small = []
        for left, right in pairwise(starts):
            if count(left, right) <= chunk_size:
                small.append((left, right))
            else:
                spans.extend(merge(small))
                small = []
                if right - left == 1:
                    raise ValueError("chunk_size cannot contain a single Unicode character")
                spans.extend(split(left, right, following))
        return spans + merge(small)

    spans = []
    for start, end in split(0, len(text), separators):
        raw = text[start:end]
        left = start + len(raw) - len(raw.lstrip())
        right = start + len(raw.rstrip())
        if left < right:
            spans.append((left, right))
    return spans
