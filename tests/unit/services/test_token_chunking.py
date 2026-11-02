"""Token budgets and exact source offsets for indexed chunks."""

import os
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import tiktoken
from pydantic import ValidationError

from scripts import build_kb_index
from src.config import get_settings
from src.config.settings import Settings
from src.services.chunking import split_token_spans
from src.services.indexing import DocumentIndexingService
from src.utils.state import DocumentUpload


def test_recursive_token_chunks_keep_paragraphs_and_original_offsets() -> None:
    text = "Paragraf satu.\n\nParagraf dua.\n\nParagraf tiga."
    spans = split_token_spans(text, chunk_size=8, chunk_overlap=0, separators=["\n\n", " ", ""])

    assert [text[start:end] for start, end in spans] == [
        "Paragraf satu.",
        "Paragraf dua.",
        "Paragraf tiga.",
    ]
    assert [start for start, _ in spans] == [0, 16, 31]


def test_token_chunks_cover_repeated_unicode_text_without_exceeding_budget() -> None:
    text = ("Syarat mahasiswa 🎓 wajib membayar administrasi. " * 100) + "<|endoftext|> akhir"
    encoder = tiktoken.get_encoding("cl100k_base")
    spans = split_token_spans(text, chunk_size=64, chunk_overlap=8, separators=[" ", ""])

    assert len(spans) > 1
    assert spans[0][0] == 0
    assert spans[-1][1] == len(text)
    for index, (start, end) in enumerate(spans):
        assert 0 < len(encoder.encode_ordinary(text[start:end])) <= 64
        if index:
            previous_start, previous_end = spans[index - 1]
            assert previous_start < start <= previous_end < end
            assert len(encoder.encode_ordinary(text[start:previous_end])) <= 8
    assert all(
        any(start <= position < end for start, end in spans)
        for position, char in enumerate(text)
        if not char.isspace()
    )


def test_chunk_settings_reject_overlap_at_or_above_size() -> None:
    with pytest.raises(ValidationError, match="CHUNK_OVERLAP_TOKENS"):
        Settings(_env_file=None, CHUNK_SIZE_TOKENS=128, CHUNK_OVERLAP_TOKENS=128)


def test_build_index_cli_sets_explicit_token_units(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["build_kb_index", "--collection", "test-tokens"])
    for name in (
        "CHUNK_SIZE_TOKENS",
        "CHUNK_OVERLAP_TOKENS",
        "CHUNK_ENCODING",
        "QDRANT_COLLECTION_NAME",
    ):
        monkeypatch.setenv(name, "before")
    captured = {}

    async def build(*args):
        captured.update(
            {
                name: os.environ[name]
                for name in ("CHUNK_SIZE_TOKENS", "CHUNK_OVERLAP_TOKENS", "CHUNK_ENCODING")
            }
        )

    monkeypatch.setattr(build_kb_index, "build", build)
    build_kb_index.main()

    assert captured == {
        "CHUNK_SIZE_TOKENS": "1024",
        "CHUNK_OVERLAP_TOKENS": "128",
        "CHUNK_ENCODING": "cl100k_base",
    }


async def test_indexing_preserves_spans_pages_and_token_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "CHUNK_SIZE_TOKENS", 1024)
    monkeypatch.setattr(settings, "CHUNK_OVERLAP_TOKENS", 128)
    monkeypatch.setattr(settings, "GRAPH_SYNC_ENABLED", False)
    monkeypatch.setattr("src.services.indexing.get_cache", Mock)
    embeddings = SimpleNamespace(
        aembed_documents=AsyncMock(side_effect=lambda texts: [[0.1, 0.2] for _ in texts])
    )
    store = Mock()
    store._get_embeddings.return_value = embeddings
    store._supports_bm25_vectors.return_value = False
    store.collection_name = "tokens"
    first = "Mahasiswa wajib memenuhi persyaratan akademik. " * 150
    second = "Ketentuan pembayaran berlaku untuk seluruh mahasiswa. " * 150
    text = first + "\n\n" + second
    document = DocumentUpload(
        document_id="token-doc",
        filename="rules.pdf",
        file_path="rules.pdf",
        file_size=1,
        mime_type="application/pdf",
        extracted_text=text,
        ocr_page_status=[{"text_length": len(first)}, {"text_length": len(second)}],
    )

    result = await DocumentIndexingService(vector_store=store).index_document(document)

    points = [
        point for call in store.client.upsert.call_args_list for point in call.kwargs["points"]
    ]
    assert result.chunk_count == len(points) > 1
    assert max(len(point.payload["text"]) for point in points) > 2000
    encoder = tiktoken.get_encoding("cl100k_base")
    for index, point in enumerate(points):
        payload = point.payload
        assert payload["chunk_id"] == f"token-doc:chunk:{index}"
        assert payload["chunk_index"] == index
        assert payload["text"] == text[payload["start_char"] : payload["end_char"]]
        assert payload["token_count"] == len(encoder.encode_ordinary(payload["text"])) <= 1024
        assert payload["chunk_encoding"] == "cl100k_base"
        pages = []
        if payload["start_char"] < len(first):
            pages.append({"page": 1})
        if payload["end_char"] > len(first) + 2:
            pages.append({"page": 2})
        assert payload["source_locations"] == pages
