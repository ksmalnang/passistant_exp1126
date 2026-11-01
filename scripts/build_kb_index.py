"""Build a reproducible Qdrant index from the raw knowledge-base PDFs.

The raw PDFs in .agent/assets/kb are never modified. OCR output is cached per
document under the configured OCR cache directory, so re-indexing with different
chunking settings does not call the OCR provider again.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

RAW_DIR = Path(".agent/assets/kb")
MANIFEST_DIR = Path("data/processed/index_manifests")

DOCUMENTS = {
    "Buku Pedoman Kemahasiswaan.pdf": ("Buku Pedoman Kemahasiswaan", "policy"),
    "Juknis UTS untuk Mahasiswa.pdf": ("Petunjuk Teknis UTS Genap 2024/2025", "policy"),
    "Kurikulum Teknik Informatika Unpas 2021 (versi Publik - Tanpa RPS).pdf": (
        "Dokumen Kurikulum Teknik Informatika 2021",
        "curriculum",
    ),
    "Wadek1-Buku Panduan Akademik FT Unpas 2023-2024 Publish.pdf": (
        "Buku Panduan Akademik FT Unpas 2023-2024",
        "policy",
    ),
}


def document_id(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return f"kb-{digest[:12]}"


async def load_pages(cache, doc_id: str, path: Path):
    from src.config import get_settings
    from src.services.ocr_cache import CachedPage, OCRCacheMeta
    from src.utils.state import DocumentType
    from src.utils.tools.ocr import GLMOCRTool

    entry = cache.load(doc_id)
    if entry is not None:
        return entry.pages

    title, doc_type = DOCUMENTS[path.name]
    result = await GLMOCRTool().extract_text(str(path), DocumentType(doc_type))
    pages = [
        CachedPage(
            page_index=index,
            page=index + 1,
            status=str(status.get("status", "unknown")),
            layout_details=list((result.layout_details or [[]])[index])
            if index < len(result.layout_details or [])
            else [],
            extracted_text=(result.page_texts or [""])[index]
            if index < len(result.page_texts or [])
            else "",
            error=status.get("error"),
        )
        for index, status in enumerate(result.page_results or [])
    ]
    settings = get_settings()
    cache.save(
        OCRCacheMeta(
            document_id=doc_id,
            filename=path.name,
            doc_title=title,
            document_type=doc_type,
            file_type="application/pdf",
            num_pages=result.num_pages,
            raw_file=path.name,
            source_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            ocr_provider="zhipu",
            ocr_model=settings.GLM_LAYOUT_MODEL,
            ocr_timestamp=datetime.now(UTC).isoformat(),
            ocr_page_status=list(result.page_results or []),
        ),
        pages,
    )
    return pages


def build_document(path: Path, pages):
    from src.utils.state import DocumentType, DocumentUpload, ProcessingStatus

    title, doc_type = DOCUMENTS[path.name]
    page_status = [
        {
            "page_index": page.page_index,
            "status": page.status,
            "text_length": len(page.extracted_text),
        }
        for page in pages
    ]
    text = "\n\n".join(page.extracted_text for page in pages if page.extracted_text).strip()
    return DocumentUpload(
        document_id=document_id(path),
        filename=path.name,
        file_path=str(path),
        document_type=DocumentType(doc_type),
        mime_type="application/pdf",
        file_size=path.stat().st_size,
        document_title=title,
        extracted_text=text,
        num_pages=len(pages),
        ocr_page_status=page_status,
        processing_status=ProcessingStatus.COMPLETED,
    )


async def build(collection: str, reset: bool, doc_names: list[str]) -> dict:
    from src.config import get_settings
    from src.services.indexing import DocumentIndexingService
    from src.services.ocr_cache import OCRCache
    from src.utils.vector_store import VectorStoreTools

    paths = [RAW_DIR / name for name in (doc_names or DOCUMENTS)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise SystemExit(f"missing raw documents: {missing}")

    vector_store = VectorStoreTools()
    if reset and vector_store.client.collection_exists(collection):
        vector_store.client.delete_collection(collection)
    settings = get_settings()
    if collection != settings.QDRANT_COLLECTION_NAME:
        raise SystemExit(
            f"collection mismatch: QDRANT_COLLECTION_NAME={settings.QDRANT_COLLECTION_NAME}"
        )

    cache = OCRCache()
    indexer = DocumentIndexingService(vector_store=vector_store)
    results = []
    for path in paths:
        started = time.perf_counter()
        pages = await load_pages(cache, document_id(path), path)
        document = build_document(path, pages)
        indexed = await indexer.index_document(document, replace_existing=True)
        results.append(
            {
                "filename": path.name,
                "document_id": document.document_id,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "pages": len(pages),
                "ocr_chars": len(document.extracted_text or ""),
                "chunks": indexed.chunk_count,
                "indexed_chars": indexed.indexed_chars,
                "seconds": round(time.perf_counter() - started, 1),
            }
        )
        print(json.dumps(results[-1], ensure_ascii=False))

    manifest = {
        "collection": collection,
        "created_at": datetime.now(UTC).isoformat(),
        "settings": {
            "chunk_size": settings.CHUNK_SIZE,
            "chunk_overlap": settings.CHUNK_OVERLAP,
            "chunk_separators": list(settings.CHUNK_SEPARATORS),
            "embedding_model": settings.EMBEDDING_MODEL,
            "vector_size": settings.VECTOR_SIZE,
            "ocr_model": settings.GLM_LAYOUT_MODEL,
        },
        "total_chunks": sum(item["chunks"] for item in results),
        "documents": results,
    }
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = MANIFEST_DIR / f"{collection}.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"manifest: {manifest_path}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument("--document", action="append", default=[])
    args = parser.parse_args()

    os.environ["QDRANT_COLLECTION_NAME"] = args.collection
    os.environ["CHUNK_SIZE"] = str(args.chunk_size)
    os.environ["CHUNK_OVERLAP"] = str(args.chunk_overlap)
    if args.reset:
        os.environ["RESET_COLLECTION"] = "1"

    asyncio.run(build(args.collection, args.reset, args.document))


if __name__ == "__main__":
    main()
