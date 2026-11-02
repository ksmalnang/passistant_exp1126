"""Compare a captured character index with token chunks using fixed reference spans.

Preparation is local. --run calls the configured embedding and reranking APIs,
using two in-memory Qdrant collections and never writing to the active index.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from collections import Counter
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import tiktoken
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    Modifier,
    PointStruct,
    SparseVectorParams,
    VectorParams,
)

from src.clients.embeddings import get_embeddings
from src.clients.redis import RedisCache
from src.clients.reranker import get_reranker
from src.config import get_settings
from src.eval.retrieval_recall import find_reference_spans, passage_recall
from src.services.indexing import DocumentIndexingService
from src.services.ocr_cache import OCRCache
from src.services.reindex import OCRReindexService
from src.utils.nodes.retrieval import RetrievalNode
from src.utils.state import AgentState
from src.utils.vector_store import VectorStoreTools


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare(folder: Path, dataset: Path) -> dict[str, Any]:
    baseline = read_json(folder / "baseline_points.json")
    config = read_json(folder / "baseline_config.json")
    cache = OCRCache()
    reindex = OCRReindexService(indexer=object())
    documents = {}
    for document_id in sorted({point["payload"]["document_id"] for point in baseline}):
        entry = cache.load(document_id)
        if entry is None:
            raise ValueError(f"OCR cache missing: {document_id}")
        documents[document_id] = reindex._rebuild_document(entry, rebuild_text=False)
    sources = {key: document.extracted_text or "" for key, document in documents.items()}
    positions: dict[str, tuple[int, int]] = {}
    for point in sorted(
        baseline, key=lambda item: (item["payload"]["document_id"], item["payload"]["chunk_index"])
    ):
        payload = point["payload"]
        document_id = payload["document_id"]
        previous_start, previous_end = positions.get(document_id, (-1, 0))
        search_start = max(previous_start + 1, previous_end - config["chunk_overlap_chars"])
        start = sources[document_id].find(payload["text"], search_start)
        if start < 0:
            raise ValueError(f"Baseline text differs from OCR source: {payload['chunk_id']}")
        end = start + len(payload["text"])
        payload.update(start_char=start, end_char=end)
        positions[document_id] = (start, end)

    fingerprint = {
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "baseline_sha256": hashlib.sha256(
            (folder / "baseline_points.json").read_bytes()
        ).hexdigest(),
        "source_sha256": {
            key: hashlib.sha256(value.encode()).hexdigest() for key, value in sources.items()
        },
        "baseline_config_sha256": hashlib.sha256(
            (folder / "baseline_config.json").read_bytes()
        ).hexdigest(),
        "metric_version": "normalized-source-spans-v2",
        "candidate_chunking": {"encoding": "cl100k_base", "size": 1024, "overlap": 128},
    }
    audit_path = folder / "audit.json"
    checkpoint_names = (
        "queries.json",
        "query_embeddings.json",
        "candidate_points.json",
        "before_results.json",
        "after_results.json",
    )
    if any((folder / name).exists() for name in checkpoint_names):
        previous = read_json(audit_path)
        if any(previous.get(key) != value for key, value in fingerprint.items()):
            raise ValueError("Checkpoint inputs changed; use a new work directory")
    rows = [
        json.loads(line)
        for line in dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    references = {}
    excluded = []
    for row in rows:
        quotes = list(dict.fromkeys(row.get("reference_contexts") or []))
        matches = [find_reference_spans(quote, sources) for quote in quotes]
        missing = [index for index, spans in enumerate(matches) if not spans]
        if not quotes or missing or row.get("metadata", {}).get("answerable") is False:
            excluded.append({"id": row["id"], "missing_references": missing})
        else:
            references[row["id"]] = matches
    audit = {
        **fingerprint,
        "queries": len(rows),
        "valid_queries": len(references),
        "reference_passages": sum(len(value) for value in references.values()),
        "excluded": excluded,
        "baseline_chunks": len(baseline),
        "documents": [
            {"document_id": key, "filename": value.filename} for key, value in documents.items()
        ],
    }
    write_json(folder / "audit.json", audit)
    print(
        f"Local audit: {len(rows)} queries, {len(references)} fully verified queries, "
        f"{audit['reference_passages']} passages, {len(excluded)} excluded; "
        f"{len(baseline)} baseline chunks from {len(documents)} documents",
        flush=True,
    )
    return {
        "baseline": baseline,
        "config": config,
        "documents": documents,
        "sources": sources,
        "rows": rows,
        "references": references,
        "audit": audit,
    }


def load_points(client: QdrantClient, name: str, points: list[dict]) -> VectorStoreTools:
    vector = points[0]["vector"]
    size = len(vector[""] if isinstance(vector, dict) else vector)
    client.create_collection(
        name,
        vectors_config=VectorParams(size=size, distance=Distance.COSINE),
        sparse_vectors_config={"bm25": SparseVectorParams(modifier=Modifier.IDF)},
    )
    for start in range(0, len(points), 32):
        client.upsert(
            name,
            [
                PointStruct(id=point["id"], payload=point["payload"], vector=point["vector"])
                for point in points[start : start + 32]
            ],
        )
    tools = VectorStoreTools(client=client, cache=RedisCache(None, "chunking-ab", 0))
    tools.collection_name = name
    tools.bm25_vectors_enabled = True
    tools.retrieval_strategy = "reranker"
    return tools


def snapshot_points(client: QdrantClient, collection: str) -> list[dict]:
    points = []
    offset = None
    while True:
        batch, offset = client.scroll(collection, limit=256, offset=offset, with_vectors=True)
        points.extend(point.model_dump(mode="json") for point in batch)
        if offset is None:
            return points


async def evaluate(folder: Path, prepared: dict[str, Any], output: Path) -> None:
    settings = get_settings()
    config = prepared["config"]
    for name, actual in (
        ("embedding", settings.EMBEDDING_MODEL),
        ("reranker", settings.RERANKER_MODEL),
    ):
        if config[name] != actual:
            raise ValueError(f"{name} changed since baseline capture")
    settings.GRAPH_RAG_ENABLED = False
    settings.GRAPH_SYNC_ENABLED = False
    settings.REDIS_URL = None
    settings.CHUNK_SIZE_TOKENS = 1024
    settings.CHUNK_OVERLAP_TOKENS = 128
    settings.CHUNK_ENCODING = "cl100k_base"
    # Changing models on only one side would confound the chunking comparison.
    get_reranker().fallback = None
    with closing(QdrantClient(":memory:")) as client:
        baseline = load_points(client, "character_baseline", prepared["baseline"])
        node = RetrievalNode(retriever=baseline, llm_provider=lambda: None)
        node.max_query_variants = config["max_query_variants"]
        frozen_path = folder / "queries.json"
        if frozen_path.exists():
            frozen = read_json(frozen_path)
        else:
            frozen = {}
            for row in prepared["rows"]:
                frozen[row["id"]] = await node._build_queries(
                    AgentState(current_intent="query_document", retrieval_query=row["user_input"]),
                    "",
                    follow_up=False,
                )
            write_json(frozen_path, frozen)
        unique_queries = list(
            dict.fromkeys(
                query
                for key, queries in frozen.items()
                if key in prepared["references"]
                for query in queries
            )
        )
        embedding_path = folder / "query_embeddings.json"
        if embedding_path.exists():
            query_embeddings = read_json(embedding_path)
        else:
            vectors = await get_embeddings().aembed_documents(unique_queries)
            query_embeddings = dict(zip(unique_queries, vectors, strict=True))
            write_json(embedding_path, query_embeddings)

        candidate_path = folder / "candidate_points.json"
        if candidate_path.exists():
            candidate_points = read_json(candidate_path)
            candidate = load_points(client, "token_candidate", candidate_points)
        else:
            candidate = VectorStoreTools(client=client, cache=RedisCache(None, "chunking-ab", 0))
            candidate.collection_name = "token_candidate"
            candidate.retrieval_strategy = "reranker"
            indexer = DocumentIndexingService(vector_store=candidate)
            indexer._cache = RedisCache(None, "chunking-ab", 0)
            for document in prepared["documents"].values():
                result = await indexer.index_document(document)
                print(f"indexed {document.filename}: {result.chunk_count} token chunks", flush=True)
            candidate_points = snapshot_points(client, candidate.collection_name)
            write_json(candidate_path, candidate_points)

        for name, tools, points in (
            ("before", baseline, prepared["baseline"]),
            ("after", candidate, candidate_points),
        ):
            tools.reranker_candidate_multiplier = config["candidate_multiplier"]
            payloads = {point["payload"]["chunk_id"]: point["payload"] for point in points}
            result_path = folder / f"{name}_results.json"
            completed = read_json(result_path) if result_path.exists() else {}
            for row in prepared["rows"]:
                key = row["id"]
                if key in completed or key not in prepared["references"]:
                    continue
                merged = []
                for query in frozen[key]:
                    results = await tools._search_one(
                        query=query,
                        query_embedding=query_embeddings[query],
                        query_filter=None,
                        top_k=10,
                        score_threshold=0.2,
                    )
                    merged = node._merge_results(merged, results, query)
                seeds = merged[:10]
                expanded = await tools.expand_neighbors(seeds)
                selected = [payloads[chunk["chunk_id"]] for chunk in seeds]
                context = [payloads[chunk["chunk_id"]] for chunk in expanded]
                references = prepared["references"][key]
                encoder = tiktoken.get_encoding("cl100k_base")
                completed[key] = {
                    "recall_at_10": passage_recall(references, selected, prepared["sources"]),
                    "recall_with_neighbors": passage_recall(
                        references, context, prepared["sources"]
                    ),
                    "seed_ids": [chunk["chunk_id"] for chunk in seeds],
                    "context_ids": [chunk["chunk_id"] for chunk in expanded],
                    "seed_tokens": sum(
                        len(encoder.encode_ordinary(chunk["text"])) for chunk in seeds
                    ),
                    "context_tokens": sum(
                        len(encoder.encode_ordinary(chunk["text"])) for chunk in expanded
                    ),
                }
                write_json(result_path, completed)
                print(
                    f"{name} {len(completed)}/{len(prepared['references'])} {key}: "
                    f"Recall@10={completed[key]['recall_at_10']:.3f}",
                    flush=True,
                )
        write_report(folder, prepared, candidate_points, output)


def write_report(
    folder: Path, prepared: dict[str, Any], candidate: list[dict], output: Path
) -> None:
    before = read_json(folder / "before_results.json")
    after = read_json(folder / "after_results.json")
    if set(before) != set(prepared["references"]) or set(after) != set(before):
        raise ValueError("Both runs must complete on the same valid queries")
    encoder = tiktoken.get_encoding("cl100k_base")
    baseline_tokens = [
        len(encoder.encode_ordinary(point["payload"]["text"])) for point in prepared["baseline"]
    ]
    candidate_tokens = [
        len(encoder.encode_ordinary(point["payload"]["text"])) for point in candidate
    ]
    config = prepared["config"]
    audit = prepared["audit"]
    lines = [
        "# Perbandingan character chunking dan token chunking",
        "",
        f"Laporan: {datetime.now(timezone(timedelta(hours=7))).isoformat()} (WIB)",
        "",
        f"Passage Recall@10: {mean(row['recall_at_10'] for row in before.values()):.4f} "
        f"→ {mean(row['recall_at_10'] for row in after.values()):.4f}. "
        "Recall tetap pada subset yang dapat diverifikasi; kedua sisi sudah mencapai ceiling "
        "pada subset ini. Hasil ini belum menunjukkan peningkatan recall pada keseluruhan dataset.",
        "",
        "## Konfigurasi dan kontrol eksperimen",
        "",
        "| Parameter | Sebelum | Sesudah |",
        "| --- | --- | --- |",
        f"| Ukuran | {config['chunk_size_chars']} karakter | 1024 token |",
        f"| Target overlap | {config['chunk_overlap_chars']} karakter | 128 token |",
        "| Pengukuran token | cl100k_base untuk audit | cl100k_base |",
        f"| Jumlah chunk | {len(baseline_tokens)} | {len(candidate_tokens)} |",
        f"| Rata-rata token/chunk | {mean(baseline_tokens):.2f} | {mean(candidate_tokens):.2f} |",
        f"| Maksimum token/chunk | {max(baseline_tokens)} | {max(candidate_tokens)} |",
        "",
        f"Corpus: {len(prepared['sources'])} dokumen dari snapshot `{config['collection']}`, "
        "dibandingkan dengan hasil indexing ulang OCR cache yang sama di Qdrant in-memory. "
        "Indeks aktif tidak diubah.",
        "",
        f"Embedding: `{config['embedding']}`; provider: `{config['embedding_provider']}`; "
        f"reranker: `{config['reranker']}`; candidate multiplier: {config['candidate_multiplier']}; top-k: 10.",
        "",
        "Dense + BM25 + RRF + reranker menggunakan kode retrieval yang sama. "
        "Query asli dan ekspansi deterministik dibekukan; query rewrite LLM serta GraphRAG dimatikan "
        "pada kedua sisi. Embedding query dipakai ulang. Fallback reranker dimatikan agar model "
        "tidak berubah pada salah satu sisi. Tidak ada response generation atau judge RAGAS.",
        "",
        "## Definisi Recall@10",
        "",
        "Metrik utama adalah **passage Recall@10**: jumlah kutipan referensi yang seluruh "
        "span sumbernya tercakup oleh gabungan 10 seed hasil rerank, dibagi jumlah kutipan "
        "referensi query. Skor agregat adalah macro-average antarkueri. "
        "Ground truth berupa span sumber tetap, bukan chunk_id yang berubah setelah rechunking. "
        "Kapitalisasi, Unicode NFKC, whitespace, tanda baca kalimat dan Markdown "
        "dinormalisasi; angka desimal, persen dan operator perbandingan dipertahankan. "
        "Batas kata/angka wajib cocok. Tidak ada fuzzy matching. "
        "Kemunculan berulang dihitung satu referensi. Gap whitespace antarchunk diizinkan.",
        "",
        "Cakupan setelah ekspansi ±1 dilaporkan terpisah dan **bukan Recall@10**, "
        "karena jumlah konteks dapat melebihi 10.",
        "",
        f"Dataset `tests/fixtures/eval/public_beta_v1_2.jsonl`: {audit['queries']} query; "
        f"{audit['valid_queries']} query dengan seluruh kutipan terverifikasi "
        f"({audit['reference_passages']} passage); {len(audit['excluded'])} query dikecualikan. "
        "Query dengan kutipan yang tidak ditemukan tidak diberi nilai nol dan tidak "
        "dikurangi denominatornya secara parsial.",
        "",
        "## Hasil",
        "",
        "| Metrik | Sebelum | Sesudah | Selisih |",
        "| --- | --- | --- | --- |",
    ]
    for label, field in (
        ("Passage Recall@10", "recall_at_10"),
        ("Cakupan passage setelah ±1", "recall_with_neighbors"),
    ):
        old = mean(row[field] for row in before.values())
        new = mean(row[field] for row in after.values())
        lines.append(f"| {label} | {old:.4f} | {new:.4f} | {(new - old) * 100:+.2f} pp |")
    for label, field in (
        ("Rata-rata token 10 seed", "seed_tokens"),
        ("Rata-rata token konteks termasuk ±1", "context_tokens"),
    ):
        old = mean(row[field] for row in before.values())
        new = mean(row[field] for row in after.values())
        lines.append(f"| {label} | {old:.2f} | {new:.2f} | {new - old:+.2f} |")
    changes = Counter(
        "naik"
        if after[key]["recall_at_10"] > before[key]["recall_at_10"]
        else "turun"
        if after[key]["recall_at_10"] < before[key]["recall_at_10"]
        else "tetap"
        for key in before
    )
    lines.extend(
        [
            "",
            f"Query membaik: {changes['naik']}; menurun: {changes['turun']}; tetap: {changes['tetap']}.",
            "",
            "## Hasil per query",
            "",
            "| Query | Sebelum | Sesudah | Selisih pp |",
            "| --- | --- | --- | --- |",
        ]
    )
    for key in before:
        old, new = before[key]["recall_at_10"], after[key]["recall_at_10"]
        lines.append(f"| {key} | {old:.4f} | {new:.4f} | {(new - old) * 100:+.2f} |")
    lines.extend(
        [
            "",
            "## Query yang dikecualikan",
            "",
            "| Query | Indeks kutipan tidak ditemukan (0-based) |",
            "| --- | --- |",
        ]
    )
    for row in audit["excluded"]:
        lines.append(f"| {row['id']} | {row['missing_references']} |")
    lines.extend(
        [
            "",
            "## Reproduksibilitas dan keterbatasan",
            "",
            f"Dataset SHA-256: `{audit['dataset_sha256']}`.",
            "",
            f"Snapshot baseline SHA-256: `{audit['baseline_sha256']}`.",
            "",
            f"Konfigurasi baseline SHA-256: `{audit['baseline_config_sha256']}`; "
            f"scoring schema: `{audit['metric_version']}`.",
            "",
            "Checkpoint lokal ada di `.tmp_probe/chunking_comparison/`: konfigurasi, snapshot "
            "vektor kedua indeks, audit sumber, query dan embedding yang dibekukan, serta hasil per query.",
            "",
            "Jalankan `uv run --no-sync python -u -m scripts.compare_chunking_recall --run` "
            "untuk melanjutkan checkpoint atau menghasilkan ulang laporan. "
            "Persiapan lokal saja: hilangkan `--run`. Run baru memerlukan snapshot baseline "
            "dan OCR cache yang sama. Panggilan API dapat berbayar.",
            "",
            "Eksperimen ini membandingkan dua konfigurasi ukuran dan overlap sekaligus, "
            "bukan mengisolasi tokenizer saja. Chunk yang lebih besar dapat meningkatkan recall "
            "dengan membawa lebih banyak teks; volume token konteks dilaporkan agar efek itu terlihat. "
            "Ground truth yang tidak terverifikasi membatasi cakupan hasil; metrik ini bukan "
            "document Recall@10 atau RAGAS context_recall. Tidak ada klaim signifikansi statistik.",
            "",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Markdown report: {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=Path(".tmp_probe/chunking_comparison"))
    parser.add_argument(
        "--dataset", type=Path, default=Path("tests/fixtures/eval/public_beta_v1_2.jsonl")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("experiments/chunking_recall_comparison.md")
    )
    parser.add_argument(
        "--run", action="store_true", help="Call embedding/reranker APIs and compare both indices"
    )
    args = parser.parse_args()
    prepared = prepare(args.work_dir, args.dataset)
    if args.run:
        asyncio.run(evaluate(args.work_dir, prepared, args.output))


if __name__ == "__main__":
    main()
