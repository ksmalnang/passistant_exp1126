# Perbandingan character chunking dan token chunking

Laporan: 2026-11-03T05:19:26.579469+07:00 (WIB)

Passage Recall@10: 1.0000 → 1.0000. Recall tetap pada subset yang dapat diverifikasi; kedua sisi sudah mencapai ceiling pada subset ini. Hasil ini belum menunjukkan peningkatan recall pada keseluruhan dataset.

## Konfigurasi dan kontrol eksperimen

| Parameter | Sebelum | Sesudah |
| --- | --- | --- |
| Ukuran | 2000 karakter | 1024 token |
| Target overlap | 300 karakter | 128 token |
| Pengukuran token | cl100k_base untuk audit | cl100k_base |
| Jumlah chunk | 456 | 268 |
| Rata-rata token/chunk | 468.57 | 782.52 |
| Maksimum token/chunk | 952 | 1024 |

Corpus: 5 dokumen dari snapshot `passistant_eval_v1`, dibandingkan dengan hasil indexing ulang OCR cache yang sama di Qdrant in-memory. Indeks aktif tidak diubah.

Embedding: `qwen/qwen3-embedding-8b:nitro`; provider: `deepinfra`; reranker: `voyageai/rerank-3-lite`; candidate multiplier: 6; top-k: 10.

Dense + BM25 + RRF + reranker menggunakan kode retrieval yang sama. Query asli dan ekspansi deterministik dibekukan; query rewrite LLM serta GraphRAG dimatikan pada kedua sisi. Embedding query dipakai ulang. Fallback reranker dimatikan agar model tidak berubah pada salah satu sisi. Tidak ada response generation atau judge RAGAS.

## Definisi Recall@10

Metrik utama adalah **passage Recall@10**: jumlah kutipan referensi yang seluruh span sumbernya tercakup oleh gabungan 10 seed hasil rerank, dibagi jumlah kutipan referensi query. Skor agregat adalah macro-average antarkueri. Ground truth berupa span sumber tetap, bukan chunk_id yang berubah setelah rechunking. Kapitalisasi, Unicode NFKC, whitespace, tanda baca kalimat dan Markdown dinormalisasi; angka desimal, persen dan operator perbandingan dipertahankan. Batas kata/angka wajib cocok. Tidak ada fuzzy matching. Kemunculan berulang dihitung satu referensi. Gap whitespace antarchunk diizinkan.

Cakupan setelah ekspansi ±1 dilaporkan terpisah dan **bukan Recall@10**, karena jumlah konteks dapat melebihi 10.

Dataset `tests/fixtures/eval/public_beta_v1_2.jsonl`: 65 query; 11 query dengan seluruh kutipan terverifikasi (21 passage); 54 query dikecualikan. Query dengan kutipan yang tidak ditemukan tidak diberi nilai nol dan tidak dikurangi denominatornya secara parsial.

## Hasil

| Metrik | Sebelum | Sesudah | Selisih |
| --- | --- | --- | --- |
| Passage Recall@10 | 1.0000 | 1.0000 | +0.00 pp |
| Cakupan passage setelah ±1 | 1.0000 | 1.0000 | +0.00 pp |
| Rata-rata token 10 seed | 4962.45 | 8181.36 | +3218.91 |
| Rata-rata token konteks termasuk ±1 | 11063.82 | 18132.64 | +7068.82 |

Query membaik: 0; menurun: 0; tetap: 11.

## Hasil per query

| Query | Sebelum | Sesudah | Selisih pp |
| --- | --- | --- | --- |
| d1-01 | 1.0000 | 1.0000 | +0.00 |
| d1-04 | 1.0000 | 1.0000 | +0.00 |
| d1-06 | 1.0000 | 1.0000 | +0.00 |
| d1-10 | 1.0000 | 1.0000 | +0.00 |
| d1-11 | 1.0000 | 1.0000 | +0.00 |
| d1-12 | 1.0000 | 1.0000 | +0.00 |
| d1-13 | 1.0000 | 1.0000 | +0.00 |
| d1-23 | 1.0000 | 1.0000 | +0.00 |
| d1-29 | 1.0000 | 1.0000 | +0.00 |
| d2-14 | 1.0000 | 1.0000 | +0.00 |
| d2-30 | 1.0000 | 1.0000 | +0.00 |

## Query yang dikecualikan

| Query | Indeks kutipan tidak ditemukan (0-based) |
| --- | --- |
| d1-02 | [1] |
| d1-03 | [0] |
| d1-05 | [1] |
| d1-07 | [1] |
| d1-08 | [0, 1] |
| d1-09 | [0] |
| d1-14 | [1] |
| d1-15 | [0, 1] |
| d1-16 | [0, 1] |
| d1-17 | [0] |
| d1-18 | [0] |
| d1-19 | [1] |
| d1-20 | [0] |
| d1-21 | [0] |
| d1-22 | [0, 1] |
| d1-24 | [1] |
| d1-25 | [1] |
| d1-26 | [0] |
| d1-27 | [0] |
| d1-28 | [0, 1] |
| d1-30 | [1] |
| d2-01 | [0] |
| d2-02 | [1, 2] |
| d2-03 | [0] |
| d2-04 | [0, 1] |
| d2-05 | [0, 1] |
| d2-06 | [1] |
| d2-07 | [0, 1] |
| d2-08 | [0, 1] |
| d2-09 | [0, 1, 2] |
| d2-10 | [0, 1] |
| d2-11 | [0] |
| d2-12 | [0, 1] |
| d2-13 | [0, 1] |
| d2-15 | [0, 1, 2] |
| d2-16 | [0, 1] |
| d2-17 | [0, 1] |
| d2-18 | [0, 1] |
| d2-19 | [0, 1] |
| d2-20 | [0, 1] |
| d2-21 | [0, 1] |
| d2-22 | [0, 1, 2] |
| d2-23 | [0, 1] |
| d2-24 | [0, 1] |
| d2-25 | [0, 1, 2] |
| d2-26 | [0] |
| d2-27 | [0, 1] |
| d2-28 | [0, 1] |
| d2-29 | [0] |
| d2-31 | [0] |
| d2-32 | [1, 2] |
| d2-33 | [0] |
| d2-34 | [0] |
| d2-35 | [0, 1] |

## Reproduksibilitas dan keterbatasan

Dataset SHA-256: `baaefb44a3231f37986c4034ed5581e0fd9e3068bc4bb3806086f76cfca14d9c`.

Snapshot baseline SHA-256: `120b41ff03e3c2ef6259572c1d4a1a74d9c509e1b93d0e1f8a6f46665401a684`.

Konfigurasi baseline SHA-256: `107c105aa271bd7e3a56749c60a6339c3e5716e6557479bd998c9596c8717218`; scoring schema: `normalized-source-spans-v2`.

Checkpoint lokal ada di `.tmp_probe/chunking_comparison/`: konfigurasi, snapshot vektor kedua indeks, audit sumber, query dan embedding yang dibekukan, serta hasil per query.

Jalankan `uv run --no-sync python -u -m scripts.compare_chunking_recall --run` untuk melanjutkan checkpoint atau menghasilkan ulang laporan. Persiapan lokal saja: hilangkan `--run`. Run baru memerlukan snapshot baseline dan OCR cache yang sama. Panggilan API dapat berbayar.

Eksperimen ini membandingkan dua konfigurasi ukuran dan overlap sekaligus, bukan mengisolasi tokenizer saja. Chunk yang lebih besar dapat meningkatkan recall dengan membawa lebih banyak teks; volume token konteks dilaporkan agar efek itu terlihat. Ground truth yang tidak terverifikasi membatasi cakupan hasil; metrik ini bukan document Recall@10 atau RAGAS context_recall. Tidak ada klaim signifikansi statistik.
