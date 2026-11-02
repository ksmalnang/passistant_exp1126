# Panduan agen — PASsistant

Instruksi ini berlaku untuk seluruh repository. Baca `README.md` dan panduan terkait di `docs/` atau `.agent/PROJECT_GUIDE.md` bila tugas membutuhkan konteks lebih dalam. Utamakan perubahan kecil yang sesuai pola kode yang sudah ada.

## Konteks proyek

PASsistant adalah chatbot layanan akademik berbasis Python 3.11+, LangGraph, FastAPI, dan RAG. Dokumen di-OCR lalu diindeks ke Qdrant; Redis dipakai untuk cache. Neo4j/GraphRAG adalah jalur opsional, bukan pengganti retrieval Qdrant. Antarmuka meliputi CLI, REST, SSE/WebSocket, dan Telegram. Dependency dikelola dengan `uv` melalui `pyproject.toml` dan `uv.lock`.

## Cari lokasi perubahan yang tepat

- `src/api/`: route, kontrak request/response, sesi, dan streaming; pertahankan perilaku antarmuka yang sudah ada.
- `src/graphs/workflow.py`, `src/utils/nodes/`, `src/utils/state.py`: routing, node, prompt, dan state LangGraph.
- `src/services/`: logika aplikasi yang dipakai lintas antarmuka; jangan taruh logika bisnis hanya di route atau adapter Telegram.
- `src/utils/vector_store/`, `src/services/indexing.py`, `src/utils/nodes/retrieval.py`: indexing, retrieval, reranking, dan sitasi. Jaga metadata sumber/halaman serta identitas chunk saat mengubah pipeline.
- `src/graph/`: GraphRAG/Neo4j; `src/graphs/` adalah workflow LangGraph utama. Qdrant adalah sumber teks chunk dan `chunk_id` harus cocok dengan node Neo4j. Lihat `docs/graphrag.md` sebelum mengubah sinkronisasi atau ekspansi graph.
- `src/clients/`, `src/config/settings.py`: integrasi eksternal dan pengaturan berbasis environment; tambahkan pengaturan baru di settings, bukan pembacaan environment yang tersebar.
- `src/guardrails/`, `src/telegram_bot/`, `src/eval/`: keamanan input/output, adapter Telegram, dan evaluasi RAGAS. Lihat `docs/evaluation.md` untuk format dataset dan mode evaluasi.
- `tests/`: tes unit, API, integration, smoke, system, regression, dan fixture. Letakkan tes dekat area perilaku yang berubah.

## Aturan kerja

- Jangan commit `.env`, cache, laporan evaluasi, atau artefak lokal.
- Gunakan konfigurasi dan dependency yang sudah tersedia; jika dependency berubah, perbarui `pyproject.toml` dan `uv.lock` bersama-sama. Ruff menargetkan Python 3.11 dengan panjang baris 100; ikuti style modul yang diubah.
- Perbarui dokumentasi yang relevan bila kontrak API, konfigurasi, deployment, retrieval, atau evaluasi berubah. Rujukan: `docs/api.md`, `docs/deployment.md`, `docs/retrieval-pipeline.md`, `docs/graphrag.md`.

## Validasi

- Jalankan tes paling spesifik dulu: `uv run pytest tests/unit/<area> -q` atau `uv run pytest tests/api/<area> -q`, lalu perluas sesuai dampak perubahan.
- Cek style dan tipe bila mengubah Python: `uv run ruff check .`, `uv run ruff format --check .`, `uv run basedpyright src/`. `src/eval/` dikecualikan dari cakupan basedpyright bawaan.
- `uv run pytest tests/unit tests/api -q` mencakup jalur offline utama. Tes di `tests/smoke/` menggunakan server HTTP lokal; tes `tests/regression/` menjalankan ulang subset pytest melalui subprocess.
- `tests/system/` opt-in melalui `PASSISTANT_SYSTEM_TESTS=1` dan dapat menggunakan Qdrant serta provider OCR/LLM sungguhan. Jalankan hanya dengan izin dan konfigurasi lingkungan yang tepat. Evaluasi RAGAS fixture pun dapat memanggil judge berbayar; jangan anggap evaluasi fixture sebagai tes offline gratis.
- Jika dependency, layanan, atau kredensial belum tersedia, laporkan batas validasi; jangan mengklaim tes atau evaluasi berhasil tanpa menjalankannya.
