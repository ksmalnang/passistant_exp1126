# RAG optimization experiment log

All scores below come from `uv run python -m src.eval evaluate` with the repository's
RAGAS 0.4.3 metrics: `faithfulness`, `context_recall`, and
`factual_correctness(mode="f1")`. Aggregate scores are the mean over every sample whose
metric produced a score, as implemented in `src/eval/reporting.py`.

## Fixed evaluation conditions

- Judge: `openai/gpt-6-luna` via OpenRouter, with `RAGAS_LLM_PROVIDER=openai`.
  Reasoning is disabled explicitly (`{"reasoning": {"enabled": false}}`).
- The earlier `deepseek/deepseek-v4-flash-0731` judge was rejected after a controlled
  check: on d1-14 it marked all four response claims unsupported even though the reference
  states the same facts (and marked the reference claims supported against the response).
  Results scored with that judge are retained below but are **not directly comparable**
  with gpt-6-luna results.
- Judge embeddings: `qwen/qwen3-embedding-8b:nitro`.
- Dataset: `tests/fixtures/eval/public_beta_v1.jsonl` (65 samples, see below) and
  `tests/fixtures/eval/public_beta_v1_1.jsonl` (same questions and pipeline outputs, five
  source-verified reference corrections).
- Index: `passistant_eval_v1`, built from the four PDFs in `.agent/assets/kb` with the
  repository's GLM-OCR ingestion, `chunk_size=1000`, `chunk_overlap=200`
  (`data/processed/index_manifests/passistant_eval_v1.json`).
- Retrieval: `RETRIEVAL_STRATEGY=reranker`, `RETRIEVAL_TOP_K=10`,
  `RETRIEVAL_MAX_QUERY_VARIANTS=2`.

## Dataset provenance

The repository's original fixtures under `tests/fixtures/` use the legacy
`id/question/contexts/answer/ground_truth` schema. They are preserved unchanged.
`scripts/convert_legacy_testset.py` converts them to the current schema without editing any
question or answer:

| file | samples | origin |
|---|---:|---|
| `d1_legacy30_v1.jsonl` | 30 | `ragas_dataset_11051234.jsonl` |
| `d2_legacy35_v1.jsonl` | 35 | `ragas_dataset_15.jsonl` |
| `public_beta_v1.jsonl` | 65 | union of both, ids namespaced |

`scripts/apply_reference_corrections.py` produced `public_beta_v1_1.jsonl` with five
references corrected against the raw OCR text (d1-03, d2-03, d2-08, d2-05, d2-15).

## Instrumentation fixes made before the baseline

1. `src/eval/openrouter.py`: when `RAGAS_LLM_REASONING_ENABLED` is false, the judge request
   now sends `{"reasoning": {"enabled": false}}` instead of omitting the field. Omitting it
   let the provider reason by default: the 3-sample smoke run spent 140,514 completion
   tokens; after the fix the same run spent 9,605. Scores are not comparable across the fix.
2. `src/utils/vector_store/search.py`: the retrieval cache key now includes the collection
   name, so two indexes cannot serve each other's cached results.
3. `src/clients/reranker.py`: 429 responses are retried, requests are serialized, and a
   minimum interval is applied before falling back to another reranker.
4. `src/eval/reporting.py`: reports now record the collection, point count, index manifest,
   retrieved contexts, and per-chunk scores/pages.

## Runs

| date | run | dataset | system | faithfulness | context recall | factual f1 | notes |
|---|---|---|---:|---:|---:|---:|---|
| 2026-10-27 | smoke3_baseline | 3 | reranker/Voyage, judge reasoning omitted | 0.743 | 0.667 | 0.387 | invalid judge config, 27 requests / 140k completion tokens |
| 2026-10-27 | smoke3_fixedjudge | 3 | reranker/Voyage, judge reasoning off | 0.958 | 0.667 | 0.350 | 27 requests / 9.6k completion tokens |
| 2026-10-27 | baseline_A_v1_1 (deepseek judge) | 65 | reranker/Voyage, original prompt | 0.8339 | 0.9051 | 0.5134 | pipeline outputs; defective judge, kept for comparison only |
| 2026-10-27 | baseline_A_luna | 65 | same responses, gpt-6-luna judge | 0.7198 | 0.7564 | 0.4836 | authoritative Baseline A (64 factual scored) |
| 2026-10-27 | ten_v4_editor_luna | 10 | v4 prompt + grounding editor, gpt-6-luna | 0.9500 | 0.8500 | 0.6570 | first 10 samples only; generation-only comparison |
| 2026-10-27 | candidate_C1_luna_v1_2 | 65 | v4 + grounding editor, graph off | blocked | blocked | blocked | full responses captured; OpenRouter key limit hit during scoring |
| 2026-10-27 | graph_offline_v1 | 65 | graph expansion ablation, lexical proxy | n/a | 0.7903 -> 0.7833 | n/a | graph adds 2.42 chunks but displaces seeds; not promoted |
