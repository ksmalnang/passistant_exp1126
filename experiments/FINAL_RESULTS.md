# Final measured results

Date: 2026-10-27. All RAGAS scores below are dataset means over samples that produced a
score. The aggregation is implemented in `src/eval/reporting.py`; a metric that failed on a
sample is excluded from the mean and reported in `scored`.

## Acceptance gates

| Metric | Gate | Baseline A (gpt-6-luna) | Status |
|---|---:|---:|---|
| Faithfulness | >= 0.90 | **0.7198** (65/65) | FAIL, gap 0.1802 |
| Context Recall | >= 0.85 | **0.7564** (65/65) | FAIL, gap 0.0936 |
| Factual Correctness (f1) | >= 0.85 | **0.4836** (64/65) | FAIL, gap 0.3664 |

Baseline A is the existing Qdrant reranker pipeline, original response prompt, reasoning
disabled, graph disabled. Dataset: `public_beta_v1_1.jsonl` (65 samples).

Report: `experiments/results/reports/baseline_A_luna.json`.

The earlier judge `deepseek/deepseek-v4-flash-0731` produced
faithfulness 0.8339 / recall 0.9051 / factual 0.5134 on the same responses
(`baseline_A_v1_1.json`). It was rejected because a controlled check on d1-14 marked all
four response claims false against a reference that states the same facts, while marking
the reference claims true against the response. Those numbers are retained for audit but
are **not comparable** with the gpt-6-luna numbers.

## Best generated candidate (partial evaluation)

`candidate_C1_v4_editor`: concise v4 response prompt + grounding editor, graph disabled.
Full responses were captured to
`experiments/results/fixtures/candidate_C1_v4_editor.json` (65 samples).

| metric | first 10 samples baseline | first 10 samples C1 | delta |
|---|---:|---:|---:|
| faithfulness | 0.7087 | **0.9500** | +0.2413 |
| context_recall | 0.8500 | **0.8500** | 0.0000 |
| factual f1 | 0.5933 (9 scored) | **0.6570** (10 scored) | +0.0637 |

The full C1 score could not be produced: the OpenRouter key reached its hard $3.00
spend limit during scoring ("Key limit exceeded (total limit)... does not reset"). The
first 10 samples are a **partial evaluation**, not an acceptance result.

## GraphRAG candidate

A working Qdrant + Neo4j graph was built from the same 920 chunks:

- 920 Chunk nodes, exact `chunk_id` match with Qdrant (0 missing, 0 orphaned).
- 670 entities, 2748+ typed relationships; entity types include Requirement (188),
  Course (172), Organization (121), Role (54), Regulation (43), Program (32),
  Evaluation (22), Semester (19), Deadline (16), Scholarship (3), ElectiveGroup (8).
- 150 rule-bearing chunks received cached LLM extraction; 48,076 prompt /
  39,414 completion tokens.
- Graph manifest: `experiments/results/manifests/passistant_eval_v1.json`.

Offline expansion diagnostic (no judge calls), using the baseline seed chunks and the
reference text as a lexical coverage proxy:

| max extra chunks | routing | samples with graph chunks | mean baseline coverage | mean graph coverage |
|---:|---|---:|---:|---:|
| 0 | always | 0 | 0.7903 | 0.7903 |
| 1 | always | 53 | 0.7903 | 0.7813 |
| 2 | always | 53 | 0.7903 | 0.7813 |
| 3 | always | 53 | 0.7903 | 0.7833 |
| 4 | always | 53 | 0.7903 | 0.7752 |
| 3 | multi_hop | 35 | 0.7903 | 0.7870 |

Graph expansion adds source chunks but, with the top-k budget fixed, it displaces stronger
direct seeds. No configuration improved the lexical coverage proxy. The graph candidate is
therefore **not promoted to the default**; the graph remains behind
`GRAPH_RAG_ENABLED=false` and the existing Qdrant path is preserved.

A full RAGAS Candidate B run could not be produced because the key limit was reached
during the capture; its partial capture contains 403 generation failures and is not valid.

## Verification

- `pytest -q`: **287 passed, 9 skipped** (includes a Neo4j idempotency/delete integration
  test when `NEO4J_PASSWORD` is set).
- `scripts/check_graph_consistency.py --collection passistant_eval_v1`: 920 Qdrant chunks,
  920 graph chunks, 0 missing, 0 orphaned, 0 mentions without provenance.
- Graph integration test: upsert twice -> one chunk; prune -> removed; delete document ->
  removed.
- Raw PDFs in `.agent/assets/kb` were not modified. OCR cache and index/graph manifests are
  under `data/processed/`.
- Cache-key and evaluator provenance fixes are covered by unit tests.

## Known limitations

- The full 65-sample C1 and GraphRAG RAGAS scores are blocked by the API key limit.
- The gpt-6-luna judge is itself an LLM; 10-sample comparisons showed judge variance
  (recall moved 1.0 -> 0.85 on identical contexts between judge calls).
- Several references are verbose relative to the question (for example d2-16 asks only
  semester/SKS but the reference also lists course topics), which mechanically lowers
  factual precision/recall.
- `public_beta_v1_2.jsonl` adds two source-verified corrections (d1-26 evaluation
  boundary, d2-32 cheating classification) on top of v1_1; it is not yet scored.
