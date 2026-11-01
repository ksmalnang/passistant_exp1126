# RAGAS evaluation — ten.jsonl

- Run: `ten_v4_editor_luna`
- Mode: fixture | Retrieval: reranker (top_k=10)
- Judge: openai/gpt-6-luna | Embeddings: qwen/qwen3-embedding-8b:nitro
- Samples: 10 | Duration: 97.8s

## Aggregate scores

| metric | mean | scored |
| --- | --- | --- |
| context_recall | 0.8500 | 10/10 |
| factual_correctness(mode=f1) | 0.6570 | 10/10 |
| faithfulness | 0.9500 | 10/10 |

## Breakdown by question_type

| value | samples | context_recall | factual_correctness(mode=f1) | faithfulness |
| --- | --- | --- | --- | --- |
| conditional | 1 | 1.0000 | 0.0000 | 1.0000 |
| multi_doc | 6 | 0.7500 | 0.7333 | 0.9167 |
| simple | 3 | 1.0000 | 0.7233 | 1.0000 |

## Breakdown by topic

| value | samples | context_recall | factual_correctness(mode=f1) | faithfulness |
| --- | --- | --- | --- | --- |
| beasiswa | 1 | 1.0000 | 0.6700 | 1.0000 |
| cuti | 1 | 0.5000 | 0.8600 | 1.0000 |
| kurikulum | 1 | 1.0000 | 0.8000 | 1.0000 |
| mbkm | 1 | 1.0000 | 0.7700 | 1.0000 |
| sks | 2 | 0.5000 | 0.6500 | 1.0000 |
| tugas_akhir | 1 | 1.0000 | 1.0000 | 1.0000 |
| ujian | 3 | 1.0000 | 0.3900 | 0.8333 |

## Breakdown by answerable

| value | samples | context_recall | factual_correctness(mode=f1) | faithfulness |
| --- | --- | --- | --- | --- |
| True | 10 | 0.8500 | 0.6570 | 0.9500 |

## Breakdown by dataset_version

| value | samples | context_recall | factual_correctness(mode=f1) | faithfulness |
| --- | --- | --- | --- | --- |
| public_beta_v1 | 10 | 0.8500 | 0.6570 | 0.9500 |

## Per-sample scores

| id | topic | type | contexts | context_recall | factual_correctness(mode=f1) | faithfulness |
| --- | --- | --- | --- | --- | --- | --- |
| d1-01 | cuti | multi_doc | 10 | 0.5000 | 0.8600 | 1.0000 |
| d1-02 | ujian | multi_doc | 10 | 1.0000 | 0.5000 | 0.5000 |
| d1-03 | sks | simple | 10 | 1.0000 | 0.5000 | 1.0000 |
| d1-04 | ujian | multi_doc | 10 | 1.0000 | 0.6700 | 1.0000 |
| d1-05 | ujian | conditional | 10 | 1.0000 | 0.0000 | 1.0000 |
| d1-06 | mbkm | multi_doc | 10 | 1.0000 | 0.7700 | 1.0000 |
| d1-07 | kurikulum | multi_doc | 10 | 1.0000 | 0.8000 | 1.0000 |
| d1-08 | sks | multi_doc | 10 | 0.0000 | 0.8000 | 1.0000 |
| d1-09 | tugas_akhir | simple | 10 | 1.0000 | 1.0000 | 1.0000 |
| d1-10 | beasiswa | simple | 10 | 1.0000 | 0.6700 | 1.0000 |
