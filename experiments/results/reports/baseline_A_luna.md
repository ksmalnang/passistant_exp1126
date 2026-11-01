# RAGAS evaluation — public_beta_v1_1.jsonl

- Run: `baseline_A_luna`
- Mode: fixture | Retrieval: reranker (top_k=10)
- Judge: openai/gpt-6-luna | Embeddings: qwen/qwen3-embedding-8b:nitro
- Samples: 65 | Duration: 624.0s

## Aggregate scores

| metric | mean | scored |
| --- | --- | --- |
| context_recall | 0.7564 | 65/65 |
| factual_correctness(mode=f1) | 0.4836 | 64/65 |
| faithfulness | 0.7198 | 65/65 |

## Breakdown by question_type

| value | samples | context_recall | factual_correctness(mode=f1) | faithfulness |
| --- | --- | --- | --- | --- |
| conditional | 10 | 0.7833 | 0.3770 | 0.5480 |
| multi_doc | 31 | 0.7339 | 0.5027 | 0.7606 |
| simple | 24 | 0.7743 | 0.5042 | 0.7386 |

## Breakdown by topic

| value | samples | context_recall | factual_correctness(mode=f1) | faithfulness |
| --- | --- | --- | --- | --- |
| beasiswa | 2 | 1.0000 | 0.6500 | 0.6014 |
| cuti | 3 | 0.5833 | 0.4600 | 0.7852 |
| dpp | 3 | 0.7222 | 0.3567 | 0.5503 |
| kurikulum | 4 | 0.7500 | 0.4600 | 0.6823 |
| lainnya | 4 | 0.8333 | 0.5425 | 0.8053 |
| mbkm | 3 | 0.6667 | 0.5500 | 0.7492 |
| organisasi | 1 | 1.0000 | 0.5600 | 0.9231 |
| pelanggaran | 2 | 1.0000 | 0.5700 | 0.7445 |
| sks | 12 | 0.6458 | 0.5367 | 0.7936 |
| tugas_akhir | 19 | 0.8114 | 0.4484 | 0.6583 |
| ujian | 11 | 0.7273 | 0.3970 | 0.7575 |
| yudisium | 1 | 0.7500 | 0.9100 | 0.6000 |

## Breakdown by answerable

| value | samples | context_recall | factual_correctness(mode=f1) | faithfulness |
| --- | --- | --- | --- | --- |
| True | 65 | 0.7564 | 0.4836 | 0.7198 |

## Breakdown by dataset_version

| value | samples | context_recall | factual_correctness(mode=f1) | faithfulness |
| --- | --- | --- | --- | --- |
| public_beta_v1 | 65 | 0.7564 | 0.4836 | 0.7198 |

## Per-sample scores

| id | topic | type | contexts | context_recall | factual_correctness(mode=f1) | faithfulness |
| --- | --- | --- | --- | --- | --- | --- |
| d1-01 | cuti | multi_doc | 10 | 0.5000 | 0.4600 | 0.8889 |
| d1-02 | ujian | multi_doc | 10 | 1.0000 | - | 0.6667 |
| d1-03 | sks | simple | 10 | 1.0000 | 0.6700 | 0.1667 |
| d1-04 | ujian | multi_doc | 10 | 1.0000 | 0.5700 | 0.7143 |
| d1-05 | ujian | conditional | 10 | 1.0000 | 0.2900 | 0.8571 |
| d1-06 | mbkm | multi_doc | 10 | 1.0000 | 0.6100 | 1.0000 |
| d1-07 | kurikulum | multi_doc | 10 | 1.0000 | 0.5700 | 0.6667 |
| d1-08 | sks | multi_doc | 10 | 0.0000 | 0.5000 | 0.5714 |
| d1-09 | tugas_akhir | simple | 10 | 1.0000 | 1.0000 | 1.0000 |
| d1-10 | beasiswa | simple | 10 | 1.0000 | 0.6700 | 0.5556 |
| d1-11 | lainnya | simple | 10 | 1.0000 | 0.3300 | 0.6000 |
| d1-12 | organisasi | multi_doc | 10 | 1.0000 | 0.5600 | 0.9231 |
| d1-13 | tugas_akhir | simple | 10 | 1.0000 | 0.5000 | 0.8000 |
| d1-14 | sks | simple | 10 | 1.0000 | 0.0000 | 0.8571 |
| d1-15 | mbkm | multi_doc | 10 | 0.5000 | 0.4600 | 0.5333 |
| d1-16 | tugas_akhir | multi_doc | 10 | 0.5000 | 0.2900 | 0.3333 |
| d1-17 | tugas_akhir | multi_doc | 10 | 1.0000 | 0.0000 | 0.7000 |
| d1-18 | tugas_akhir | conditional | 10 | 1.0000 | 0.4400 | 0.4286 |
| d1-19 | cuti | multi_doc | 10 | 0.5000 | 0.2500 | 0.8000 |
| d1-20 | ujian | multi_doc | 10 | 1.0000 | 0.7500 | 1.0000 |
| d1-21 | dpp | conditional | 10 | 1.0000 | 0.4000 | 0.4286 |
| d1-22 | dpp | multi_doc | 10 | 0.5000 | 0.0000 | 0.5556 |
| d1-23 | tugas_akhir | simple | 10 | 0.5000 | 0.4000 | 0.5000 |
| d1-24 | kurikulum | multi_doc | 10 | 1.0000 | 0.4000 | 0.7500 |
| d1-25 | lainnya | simple | 10 | 1.0000 | 0.8000 | 1.0000 |
| d1-26 | ujian | simple | 10 | 0.0000 | 0.0000 | 1.0000 |
| d1-27 | pelanggaran | multi_doc | 10 | 1.0000 | 0.4300 | 0.6000 |
| d1-28 | ujian | multi_doc | 10 | 1.0000 | 0.4000 | 1.0000 |
| d1-29 | tugas_akhir | multi_doc | 10 | 1.0000 | 0.5500 | 0.8182 |
| d1-30 | sks | multi_doc | 10 | 1.0000 | 0.3600 | 1.0000 |
| d2-01 | ujian | simple | 10 | 1.0000 | 0.4000 | 0.3750 |
| d2-02 | cuti | multi_doc | 10 | 0.7500 | 0.6700 | 0.6667 |
| d2-03 | sks | simple | 10 | 0.7500 | 0.7500 | 1.0000 |
| d2-04 | tugas_akhir | multi_doc | 10 | 1.0000 | 0.5700 | 0.7500 |
| d2-05 | mbkm | conditional | 10 | 0.5000 | 0.5800 | 0.7143 |
| d2-06 | tugas_akhir | simple | 10 | 0.5000 | 0.4000 | 0.5714 |
| d2-07 | ujian | conditional | 10 | 0.7500 | 0.3300 | 0.3333 |
| d2-08 | yudisium | multi_doc | 10 | 0.7500 | 0.9100 | 0.6000 |
| d2-09 | ujian | multi_doc | 10 | 0.3333 | 0.5900 | 0.7500 |
| d2-10 | sks | simple | 10 | 1.0000 | 1.0000 | 0.5714 |
| d2-11 | sks | simple | 10 | 0.5000 | 0.5000 | 1.0000 |
| d2-12 | kurikulum | conditional | 10 | 0.6667 | 0.2000 | 0.5000 |
| d2-13 | dpp | multi_doc | 10 | 0.6667 | 0.6700 | 0.6667 |
| d2-14 | tugas_akhir | simple | 10 | 1.0000 | 0.4300 | 0.7143 |
| d2-15 | sks | multi_doc | 10 | 1.0000 | 0.6700 | 0.5333 |
| d2-16 | sks | simple | 10 | 1.0000 | 0.3100 | 1.0000 |
| d2-17 | sks | simple | 10 | 0.0000 | 0.5000 | 1.0000 |
| d2-18 | tugas_akhir | simple | 10 | 1.0000 | 0.5500 | 0.7778 |
| d2-19 | kurikulum | multi_doc | 10 | 0.3333 | 0.6700 | 0.8125 |
| d2-20 | sks | multi_doc | 10 | 0.5000 | 0.5900 | 0.8235 |
| d2-21 | sks | multi_doc | 10 | 0.0000 | 0.5900 | 1.0000 |
| d2-22 | ujian | multi_doc | 10 | 0.2500 | 0.6400 | 1.0000 |
| d2-23 | lainnya | simple | 10 | 1.0000 | 0.3700 | 0.9545 |
| d2-24 | tugas_akhir | simple | 10 | 0.0000 | 0.0800 | 0.8421 |
| d2-25 | tugas_akhir | conditional | 10 | 0.2500 | 0.0000 | 0.0000 |
| d2-26 | tugas_akhir | simple | 10 | 1.0000 | 0.5000 | 0.2727 |
| d2-27 | tugas_akhir | multi_doc | 10 | 1.0000 | 0.6400 | 0.9286 |
| d2-28 | pelanggaran | multi_doc | 10 | 1.0000 | 0.7100 | 0.8889 |
| d2-29 | beasiswa | conditional | 10 | 1.0000 | 0.6300 | 0.6471 |
| d2-30 | tugas_akhir | simple | 10 | 1.0000 | 0.4700 | 0.5000 |
| d2-31 | tugas_akhir | conditional | 10 | 1.0000 | 0.3100 | 1.0000 |
| d2-32 | ujian | multi_doc | 10 | 0.6667 | 0.0000 | 0.6364 |
| d2-33 | tugas_akhir | conditional | 10 | 0.6667 | 0.5900 | 0.5714 |
| d2-34 | lainnya | simple | 10 | 0.3333 | 0.6700 | 0.6667 |
| d2-35 | tugas_akhir | simple | 10 | 1.0000 | 0.8000 | 1.0000 |
