# RAGAS evaluation — public_beta_v1_1.jsonl

- Run: `baseline_A_v1_1`
- Mode: live | Retrieval: reranker (top_k=10)
- Judge: deepseek/deepseek-v4-flash-0731 | Embeddings: qwen/qwen3-embedding-8b:nitro
- Samples: 65 | Duration: 1347.5s

## Aggregate scores

| metric | mean | scored |
| --- | --- | --- |
| context_recall | 0.9051 | 65/65 |
| factual_correctness(mode=f1) | 0.5134 | 65/65 |
| faithfulness | 0.8339 | 65/65 |

## Breakdown by question_type

| value | samples | context_recall | factual_correctness(mode=f1) | faithfulness |
| --- | --- | --- | --- | --- |
| conditional | 10 | 0.8583 | 0.5310 | 0.6854 |
| multi_doc | 31 | 0.9113 | 0.5103 | 0.8628 |
| simple | 24 | 0.9167 | 0.5100 | 0.8583 |

## Breakdown by topic

| value | samples | context_recall | factual_correctness(mode=f1) | faithfulness |
| --- | --- | --- | --- | --- |
| beasiswa | 2 | 1.0000 | 0.7400 | 0.8637 |
| cuti | 3 | 1.0000 | 0.4133 | 0.8194 |
| dpp | 3 | 1.0000 | 0.5233 | 0.6310 |
| kurikulum | 4 | 0.7500 | 0.4750 | 0.7585 |
| lainnya | 4 | 1.0000 | 0.6325 | 0.8732 |
| mbkm | 3 | 1.0000 | 0.5733 | 0.6147 |
| organisasi | 1 | 1.0000 | 0.5700 | 1.0000 |
| pelanggaran | 2 | 1.0000 | 0.5150 | 0.8125 |
| sks | 12 | 0.8889 | 0.5067 | 0.9345 |
| tugas_akhir | 19 | 0.8553 | 0.5189 | 0.8261 |
| ujian | 11 | 0.9015 | 0.4073 | 0.8530 |
| yudisium | 1 | 1.0000 | 0.9100 | 0.8333 |

## Breakdown by answerable

| value | samples | context_recall | factual_correctness(mode=f1) | faithfulness |
| --- | --- | --- | --- | --- |
| True | 65 | 0.9051 | 0.5134 | 0.8339 |

## Breakdown by dataset_version

| value | samples | context_recall | factual_correctness(mode=f1) | faithfulness |
| --- | --- | --- | --- | --- |
| public_beta_v1 | 65 | 0.9051 | 0.5134 | 0.8339 |

## Per-sample scores

| id | topic | type | contexts | context_recall | factual_correctness(mode=f1) | faithfulness |
| --- | --- | --- | --- | --- | --- | --- |
| d1-01 | cuti | multi_doc | 10 | 1.0000 | 0.4000 | 1.0000 |
| d1-02 | ujian | multi_doc | 10 | 1.0000 | 0.3300 | 1.0000 |
| d1-03 | sks | simple | 10 | 1.0000 | 0.5700 | 0.5000 |
| d1-04 | ujian | multi_doc | 10 | 1.0000 | 0.5700 | 0.7143 |
| d1-05 | ujian | conditional | 10 | 1.0000 | 0.4000 | 0.9231 |
| d1-06 | mbkm | multi_doc | 10 | 1.0000 | 0.5800 | 1.0000 |
| d1-07 | kurikulum | multi_doc | 10 | 1.0000 | 0.5700 | 0.7500 |
| d1-08 | sks | multi_doc | 10 | 1.0000 | 0.5000 | 1.0000 |
| d1-09 | tugas_akhir | simple | 10 | 1.0000 | 1.0000 | 1.0000 |
| d1-10 | beasiswa | simple | 10 | 1.0000 | 0.7700 | 1.0000 |
| d1-11 | lainnya | simple | 10 | 1.0000 | 0.5700 | 0.6000 |
| d1-12 | organisasi | multi_doc | 10 | 1.0000 | 0.5700 | 1.0000 |
| d1-13 | tugas_akhir | simple | 10 | 1.0000 | 0.5000 | 0.8000 |
| d1-14 | sks | simple | 10 | 1.0000 | 0.0000 | 1.0000 |
| d1-15 | mbkm | multi_doc | 10 | 1.0000 | 0.3600 | 0.2727 |
| d1-16 | tugas_akhir | multi_doc | 10 | 1.0000 | 0.6700 | 0.8000 |
| d1-17 | tugas_akhir | multi_doc | 10 | 1.0000 | 0.1500 | 0.8182 |
| d1-18 | tugas_akhir | conditional | 10 | 1.0000 | 0.7500 | 0.5714 |
| d1-19 | cuti | multi_doc | 10 | 1.0000 | 0.2500 | 0.8333 |
| d1-20 | ujian | multi_doc | 10 | 1.0000 | 0.6700 | 0.8000 |
| d1-21 | dpp | conditional | 10 | 1.0000 | 0.5000 | 0.4286 |
| d1-22 | dpp | multi_doc | 10 | 1.0000 | 0.5000 | 0.7143 |
| d1-23 | tugas_akhir | simple | 10 | 1.0000 | 0.4000 | 0.8000 |
| d1-24 | kurikulum | multi_doc | 10 | 1.0000 | 0.4000 | 0.9091 |
| d1-25 | lainnya | simple | 10 | 1.0000 | 0.8800 | 1.0000 |
| d1-26 | ujian | simple | 10 | 1.0000 | 0.0000 | 1.0000 |
| d1-27 | pelanggaran | multi_doc | 10 | 1.0000 | 0.3600 | 0.6250 |
| d1-28 | ujian | multi_doc | 10 | 1.0000 | 0.4000 | 1.0000 |
| d1-29 | tugas_akhir | multi_doc | 10 | 1.0000 | 0.4000 | 0.7500 |
| d1-30 | sks | multi_doc | 10 | 1.0000 | 0.7300 | 1.0000 |
| d2-01 | ujian | simple | 10 | 1.0000 | 0.4600 | 0.6667 |
| d2-02 | cuti | multi_doc | 10 | 1.0000 | 0.5900 | 0.6250 |
| d2-03 | sks | simple | 10 | 1.0000 | 0.7500 | 1.0000 |
| d2-04 | tugas_akhir | multi_doc | 10 | 1.0000 | 0.7700 | 1.0000 |
| d2-05 | mbkm | conditional | 10 | 1.0000 | 0.7800 | 0.5714 |
| d2-06 | tugas_akhir | simple | 10 | 0.5000 | 0.5500 | 0.4000 |
| d2-07 | ujian | conditional | 10 | 1.0000 | 0.6000 | 0.6429 |
| d2-08 | yudisium | multi_doc | 10 | 1.0000 | 0.9100 | 0.8333 |
| d2-09 | ujian | multi_doc | 10 | 1.0000 | 0.3800 | 1.0000 |
| d2-10 | sks | simple | 10 | 1.0000 | 0.7100 | 0.7143 |
| d2-11 | sks | simple | 10 | 0.5000 | 0.4600 | 1.0000 |
| d2-12 | kurikulum | conditional | 10 | 0.6667 | 0.2200 | 0.3750 |
| d2-13 | dpp | multi_doc | 10 | 1.0000 | 0.5700 | 0.7500 |
| d2-14 | tugas_akhir | simple | 10 | 1.0000 | 0.6200 | 1.0000 |
| d2-15 | sks | multi_doc | 10 | 1.0000 | 0.7000 | 1.0000 |
| d2-16 | sks | simple | 10 | 1.0000 | 0.3100 | 1.0000 |
| d2-17 | sks | simple | 10 | 1.0000 | 0.5000 | 1.0000 |
| d2-18 | tugas_akhir | simple | 10 | 1.0000 | 0.5000 | 1.0000 |
| d2-19 | kurikulum | multi_doc | 10 | 0.3333 | 0.7100 | 1.0000 |
| d2-20 | sks | multi_doc | 10 | 0.5000 | 0.1800 | 1.0000 |
| d2-21 | sks | multi_doc | 10 | 0.6667 | 0.6700 | 1.0000 |
| d2-22 | ujian | multi_doc | 10 | 0.2500 | 0.6700 | 1.0000 |
| d2-23 | lainnya | simple | 10 | 1.0000 | 0.4100 | 0.8929 |
| d2-24 | tugas_akhir | simple | 10 | 0.0000 | 0.1200 | 0.8095 |
| d2-25 | tugas_akhir | conditional | 10 | 0.2500 | 0.1300 | 0.9474 |
| d2-26 | tugas_akhir | simple | 10 | 1.0000 | 0.2900 | 0.5000 |
| d2-27 | tugas_akhir | multi_doc | 10 | 0.8333 | 0.5900 | 0.9167 |
| d2-28 | pelanggaran | multi_doc | 10 | 1.0000 | 0.6700 | 1.0000 |
| d2-29 | beasiswa | conditional | 10 | 1.0000 | 0.7100 | 0.7273 |
| d2-30 | tugas_akhir | simple | 10 | 1.0000 | 0.4000 | 0.9167 |
| d2-31 | tugas_akhir | conditional | 10 | 1.0000 | 0.5900 | 1.0000 |
| d2-32 | ujian | multi_doc | 10 | 0.6667 | 0.0000 | 0.6364 |
| d2-33 | tugas_akhir | conditional | 10 | 0.6667 | 0.6300 | 0.6667 |
| d2-34 | lainnya | simple | 10 | 1.0000 | 0.6700 | 1.0000 |
| d2-35 | tugas_akhir | simple | 10 | 1.0000 | 0.8000 | 1.0000 |
