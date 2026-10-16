# Primary-source PDF and repository reconciliation

Checked 15 October 2026. This addendum resolves page-number gaps in three existing paper audits and narrows two repository discrepancies. It reports **source content**, not an independent reproduction of experiments. Page numbers below are one-based PDF pages; each of these PDFs also prints the same page number. The exact arXiv version in each link matters.

## Source access and integrity

The web PDF reader rejected the three PDFs on size (about 12–18 MB), so I downloaded the official versioned arXiv PDFs into local `work/paper_pdfs/` and inspected them with `pdfinfo` and per-page `pdftotext -layout`. The downloaded PDFs are research scratch files, not repository artifacts. Their SHA-256 hashes allow a later reader to identify the files checked:

| Source | PDF pages | SHA-256 |
|---|---:|---|
| [Med-R1, arXiv:2503.13939v5](https://arxiv.org/pdf/2503.13939v5) | 11 | `cf0d4d388604384a3886a150ffa144755e8558a8c953d156d42b8b42d785f463` |
| [SGMRI-VQA, arXiv:2604.15808v1](https://arxiv.org/pdf/2604.15808v1) | 16 | `7b40d63208d35621716c1570c0d7d304898a87356453f079063475e57cf897e3` |
| [NeuroQA, arXiv:2605.20525v1](https://arxiv.org/pdf/2605.20525v1) | 30 | `b4b352df3e62adfcf5f77948d24e9fd7eb00cb26fd9d151e383c56b1de8f519f` |

PDF text extraction verified the locations and transcribed values below. I did not execute paper code or inspect underlying medical images. A PDF page citation confirms what authors reported, not the validity of its labels, split, or clinical interpretation.

## Med-R1 v5: verified PDF pointers

| Existing audit claim | Exact PDF location and check |
|---|---|
| Cross-modality matrix; MRI-trained model gives 98.57% on MRI, 72.79% on CT, 74.33% row average | Table I, **PDF p. 5**. Row and column headers confirm the train/test direction. |
| Same-family modality comparison: Qwen2-VL-2B SFT 54.07%, Think 69.91%, No-Think 72.49%; Qwen2.5-VL-3B SFT 60.85%, Think 71.14%, Think-After 71.62%, No-Think 72.57% | Table IV, **PDF p. 6**. These are the table's `Overall` column across eight modalities. |
| Cross-task Qwen2.5-VL-3B No-Think 77.31%, Think 74.29%, Think-After 75.47% | Table VIII, **PDF p. 8**, `Overall` column. The cross-task matrices themselves are Tables V–VII, **PDF pp. 7–8**. |
| Dataset and training setup | §IV-A, **PDF p. 6**: 82,059 images, 88,996 VQA pairs, 31,877 MRI pairs; 328×328 input images; four sampled completions and one training epoch. `pdftotext` spaces some thousands separators (for example `82, 059`), so simple exact-string search misses them. |
| Uncertainty method | §IV-A, **PDF p. 7**: 10,000 bootstrap resamples of binary accuracy labels, with 95% intervals. This does not verify patient-clustered uncertainty. |

The paper's **328×328** input statement is now PDF verified. The author [repository](https://github.com/Yuxiang-Lai117/Med-R1) uses **384×384** in its README as noted in the existing audit; exact configuration/checkpoint parity remains **NOT_VERIFIED**. Neither PDF pagination nor extraction resolves this paper-versus-code setting difference.

## SGMRI-VQA v1: verified PDF pointers

| Existing audit claim | Exact PDF location and check |
|---|---|
| Source-image construction and annotation count | §3.1.1, **PDF p. 4**: 1,970 retained volumes (996 brain, 974 knee), 50,672 slices, 23,724 boxes. |
| Physician review scope and final QA totals | §3.3.1, **PDF p. 6**: all 222 brain and 155 knee **test** volumes reviewed; **over 300 training volumes** reviewed iteratively. The same page gives 41,307 QA pairs, 32,142 image-level and 9,165 volume-level, with 32,014 train and 9,293 test. It does **not** establish exhaustive manual review of the training set. |
| Dataset split table and training recipe | Table 2 and §4.1, **PDF p. 7**. The SFT description specifies Qwen3-VL-Instruct-8B and eight H100 GPUs; this is an SFT baseline, not an RL result. |
| Image-level results | Table 3, **PDF p. 9**: starting Qwen3-VL-8B average 41.66 and V-Score 3.11; fine-tuned model average 59.45 and V-Score 15.51. Best listed zero-shot V-Score is 5.95. |
| Volume-level results | Table 4, **PDF p. 9**: starting Qwen3-VL-8B average 33.18 and V-Score 0.16; fine-tuned model average 49.23 and V-Score 5.97. Best listed zero-shot volume V-Score is 1.83. |

These are frame-indexed 2D box metrics. `V-Score` in Tables 3–4 must not be relabeled as native 3D Dice or evidence of longitudinal progression. The title wording difference between the [arXiv v1 paper](https://arxiv.org/abs/2604.15808v1) and [official repository](https://github.com/lamawmouk/SGMRI-VQA) does not by itself establish a different underlying study; code/data snapshot parity remains **NOT_VERIFIED**.

## NeuroQA v1: verified PDF pointers and narrower mismatch

| Existing audit claim | Exact PDF location and check |
|---|---|
| Construction and subject-separated split | §3 and Figure 2, **PDF pp. 5–6**: 56,953 items, 12,977 subjects; 40,355 train, 5,493 validation, 5,524 test-public, 5,581 test-private items. Figure 2 caption is on **p. 6**. This is an author-reported split, not independent cross-cohort identity verification. |
| Test-public baselines and clinician reference | Table 2, **PDF p. 8**: majority-per-template text-only 49.4% closed, supervised 3D CNN 43.7%, Gemini-3.1-Pro 47.5%, two-rater closed mean 48.9% on a distinct 100-item subset. Table 3, same page: image-grounded 50.0% (62 items) and image-informed 42.1% (38 items). These denominators differ and must not be pooled as though paired. |
| Full-pool shortcut control | §3.3, **PDF p. 6**, and Table 4, **PDF p. 9**: 44.6% closed text-only versus 39.5% random, a **5.1 percentage-point** margin. This is a different denominator from test-public's 49.4%. |
| Image-grounding stress-test definition | Figure 4, **PDF p. 9**: image absent, fabrication, and hallucination conditions. The figure specifies a protocol; it does not establish that each model passed it. |
| Longitudinal inventory | Appendix Table 14, **PDF p. 27**: 14 templates and 1,166 items. This confirms task inventory, not a successful model result or independently adjudicated change. |
| First and second expert review rounds | Appendix E.1, **PDF p. 17**: three experts reviewed **24 question-only items** in round 1. Appendix E.2, **PDF p. 20**: two clinicians reviewed **57 image-paired items** in round 2. The 57 figure is therefore genuine for round 2; it is not an alternative total for the same review stage. |

**SOURCE_MISMATCH, narrowed:** the current [NeuroQA README](https://github.com/mhabbasiit/neuroqa#construction-pipeline) says **57 question-only items in round 1**, whereas arXiv v1 Appendix E.1 says **24**. Both sources agree that round 2 used 57 image-paired items. The README also says the 44.6% full-pool text-only result is **5.6** points above random; arXiv v1 Table 4 gives **39.5%** random and **5.1** points. These conflicts concern documentation/revision parity; they do not license silently replacing versioned paper values or claiming which underlying review happened. A pinned code/data release or author clarification would be needed to settle provenance.

## BrainTRACE: dataset and judge discrepancies remain open

The [BrainTRACE arXiv v1 PDF](https://arxiv.org/pdf/2610.06571v1) has 35 pages. Its availability footnote on **PDF p. 1** points to [`lanqz7766/BrainTRACE`](https://huggingface.co/datasets/lanqz7766/BrainTRACE). The current [code README](https://github.com/lanqz7766/BrainTRACE#1-install-and-download-metadata) instead downloads [`BrainTRACE-anon/BrainTRACE`](https://huggingface.co/datasets/BrainTRACE-anon/BrainTRACE). Both public dataset pages existed when checked, and their public Git `HEAD` revisions differed (`1861efc91c93e1de87f607c8d4a973dee1f9e898` versus `d2088bfbcbb01261f8026b3f7bb1b23dcdce3a19`). Different revision IDs do **not** prove different item contents. Dataset-row equivalence, intended canonical release, and correspondence to the paper's evaluated snapshot remain **NOT_VERIFIED**. I did not download patient-linked metadata or MR-RATE imagery to compare rows.

Appendix C.6 of the paper, **PDF p. 24**, explicitly calls **`gpt-4o`** the primary open-ended evaluator in the main experiments. The current [scoring README](https://github.com/lanqz7766/BrainTRACE/blob/main/scoring/README.md) documents **`gpt-4o-mini-2024-07-18`** as its default and says leaderboard reproduction should use the recorded judge setting; the [root README quickstart](https://github.com/lanqz7766/BrainTRACE) also passes the mini model explicitly. The released [run manifest description](https://github.com/lanqz7766/BrainTRACE/blob/main/reproduction/README.md) points to per-run settings, but I did not verify a pinned paper-era judge record or rerun scoring. Thus this is a **SOURCE_MISMATCH for exact leaderboard reproduction**, not evidence that either judge produced a particular published score. Keep BrainTRACE evaluation-only and freeze dataset revision, scorer commit, judge model and decoding before any use.

## Remaining limits

- Paper-reported numbers above were checked against exact-version PDF pages; author-code execution, dataset row hashes, hidden split lineage, and clinical ground-truth adjudication are **NOT_VERIFIED**.
- The local PDF hashes identify the downloaded source files. They are not claims that arXiv itself publishes those hashes or that future URLs will serve identical bytes.
- No patient-level identifiers, medical images, restricted labels, model weights or credentials are included here.
