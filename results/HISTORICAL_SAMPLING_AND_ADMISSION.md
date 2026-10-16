# Historical sampling, separate B/C controls, and task admission

Status: **TESTED_CPU**; historical schedule identity reconstructed from the frozen legacy selector and saved step/category/patient receipt, not originally logged image IDs. This is a sampling audit, not a new MRI-performance result. Private per-example references remain in `local_data/ds007045_bounded/safe_resume_v1/historical112_sampling_v2_PRIVATE.json`.

## Actual historical 112-example schedule

The saved receipt matches all 112 reconstructed step/category/patient tuples. Each category contains the same 28 distinct training patients, each once. Affines are LAS for every selected volume; voxel axis 2 is the axial axis, increasing superiorly. Display rendering applies `rot90` and 2× scaling. Slice indices below are zero-based native voxel indices, not anatomical names or standardized physical slice positions across patients.

| Category | Actual labels | Axial indices | Relevant mask pixels: min / median / max | Positive minimum |
|---|---|---|---|---|
| Sequence | FLAIR 28; T1 0 | 45: 28 | Tumor 0 / 134 / 2814 | 268 |
| Tumor presence | yes 14; no 14 | 45: 28 | Tumor 0 / 134 / 2814 | 268 |
| Enhancing presence | yes 11; no 17 | 45: 28 | Enhancing 0 / 0 / 1808 | 257 |
| Tumor centroid grid | left-middle 11, right-middle 8, center-middle 5, center-top 3, left-bottom 1 | 45: 14; 91: 14 | Tumor 268 / 1797.5 / 4252 | 268 |

All rows additionally retain both tumor and enhancing areas, image/question/mask digests, target, patient, slice and orientation in the private audit. Sequence and presence rows have 14 zero-tumor and 14 positive-tumor slices. Grid rows are all tumor-positive by eligibility. Positive-area medians are 1509 for the first two routes, 698 for enhancement, and 1797.5 for grid. The selected historical positives contain no areas ≤100 pixels; tiny one-voxel positives elsewhere in the broader pack must not be attributed to these 112 selections.

There are **70 distinct image hashes**, 42 image-reuse excess occurrences across tasks, and **zero repeated image-plus-question pairs**. Each route individually uses 28 distinct images. Reuse across different tasks is not automatically leakage, but 112 examples do not mean 112 distinct images. Repeated target strings exceed distinct targets by 27/26/26/23 in sequence/presence/enhancement/grid respectively.

**The low-slice selection defect is verified.** Lexical minimum record selection concentrated three routes entirely at z45. A training-only slice-index rule reaches 10/12 development tumor-presence answers and 10/12 enhancement answers, versus historical SFT28's 10/12 and 8/12 respectively. These four repeatedly examined development patients are diagnostic evidence of shortcut risk, not a confirmatory population estimate. A fixed native slice index is not identical anatomy across patients.

## Visually supported versus programmatically derived

Every audited row has a verified source-mask/sequence reference, but carries `SOURCE_LABEL_VERIFIED_VISIBILITY_NOT_ADJUDICATED`. A positive mask is not proof that a learner can reliably see that feature in the rendered slice. Negative slices from tumor patients are not healthy-patient labels. Enhancement in postcontrast-only images does not establish a clinically calibrated enhancement comparison.

The lead inspected the two rendered historical binary-positive examples with the smallest relevant mask areas (268 tumor and 257 enhancing pixels). Both real paired images decode and have visible MRI panels. The first is a low, predominantly posterior-fossa slice and does not establish clear lesion answerability from unaided inspection. The second contains conspicuous between-panel signal differences, but this is not independent expert annotation. Two visual checks do not validate all 112 rows. No thresholds or labels were changed based on held-out predictions or these informal observations.

## Control B: label-balanced sequence exposure

`balanced_sequence_v2` retains 28 patients per route, their order, 112 examples, and **all 84 non-sequence examples unchanged**. Sequence becomes 14 FLAIR / 14 contrast-enhanced T1 by replacing 14 sequence image/target pairs. This is a sequence-exposure intervention, not merely changing text labels or loss weights. It intentionally retains z45 selection. A completed matched B28 would estimate the effect of this intervention under the historical design, not solve general MRI answerability.

The isolated Codex reviewer independently replayed B and verified the 14 replacements and identical remaining 84. The pending post-panic model stage is only the first four examples/one optimizer update of the frozen B schedule. Its executed prefix is separately hashed; it cannot establish a balanced-exposure outcome or stand in for B28.

## Candidate C: separate slice/label-aware design

`slice_label_candidate_v1` uses only the 382 training references, not development/test model predictions. It is a new intervention in image selection and eligibility, and is not mixed into B.

- Sequence: 28 patients, 14/14 modality labels, levels z45/z91/z136 = 10/9/9.
- Binary candidates: demand 28 distinct patients, exact 14/14 labels, within-level positive/negative balance at all three levels, zero-mask negatives and positives with ≥1000 native reference pixels. This geometry threshold is an engineering filter, not a new clinical gold standard or sufficient visual-admission criterion.
- Neither binary route has a feasible full assignment under those constraints. Independent reviewer max-flow enumeration confirms best flows 27 for tumor and 19 for enhancement. Enhancement has eligible positive row counts 2/3/0 across the three levels. This does **not** imply that every smaller or differently defined slice-aware task is infeasible.
- Grid: 21 eligible patients after tumor-area ≥1000 and centroid distance ≥10 display pixels from grid boundaries; levels 10/10/1. Labels remain imbalanced (8/7/3/2/1 in left-middle/right-middle/center-middle/center-top/left-bottom). It remains a mask-centroid engineering endpoint, not anatomical localization.

**C is not admitted to model training**: visual adjudication is incomplete and the full binary design is infeasible. Next data work should seek additional appropriate training slices, or preregister a smaller, matched subset that preserves within-level support and patient isolation. Do not weaken the area rule solely to fill quotas or turn uncertain examples into clear positives. A revised C needs its own version and independent review.

## Scientific implications and next comparisons

- **Scientific significance:** this audit identifies a sampling failure that can make apparent lesion reasoning consistent with an axial-level rule. It does not establish new disease biology or clinical accuracy.
- **AI/ML significance:** the distinction between category balance, label balance, image eligibility and slice support is necessary for interpretable Vero-to-MRI transfer. It is not a new RL algorithm.
- **Experimental significance:** preserve historical A, label-only B, and separately versioned C; compare identical starting weights and budgets within each prespecified contrast. Do not interpret format learning or a one-step parameter update as cross-task generalization.

The full multi-task GSPO transfer matrix remains pending corrected data admission and independently verified optimization continuity. The completed 13-paper audit is preserved without repetition. Claude did not conduct this review; an isolated Codex reviewer was used under the authorized fallback policy.
