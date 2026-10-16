# Balanced SFT: independent data and statistical audit

**Status: `TESTED_CPU`.** Independent Codex MRI data/statistics agent, 2026-10-15 local. This audit reads actual private manifests, historical receipts, saved predictions and pinned source metadata. It loads no model, modifies no source-module implementation, and does not inspect new MRI/mask payloads. Corrected-model results are pending its actual matched inference; no training completion is asserted here.

## Supervision and patient isolation

The executable [CPU audit](../scripts/audit_balanced_sft_comparison.py) verifies 382 internal-training rows from 28 patients, 54 internal-development rows from four patients and eight historical-test patients. All three patient sets are disjoint. The 28+4 patients exactly reconstruct the original source-designated 32-patient training partition. Historical test **membership metadata** is read to verify exclusion; historical test labels are not used to fit baselines or select settings.

The independent reconstruction of `balanced_sequence_v2`, using seed 7 and each category's hashed patient order, matches the production schedule at all 112 positions. It yields 14 FLAIR and 14 contrast-enhanced T1 sequence examples. All 84 non-sequence records are byte-for-byte equal to `legacy_lexical_v1` selections, and category-specific patient ordering is identical. Fourteen sequence record IDs/images change. The intervention is sequence-label exposure **and its associated visual examples**, not an isolated numerical reweighting of unchanged images.

Each new selection's record ID, patient key, label and actual image SHA-256 is saved privately. All 112 legacy step/category/patient tuples match the historical receipt. The old receipt did not save record IDs or image hashes; exact old IDs are reconstructed from the preserved training manifest and lexical selection implementation. That distinction prevents overstating the evidence.

Private evidence directory: `local_data/ds007045_bounded/balanced_sft_independent_audit_v1/`. Its `schedule_split_baselines_PRIVATE.json` SHA-256 is `84c6eea12aa28aef30fe80d04eba3be9c7c5ef91e51c319aa042db6d68719991`.

## Existing matched development results

Both saved runs pass the original manifest/prompt/image/adapter/native-template verification. The additional scorer also rejects cross-model changes to base configuration, native template, LoRA configuration, temperature or token cap. The same fixed 54 questions use temperature 0 and maximum 256 tokens. This verifies the recorded configuration; historical receipts do not separately fingerprint every processor artifact.

| Metric | Pre-SFT adapter | Historical SFT28 | Corrected SFT28 |
| --- | ---: | ---: | --- |
| Reward-parser correct | 0/54 | 36/54 | Pending actual output |
| Strict-format valid | 0/54 | 54/54 | Pending |
| Sequence | 0/24 | 15/24 | Pending |
| Tumor presence | 0/12 | 10/12 | Pending |
| Enhancing presence | 0/12 | 8/12 | Pending |
| Tumor grid | 0/6 | 3/6 | Pending |
| Equal-task macro accuracy | 0% | 65.625% | Pending |
| Patient-averaged accuracy | 0% | 66.346% | Pending |
| Worst-task accuracy | 0% | 50% | Pending |
| Sequence predictions, FLAIR / T1 | 0 / 0 strict-parser labels; 24 invalid | 21 / 3 | Pending |

Pre-SFT's zero is under the required parser, not a conclusion that every raw response lacks semantic information. The saved audit separately records exploratory single-box extraction. Invalid-answer counts and per-task confusion matrices retain the strict parser's invalid class.

Historical sequence confusion: all 12 true FLAIR images predict FLAIR; among 12 T1 images, nine predict FLAIR and three predict T1. Balanced sequence accuracy is 62.5%, compared with 50% for the constant-label baseline. The modal class occupies 87.5% of sequence predictions. These values quantify the remaining prior bias; they do not establish MRI appearance learning.

Historical tumor presence confusion is true-no → five no/one yes, true-yes → one no/five yes. Enhancement is true-no → five no/three yes, true-yes → one no/three yes. Grid confusion is true-right-middle → two center-middle/one right-bottom/one right-middle; true-center-top → one center-top; true-center-middle → one center-middle.

The historical per-patient correct totals, in fixed private manifest order, are 7/13, 8/13, 11/14 and 10/14. The pre-SFT totals are 0/13, 0/13, 0/14 and 0/14. The private scorer saves each paired patient outcome and each record's gained/lost status. Four development patients support descriptive diagnostic comparisons; they do not support a confirmatory generalization claim.

Actual response-character ranges (mean) for historical SFT are sequence 63–81 (67.75), tumor presence 63–64 (63.5), enhancement 63–64 (63.5), and grid 71–74 (73.17). Generated-token counts and finish reasons were absent from historical receipts: **token lengths and truncation are `NOT_MEASURED`**. A well-formed answer is not evidence that generation avoided its cap.

The initial two-model `matched_comparison_PRIVATE.json` SHA-256 is `075380dbf0a24acc006dc4442aa114957cfa17e6c8d3c41fe1674b461d7b48d6`. Rerun with `--balanced-outputs PATH` into a distinct audit directory after corrected inference to preserve this snapshot.

## Train-only shortcut controls on the same four development patients

The new baseline fits **only the 28 internal training patients**, not the original 32-patient source-training partition. Each axial level uses the most frequent label with a prespecified lexical tie-break; at tumor z45 the 14/14 tie resolves to `no`. Neither development nor historical-test labels select this rule.

| Task | Global training-majority development score | Training axial-level majority development score | Historical SFT28 |
| --- | ---: | ---: | ---: |
| Sequence | 12/24 | 12/24 | 15/24 |
| Tumor presence | 6/12 | 10/12 | 10/12 |
| Enhancing presence | 8/12 | 10/12 | 8/12 |
| Tumor grid | 0/6 | 1/6 | 3/6 |

For both presence tasks the train-fit rule is z45=no, z91=yes, z136=no. Tumor training no/yes counts are 14/14, 3/25 and 21/7. Enhancement no/yes counts are 17/11, 10/18 and 26/2. Historical tumor presence ties this anatomy-level baseline; enhancement falls two questions below it. These results do not justify a lesion-detection claim. All-tumor case selection, tiny positive masks, shared paired images, conditional grid eligibility and grid-boundary sensitivity remain as documented in the [visual answerability audit](../data/MRI_VISUAL_ANSWERABILITY_AND_SHORTCUT_AUDIT.md).

## Metadata-only prospective cohort preparation

The audit independently rereads committed metadata and Git-annex pointers at source commit `283007b96977b5b3285bd22cd67b7d97114b1eed`. It verifies 363 unique subjects and the required processed FLAIR/T1ce/mask pointers plus raw modality JSON paths. Removing the existing 40 leaves **323 metadata-eligible patients**, with reserve counts BO=164, FO=3, JS=6, KA=103, ME=3, NSK=34, TO=10. Their 969 referenced payloads total 4,195,990,392 bytes. No new MRI or mask is downloaded or opened.

A bounded **14-patient metadata selection** is prepared: two patients from each of the seven reserve sites, ordered within site by SHA-256 of `ds007045-prospective-site2-v1:<source commit>:<subject>` with a lexical subject tie-break. The selected 42 payloads total **181,494,498 bytes (~173.1 MiB)** according to annex metadata. This is a site-balanced estimand; it does not represent the source's proportional site distribution. It leaves one metadata-eligible patient each in FO and ME and cannot include exhausted BA.

The private selection receipt includes exact subject keys, source paths, official pointed-to SHA-256 values and declared byte sizes. Its SHA-256 is `27a9fa79c3e2f0ad9fe03b0f15ab2fb7f3a3257fedb82215b46dd7139312f50d`. Status is **`METADATA_SELECTION_PREPARED_NOT_TEST_RELEASED`**. An earlier 21-candidate private draft is preserved as superseded; this 14-candidate revision was requested before any payload access or outcome inspection. Selection is reproducible, but evaluation remains sealed until the following concrete protocol is finalized:

1. Bind all comparison models to actual checkpoint hashes and freeze model loading, system instruction, native template, processor, temperature 0 and 256-token cap before fetching any selected payload. Include pre-SFT, historical SFT28, corrected SFT28 and only RL arms that have already passed their training gates.
2. Freeze the renderer source hash and settings: original fixed z45/91/136 geometry, rotation, per-volume nonzero 0.5/99.5 percentile intensity scaling, bilinear enlargement and original panel layout. Use opaque transport IDs and prevent image path/record metadata from entering text. Freeze a fixed same-patient, same-level modality-swap and blank/no-image protocol for every compared model.
3. Preserve the existing mask rules as explicitly mask-derived **slice** endpoints: tumor `mask>0`, enhancement `mask==3`, grid only on positive tumor slices. Do not invent a minimum lesion area after examining prospective labels. Any minimum-area task or varied-level benchmark is a separately preregistered endpoint fixed using training/development evidence before release. This initial cohort alone does not cure the fixed-level shortcut.
4. Verify downloaded bytes against every frozen annex SHA-256 and size, then check dimensionality/geometry/source alignment. Record failures and exclusions with reason; do not substitute easier cases based on images, masks or model behavior. Report selected, downloaded, valid and evaluated denominators.
5. Freeze strict parser and scoring source hashes. Report correct/total, strict format, class confusion, prediction frequencies, balanced sequence accuracy, macro/worst-task accuracy, patient means and matched pixel-control changes. Compare against the unchanged train-only global and level-only baseline rules above. Use the same frozen questions for all models.
6. Make patient the uncertainty unit. Prespecify a site-stratified paired bootstrap (10,000 replicates, seed 7), resampling two complete patients with replacement within each of seven sites, carrying all questions/conditions/models together. Report percentile 95% intervals for patient-mean paired differences and primary sequence swap sensitivity; include site-specific estimates. Treat only corrected versus historical SFT sequence balanced accuracy as the primary contrast; mark additional model/task comparisons exploratory. Small strata and 14 patients give limited precision; no clinical validation claim follows.

Freeze the complete protocol and checkpoint manifest with a timestamp and content hashes before new payload access. No model/reward/prompt/checkpoint selection is allowed on this cohort. “Prospective” refers to this project's use, not prospective scan collection. Source patients all have tumors; healthy-control generalization and pretraining/cross-source nonoverlap remain unverified.

## Reproduction and checks

```sh
PYTHONPATH=src python3 scripts/audit_balanced_sft_comparison.py \
  --root local_data/ds007045_bounded \
  --output-dir local_data/ds007045_bounded/balanced_sft_independent_audit_v1
PYTHONPATH=src python3 -m pytest -q \
  tests/test_balanced_sft_comparison_audit.py \
  tests/test_ds007045_internal_dev_eval.py
```

**16 tests passed.** Synthetic tests verify paired patient weighting, balanced accuracy, rejection of cross-model token-cap changes and duplicate predictions, and a training-only axial-rule tie. The real-data executable independently checks all selected image hashes and recorded evaluation provenance. The optional `--metadata-repo` argument reproduces the pinned metadata selection with Git blob reads only. All linked patient-level audit files remain ignored under `local_data`; this report contains aggregate evidence only.
