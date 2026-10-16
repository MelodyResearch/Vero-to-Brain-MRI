# B28 sequence image controls — independent complete review

**Status:** `REPRODUCED` from the complete saved original/no-image/swap outputs. **Verdict:** `PASS_WITH_LIMITATIONS` for frozen input linkage and descriptive scoring. Scope: same-family A28/B28 checkpoint comparison. Scientific generalization and private distinct-model-family review remain `NOT_VERIFIED`.

The independent reviewer waited for all24 no-image and24 swap outputs before reading/scoring the new controls. Initial results were frozen before reading `complete_analysis_PRIVATE.json`. No model was rerun. Public aggregates contain no patient identifiers, response text or per-case outcomes; these remain private.

## Complete observed comparison

Each condition contains24 sequence questions from the same four internal-development patients. Original and swapped images have12 FLAIR and12 T1ce targets. Swap correctness uses the **displayed replacement image's label**.

| Metric | A28 original | B28 original | A28 no image | B28 no image | A28 swap | B28 swap |
|---|---:|---:|---:|---:|---:|---:|
| Correct with full format |15/24|15/24|12/24|12/24|15/24|15/24|
| Full format |24/24|24/24|24/24|24/24|24/24|24/24|
| Invalid/unparsed |0|0|0|0|0|0|
| Predicted FLAIR /T1ce |21/3|21/3|24/0|24/0|21/3|21/3|
| Mean patient accuracy |62.5%|62.5%|50%|50%|62.5%|62.5%|

Unchanged Vero task accuracy and full-format correctness agree for every actual control output. Each patient contributes six sequence questions per condition, so pooled and patient-mean accuracy coincide here.

**A28 versus B28:** zero newly correct, zero newly incorrect and zero changed parsed answers in each condition. All24 correctness outcomes and every patient's mean are unchanged, for each of original/no-image/swap. This statement concerns paired answers and correctness; it does not claim all A/B explanatory response text is identical.

### Classwise confusion

The original and swap conditions have the same confusion matrix for both checkpoints: all12 FLAIR images are called FLAIR; three of12 T1ce images are called T1ce and nine are called FLAIR. Without an image, all24 answers are FLAIR:12 correct FLAIR and12 incorrect T1ce targets. Thus equal 50% no-image accuracy reflects a constant class response on a balanced target set, not independent successful image reasoning.

## Directional changes and the correct statistical unit

There are **12 unordered same-patient, same-slice FLAIR/T1ce pairs**, each evaluated in two swap directions. Both checkpoints show:

- **6/24 correct-direction flips**, covering **3/12 unordered pairs**. A correct-direction flip requires that both the original and replacement-image answers are correct and different.
- Six changed valid answers overall; in this dataset all six meet the correct-direction criterion.
- Nine of12 pairs receive FLAIR for both modalities. The remaining three pairs are distinguished correctly in both directions.

The 24 swap directions are not24 independent image pairs. Multiple pairs also belong to the same four patients. No independent-question p-value, confirmatory interval or population-level claim is justified.

## Identical visible-input consistency

Every swap is an exact permutation of an original image/prompt pair. The reviewer checked the image hash and structured prompt, and checked rendered prompt hashes where recorded. For **both A28 and B28**, each swapped output is **exactly text-identical and answer-identical to the original output on that same visible image and prompt:24/24**.

This is a useful execution-consistency check. It also explains why original and swap accuracy match: under identical deterministic input/output mapping, the aggregate swap evaluation is a relabeling/permutation of the original24 cases. It is **not an additional independent accuracy cohort** or new patient evidence.

All24 no-image records within each checkpoint likewise share one identical model prompt and produce one distinct response/answer. Their repetition tests consistent execution; it does not create24 independent model observations. Target labels are scoring information and were not inserted into these prompts.

## Input and checkpoint verification

The independent replay verifies exact frozen control manifest hashes, complete unique IDs, category and patient/slice opposite-sequence lineage, image bytes and placeholder counts. No-image removes only the structured image content; swap changes only the image reference. Prompt text remains unchanged. Each control preserves its checkpoint's model/configuration/native template,504 loaded adapter tensors, greedy decoding and256-token cap. B controls also match the primary run's completed training receipt, optimizer-state manifest and processor/tokenizer artifact hashes. Historical A28 outputs match their immutable hashes.

Historical artifacts do not contain all newer processor fingerprints or actual stopping telemetry. Matching their recorded configuration does not retrospectively supply those missing measurements. Full optimizer-chain approval remains in the separate engineering audit.

## Actual stopping evidence

All48 newly generated B28 controls record `finish_reason=stop`, no length termination and no unknown finish reason. No-image responses contain17 tokens each. Swap responses contain17–20 tokens, mean17.417, matching the primary sequence subset's token-count range and mean. A28 stopping reasons and token counts remain **unknown/unmeasured**; valid formatting is not substituted for missing telemetry.

## Scientific interpretation

This run shows **limited, existing image-conditioned sequence response sensitivity**, unchanged by B's sequence exposure correction: three image pairs are distinguished, while nine collapse to FLAIR. Original accuracy exceeds no-image accuracy by3/24 for each checkpoint, and correct directional changes occur on those three pairs. This does not establish the image feature used, robust sequence recognition, lesion understanding or independent-patient generalization.

The hypothesis that this specific14/14 supervision correction alone would improve sequence discrimination at the matched28-update budget is **not supported by these observations**. No-image predictions, primary sequence answers and directional behavior remain unchanged. The full primary task comparison separately fell from36/54 to34/54, with losses in enhancing-mask presence and grid localization. Neither this single SFT run nor these controls establish reproducible negative transfer or identify whether representation limits, fixed generic targets, optimization, short training or data shortcuts caused the result.

B remains an SFT engineering control supporting the Vero-to-MRI program. It is not multi-task RL, a completed cross-task RL transfer matrix or an MRI-specific algorithm contribution. The next mechanism test should use frozen training-only cases and a separately approved scope; these development results must not be used to search repeatedly for favorable evaluation subsets.

## Independent crosscheck and safe evidence

After the initial independent freeze, the production complete analyzer agreed on all**144 model/condition/question outcomes**, all six condition confusion matrices, full-format metrics, patient means, termination counts and directed changes. These144 outcome checks are accounting units, not144 independent statistical observations. The exact same-visible-input response consistency check is additional independent analysis beyond the production analyzer's reported fields.

Machine-readable safe aggregates: `results/B28_CONTROL_INDEPENDENT_AGGREGATE.json`. Its complete key set was checked against an explicit whitelist, and it contains no patient IDs, record IDs, raw responses or private row objects.

- Aggregate SHA-256: `d264ecfe0b514c9778fa125540aea47ba8865175b2078bcd5c9f31322ed9c130`.
- Frozen independent private result SHA-256: `2b061e58189e6eaa978ef117bca2838292c6813d32aede0e9f0aa68570ba1bef`.
- B28 no-image outputs SHA-256: `6226bb9ca175e06bf7a59f775e32e4316ad370f94d1275b1b6d44b2ee460f599`.
- B28 swap outputs SHA-256: `58e401f28f25acd359b6b8fadce63145e967c5d1480b5a9e01316ee9b099d5ee`.
- Production complete analysis SHA-256: `2f8c4af829b118962500fa63f4ca69f6e5587654b4c53dd036d3b265e5d9a074`.

The independent script, row-level results and agreement receipt remain under `local_data/ds007045_bounded/overnight_method_stats_v1/`. No prospective patient payload or test cohort was opened.
