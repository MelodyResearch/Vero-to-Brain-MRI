# Overnight independent method and statistical review

Reviewer: isolated Codex agent `/root/overnight_method_stats`. This is an independent local review of source, frozen manifests and saved predictions; it is not a Claude review. No model was loaded, no MRI inference was repeated, and no historical-test labels were used. The two responsibilities below have distinct conclusions. Engineering authorization for a particular model stage belongs to the separate engineering reviewer.

## 1. RL methodology review

**Verdict: `PASS_WITH_LIMITATIONS` for the scientific definition of B as a controlled SFT comparison; `NOT_VERIFIED` for full Vero recipe transfer.**

An independent CPU replay of `scripts/audit_balanced_sft_comparison.py` reproduces all 112 schedule positions: 14 sequence image/target replacements, exactly 84 unchanged non-sequence examples, unchanged category-specific patient order, and 28/4/8 disjoint training/development/historical-test patients. B changes sequence exposure together with the corresponding images; it is not label editing on fixed images. It retains the low-slice bias intentionally. A completed B28 can test this narrowly defined intervention against historical A28. It cannot by itself validate C or explain Vero category balancing.

The SFT code optimizes supervised next-token cross entropy on a fixed response containing `Visible MRI evidence.` and a reference label. This fixed rationale is not a verified visual explanation. Strict parser adherence, a lower loss or a changed checkpoint demonstrates engineering progress, not lesion reasoning. The hidden reference is used in supervised targets as intended; the training question is identical within each route and contains no patient key or record ID. The image-bearing JSONL also contains private image paths as structured transport fields; the SFT runner passes only question text and image data to the native chat formatter. This checks the inspected route, not every possible future renderer or source.

Source inspection confirms the local GSPO diagnostic uses response-mean log ratios, a detached sequence ratio with token gradients, asymmetric clipping and sample-standard-deviation group advantages. It creates fresh Adam state for each diagnostic group. The multiarm wrapper explicitly reports this reset. Implementing SFT restoration does not repair the RL wrapper automatically. **P1:** do not promote reset-per-group runs to matched continuous Vero arms; integrate independently verified optimizer persistence before that comparison.

One fresh on-policy update begins with current/old ratios near one, where clipping normally has no effect. A successful single update cannot isolate GSPO clipping or distinguish it from related on-policy policy-gradient objectives. The existing pinned-source audit reports official eight-rollout groups, continuous optimization, larger batches/response budgets and full-model training; local two-rollout, four-bit language-LoRA diagnostics freeze vision and add an SFT warmstart. No new literature audit was performed, and no full official parity is claimed.

### Focused defect ledger

| Priority | Observed issue and evidence | Consequence | Minimal correction / verification |
|---|---|---|---|
| P1 | Historical lexical minimum selection puts three categories entirely at z45; independent schedule replay agrees with the sampling audit. | A label-balanced sequence schedule does not remove lesion-level shortcuts. | Preserve B unchanged as the controlled intervention; version C separately and require within-level support and visibility admission. Do not silently replace B examples. |
| P1 | A28 presence 10/12 equals the training-fit axial-level rule; enhancement 8/12 is below that rule's 10/12. | Existing scores do not establish lesion recognition. | Carry level-only, no-image and image-counterfactual controls into scientific task admission and final comparisons. |
| P1 | Local RL children instantiate fresh Adam; wrapper records `optimizer_reset_each_group=True`. | Adaptive moments and training budgets do not reproduce official continuous training. | Keep this as an explicit engineering approximation until a reviewed RL continuation implementation exists. B SFT continuation is not GSPO. |
| P2 for engineering / P1 before truncation-aware evaluation | Saved historical and B-step1 generations lack actual token counts, finish reasons, and a post-generation resource boundary. | Cannot measure truncation or certify the final generation's observed memory boundary. | Installed `mlx_vlm/generate/dispatch.py` returns actual `generation_tokens`, `finish_reason`, and `peak_memory`. Record them in future receipts, distinguish stop from length, add a boundary after generation, and regression-test missing/invalid metadata. Do not backfill historical unknowns. |

These findings block only the claims or interfaces identified. None justifies blocking a separately approved, safely bounded B SFT continuation.

## 2. Scientific and statistical review

**Verdict: `PASS_WITH_LIMITATIONS` for the saved A0/A28 descriptive comparison; corrected B28 evaluation remains `NOT_RUN`.**

The independent replay verifies image and prompt provenance, native-template receipts, exact paired record sets, adapter identities, common generation cap/temperature, and model/config/template hashes recorded in the historical outputs. The historical receipts do not independently fingerprint every processor artifact. Four repeatedly examined development patients are not a confirmatory test, and the 54 questions are clustered within those patients.

| Metric | A0 | Historical A28 |
|---|---:|---:|
| Strict correct | 0/54 | 36/54 |
| Strict format valid | 0/54 | 54/54 |
| Sequence | 0/24 | 15/24 |
| Tumor presence | 0/12 | 10/12 |
| Enhancement presence | 0/12 | 8/12 |
| Tumor grid | 0/6 | 3/6 |
| Equal-category macro accuracy | 0% | 65.625% |
| Worst-category accuracy | 0% | 50% |
| Patient-averaged accuracy | 0% | 66.346% |
| Sequence FLAIR / T1 / invalid | 0 / 0 / 24 | 21 / 3 / 0 |

A0's parser failure must not be interpreted as a measured absence of all semantic MRI capability. Historical A28 patient correct counts are 7/13, 8/13, 11/14 and 10/14. The same image is reused across some task questions, further precluding an independent-question uncertainty calculation.

The user quotes historical level-only scores of 19/24 and 20/24. Those are not the denominator of this four-patient matched development set. This replay's corresponding train-fit development scores are 10/12 and 10/12; preserve the different populations rather than mixing them in a single comparison.

### Prespecified interpretation for subsequent B evaluation

- Compare A28 to B28 only after 28 verified updates with the frozen B schedule and continuous optimizer/RNG/sampler history. Partial B outputs are engineering observations and cannot be described as the equal-budget B28 effect.
- Report paired gained/lost questions and patient-mean differences, alongside format validity, category confusion, worst task and macro metrics. No binomial confidence interval over 54 independent trials is justified.
- The primary descriptive sequence contrast is a prediction-prior and accuracy change. A move toward 12/12 predictions alone is not evidence of better image use.
- For same-patient sequence swaps, score against the visible replacement image's label. Compare correct-direction changes to original/no-image/blank controls. Arbitrary answer changes are not visual understanding.
- A later prospective evaluation must freeze checkpoints, all selection rules and the scorer before payload release; exclude reserved prospective subjects from C training expansion. Site-balanced selection changes the estimand and should be stated.

### Reproducible evidence

Independent CPU replay output is privately retained under `local_data/ds007045_bounded/overnight_method_stats_v1/`:

- `schedule_split_baselines_PRIVATE.json`: SHA-256 `84c6eea12aa28aef30fe80d04eba3be9c7c5ef91e51c319aa042db6d68719991`.
- `matched_comparison_PRIVATE.json`: SHA-256 `075380dbf0a24acc006dc4442aa114957cfa17e6c8d3c41fe1674b461d7b48d6`.

Both exactly reproduce the prior independently recorded audit digests. This is new verification of existing evidence, not new model performance.

## 3. Direction C and new-stage review status

**C1 geometry/subset verdict: `PASS_WITH_LIMITATIONS`; scientific image-answerability: `NOT_VERIFIED`.** Independently inspected `scripts/build_c1_matched_subsets.py` and its `overnight_c1_v2/matched_subsets_v2_PRIVATE.json` output. A separate source→patient→level/label→sink Edmonds–Karp implementation exhaustively checked every admissible quota combination, rather than importing the producer's slot-matching routine.

| Candidate | Exhaustively checked quotas | Maximum distinct patients | Labels | Level counts | Metadata association, level / site / site+level |
|---|---:|---:|---|---|---|
| Tumor presence | 33 | **26** | 13 yes / 13 no | 45:18, 91:6, 136:2 | 13/26, 17/26, 18/26 |
| Enhancement presence | 6 | **10** | 5 yes / 5 no | 45:4, 91:6 | 5/10, 7/10, 7/10 |

Every selected reference was checked against the training-only source inventory, every selected patient occurs once within its route, positives retain the ≥1000-pixel threshold, negatives have zero relevant mask pixels, and each represented level has equal yes/no counts. Enhancement has zero eligible positives at level136, so no nonempty design requiring all three original levels can meet the exact per-level balance criterion. Its feasible two-level result is an explicit design revision, not success under the original constraint. Tumor26 is the maximum under all three supported levels; the preserved 28-patient candidate is not silently relabeled as feasible.

These are finite-sample geometry constraints, not evidence that the lesions are visible to a human or model. Level balance reduces the direct level/label association to 50% within each selected set. It does not remove site/protocol or within-level anatomy shortcuts: site and site+level associations remain as tabulated. These values are **resubstitution majority accuracies**, not independently tested prediction scores. The producer's leave-one-patient-out tiny-stratum diagnostics are descriptive and particularly sensitive to the exact matched selection; do not use them as evidence that a site shortcut is absent. Actual no-image model baselines remain `NOT_RUN`.

Independent certificate: `local_data/ds007045_bounded/overnight_method_stats_v1/independent_c1_flow_PRIVATE.json`, SHA-256 `d8ad6f3f0f21457ee9dcaee96a2ed4bf9ebcb6bd0b3362833e3b08e2a11b8fa3`. This is a separately implemented, executed CPU feasibility proof, not a model result.

**C2 source boundary:** the candidate metadata excludes all40 existing patients and14 reserved prospective subjects, leaving309 metadata candidates; its deterministic bounded selection is12 patients across seven sites, because two sites have only one remaining candidate. Selected image+mask pointers total149,834,294 bytes and masks318,016 bytes. This report has inspected that selection logic and aggregate receipt; it does not claim new images were downloaded or independently visually validated.

**Task-card review: `PASS_WITH_LIMITATIONS` for accurate engineering status and design boundaries.** Independently read `data/C_TASK_ADMISSION_V2.md` and `results/C1_FEASIBILITY_RESULTS.md`. All four cards retain `ENGINEERING_ONLY`. Their C1 label/level/site counts match the private selected records; the sequence candidate remains14/14 with levels10/9/9 and explicitly lacks guaranteed within-level class balance; the21-patient grid candidate is explicitly imbalanced. The cards distinguish sequence/contrast discrimination from spatial localization as the two promising capability families, and avoid claiming that the two mask-presence questions establish independent clinical skills. They correctly limit postcontrast-only enhancement interpretation, display-grid coordinates, mask visibility and 2D-versus-3D conclusions.

The card conditions are a defensible admission plan, not completed visual adjudication. No actual new no-image VLM baseline was observed by this reviewer. The corrected structured-prompt audit excludes transport paths while checking actual text; its earlier false positives remain preserved and do not establish observed leakage. Keep future rendered-input and image-evidence checks separate.

### Executed C1 simple real-image baselines

Subsequently inspected `scripts/c1_image_shortcut_baseline.py` and independently recomputed every prediction without importing its prediction function. The independent replay uses source-hash-verified PNGs,16×16 bilinear grayscale thumbnails and, separately,16-bin normalized whole-image grayscale histograms. Euclidean nearest-neighbor classification excludes every row from the query patient's identity; deterministic record-ID order resolves ties. Histogram binning was independently implemented with integer bin counts rather than the producer's histogram call. All predictions match.

| Leave-one-patient-out nearest neighbor | Tumor presence | Enhancement presence |
|---|---:|---:|
|16×16 grayscale thumbnail|12/26|6/10|
|16-bin intensity histogram|13/26|4/10|

These are actual image-input CPU baselines, not VLM results. They use the training-selected C1 cohort and have not been tuned or evaluated on a new patient cohort. The weak scores neither establish lesion understanding nor show that all image shortcuts have been removed; they quantify these two deliberately simple classifiers only. Raw thumbnails include panel layout and anatomy, and histograms discard localization. The dataset is too small for a strong comparison of these feature families. C3 sequence/spatial simple-image baselines remain pending.

Producer receipt SHA-256: `b00d7446243ed4a08bc55a5bb2894624ffefe470c2f9ca418794ff2f6bc04711`. The independent per-method agreement receipt remains private at `overnight_method_stats_v1/independent_image_baselines_PRIVATE.json`.

New-stage training receipts remain subject to subsequent review. Scientific eligibility is separate from a programmatic label check or a successful training step.

## 4. Actual C2 mask expansion: independent verification

**Verdict: `PASS_WITH_LIMITATIONS` for source payload integrity and mask geometry.** After the metadata-only stage, the data agent retrieved12 new masks totaling318,016 bytes. This reviewer independently reread every corresponding annex pointer at pinned source commit `283007b96977b5b3285bd22cd67b7d97114b1eed`, verified source size/SHA-256 against both downloaded bytes and the frozen selection, and checked exclusion of all40 existing subjects and14 reserved prospective subjects by source identity. No prospective payload was read. The new cohort's sites are BO2, FO1, JS2, KA2, ME1, NSK2 and TO2.

Each mask was opened serially. All have182×218×182 voxels, the expected LAS affine with1-mm spacing, finite values and labels restricted to0/1/2/3. Direct array recounts reproduce every recorded tumor and enhancing-component area at levels45/91/136.

| New12-mask eligibility | z45 | z91 | z136 |
|---|---:|---:|---:|
| Tumor positive area≥1000 |1|8|1|
| Tumor zero-area negative |8|2|5|
| Tumor small positive excluded |3|2|6|
| Enhancing positive area≥1000 |0|1|0|
| Enhancing zero-area negative |11|5|8|
| Enhancing small positive excluded |1|6|4|

The expansion **does not fix the missing upper-level enhancing-positive support**. It supplies one new eligible enhancement-positive at the middle level only. It does supply one upper-level tumor-positive. Combining existing28 training patients with these12 mask-only candidates, an independent flow replay across130 tumor quota combinations and eight enhancement quota combinations confirms maximum balanced subsets of32 tumor patients and12 enhancement patients. The latter still requires only levels45/91; exact balanced three-level enhancement remains infeasible under the unchanged threshold.

These combined maxima are mask-geometry feasibility results, not MRI-ready training examples. New corresponding MRI frames were not downloaded or loaded in this verification, so no new visibility or image alignment claim follows. A useful next retrieval is bounded paired MRI for selected new eligible cases, followed by alignment/visible-evidence checks. The negative enhancement result argues against downloading many additional subjects under an assumption that the original fixed upper level will automatically become viable.

Actual mask receipt SHA-256: `cbaff83e42b1b146760439d08570af1787638013d7af77d7ebc3e2195c1c5c3b`. The separate verification and exhaustive combined-flow result are retained privately at `overnight_method_stats_v1/independent_c2_masks_PRIVATE.json`.

### Subsequently executed: targeted real MRI pair

During a confirmed GPU-idle interval, this reviewer independently verified the new C2 upper-level FLAIR/T1ce pair: **two payloads, one training-source patient,10,317,786 bytes**. Source Git-annex pointers at the frozen commit, local sizes/hashes, selection lineage, existing40/reserved14 exclusion, image/mask affine equality,182×218×182 LAS geometry and finite intensities all agree. No prospective image was read. The selected subject is the only new candidate meeting the unchanged upper-level tumor-area criterion.

The slice136 mask recount is1867 tumor pixels and733 enhancing pixels. Thus it supports the tumor geometry filter but still fails the enhancing≥1000 criterion. Independent serial array preprocessing reproduces each saved PNG panel pixel-for-pixel using the recorded percentile subsampling, rotation and bilinear display transform; both panels also match their locations in the combined PNG. Combined PNG SHA-256: `45bec4d19bcbf5ce9309ef81d40d723403d9da1fc45be0a51dc4a0a6fa426a57`.

**Verdict: `PASS_WITH_LIMITATIONS` for actual source-linked MRI bytes, geometry and rendering.** This verifies a real image pair beyond mask-only feasibility. It is not a blinded visible-lesion judgment, a new model result or scientific task admission. Status remains `ENGINEERING_ONLY`, clinical visibility `NOT_VERIFIED`. Private independent receipt: `overnight_method_stats_v1/independent_c2_image_PRIVATE.json`.

## 5. Complete B28 comparison analysis prepared; new B28 outcomes not yet observed

**Implementation status: tested on synthetic fixtures and replayed historical saved outputs. Scientific verdict remains `NOT_VERIFIED`.** New analysis-only `scripts/analyze_b28_paired.py` requires the completed 28-update receipt, 112 executed exposures, original A0 adapter, frozen schedule and final B adapter hash before naming predictions B28. It checks all 54 IDs, 24/12/12/6 task counts, four patients, image hashes, prompts, native templates, adapter loading and common model/decoding configuration. The complete optimizer chain still requires separate engineering verification. No training code or historical outputs were changed.

Two metrics are deliberately distinct:

- Equal-task macro of question accuracies: pool questions within each task, then equally average the four task accuracies.
- Equal-task macro of within-patient task accuracies: average each patient's questions within a task, equally average observed patients for that task, then equally average tasks. A patient without an eligible question in a task is omitted for that task; no invented zero is assigned. Coverage counts accompany this metric.

Actual saved A28 outputs give **65.625%** for the first and **62.5%** for the second. The discrepancy comes from grid questions: six questions across all four patients, with counts 1/1/2/2. Grid pooled question accuracy is 50%, while the mean of within-patient grid accuracies is 37.5%. Sequence, tumor presence and enhancement presence each cover all four patients with equal counts per patient (6, 3 and 3 respectively). A0 remains zero under both definitions. These are descriptive historical reanalyses, not new model inference or independent evidence of generalization.

The unchanged Vero reward contract allows its partial-format branch to receive task accuracy. Therefore contract accuracy and full-format correctness are reported separately, including separate confusion matrices and paired changes. Full-format correctness requires format reward exactly 1 and correct task answer. Diagnostic answer-only extraction never contributes to either primary score. Invalid/unparsed predictions retain an explicit `INVALID` bucket. Historical A28 has full format on all 54, so its two correctness definitions agree.

Both sequence controls must contain all 24 records. The analyzer reconstructs the intervention from the original same-patient, same-slice sequence pair: no-image removes only structured image content; swap substitutes only the opposite sequence image. All prompt text and generation settings remain matched. Swap correctness uses the **replacement image's label**. Correct-direction flips require both original and swapped predictions to be correct and different; arbitrary answer changes are reported separately.

Replaying actual historical outputs reproduces original 15/24, no-image 12/24 and swapped-image 15/24, with 6/24 correct-direction flips. These saved controls do not constitute a new B28 experiment. Observed generation token counts are preserved even when finish reason is unknown; historical termination remains unknown, and a cap-length output does not imply truncation without a recorded finish reason.

CPU verification: `tests/test_b28_paired_analysis.py` plus `tests/test_balanced_sft_comparison_audit.py`: **22 passed**. Fixtures explicitly contain synthetic non-loadable image and adapter bytes; no fixture is entered into real experiment output directories. Tests reject missing/duplicate records and image, prompt, checkpoint, configuration and partial-training mismatches; they exercise replacement-label scoring, partial-format handling, unknown termination and unequal patient coverage. The independent historical replay is private at `overnight_method_stats_v1/historical_metric_control_recheck_v2_PRIVATE.json`. Actual B28 prediction collection and its paired statistical comparison remain pending until the genuine completed run and all outputs exist. No p-values, confirmatory intervals or independence assumptions over 54 questions are made.

### Independent engineering correction and Step D

The first analysis implementation called its scope `CROSS_MODEL_REVIEW`, despite comparing checkpoints from one model family. Following independent review, this was corrected to `PAIRED_INTERNAL_DEV_CHECKPOINT_ANALYSIS`, with `cross_model_review: NOT_VERIFIED` explicit. The CLI also now pins the immutable A0 and A28 output-file SHA-256 values before analysis, so edited response text cannot pass merely by retaining matching configuration fields. Two regression tests first failed and subsequently passed; the combined suite now has **24 passing tests**. The independent engineering reviewer inspected the corrected source and independently ran all 20 paired-analysis tests with MLX imports blocked, confirming both corrections. This is engineering verification of the analyzer, not actual B28 results or a cross-family scientific review.

## 6. Frozen C cohort: independent source, header and aggregate audit

**Verdict: `PASS_WITH_LIMITATIONS` for the linked payloads, reference geometry, frozen selection and aggregate counts.** Independently streamed the hashes of all 20 image volumes (124,875,617 bytes) and checked each against its official Git-annex pointer at the frozen source commit. The scope correctly reuses two previously verified volumes and downloads 18 new volumes totaling 114,557,831 bytes. The 10 new patients exactly match the union selected by the frozen geometry match and exclude the existing 40 and reserved 14 source subjects.

Header-only checks independently confirm that every image and its source mask share the exact expected 182×218×182 LAS affine and 1-mm spacing. No full MRI array was materialized during this review. The 11 new paired PNGs have the recorded hashes and 752×470 RGB geometry. The complete rendered cohort contains 44 task rows, **35 distinct patients and 38 distinct images**; these counts are not interchangeable. Every row retains its frozen patient, label, level and mask areas. Existing question hashes are preserved where present; newly linked geometry rows are not automatically complete training prompt records.

| Category | Rows | Labels | Level counts z45/z91/z136 | Site resubstitution | Site × level resubstitution |
|---|---:|---|---|---:|---:|
| Tumor presence |32|16 yes /16 no|20/8/4|21/32|22/32|
| Enhancement presence |12|6 yes /6 no|4/8/0|9/12|9/12|

These metadata-only baselines were independently recomputed and reproduce the producer's counts. They are resubstitution diagnostics, not held-out predictive estimates. Site-related shortcuts remain despite within-level label balance. The enhancement task still lacks upper-level positive support. No new C training or clinical visibility adjudication occurred. A fresh pixel-by-pixel rerender of all 20 volumes was not run in this header/hash audit; the earlier single-patient pair reproduction remains separate evidence.

Private audit receipt: `overnight_method_stats_v1/independent_c2_cohort_headers_PRIVATE.json`. Render receipt SHA-256: `5786681d14498f4ac2545d858e293a5f12e5ad0cd31234e8c2b6a79ef030b06b`. No reserved prospective payload was read.

### C2 actual PNG shortcut baselines: independent replay

**`REPRODUCED`; `PASS_WITH_LIMITATIONS` for the recorded fixed-feature CPU experiment.** The expanded cohort was subsequently tested with the same prespecified C1 features. This reviewer independently extracted 16×16 bilinear grayscale thumbnails and 16-bin full-PNG grayscale distributions, then independently chose nearest neighbors using scalar squared distances and deterministic record-ID ties. Histogram bins were accumulated from integer grayscale counts rather than the producer's NumPy histogram call. All 88 predictions, nearest-neighbor identities, distances and confusion counts match the saved primary receipt. Every neighbor belongs to another patient; categories are evaluated separately, and no within-category duplicate image hash occurs.

| Fixed leave-one-patient-out 1NN | Tumor presence | Enhancing-mask presence |
|---|---:|---:|
| 16×16 grayscale thumbnail |18/32|5/12|
| 16-bin grayscale histogram |18/32|6/12|
| Fixed-no reference |16/32|6/12|

The source/C1 feature definition and frozen cohort hashes agree with the receipt. Features depend on image pixels only; labels enter through the nearest training neighbor, not feature extraction or query selection. No image filenames or subject identifiers enter distance calculations. The full PNG still includes panel borders, constant modality captions, intensity distributions and anatomy, so these are deliberately simple shortcut probes rather than lesion-specific classifiers. Site structure is not removed by leave-one-patient-out exclusion, and all cohorts were selected using training-label geometry. Weak scores do not prove absence of shortcuts, establish clinical visibility or demonstrate independent generalization. C1 and C2 scores concern different selected cohorts and must not be interpreted as a controlled improvement comparison.

This replay used one small PNG at a time with single-thread caps and explicit MLX/MLX-VLM/Torch/Transformers/nibabel import blocking. No MRI volumes or models were loaded. Primary receipt SHA-256: `59d15c50d24537a066f720c311a81b22decc529499ffef99a63ebf643cff6a75`. Independent evidence is private at `overnight_method_stats_v1/independent_c2_pixel_shortcuts_PRIVATE.json`. Scientific task admission remains `NOT_VERIFIED`.
