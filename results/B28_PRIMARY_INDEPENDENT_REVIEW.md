# B28 primary matched evaluation — independent initial review

**Evidence:** `REPRODUCED` from actual saved outputs and frozen targets. **Scope:** `PAIRED_INTERNAL_DEV_CHECKPOINT_ANALYSIS`. **Engineering/data verdict:** `PASS_WITH_LIMITATIONS` for complete input/checkpoint linkage and scoring. **Scientific generalization:** `NOT_VERIFIED`. **Distinct model-family review of these private outputs:** `NOT_VERIFIED`.

This initial review was formed from primary artifacts and an independent parser/statistical implementation before crosschecking the production paired analyzer. It does not rely on the implementation lead's diagnosis or aggregate summary. No model was loaded or rerun.

## Actual results

All three checkpoints have exactly the same 54 records from four internal-development patients, with question counts 13/13/14/14. A0/A28 output bytes match their frozen hashes. B28 outputs bind the genuinely completed 28-update/112-exposure receipt and final adapter. All image bytes, prompts, categories, native-template receipt, 504 loaded adapter tensors, model configuration, greedy decoding and 256-token cap match. The internal-development patients are disjoint from the 28 training patients.

| Metric | A0 | Historical A28 | Corrected B28 |
|---|---:|---:|---:|
| Full format |0/54|54/54|54/54|
| Correct with full format |0/54|36/54|34/54|
| Sequence |0/24|15/24|15/24|
| Tumor presence |0/12|10/12|10/12|
| Enhancing-mask presence |0/12|8/12|7/12|
| Tumor grid |0/6|3/6|2/6|
| Equal-task macro of question accuracy |0%|65.625%|59.375%|
| Equal-task macro of within-patient task means |0%|62.500%|57.292%|
| Worst pooled task accuracy |0%|50.000%|33.333%|
| Mean of four patient question accuracies |0%|66.346%|62.637%|

Full-format and unchanged Vero task-accuracy metrics coincide for these actual A28/B28 outputs because every output passes full format. A0's zero strict score is a format-gated result and must not be interpreted as absence of latent MRI knowledge.

B28 versus A28: **zero newly correct, two newly incorrect, 52 unchanged correctness outcomes**. Two patients worsen and two remain unchanged; none improves. The mean paired patient change is **−3.709 percentage points**. Equal-task question macro changes by **−6.250 points**; equal-task patient macro changes by **−5.208 points**. These are different estimands, not inconsistent arithmetic. Every task covers all four patients; grid question counts are 1/1/2/2, while the other tasks have equal per-patient counts.

## Classwise evidence

| Task / true class | A28 predictions | B28 predictions |
|---|---|---|
| Sequence: FLAIR, n=12 |12 FLAIR|12 FLAIR|
| Sequence: T1ce, n=12 |9 FLAIR, 3 T1ce|9 FLAIR, 3 T1ce|
| Tumor: no, n=6 |5 no, 1 yes|5 no, 1 yes|
| Tumor: yes, n=6 |1 no, 5 yes|1 no, 5 yes|
| Enhancing mask: no, n=8 |5 no, 3 yes|4 no, 4 yes|
| Enhancing mask: yes, n=4 |1 no, 3 yes|1 no, 3 yes|
| Grid: right-middle, n=4 |2 center-middle, 1 right-bottom, 1 right-middle|same counts|
| Grid: center-top, n=1 |1 center-top|1 center-middle|
| Grid: center-middle, n=1 |1 center-middle|1 center-middle|

There are no invalid/unparsed A28/B28 predictions. Sequence predictions remain strongly asymmetric: 21 FLAIR and three T1ce across the balanced 12/12 evaluation. The controlled sequence-example exposure change did not improve sequence accuracy or the overall macro in this observed run. This is evidence against sufficiency of this particular correction at this budget, not proof that class balancing cannot help or that a particular architectural mechanism caused the result.

## Termination and limitations

All 54 B28 outputs record `finish_reason=stop`, zero reported truncations, and 16–20 generated tokens (mean 16.741). Recorded total token counts equal prompt plus generation counts. Historical A0/A28 termination and generation token counts remain unmeasured; character lengths or apparent completeness are not substituted for missing telemetry.

The B intervention changes sequence-example/image/label exposure while retaining the other 84 scheduled examples. It is an SFT engineering control, not Vero RL transfer. The small decreases in two other routes must be reported, but one run on four repeatedly inspected development patients does not establish reproducible negative transfer, causal interference, site generalization or clinical validity. No question-level independence, p-value or confirmatory confidence interval is claimed. Three tumor-related routes share source mask/image provenance.

**At initial-review freeze, B28 no-image and swap controls are `NOT_RUN`.** No conclusion about new B28 pixel dependence follows from primary accuracy. The later same-patient sequence swaps permute existing original image/prompt pairs; their correctness must use the displayed replacement label, and identical-input response consistency must be audited. They do not create independent new patients or an independent accuracy cohort. Historical controls remain separate evidence.

## Evidence links

Raw per-patient outcomes, independent parser, input verification and paired differences remain private in `overnight_method_stats_v1/independent_b28_primary_PRIVATE.json` and `independent_b28_primary.py`.

- Actual B28 outputs SHA-256: `6f7a3bf69600aa424693e31c68a5026edc38f866df09c67048da3a37ba84017b`.
- B28 adapter SHA-256: `b316db3d5eadf2e1fc0cbe80ccc93c7f7405858769f7d32b39a137dea3d6c572`.
- B28 completed training receipt SHA-256: `2d159e48a55f3c93057bdebcc083f544c2f82f9ca56d8d5c6c1403ac04b1e1cb`.
- B28 optimizer-state manifest SHA-256: `ba1f0798372497927fb6ba56fb440a2021bfe4e0f66bb27b66c530511889f33d`.

This review checks final receipt linkage and scoring; the complete optimizer chain is covered by the separate engineering audit. It does not claim a second full-model uninterrupted training replay.

### Subsequent production-analyzer crosscheck

After freezing the independent results above, ran `scripts/analyze_b28_paired.py` on the actual complete artifacts. All **162 checkpoint-question outcomes**, 12 task confusion matrices, full-format correctness, macro definitions, patient means, termination counts and paired differences agree with the independent implementation. The private agreement receipt is `overnight_method_stats_v1/b28_independent_production_agreement_PRIVATE.json`. This crosscheck does not add patients, model runs or independent statistical observations.
