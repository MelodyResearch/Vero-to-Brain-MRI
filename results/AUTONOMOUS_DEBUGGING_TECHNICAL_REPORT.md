# Autonomous debugging: Vero-to-MRI technical progress

**Active sprint: 2026-10-15–16, America/Los_Angeles. Report state: completed B primary and image-control analysis; research ongoing.** B28 training and all **54/54 primary development predictions** are complete and independently audited. B28 scores 34/54 versus historical A28's 36/54; sequence remains 15/24. All 48 frozen image-control generations are complete and independently rescored, with no A28-to-B28 sequence improvement. This snapshot is not the conclusion of the research project.

## 1. Research question and experiment separation

The primary question remains whether Vero's category-balanced sampling, task-routed rewards, open-ended retention and GSPO training recipe transfer to heterogeneous, image-grounded MRI tasks under matched starting weights and budgets. The eventual evidence must include specialist/proportional/uniform comparisons, SFT and reward-normalization controls, a cross-task transfer matrix and genuinely independent patients. The completed 13-paper audit is preserved; it was not repeated for this sprint.

Three evidence streams must remain separate. **A** denotes immutable historical controls. **B** changes 14 sequence images and their corresponding labels in SFT while preserving the other 84 examples. It is a sequence-example/label exposure control, not a text-only label intervention. **C** develops a separately versioned, slice-aware task cohort. B is a controlled supervision experiment; C is task/data preparation. Neither is a completed Vero RL transfer experiment. See the [research covenant](../research/VERO_MRI_RESEARCH_COVENANT.md).

## 2. Actual B28 training and exact scope

The corrected `balanced_sequence_v2` run completed **28 optimizer updates and 112 image/question presentations from 28 training patients**. It started from the historical common initialization with seed 7. Sequence presentations were 14 FLAIR and 14 contrast-enhanced T1, replacing historical A28's 28 FLAIR/0 T1 exposure. The intervention replaced 14 FLAIR images with T1ce images and changed their labels accordingly. The other three routes retained their original 84 examples; B does not repair their slice-selection or clinical-answerability limitations.

| Parameter | Executed B configuration |
|---|---|
| Base | Qwen3-VL-8B; local MLX path |
| Precision/adaptation | 4-bit base, group size 64; rank-4 language LoRA, alpha 8 (scale 2.0); frozen vision |
| Trainable adapter inventory | 504 tensors |
| Optimizer | Adam; learning rate 1e-5; betas (0.9, 0.999); epsilon 1e-8; bias correction disabled; no weight decay; constant schedule |
| Exposure | Four task-route presentations per update; 28 updates, 112 total |
| Sequence labels | 14 FLAIR / 14 contrast-enhanced T1 |
| Presence labels | 14 no / 14 yes |
| Enhancing-mask labels | 17 no / 11 yes |
| Checkpoints | Adapter, complete Adam state, RNG and schedule cursor at each completed boundary |
| Final boundary | Cursor 112; 504 finite changed adapter tensors; 504 Adam moment pairs |

The final update had gradient L2 **1.367220902**. The independently recomputed mean of all 112 teacher-forced training losses was **2.6830218647**; this is a training diagnostic, not an evaluation score. The final adapter SHA256 is:

`b316db3d5eadf2e1fc0cbe80ccc93c7f7405858769f7d32b39a137dea3d6c572`

The complete 28-boundary chain was independently rehashed against preceding adapters, optimizer-state checksums, consumed approvals, frozen schedule prefixes and cumulative receipts. Tiny native-fixture restored-versus-uninterrupted equality and real state restoration support continuity within the tested scope. **Uninterrupted full-Qwen numerical equivalence was not measured.** The independent verdict is `PASS_WITH_LIMITATIONS`, not a scientific learning verdict. Source: [engineering review](OVERNIGHT_ENGINEERING_REVIEW.md), private `overnight_resume_review_v1/full28_chain_independent_verification.json`, and the immutable final training receipt under `balanced_sft_resume_step028_v1`.

## 3. Complete matched primary evaluation and image controls

The primary manifest is frozen at **54 questions from four development patients**: 24 sequence, 12 tumor presence, 12 enhancing-mask presence and six grid questions. It uses the same native chat template, system instruction, image bytes, 4-bit base and greedy 256-token decoding profile for the compared local adapters. These are internal-development patients, not a fresh prospective test cohort.

| Checkpoint / condition | Full strict format | Strict task-correct | Sequence | Presence | Enhancement | Grid |
|---|---:|---:|---:|---:|---:|---:|
| Historical A0, matched system instruction | 0/54 | 0/54 | 0/24 | 0/12 | 0/12 | 0/6 |
| Historical A28, matched system instruction | 54/54 | 36/54 | 15/24 | 10/12 | 8/12 | 3/6 |
| Corrected B28 | **54/54** | **34/54** | **15/24** | **10/12** | **7/12** | **2/6** |

A0's zeros are strict-contract results; they do not demonstrate zero semantic answer knowledge. Historical A28's four-patient mean accuracy was 0.6635. Its user-only prompt probe produced 0/54 strict wrappers with the same weights, whereas restoring the training system instruction produced 54/54. This verified prompt mismatch must not be confused with the separate effect of sequence-label exposure.

Historical A28 sequence controls scored **15/24** with original images, **12/24** without images, and **15/24** with the opposite sequence from the same patient and slice. Only **6/24** swaps changed the answer in the correct direction. These historical observations motivated B and are now matched by B's completed controls below. See [historical matched evaluation](FORMAT_SFT_INTERNAL_DEV_EVALUATION.md).

The [independent primary review](B28_PRIMARY_INDEPENDENT_REVIEW.md) reproduced all 162 checkpoint-question outcomes and subsequently agreed with the production analyzer. B versus A28 has **zero newly correct, two newly incorrect and 52 unchanged correctness outcomes**. Two of four patients worsen and two remain unchanged. Mean patient accuracy falls from 66.346% to 62.637% (−3.709 percentage points). Equal-task question macro is 65.625% versus 59.375%; equal-task macro of within-patient task means is 62.500% versus 57.292%; worst pooled task accuracy is 50.000% versus 33.333%. These summarize different weighting schemes and must not be interchanged.

Both A28 and B predict 21 FLAIR and three T1ce on the balanced 12/12 sequence set. B's 14/14 supervised exposure therefore did not improve this run's sequence accuracy or remove its observed class asymmetry. This refutes sufficiency of this specific exposure correction at the tested budget; it does not prove class balancing is generally ineffective or establish an architectural cause. The two other-task errors are reported as small observed declines, not reproducible negative transfer.

All 54 B outputs recorded `finish_reason=stop`, zero reported truncations and 16–20 generation tokens (mean 16.741). Historical A0/A28 token/termination telemetry is missing and remains unknown. The B output digest is `6f7a3bf69600aa424693e31c68a5026edc38f866df09c67048da3a37ba84017b`.

### Completed image controls

The [independent complete control review](B28_IMAGE_CONTROL_INDEPENDENT_REVIEW.md) and its [aggregate evidence](B28_CONTROL_INDEPENDENT_AGGREGATE.json) reproduce the production analysis from all actual frozen outputs, with `PASS_WITH_LIMITATIONS` for input linkage and descriptive scoring.

| Sequence condition | Historical A28 | Corrected B28 | Full format, each checkpoint |
|---|---:|---:|---:|
| Original image | 15/24 | 15/24 | 24/24 |
| No image | 12/24 | 12/24 | 24/24 |
| Same-patient/slice opposite sequence | 15/24 | 15/24 | 24/24 |
| Correct directional answer changes | 6/24 | 6/24 | — |

Both checkpoints answer FLAIR for all 24 no-image prompts. There are **zero answer or correctness changes from A28 to B28 across all 72 sequence-condition rows**, with zero patient-level differences in each condition. The six directed changes correspond to **three of 12 unordered image pairs**, not six independent pairs. All 24 swapped responses exactly match the original response for the identical visible image and prompt, for both checkpoints. The 24 no-image prompts are identical and yield one distinct response per checkpoint. These are implementation/paired image controls, not additional independent patients or a new accuracy cohort.

All **102 B generations** (54 primary plus 48 controls) record `stop`, with zero recorded length terminations. Historical termination remains unknown. The complete controls provide no improvement in observed sequence accuracy or the measured image-dependence pattern after the B exposure correction. They do not establish a capacity limit or prove that the model has no MRI understanding. Four repeatedly inspected development patients remain the scope of inference.

B no-image output SHA256: `6226bb9ca175e06bf7a59f775e32e4316ad370f94d1275b1b6d44b2ee460f599`; B swap output SHA256: `58e401f28f25acd359b6b8fadce63145e967c5d1480b5a9e01316ee9b099d5ee`.

### Target exposure and online training losses

A bounded tokenizer/source audit verified B's 14/14 targets, exact saved target hashes and assistant-only loss indexing. Each FLAIR target has 16 tokens (14 fixed observation/format, two answer); each T1 target has 19 (14 fixed, five answer). A uses 1,736 supervised tokens overall and B 1,778, a **42-token increase**. Updates/presentations are matched; **supervised token counts and wall time are not equal**.

Saved sequence mean losses are 2.816188 for A and 2.755610 for B; last-four sequence means are 0.179735 and 0.248374. These are **ONLINE whole-target losses at evolving checkpoints**, not final-B28 teacher-forced evaluation. Per-token losses were not saved, so token counts cannot establish answer-loss or gradient dominance. The native assistant prefix contains no prefilled think tag, direct loss excludes prompt tokens, and the target does not append EOS. Missing EOS supervision is a source fact, not an established defect; all 54 primary B outputs stopped normally. The significance of absent EOS supervision remains unmeasured. Historical A labels are reconstructed, whereas B target hashes were logged. An independent reviewer reproduced all 224 target lengths, 112 B target hashes, 84 unchanged nonsequence examples, token partitions and scalar-loss summaries. See [token exposure diagnostic](B28_TARGET_EXPOSURE_DIAGNOSTIC.md).

## 4. Actual C data expansion and shortcut results

The frozen C2 geometry matches now have real images: **32 tumor-presence rows and 12 enhancing-mask rows, across 35 distinct patients and 38 unique PNG hashes**. Nine patients occur in both categories. The selected new-source union contains ten patients; its 20 FLAIR/T1ce volumes total **124,875,617 bytes**, including a previously verified two-volume pair. The newly fetched 18 files total **114,557,831 bytes**. Exact official annex hashes, sizes, source exclusions and mask/image geometry were checked. Eleven paired PNG frames were newly rendered in an explicitly reserved model-idle window.

Labels retained the ≥1,000-native-mask-pixel positive threshold and exactly-zero negatives. Tumor uses levels 45/91/136 with 20/8/4 rows; enhancement uses levels 45/91 with 4/8 rows. Every included level is class-balanced; enhancement still lacks eligible positive support at level 136. The source 40 and reserved prospective 14 were excluded from new-source selection. Geometry eligibility is not clinical visibility.

| C2 control | Tumor | Enhancing mask |
|---|---:|---:|
| Fixed no / level-only majority | 16/32 | 6/12 |
| Site-only same-sample association | 21/32 | 9/12 |
| Site × level same-sample association | 22/32 | 9/12 |
| Fixed 16×16 grayscale thumbnail LOPO 1NN | 18/32 | 5/12 |
| Fixed 16-bin grayscale histogram LOPO 1NN | 18/32 | 6/12 |

The PNG controls reused exact C1 features without tuning. Neighbors were restricted to the same task and excluded the query patient. All **88 predictions**, neighbor IDs, distances and confusion counts were independently reproduced; no same-patient neighbor or within-category duplicate image hash was found. The actual single-thread PNG pass took 0.22 seconds and blocked MLX/Torch/volume imports. Site associations remain descriptive resubstitution results. The LOPO controls are internal cross-validation on a selected training cohort; they are not fresh prospective validation or lesion-understanding tests. C remains `ENGINEERING_ONLY`, with clinical visibility `NOT_VERIFIED`. See [C2 evidence and mission check](C2_IMAGE_LINKED_COHORT_RESULTS.md) and [methods review](OVERNIGHT_METHOD_STATS_REVIEW.md).

## 5. Corrected defects and remaining barriers

| Issue | Actual resolution or outstanding scope |
|---|---|
| SFT adapter-only continuation loses optimizer state | Complete SFT state restore and boundary checks implemented; tiny-fixture parity and actual B28 chain independently verified. Full-Qwen uninterrupted parity remains unmeasured. |
| Training/inference system-prompt mismatch | Reproduced with the same historical A28 checkpoint; matched native prompt contract retained. |
| Fixed manifest paths did not freeze pixels | Failing regression reproduced acceptance of altered pixels; primary evaluator now verifies current images/prompts/IDs/categories/profile against both immutable A0/A28 output receipts. Independent recheck passed. Actual historical pixels had not drifted. |
| Sequence exposure confound | B28 executed the prespecified 14/14 intervention. Complete primary and image-control evaluation show no improvement in the observed sequence performance or image-dependence pattern. |
| Local RL optimizer restart | **Unresolved for the old RL driver.** It creates Adam per prompt group; SFT repairs do not repair this path. A minimal continuous-state correction is proposed, not launched. |
| Full Vero recipe fidelity | Local LoRA/MLX, short rollouts, serial update schedule, optimizer settings, active-group diagnostic selection and absent judge-backed open-ended retention differ from upstream. Full recipe parity and transfer matrix remain unrun. |

The lead reports the current CPU regression suite at **253 passed, two skipped**. Skips are not executed validation; this report author did not rerun the full suite during model inference. Focused C2/C1 tests passed with heavyweight imports blocked. The [root-cause registry](../research/ROOT_CAUSE_REGISTRY.md) and [RL continuity proposal](../research/RL_CONTINUITY_MINIMAL_CORRECTION.md) preserve evidence and remaining limitations. A single on-policy update at approximately unit ratio does not empirically validate clipping under a changed policy.

**New RL status:** no new RL run was launched in this sprint. B28 group-wise RL reward variance is `NOT_MEASURED`; greedy primary correctness and training losses do not estimate stochastic rollout-group variance. Historical RL experiments remain separately documented.

## 6. Independent verification and resource scope

A genuine authenticated Claude Code review used **public upstream Vero excerpts only**, at commit `025ca3c800fc95224c86323c1f7fd33dceed3090`, with tools disabled. Codex froze its independent initial review before comparing disagreements. Small exact-helper/scalar fixtures resolved selected claims; inferred runtime/security consequences remain conditional. See the [public Codex–Claude comparison](CODEX_CLAUDE_PUBLIC_COMPARISON.md). Claude did not receive private MRI data or B/C receipts. **Private-interface cross-model review remains `NOT_VERIFIED`; isolated Codex reviewers performed the private engineering/statistical checks.**

One heavy model process ran at a time, with bounded stages, state saving and live resource gates. The [final control engineering audit](B28_CONTROLS_ENGINEERING_AUDIT.json) gives `PASS_WITH_LIMITATIONS` for all **48 controls across 12 stages and 144 resource decisions**, with zero HOLD decisions. Minimum recorded available memory was 9.196 GiB, minimum pressure-free fraction 50%, and maximum MLX peak 6.845 GiB. Every recorded stage was on AC with no other heavy worker.

After the final model exited, fresh checks showed **15.072 GiB available, pressure-free 66%, stable swap 9.325 GiB, AC 100% and zero matching model workers**. Thermal telemetry remained unavailable. No further model launch is approved by this audit. These observations do not identify the prior kernel-panic cause or guarantee future stability.

## 7. Three separate significance judgments and next decision

**Scientific significance:** clinical MRI benefit is not established. C now supports verifiable image-linked engineering labels and exposes residual site associations; visual answerability and independent task capabilities remain to be established. Binary tumor routes share mask lineage.

**AI/ML significance:** no full Vero transfer benefit, novel MRI optimizer, negative-transfer mechanism or balanced-mixture advantage is established. B28 creates a reproducible matched supervision control; the complete primary and image controls show no sequence accuracy or measured image-dependence gain in this run.

**Experimental significance:** substantive progress is observed: completed real-image training with an independently audited state chain, frozen comparison interfaces, actual C image expansion and independently reproduced pixel baselines. These reduce specific confounds and enable the next controlled experiment.

The next proposed decisive test is a frozen **training-only** answer-likelihood and generation diagnostic from saved checkpoints, under real, no-image and same-patient sequence-swap inputs. It should save answer-token versus fixed-token losses and answer likelihood margins, to distinguish training underfit, label-prior preference and failure to generalize. No such new model test has run, and none of these mechanisms is established by current token counts. Separately, the confirmed RL optimizer-continuity defect needs a minimal implementation correction, regression and independent review before specialist/proportional/uniform comparisons from common weights. C task admission, open-ended retention, MedGRPO-style controls and untouched-patient evaluation remain required. No full-transfer or clinical-completion claim is warranted.

## 8. Reproducibility checkpoint

The actual repository baseline for reproducing this sprint is commit `39a79a41acf9320b4adb077d942db45dd0dc0496`. It precedes final report integration and must not be described as containing its own subsequent report edits. The lead will supply the final integration SHA in the delivery message. Raw images, source identities, labels, predictions and checkpoint payloads remain private outside Git. Versioned reports in the private repository contain permitted aggregate evidence and artifact digests.
