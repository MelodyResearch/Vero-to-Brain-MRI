# Longitudinal Brain MRI Verifiable RL

Independent research inspired by Princeton's [Vero](https://github.com/zlab-princeton/vero). This is **not** an official Princeton collaboration. The falsifiable question is whether task-balanced verifiable RL improves cross-skill transfer in real brain MRI without harming grounding or explanation quality.

## Primary research direction — Vero-to-MRI recipe transfer

**The researcher's current priority is to transfer and experimentally test Vero's full *training recipe* on genuine brain MRI:** heterogeneous image-grounded task categories, category-balanced sampling, task-routed verifiable rewards, open-ended ability retention, and **actual GSPO model training**. The central endpoint is a held-out **train-task × test-task transfer matrix** against same-data SFT, single-task RL, proportional mixtures and MedGRPO/active-group controls. **A longitudinal-change verifier is one task-specific component, not a gate preventing other eligible MRI categories from being trained.**

**Execution source of truth:** [Vero-to-MRI primary execution contract](research/VERO_TO_MRI_PRIMARY_EXECUTION_CONTRACT.md). Prior literature-review decisions and experiments remain archived as historical evidence. Real-image frozen comparisons, a small SFT update, and three attempted MRI GSPO sampling arms have run. Historical strict Vero-format arms yielded no RL updates; the subsequent post-SFT28 bounded presence diagnostic produced one verified strict task-reward update. No controlled balancing or generalization gain has been demonstrated.

## Latest execution (2026-10-17)

The project remains active. These results supersede the incomplete cursors in the archived snapshot below.

| Experiment | Actual status | Verified evidence and limits |
|---|---|---|
| A0 / historical A28 / balanced B28 likelihood diagnosis | `INFERENCE_COMPLETED`: 57/57 each, 171 total | [Independent numerical verification](results/COMPLETE171_INDEPENDENT_NUMERICAL_VERIFICATION.json), [scientific findings](results/LIKELIHOOD_COMPLETE171_SCIENTIFIC_FINDINGS.md). Training-image diagnostics show a dominant B28 answer-prior shift; they do not establish independent-patient generalization. |
| Native strict-reward GSPO engineering update | One genuine update on one uncertainty-screened training image | [Independent Adam/weight verification](results/NATIVE_T1_G8_ONE_UPDATE_INDEPENDENT_VERIFICATION.json), [actual fresh native state restore](results/NATIVE_G8_ACTUAL_RESTORE_INDEPENDENT_VERIFICATION.json). All 504 adapter tensors changed; Adam advanced0→1. This selected engineering diagnostic is not a primary training arm or a demonstrated accuracy gain. |
| Common-B28 two-task internal-development baseline | `INFERENCE_COMPLETED`:24 authenticated original sequence rows + 16 new Dallas rows | [Actual baseline](results/C40_B28_COMMON_START_BASELINE.json): sequence 15/24 (62.5%, 4 patients); correspondence 8/16 (50%, 8 people / 4 cycles); equal-task macro 56.25%; format 40/40. Dallas predicts **different for every case**, matching its class-prior baseline. Old rows retain their original 13 approvals; none were regenerated or relabeled. |
| First finite two-task GSPO arm: U_raw, seed7 | `IN_PROGRESS`, not completed | [Exact scoped release](results/FIRST_FULL_U_RAW_SEED7_SCOPED_RELEASE.json): common B28, frozen 76-row pool, group 4, 169216 native-input-plus-generated-token cap, maximum 198 occurrences. One resident model saves adapter, full Adam/RNG and a durable receipt after each group. Zero-variance groups are recorded without an update. |
| Remaining RL arms, same-C-data SFT and final transfer matrix | `NOT_RUN` until their actual receipts exist | [Frozen finite protocol](research/execution/FULL_GSPO_TWO_TASK_FINITE_PROTOCOL_V1.md). CPU implementations, scoped approvals and numerical checks are not completed training. Prospective test pixels remain sealed. |

Authentic Claude Code implemented the new training interfaces; independent Codex reviewers reproduced and checked the corrections. Codex locally executed the new 16 baseline predictions and the current full arm while Claude's session quota was exhausted. No Claude raw-result review or denied-data access is claimed. The MRI correspondence task uses whole-head views and retains face/skull/neck and source-confounding limitations; it does not isolate brain morphology or establish clinical validity. The local language-LoRA/Metal study is not the official distributed Vero CUDA actor/vision configuration, and the token cap is not an exact FLOP match.

## Archived evidence snapshot (2026-10-15)

| Gate | State | Evidence |
|---|---|---|
| Dallas ds004856 real MRI ingestion | `DATA_VERIFIED` | 20 people, 40 sessions, 44 source-hash-matched volumes: 40 paired T1 and 4 FLAIR; [cohort audit](data/DALLAS_EXPANSION_AUDIT.md) |
| MSD Task04 Hippocampus | `TESTED_CPU` | 260 real MRI/mask pairs verified; 520 image-only task records, but unknown patient linkage and 94.6% dominance majority baseline; [task audit](data/MSD_TASK04_TASK_AUDIT.md) |
| MSD Task01 BrainTumour | `TESTED_CPU` | 8 verified four-sequence MRI/mask pairs and 40 fixed-midpoint prompts; no patient crosswalk; [task audit](data/MSD_TASK01_8CASE_AUDIT.md) |
| OpenNeuro ds007045 patient-keyed MRI | `DATA_VERIFIED` | 40 public BIDS patients with hash-verified T1ce, FLAIR and reviewed tumor masks; 32 source train / 8 heldout, with four image-linked task routes; [eligibility audit](data/PATIENT_KEYED_MRI_ELIGIBILITY.md). |
| Paired T1 registration and image QC | `TESTED_CPU` | 20/20 provisional numeric passes; 1 artifact hold, 19 not certified; [registration audit](data/LONGITUDINAL_QC_RESULTS.md) |
| Real-image verifier challenges | `TESTED_CPU` | 20/20 correct-pair provenance passes; unregistered grids still pass the coarse gate in 2/20; all biological change claims abstain; [challenge report](results/VERIFIER_CHALLENGE_RESULTS.md) |
| Image-only identity retrieval | `TESTED_CPU` | Spatial T1 match 8/20 versus nonspatial intensity summary 10/20; [shortcut audit](results/IDENTITY_BASELINE_RESULTS.md) |
| Reward/optimizer replay | `TESTED_CPU`, fixture only | Exact-affine, nonlinear calibration, ties and format-only gradients; [replay report](results/REWARD_REPLAY_RESULTS.md) |
| Matched base and released-Vero frozen MRI inference | `INFERENCE_COMPLETED` | Same 24 image-linked questions plus six matched no-image/blank controls per model. Shared-template base 12/24 versus Vero 9/24; Vero tumor-grid format invalid in 7/8. This small source-confounded diagnostic is **not** recipe transfer; [independent audit](results/INDEPENDENT_FROZEN_MODEL_AUDIT.md). |
| Official Vero GSPO MRI adapter | `IMPLEMENTED`, `TESTED_CPU` | Pinned veRL source, real-image JSONL/reward callback and Hydra dry run; CUDA hardware unavailable for upstream launcher; [adapter audit](results/VERO_OFFICIAL_TRAINING_ADAPTER.md). |
| Patient-heldout base versus released Vero | `INFERENCE_COMPLETED` | On ds007045's 107 fixed prompts from eight heldout patients, base 45 correct versus Vero 44; four-category patient macro 33.3% versus 32.8%, below the 45.3% training-majority macro. Image-swap and blank controls show answer priors; [independent audit](results/DS007045_INDEPENDENT_FROZEN_AUDIT.md). |
| Four-step image SFT control | `TRAINING_COMPLETED`, exploratory | 16 real-image examples from 14 training patients changed 4-bit LoRA weights. Against its **matched 4-bit starting adapter**, heldout correct rose 53/107 to 54/107, entirely one grid answer; [audit](results/DS007045_INDEPENDENT_FROZEN_AUDIT.md). |
| Strict Vero-style MRI RL sampling arms | `ROLLOUTS_COMPLETED`, `NO_UPDATE` | Specialist, proportional and uniform each ran eight groups/two rollouts on real training images from the same SFT start. All 48 rewards were zero under the strict format gate, so all final weights remained identical; [independent audit](results/DS007045_MULTIARM_RL_AUDIT.md). Official CUDA veRL training remains unavailable on this Metal host. |
| Explicit single-box reward ablation | `TRAINING_COMPLETED`, exploratory | Proportional and uniform mixtures obtained one and two task-driven 4-bit LoRA updates. On 107 heldout prompts they scored 54/107 and 53/107 versus 54/107 for the same SFT start; one uniform grid answer regressed. Active-update and token budgets differ, so this does **not** establish a balancing effect; [audit](results/DS007045_MULTIARM_RL_AUDIT.md). |
| 28-step format SFT and matched internal development | `TRAINING_COMPLETED`, `INFERENCE_COMPLETED`, exploratory | From 28 train patients, 112 real-image examples and 28 verified optimizer steps changed LoRA weights. On 54 questions from four disjoint development patients, the same-prompt starting adapter had 0/54 strict-valid, the final adapter 54/54 strict-valid and 36/54 correct. Its first user-only probe had 0/54 strict-valid because it omitted the SFT system message; [paired audit](results/FORMAT_SFT_INTERNAL_DEV_EVALUATION.md). |
| SFT28 sequence pixel-dependence controls | `INFERENCE_COMPLETED`, exploratory | On 24 balanced development sequence questions, real image 15/24 versus no-image 12/24 (always FLAIR); only 6/24 same-patient/slice opposite-sequence swaps caused a correct directional answer change. This is partial, weak visual dependence; [control audit](results/FORMAT_SFT_INTERNAL_DEV_EVALUATION.md). |
| Tiered execution policy and independent review | `TESTED_CPU`; scoped Tier B runs executed | Versioned A/B/C policy replaces unconditional rejection. Separate Codex review binds exact contracts, code, adapters and budgets; stale/altered evidence fails. The first capped diagnostic does not require a previous update. Claude did not review these runs. See [preflight](results/EXPERIMENT_CONTRACT_PREFLIGHT.md). |
| Post-SFT28 strict task-reward activation | `TRAINING_COMPLETED`, exploratory | Sequence probes: 14/14 strict outputs, all FLAIR, zero updates. A separately reviewed same-level tumor-presence diagnostic produced one strict task-driven update, finite gradient 1.3903, changed weights and fresh exact checkpoint reload. On fixed internal dev, 36→37/54 correct, equal-category accuracy unchanged, grid 3→2/6. See [activation results](results/STRICT_REWARD_ACTIVATION_SFT28.md). |
| Historical sampling defect and separate B/C designs | `TESTED_CPU` | Sequence/presence/enhancement all used axial z45. B changes sequence exposure to 14/14 and preserves all other 84 examples. Separate slice-aware C is not yet admitted: binary constraints are infeasible and visual adjudication is pending. [Sampling audit](results/HISTORICAL_SAMPLING_AND_ADMISSION.md). |
| Post-panic paced SFT compatibility | `TRAINING_COMPLETED`, one-step engineering prefix | Four real training images, one update, 252/504 tensors changed, exact fresh reload of all 504 tensors. Strict reward on the single reload output remains zero. Corrected B28 and cross-process exact continuation remain incomplete. [Actual result](results/PACED_SFT_PREFIX1_RESULT.md), [independent review](results/SAFE_RESUME_INDEPENDENT_REVIEW.md). |
| Full Vero recipe and scientific/clinical validation | `NOT_RUN` | The official CUDA GSPO actor/vision training, matched effective-update comparisons, an expert-valid open-ended retention judge and independent clinical validation remain outstanding. |

The local machine is an M3 Pro with 36 GB unified memory. MRI payloads and patient-linked receipts remain ignored locally. Earlier authenticated Claude Code aggregate/code-only reviews remain archived, but **Claude did not review the new ds007045 SFT28 results** because an unpublished-data sharing attempt was rejected before execution. This round used independently identified Codex agents. The [source audit](data/INDEPENDENT_REFERENCE_SOURCE_AUDIT.md) identifies a restricted-registration MIRIAD scan–rescan route for measuring noise before training. The project remains active; passing data and engineering gates is not evidence of Vero transfer.

## Working CPU path

Requires Python 3.11+ and dependencies in `pyproject.toml`.

```bash
python -m pip install -e '.[test]'
pytest -q
mri-rl audit --train local_data/train.jsonl --validation local_data/validation.jsonl --test local_data/test.jsonl
mri-rl render --record local_data/one_record.json --output local_data/one_render.png
mri-rl score --records local_data/test.jsonl --predictions local_data/predictions.jsonl
PYTHONPATH=src python -m mri_rl.dallas_pilot --images-root local_data/ds004856 --bids-root PATH_TO_PINNED_DS004856_METADATA --output-dir local_data/ds004856_evidence
```

`audit` exits nonzero if any record lacks subject/study lineage, has undecodable images, duplicates an ID, or shares a subject across splits. The strict input schema is in `data/TRAIN_VALIDATION_TEST_CONTRACT.md`. `score` accepts *saved outputs from a real model*; it does not perform model inference. `render` orders real frames by frame index; it does not convert numerical tables to pretend images. NIfTI utilities reorient into RAS+ and retain the affine and voxel spacing.

## Research path

The project-local [paper-reading protocol](research/methodology/NORMAN_PAPER_READING_PROTOCOL.md) and [13 primary-paper audits](research/paper_reading/) reconstruct research choices, check claims against full texts and reported tables, and propose falsifying MRI experiments. The [independent Claude review](research/paper_reading/CLAUDE_INDEPENDENT_METHOD_REVIEW_RAW.md) is archived with its source-access limits; cross-paper conclusions reconcile it with the PDF audits. The installed global `$norman-paper-reading` skill and the retained project-local adaptation informed these reviews; the global skill was not reinstalled or overwritten.

1. Preserve historical SFT28, strict-reward activation, the interrupted corrected run, and the completed paced one-step prefix. Verify optimizer/RNG/sampler restoration before a separately reviewed one-update continuation. Keep label-only B and slice-aware C distinct; no automatic full-run expansion. Then progress toward same-starting-weight specialist, proportional and uniform RL with matched attempted/informative updates and visual-token budgets.
2. Improve visibly answerable labels and image-grounding controls before any scientific transfer claim. Four internal-development patients support diagnostics; the previously examined eight patients are historical exploratory evidence, not an untouched confirmatory test. Obtain an independent frozen patient cohort for generalization.
3. Test open-ended retention with an independent factual rubric and compare MedGRPO calibration at a group size where it can affect training. Add longitudinal change only when its target is defensible; BrainTRACE remains untouched until image access and a frozen protocol exist.

No MRI images, restricted metadata, source reports, checkpoints or credentials belong in this repository. Local dataset artifacts are gitignored.
