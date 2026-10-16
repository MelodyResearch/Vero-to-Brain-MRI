# Longitudinal Brain MRI Verifiable RL

Independent research inspired by Princeton's [Vero](https://github.com/zlab-princeton/vero). This is **not** an official Princeton collaboration. The falsifiable question is whether task-balanced verifiable RL improves cross-skill transfer in real brain MRI without harming grounding or explanation quality.

## Primary research direction — Vero-to-MRI recipe transfer

**The researcher's current priority is to transfer and experimentally test Vero's full *training recipe* on genuine brain MRI:** heterogeneous image-grounded task categories, category-balanced sampling, task-routed verifiable rewards, open-ended ability retention, and **actual GSPO model training**. The central endpoint is a held-out **train-task × test-task transfer matrix** against same-data SFT, single-task RL, proportional mixtures and MedGRPO/active-group controls. **A longitudinal-change verifier is one task-specific component, not a gate preventing other eligible MRI categories from being trained.**

**Execution source of truth:** [Vero-to-MRI primary execution contract](research/VERO_TO_MRI_PRIMARY_EXECUTION_CONTRACT.md). Prior literature-review decisions and experiments remain archived as historical evidence. Real-image frozen comparisons, a small SFT update, and three attempted MRI GSPO sampling arms have run. Historical strict Vero-format arms yielded no RL updates; the subsequent post-SFT28 bounded presence diagnostic produced one verified strict task-reward update. No controlled balancing or generalization gain has been demonstrated.

## Current evidence (2026-10-15)

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
