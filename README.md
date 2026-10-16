# Longitudinal Brain MRI Verifiable RL

Independent research inspired by Princeton's [Vero](https://github.com/zlab-princeton/vero). This is **not** an official Princeton collaboration. The falsifiable question is whether task-balanced verifiable RL improves cross-skill transfer in real brain MRI without harming grounding or explanation quality.

## Primary research direction — Vero-to-MRI recipe transfer

**The researcher's current priority is to transfer and experimentally test Vero's full *training recipe* on genuine brain MRI:** heterogeneous image-grounded task categories, category-balanced sampling, task-routed verifiable rewards, open-ended ability retention, and **actual GSPO model training**. The central endpoint is a held-out **train-task × test-task transfer matrix** against same-data SFT, single-task RL, proportional mixtures and MedGRPO/active-group controls. **A longitudinal-change verifier is one task-specific component, not a gate preventing other eligible MRI categories from being trained.**

**Execution source of truth:** [Vero-to-MRI primary execution contract](research/VERO_TO_MRI_PRIMARY_EXECUTION_CONTRACT.md). Prior literature-review decisions and experiments remain archived as historical evidence, not silent replacements for this researcher-directed objective. Real-image Qwen3-VL base inference has now run; matched released-Vero inference, weight-updating GSPO and a cross-task matrix remain pending verified receipts.

## Current evidence (2026-10-15)

| Gate | State | Evidence |
|---|---|---|
| Dallas ds004856 real MRI ingestion | `DATA_VERIFIED` | 20 people, 40 sessions, 44 source-hash-matched volumes: 40 paired T1 and 4 FLAIR; [cohort audit](data/DALLAS_EXPANSION_AUDIT.md) |
| MSD Task04 Hippocampus | `TESTED_CPU` | 260 real MRI/mask pairs verified; 520 image-only task records, but unknown patient linkage and 94.6% dominance majority baseline; [task audit](data/MSD_TASK04_TASK_AUDIT.md) |
| MSD Task01 BrainTumour | `TESTED_CPU` | 8 verified four-sequence MRI/mask pairs and 40 fixed-midpoint prompts; no patient crosswalk; [task audit](data/MSD_TASK01_8CASE_AUDIT.md) |
| Paired T1 registration and image QC | `TESTED_CPU` | 20/20 provisional numeric passes; 1 artifact hold, 19 not certified; [registration audit](data/LONGITUDINAL_QC_RESULTS.md) |
| Real-image verifier challenges | `TESTED_CPU` | 20/20 correct-pair provenance passes; unregistered grids still pass the coarse gate in 2/20; all biological change claims abstain; [challenge report](results/VERIFIER_CHALLENGE_RESULTS.md) |
| Image-only identity retrieval | `TESTED_CPU` | Spatial T1 match 8/20 versus nonspatial intensity summary 10/20; [shortcut audit](results/IDENTITY_BASELINE_RESULTS.md) |
| Reward/optimizer replay | `TESTED_CPU`, fixture only | Exact-affine, nonlinear calibration, ties and format-only gradients; [replay report](results/REWARD_REPLAY_RESULTS.md) |
| Qwen3-VL base real MRI inference | `INFERENCE_COMPLETED` | Local Metal execution on 24 real image-linked Task04, Dallas and Task01 questions plus 6 no-image/blank controls; [diagnostic](results/FROZEN_MODEL_DIAGNOSTIC.md). These are small engineering probes, not held-out science. |
| Released Vero MRI inference | `INFERENCE_COMPLETED` for one-image compatibility; full paired scores pending | Pinned checkpoint generated on real Task04 MRI. Its native template's long response exceeded 256 tokens before the final answer; shared-template sensitivity and the 24-question comparison are in progress. |
| Official Vero GSPO MRI adapter | `IMPLEMENTED`, `TESTED_CPU` | Pinned veRL source, real-image JSONL/reward callback and Hydra dry run; CUDA hardware unavailable for upstream launcher; [adapter audit](results/VERO_OFFICIAL_TRAINING_ADAPTER.md). |
| Multi-task RL and scientific/clinical validation | `NOT_RUN` | No completed model weight updates, patient-independent matrix or eligible open-ended judge. |

The local machine is an M3 Pro with 36 GB unified memory. MRI payloads and patient-linked receipts remain ignored locally. Authenticated Claude Code completed an independent aggregate scientific critique and a code-only reward review; neither received subject-linked data. The [source audit](data/INDEPENDENT_REFERENCE_SOURCE_AUDIT.md) identifies a restricted-registration MIRIAD scan–rescan route for measuring noise before training. The project remains active; passing data and engineering gates is not evidence of Vero transfer.

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

1. Finish the matched base versus released-Vero frozen comparison on the same verified Task04, Task01 and Dallas images, with identical prompts/decoding and visual shortcut controls.
2. Extend independently labeled MRI categories and resolve patient lineage. Keep Task04 shared-mask tasks and Task01 eight-volume probes as engineering data until that happens.
3. Run a bounded weight-changing GSPO engineering smoke, then same-starting-weight SFT, specialist, proportional and uniform task mixtures when patient-independent data and compatible compute permit. Evaluate all models across all admitted categories.
4. Compare MedGRPO calibration and active-group filtering at matched effective updates. Add longitudinal change as one category only when its target is defensible. BrainTRACE remains untouched until image access and a frozen protocol exist.

No MRI images, restricted metadata, source reports, checkpoints or credentials belong in this repository. Local dataset artifacts are gitignored.
