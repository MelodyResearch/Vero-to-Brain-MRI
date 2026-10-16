# Longitudinal Brain MRI Verifiable RL

Independent research inspired by Princeton's [Vero](https://github.com/zlab-princeton/vero). This is **not** an official Princeton collaboration. The falsifiable question is whether task-balanced verifiable RL improves cross-skill transfer in real brain MRI without harming grounding or explanation quality.

## Current evidence (2026-10-15)

| Gate | State | Evidence |
|---|---|---|
| Dallas ds004856 real MRI ingestion | `DATA_VERIFIED` | 20 people, 40 sessions, 44 source-hash-matched volumes: 40 paired T1 and 4 FLAIR; [cohort audit](data/DALLAS_EXPANSION_AUDIT.md) |
| Paired T1 registration and image QC | `TESTED_CPU` | 20/20 provisional numeric passes; 1 artifact hold, 19 not certified; [registration audit](data/LONGITUDINAL_QC_RESULTS.md) |
| Real-image verifier challenges | `TESTED_CPU` | 20/20 correct-pair provenance passes; unregistered grids still pass the coarse gate in 2/20; all biological change claims abstain; [challenge report](results/VERIFIER_CHALLENGE_RESULTS.md) |
| Image-only identity retrieval | `TESTED_CPU` | Spatial T1 match 8/20 versus nonspatial intensity summary 10/20; [shortcut audit](results/IDENTITY_BASELINE_RESULTS.md) |
| Reward/optimizer replay | `TESTED_CPU`, fixture only | Exact-affine, nonlinear calibration, ties and format-only gradients; [replay report](results/REWARD_REPLAY_RESULTS.md) |
| Qwen3-VL versus matched Vero MRI inference | `NOT_RUN` | A local MLX GPU self-test succeeded outside the sandbox; no image-evaluation-eligible biological-change set exists. Checkpoint acquisition/runtime parity is being audited. |
| Multi-task RL and scientific/clinical validation | `NOT_RUN` | No two independently validated MRI rewards, held-out cohort or trained weights. |

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

1. Expand the real Dallas MRI cohort with subject-first splits and acquisition/QC checks. Use image-only versus metadata, blank-image and shuffled-image controls. Keep FreeSurfer-derived change targets indeterminate until a defensible repeatability threshold is frozen.
2. Obtain approved fastMRI source images if pursuing SGMRI-VQA, then link to public QA rows and audit every frame, box and subject split. Freeze an eligible held-out set before matched Qwen3-VL and Vero inference with the same rendered evidence and decoding budget.
3. If baseline competence and task diversity justify RL, train same-starting-weight, matched-budget single-task and proportional/uniform mixture arms. Add longitudinal tasks only with actual repeat visits and independent patients.
4. Compare MedGRPO calibration faithfully. Evaluate patient-level transfer intervals, worst-task scores, grounding and hallucination. BrainTRACE remains untouched until image access and a frozen protocol exist.

No MRI images, restricted metadata, source reports, checkpoints or credentials belong in this repository. Local dataset artifacts are gitignored.
