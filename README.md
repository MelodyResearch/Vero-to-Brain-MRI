# Longitudinal Brain MRI Verifiable RL

Independent research inspired by Princeton's [Vero](https://github.com/zlab-princeton/vero). This is **not** an official Princeton collaboration. The falsifiable question is whether task-balanced verifiable RL improves cross-skill transfer in real brain MRI without harming grounding or explanation quality.

## Current evidence (2026-10-15)

| Gate | State | Evidence |
|---|---|---|
| SGMRI-VQA public QA metadata | SOURCE_AUDITED | Official card and file-tree audit in `data/DATASET_ACCESS_AND_LICENSE_AUDIT.md` |
| Legally linked MRI images and patient splits | DATA_BLOCKED | Official listing lacks promised image archives; source images require fastMRI agreement; no patient-disjoint lineage verified |
| BrainTRACE longitudinal image evaluation | DATA_BLOCKED | Public benchmark is test-only; MR-RATE volumes require upstream access |
| Geometry, image integrity, 2D frame IoU, task rewards, sampling, report code | IMPLEMENTED, TESTED_CPU | `src/mri_rl/`; `pytest` |
| Qwen3-VL vs matched Vero inference | NOT_RUN, GPU_BLOCKED, DATA_BLOCKED | No verified image-linked evaluation set; no authorized GPU endpoint found |
| RL model weight updates / transfer matrix | NOT_RUN | No eligible multi-task training set or authorized training compute |
| Scientific hypothesis or method validation | NOT_RUN | No model outcomes exist |

The local machine is an M3 Pro with 36 GB unified memory. It supports CPU integrity checks; this repository makes no claim about GPU inference or RL training performance. Claude Code was found locally but was **not authenticated**; no Claude review occurred. Independent Codex research reviewers examined prior art, data, and statistical design; their findings are in `research/` and `reviews/`.

## Working CPU path

Requires Python 3.11+ and dependencies in `pyproject.toml`.

```bash
python -m pip install -e '.[test]'
pytest -q
mri-rl audit --train local_data/train.jsonl --validation local_data/validation.jsonl --test local_data/test.jsonl
mri-rl render --record local_data/one_record.json --output local_data/one_render.png
mri-rl score --records local_data/test.jsonl --predictions local_data/predictions.jsonl
```

`audit` exits nonzero if any record lacks subject/study lineage, has undecodable images, duplicates an ID, or shares a subject across splits. The strict input schema is in `data/TRAIN_VALIDATION_TEST_CONTRACT.md`. `score` accepts *saved outputs from a real model*; it does not perform model inference. `render` orders real frames by frame index; it does not convert numerical tables to pretend images. NIfTI utilities reorient into RAS+ and retain the affine and voxel spacing.

## Research path

1. Obtain approved fastMRI source images or another legitimately accessible MRI source and link them to the public QA rows. Audit every frame, box and subject split.
2. Freeze an eligible held-out set; run the matched Qwen3-VL and Vero checkpoints with the same rendered evidence and decoding budget.
3. If baseline competence and task diversity justify RL, train same-starting-weight, matched-budget single-task and proportional/uniform mixture arms. Add longitudinal tasks only with actual repeat visits and independent patients.
4. Compare MedGRPO calibration faithfully. Evaluate patient-level transfer intervals, worst-task scores, grounding and hallucination. BrainTRACE remains untouched until image access and a frozen protocol exist.

No MRI images, restricted metadata, source reports, checkpoints or credentials belong in this repository. Local dataset artifacts are gitignored.
