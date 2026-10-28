# Vero-to-Brain-MRI

**Verifiable Reinforcement Learning for Brain MRI Visual Reasoning**

An independent research project investigating how Princeton Vero-style heterogeneous verifiable reinforcement learning and GSPO can be adapted to improve image-grounded brain MRI reasoning.

This project is independent. It is not an official Princeton collaboration and does not use released Vero weights. The local recipe is Qwen3-VL-8B with a B28 language-LoRA adapter, trained and evaluated with MLX. It is not an official veRL-equivalent implementation.

## Overview

Vero trains a vision-language model on heterogeneous, image-grounded task categories with task-routed verifiable rewards and GSPO. This project asks whether that recipe, adapted with evidence, improves real brain MRI accuracy and training efficiency over direct Vero-style transfer and matched controls.

The authoritative route and next experiment are in the [current research plan][plan] (immutable link; not yet on `main`).

## Research Objectives

1. Test direct Vero-style transfer: category-balanced sampling, task-routed verifiable rewards and GSPO on real MRI images.
2. Measure the train-task × test-task transfer matrix, including negative transfer, against matched controls (SFT, single-task and mixture RL).
3. Evaluate on independent patients, with uncertainty clustered by patient or matched cycle.
4. Adapt only by naming the Vero mechanism changed and comparing with direct transfer; supervised-loss gains are not GSPO gains.

A complete heterogeneous-transfer claim needs at least two scientifically eligible MRI visual tasks. Sequence identification (FLAIR vs. contrast-enhanced T1) is the currently eligible scientific task; Dallas is used only as an engineering control.

## Current Progress

- **Training.** 18 independent RL arms and 3 SFT controls completed, all starting from the common B28 adapter. They are separate controls, not one chained model. The 18 arms produced 2,286 rollout groups, of which 362 were accepted for updates.
- **Evaluation.** 22 unique endpoints (B28, 18 RL, 3 SFT) were each evaluated on the same 100 frozen formal test questions: 84 sequence questions from 14 patients and 16 Dallas identity-correspondence questions from 8 people and 4 matched cycles.
- **Result.** No stable advantage of GSPO MRI accuracy over the common starting point was found (details below).
- **Numerical evidence.** Finite-context F32 native evidence meets the original 0.002 tolerance in tested TRAIN contexts only. It is not official parity.
- **Paired data preparation.** 28 TRAIN people now have 168 paired MRI images (84 FLAIR/T1 pairs), and 56 z045 metadata records have been prepared. 16 new people (32 images) were technically obtained and sealed. Sealed outputs support no generalization claim.
- **Matched accuracy results completed.** On the same repeatedly explored DEV24 (four TRAIN-disjoint people), Adam52 Parent scored18/24, Direct GSPO16/24andPaired-Prompt16/24. A newly completed56-image patient-paired supervised continuation scored15/24 after28updates (FLAIR12/12,T13/12,invalid0). It failed the frozen replication rule; no additional seeds run. **No new MRI accuracy gain is established.** This SFT is not a GSPO improvement. [Actual results and model selection](research/FINAL_ACCURACY_RESULTS_V1.md).
- **Retained model.** Parent Adam52 remains best among the same-contract DEV24 endpoints. Historical SFT formal84scores are not a matched comparator. Independent16-person predictions remain sealed pending final endpoint/protocol confirmation. [Handoff and evaluation boundary](research/FINAL_MODEL_HANDOFF_V1.md).
- **Dallas task.** Whole-head identity correspondence stays `ENGINEERING_ONLY`: not clinical change, not brain-specific, and not an established second scientific task.

## Methodology

- **Data:** public OpenNeuro ds007045 patient-keyed MRI and Dallas ds004856 paired sessions, with patient-level train/dev/test separation within each source namespace. Global cross-source person independence is not verified. MRI payloads are never committed.
- **Tasks and training:** sequence identification (FLAIR vs. contrast-enhanced T1) and Dallas same/different identity correspondence; Vero-style GSPO with 4 responses per prompt, strict-format task rewards, a fixed token budget and persistent Adam/RNG state. The original formal22 cohort shares B28 and uses seeds 7, 11 and 13. Later matched pilots restore Adam52; the new SFT uses order seed0, as separately reported.
- **Evaluation:** frozen greedy strict-format accuracy, per-task and equal-task macro, class recall, and patient- or cycle-clustered paired bootstrap intervals. Intervals are not corrected for multiple comparisons.
- **Evidence labels:** `OBSERVED`, `REPRODUCED`, `INFERRED`, `PROPOSED`, `NOT_VERIFIED`. Cross-model review is not claimed unless it was actually performed.

## Experimental Results

Every seed and endpoint is reported, with no best-seed selection.

| Endpoint | Sequence correct / 84 (seeds 7/11/13) | Dallas correct / 16 (seeds 7/11/13) | Equal-task macro (3 seeds) |
|---|---|---|---:|
| B28 start (one output reused) | 52 / 52 / 52 | 8 / 8 / 8 | 55.95% |
| Sequence-only RL | 51 / 42 / 51 | 8 / 9 / 8 | 54.61% |
| Dallas-only RL | 52 / 52 / 52 | 8 / 8 / 8 | 55.95% |
| Proportional mixture RL, raw | 52 / 47 / 53 | 8 / 8 / 9 | 56.20% |
| Uniform mixture RL, raw | 53 / 51 / 51 | 8 / 8 / 8 | 55.75% |
| Proportional mixture RL, calibrated | 51 / 52 / 33 | 8 / 8 / 8 | 51.98% |
| Uniform mixture RL, calibrated | 50 / 50 / 52 | 8 / 8 / 8 | 55.16% |
| Same-data SFT | 54 / 57 / 52 | 7 / 8 / 8 | 56.30% |

![Strict accuracy of all endpoints relative to B28](research/figures/ORIGINAL22_FINAL100_TRANSFER_MATRIX_V1.svg)

**Conclusions (limited to this local task and implementation):**

- No stable, cross-seed MRI accuracy advantage from GSPO was shown. Uniform raw GSPO had a 55.75% macro against 55.95% for B28.
- B28 recall was 42/42 FLAIR (100%) and 10/42 contrast-enhanced T1 (23.81%). It answered "different" for every Dallas pair; its 50% Dallas accuracy does not demonstrate learned correspondence.
- More accepted updates did not mean higher accuracy. Zero-variance groups, the strict format gate and class priors were observed; their causal shares are undetermined.
- These results do not show that Vero-style methods fail in general. They are small samples (14 patients, 4 Dallas cycles) on one local implementation.

Details: [formal results report](research/ORIGINAL22_FINAL100_ACTUAL_RESULTS_2026-10-18.md), [full matrix](results/ORIGINAL22_FINAL100_ACTUAL_MATRIX_V1.json), [class/format/cost supplement](results/ORIGINAL22_FINAL100_CLASS_FORMAT_COST_SUPPLEMENT_V1.json), [training completion](results/ORIGINAL21_TRAINING_ACTUAL_COMPLETION_V1.json).

## Repository Structure

| Path | Content |
|---|---|
| `src/mri_rl/` | Library: task builders, rewards, GSPO training and evaluation, verifiers |
| `scripts/` | Reproducible analysis, verification and run entry points |
| `tests/` | CPU tests |
| `data/` | Dataset audits, eligibility and split contracts (no MRI payloads) |
| `results/` | Aggregate verified results and independent reviews (JSON/Markdown) |
| `research/` | Contracts, protocols, reports, paper-reading audits, figures |
| `reviews/`, `configs/` | Independent critiques; recipe configurations |

Research-branch material (not yet on `main`; pinned to commit `c24cb1d`): [Paired-Prompt grouping](https://github.com/MelodyResearch/Vero-to-Brain-MRI/blob/c24cb1d8c02679ccaf5804064483863e74d144f2/src/mri_rl/paired_prompt_groups_v1.py), [shared-rollout GSPO core](https://github.com/MelodyResearch/Vero-to-Brain-MRI/blob/c24cb1d8c02679ccaf5804064483863e74d144f2/src/mri_rl/shared_rollout_gspo_core_v1.py), [matched experiment preparation](https://github.com/MelodyResearch/Vero-to-Brain-MRI/blob/c24cb1d8c02679ccaf5804064483863e74d144f2/research/PAIRED_PROMPT_MATCHED_EXPERIMENT_V1.md), [paired z045 preparation](https://github.com/MelodyResearch/Vero-to-Brain-MRI/blob/c24cb1d8c02679ccaf5804064483863e74d144f2/research/PAIRED_Z045_PREPARATION_2026-10-20.md), [holdout-16 technical check](https://github.com/MelodyResearch/Vero-to-Brain-MRI/blob/c24cb1d8c02679ccaf5804064483863e74d144f2/results/PAIRED_SIGNAL_HOLDOUT16_TECHNICAL_V1.json) and [second-task eligibility review](https://github.com/MelodyResearch/Vero-to-Brain-MRI/blob/c24cb1d8c02679ccaf5804064483863e74d144f2/results/SECOND_TASK_ELIGIBILITY_INDEPENDENT_REVIEW_V1.json). Earlier auxiliary-loss variants (v1/v2/v3) are archived exploratory work, not the mainline ([note](https://github.com/MelodyResearch/Vero-to-Brain-MRI/blob/c24cb1d8c02679ccaf5804064483863e74d144f2/research/EXPLORATORY_METHOD_ARCHIVE.md)).

## Current Research Plan

The [frozen research plan][plan] and its lock are preserved. Later explicit researcher decisions authorized the completed matched GSPO and final bounded supervised accuracy experiment. The current decision is **stop same-configuration expansion and retain Parent**, because no candidate improved the explored development score. See the [final actual results](research/FINAL_ACCURACY_RESULTS_V1.md) and [internal paper draft](research/FINAL_ACCURACY_PAPER_DRAFT_V1.md). A once-only sealed16 evaluation requires researcher confirmation before any outputs are opened; it has not run.

## Reproducibility & Limitations

CPU path (Python 3.11+):

```bash
python -m pip install -e '.[test]'
pytest -q
```

Native MLX runs need Apple silicon; they were executed serially on an M3 Pro with 36 GB. Original images, patient-linked receipts, weights, optimizer and RNG state stay in private local storage and are not committed.

Limitations:

- The local MLX/language-LoRA study is not the official distributed Vero CUDA configuration, and the token cap is not an exact FLOP match.
- The formal test is small; three training seeds do not add patients.
- Dallas correspondence uses whole-head views containing face, skull and neck. It is not brain-specific or clinical change reasoning.
- Cross-model verification is `NOT_VERIFIED` for the private final results. Independent checks were performed by separately identified Codex reviewers.
- Final-test results must not be used to select prompts, rewards, checkpoints or seeds. Earlier operational notes and CLI examples remain available in the [historical README](https://github.com/MelodyResearch/Vero-to-Brain-MRI/blob/6ee3d1668f17189b24eaf1e850644a2cfbe5f69b/README.md).

[plan]: https://github.com/MelodyResearch/Vero-to-Brain-MRI/blob/05b8ba501c03db2782c34c9a6b4d8cae07e801c8/research/CURRENT_RESEARCH_PLAN.md
