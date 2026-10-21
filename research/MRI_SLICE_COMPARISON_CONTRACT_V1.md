# Frozen slice-input comparison contract v1 — PREPARATION ONLY, `NOT_RUN`

Code `src/mri_rl/mri_slice_comparison_v1.py`; CLIs `scripts/prepare_mri_slice_comparison_v1.py`, `scripts/analyze_mri_slice_comparison_v1.py`; tests `tests/test_mri_slice_comparison_v1.py` (14, synthetic). **No model call, no GPU/MLX authorization, no rerender.** Certified base cache v3 is `STOP` (not PASS) and its permission is not reusable. The `run` action is dormant and always raises `NotAuthorized`; the existing frozen kernels (`paired_z045_matched_f32_eval_v1.run_checkpoint`/`run_row`, wired as in `accuracy_recovery_eval_v1.execute`) can consume the prepared records only after a separate exact-scope review and fresh human authorization.

## What is compared (data input only)

- **inA** = the single legacy z045 image (the existing old PNG; pixel-identical check against the old row image). **inB** = the single predeclared physical-support median (`coverage_fraction_v1`, f = 0.50, reference-modality world-z mapped to the partner) from the already-rendered 6-image TRAIN pilot (3 people × FLAIR/T1ce). Names differ from the training arms (`trainedA`/`trainedB`) on purpose.
- Held constant: person, modality, parent checkpoint (Adam52), processor, prompt text (**bit-exact**, verified), normalization, image size, decoder (`max_tokens 256`, temperature 0), scorer (frozen `PREFIX`, mean/sum/first), image count (1).
- Each record binds the image in **both** the top-level `images` and the typed prompt `image` entry (asserted equal, one entry each). Only the image path and record id differ between inputs. No labels, patient names, filenames or sites appear in text. `B_abstain` (pixel-QC/coverage abstain from the pilot) is carried and reported, never used to drop rows silently; `B_identical_to_A_pixels` is counted.
- Swap controls are the fixed same-person opposite-modality image **within the same input**; their outputs are reused (0 extra calls).

## Frozen caps (per checkpoint, 3 people → 12 records)

12 greedy calls, 36 candidate-score calls (3/record), ≤3072 generated tokens, ≤15300 teacher-forced positions (425/call), 1 model load, serial. Protocol digest is stable and written to the public prepare output.

## Endpoints and analyzer (frozen before any output)

Strict-format greedy accuracy and balanced accuracy, FLAIR and T1ce recall, invalid counts; candidate **mean (primary) / sum / first** reported separately with ties INVALID and correct-minus-wrong margin by class and aggregate; swap sensitivity (changed / valid opposite-class / both-correct); cost fields. Resolution: one row moves a class recall by 1/3 — engineering preparation, not statistical evidence.

Three effects are never merged: (1) `data_input_effect_at_<checkpoint>` = inB−inA at the same checkpoint — **a data effect only, not an RL gain**; (2) `training_arm_effect_*` = trainedB (Paired-Prompt) − trainedA (Direct same-prompt), or either − parent, at a fixed input; (3) `input_x_training_interaction` (exploratory difference of differences). Effects are produced only for conditions actually supplied. `analyze` requires a declared provenance; real rows need a receipt SHA-256; `SYNTHETIC_FIXTURE` outputs set `model_prediction_claim: false`. `MRI_accuracy_improvement_claim` is always `NOT_ESTABLISHED` here. new16 and final100 are never opened.

## Binding (PRIVATE, local code only; schema `mri-slice-comparison-binding-v1`)

`rows[]`: `opaque_id, pair_key, modality (flair|t1ce), truth, old_row_PRIVATE (original VERL row), A_image, B_image, A_sha256?, B_sha256?, B_abstain`. `A_image` = pilot `images/<id>_legacy_z045.png`; `B_image` = pilot `images/<id>_prim_f50.png`. Run: `python3 scripts/prepare_mri_slice_comparison_v1.py --binding <PRIVATE.json> --private-out <NEW_DIR> --public-out <NEW_FILE.json>`.
