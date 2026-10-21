# MRI spatial task card v1 (CPU pilot; extension of `spatial_ontology_resolution`, `second_task_eligibility`, `mri_data_prep_v1`)

Code `src/mri_rl/mri_spatial_task_pilot_v1.py`, runner `scripts/run_mri_spatial_task_pilot_v1.py`, tests `tests/test_mri_spatial_task_pilot_v1.py` (37, CPU, synthetic fixtures). **Execution status: `IMPLEMENTED`, `TESTED_CPU` on synthetic fixtures; the real-TRAIN run is `NOT_RUN` until root freezes `EXECUTION_INPUT_PRIVATE.json` and reviews the code.** No model, GPU, MLX, optimizer, resampling, new render or mask overlay is involved.

## Question and hypotheses (bounded, falsifiable)

- **Task:** the existing `mri_tumor_grid` (3×3 display-grid cell of the lesion), Vero-format route `mri_tumor_grid` in `vero_mri_reward.py` (0.8·accuracy + 0.2·format; no new reward).
- **H1 (technical):** for ≥1 of the 3 fixed cases the source mask yields arithmetic gold with no abstain reason. **Falsifier:** every case abstains — reported as a result, no replacement.
- **H2 (scope):** arithmetic gold ≠ visible truth. Every case stays `EXPERT_ADJUDICATION_REQUIRED`, `scientific_admitted = 0`; no AI expert assertion is written.
- **Controls:** original hidden target vs both coordinate conventions (retained, never rewritten); union vs label-3 component reported as separate endpoints; exact existing-PNG pixel comparison; verifier run on synthetic response strings only (labelled `synthetic_response_fixtures_only`).

## Endpoint (minimal, defensible) + ABSTAIN

Primary endpoint `whole_lesion_union(label>0)_centroid_cell`: edge coordinates `2·(mean idx + .5)` on the 364×436 panel, thirds half-open, boundary → right/bottom (exactly `spatial_ontology_resolution_v1.endpoints`). Components (8-connected) and the label-3 component are kept as separate fields; `visible_focus_center` stays `NOT_ADJUDICATED`. A case **abstains** (reasons kept, cell still recorded) on: `EMPTY_MASK_NOT_NEGATIVE_GOLD`, `SMALL_SOURCE_MASK` (<1000 native px, inherited screen), `MULTICOMPONENT_2D`, `CENTROID_NEAR_GRID_BOUNDARY` (<10 display px), `CENTROID_OFF_MASK`, `COMPONENT_GRID_DISAGREES_WITH_UNION`, `CONVENTION_SENSITIVE_GRID`, `ORIGINAL_TARGET_DISAGREES_WITH_ARITHMETIC`, `UNKNOWN_ONTOLOGY_LABEL`, `MASK_NONINTEGER_LABELS`, `MASK_OUTSIDE_IMAGE_SUPPORT` (<80 % inside nonzero support), `QC_*`, `LEGACY_PNG_MISMATCH`, `VISIBLE_TEXT_LEAKAGE`, `SOURCE_HASH_MISMATCH`, `GEOMETRY_IMAGE_*`/`GEOMETRY_MASK_*` (fail-closed `geometry_eligibility`, incl. non-LAS), `IMAGE_MASK_GRID_MISMATCH_NO_RESAMPLE`, `LEVEL_OUT_OF_RANGE`, `DISPLAY_SCALE_NOT_2X`. The 1000 px / 10 px screens are inherited engineering screens, not validated visibility thresholds; small positives are never turned into negatives.

## Ontology and frames

Formal labels 0 background, 1 necrotic core, 2 non-enhancing/edematous, 3 contrast-enhancing (paper, ds007045 **v1.0.1**, 337 people); local data are **v2.0.1** (363). Numeric-3 semantics are per case via `annotation_release_lineage` (`PAPER_RELEASE_SAME_ANNOTATION` → verified; `NEW_RELEASE_LABEL3_NOT_VERIFIED`/`NOT_SUPPLIED` → `NOT_VERIFIED`); this does not invalidate the whole-lesion union arithmetic. Labels outside {0..3} hold the case (no cast, no remap). Display = `np.rot90(volume[:,:,z])` on LAS storage: display right = anatomical **Left**, up = **Anterior**; therefore the words *left/right* in the grid label are **display** columns (image-left = anatomical Right). `display_frame` states this per case.

## Geometry and interpolation policy

Image and mask must share shape and affine (`compare_grids`, tolerance 1e-4); otherwise the case holds with `IMAGE_MASK_GRID_MISMATCH_NO_RESAMPLE` — **no resampling is ever performed** (`resampled: false` is a recorded field), so no interpolation or nearest-label step is needed. Registration is never claimed (`registration_proven: false`). Plane world z (mm) and the union-centroid world position are recorded privately.

## Leakage and independence

The mask is never a model input and nothing mask-derived is rendered (the existing PNG must equal the unmasked legacy render). Optional `visible_prompt_text` is checked with `bind_source_label` against paths, basenames, pair/opaque ids. Presence, enhancement and grid targets from one mask count as **one** mask-derived source, never independent skills.

## Resources and outputs

≤3 cases (fixed order, no replacement; failures kept in `case_<id>_FAILED_PRIVATE.json`), threads=1, 512 MiB peak-RSS and 120 s/case soft budgets (checked between cases after an atomic save; over-budget stops the *remaining* cases), new/empty private dir, sources read-only (size+mtime checked after). Private: `case_*_PRIVATE.json`, `expert_packet_PRIVATE.json` (all fields `null`, `ai_expert_assertions: 0`), `run_receipt_PRIVATE.json` (source SHA-256, input hash, code hash). Public JSON: opaque ids, status, reasons, counts, timing, RSS, 16-hex render hash — no paths, filenames, source hashes or images. Exceptions are written privately; stdout prints only counts and an error class.

Command (after root review and binding): `python3 scripts/run_mri_spatial_task_pilot_v1.py --input <EXECUTION_INPUT_PRIVATE.json> --private-out <NEW_DIR> --public-out <NEW_FILE.json>`. Input (root safe schema; list under `cases`/`rows` or bare list; aliases `flair_path`/`mask_path`/`level` also normalized): per case `opaque_id, split:"train", image_nifti_path, image_expected_sha256, mask_nifti_path, mask_expected_sha256, legacy_slice_index, legacy_png_path, legacy_png_sha256`, optional `image_modality` (`t1ce`/`flair`; the fixed pilot images are **T1ce**), `original_ground_truth`, `annotation_release_lineage`, `visible_prompt_text`, `pair_key`. `--describe-keys` prints key names only. `select_first_distinct_train_grid` encodes the pre-voxel selection rule (first 3 TRAIN `mri_tumor_grid` records in source order with distinct `patient_key`).

## Paired-canvas correction (root review, applied)

The historical spatial PNGs are **paired** 752×470 canvases (FLAIR panel box (4,28,368,464); T1ce panel box (384,28,748,464); each 364×436; title strip above y=28). The new pilot images are **T1ce**, so the T1ce render (`dallas_spatial_qualification_v1.render_slice`, float32 read as the historical builder) is compared with the **right panel crop only** (declared modality must match its panel; undeclared modality needs exactly one matching panel; a bare 364×436 PNG is accepted as the panel). All endpoint coordinates are **panel-relative**; the canvas offset is never applied (`canvas_to_panel_xy`, tested). For a technically clean case the mask-free, header-free T1ce panel crop is saved privately as a **NEW** model-input candidate with a **NEW** single-image prompt state (`SINGLE_IMAGE_PROMPT`, `NEW_SINGLE_IMAGE_CANDIDATE_NOT_MODEL_TESTED`; image bound in typed prompt and top-level `images`); abstaining cases create no row. This is not the historical paired-frame input, and the old FLAIR-panel question/target remain private provenance only. T1ce contrast and visible answerability are **not** assumed equal to FLAIR. Numeric-3 is reported as a separate `numeric3_component` (route disabled; lineage `NOT_REATTESTED` → `NOT_VERIFIED`; not called clinically enhancing).
