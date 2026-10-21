# MRI data preparation contract v1 (versioned CPU pipeline; not a new main plan)

Code: `src/mri_rl/mri_data_prep_v1.py` (version `mri-data-prep-v1`), CLI `scripts/run_mri_data_prep_v1.py`, tests `tests/test_mri_data_prep_v1.py`. numpy/nibabel/Pillow/scipy only; threads fixed to 1; no model, torch, MLX, MONAI, fitting or training. Historical PNGs/SHAs are never written or overwritten.

## 1. Findings on existing preparation (classification)

| # | Question | Class | Evidence / scope |
|---|---|---|---|
| 1 | Is current orientation "wrong"? | `ALREADY_CORRECT` as a consistent display; physical-correctness beyond headers `NOT_VERIFIED` | 56/56 LAS storage; `np.rot90(vol[:,:,z])` shows right = anatomical Left, up = Anterior (radiological convention), derivable by `legacy_display_axes`. LAS-vs-RAS is a storage choice, not an error. |
| 2 | L/R flip or physical inconsistency? | `ALREADY_CORRECT` for header consistency; anatomical truth `NOT_VERIFIED` | same shape+affine in 28/28 pairs; no flip evidence. qform/sform conflict and units were not previously recorded: the pilot now reports them (`NOT_VERIFIED` until run). |
| 3 | z-index/spacing assumption | `ALREADY_CORRECT` for same-grid comparability *conditional on* upstream MNI registration quality (`NOT_VERIFIED`); z045 meaning in mm is reported by the pilot | Fixed index = fixed physical height only if all affines are identical. Cross-patient anatomical comparability of the same z needs upstream registration evidence, not geometry alone. |
| 4 | FLAIR/T1ce truly aligned? | `NOT_VERIFIED` | Grid identity is OBSERVED; registration is not proven. Pilot adds an NMI shift-sensitivity proxy (reports `registration_proven: false` always). |
| 5 | Fixed slice loses anatomy? | `CONFIRMED_DATA_DESIGN_DEFECT` | One pair abstained for insufficient coverage; z045 is a low/inferior level and historical SFT sequence/tumour/enhancement items all came from a low slice (AGENTS.md audit). Not a pixel/extraction bug. |
| 6 | Normalization → site/scanner shortcut? | `POTENTIAL_CONFOUND` | 0.905 LOPO fixed-feature retrieval; genuine sequence contrast, normalization, and protocol cannot be separated by this probe. Per-volume window alternatives are provided as named controls; none presumed better. |
| 7 | Labels inferable from patient/filename? | `CONFIRMED_DATA_DESIGN_DEFECT` (latent) | Filenames expose target 168/168 if ever serialized; native path does not. Mitigation: `bind_source_label` keeps truth/path/patient in hidden metadata and raises on visible leakage. |
| 8 | Image/patient/volume leakage or duplicates | within-source: `ALREADY_CORRECT` (LOPO excludes same person); global: `NOT_VERIFIED` | `check_patient_disjoint` reports within-source violations and prints `global_cross_source_identity: NOT_VERIFIED`; `duplicate_guard` flags exact/near cross-patient pixels. |
| 9 | Answers visible from the image? | A: partial; B/C/D: `NOT_VERIFIED`; Dallas identity: engineering only | See §5. |
| 10 | Risk to GSPO | `INFERRED` | Sparse/low slices and shortcut-able contrast reduce the rate of verifiable correct-vs-wrong differences; ineligible items add reward noise. Quantification needs the matched experiment in §6. |

No finding is declared a "confirmed error": nothing here shows a flipped, mis-scaled or mis-registered render.

## 2. Frozen primary candidate (slice selection only) and named controls

Frozen before any pilot output exists:
- **Primary** `coverage_fraction_v1`: for each volume, axial support extent = slices with foreground area ≥ 10 % of the max slice area; pick the slices nearest world z = zmin + f·(zmax−zmin) for f ∈ (0.30, 0.50, 0.70), inferior=0. Slice with area < 50 % of max is flagged `LOW_COVERAGE` and **abstains** (never substituted by outcome). Normalization, resize (364×436 bilinear), display frame and 3-slice count are **exactly Legacy** (`legacy_percentile`). At an identical slice it is byte-identical to the legacy render (tested).
- **Legacy Control**: `legacy_render` (z045/091/136), byte-identical to `source_sequence_render` (tested), unchanged.
- **Separate named controls, never combined with the primary in a causal arm:** normalization `percentile_nonzero_full`, `zscore_nonzero` (window mean−2σ…mean+4σ of foreground), `minmax_nonzero`, `percentile_given_mask` (needs an explicit mask; nonzero≠brain-mask is never assumed); 2.5D offsets in mm (`offsets_2p5d`, default −8/0/+8 mm, floor(mm/step+.5), out-of-volume flagged); all per volume only, no cross-volume or test statistics.
- Aspect: `display_aspect_report` computes mm/pixel per axis from header spacing; for 1 mm 182×218 → 364×436 distortion = 1.0 (computed, not assumed). Non-unit distortion is reported, not silently corrected.
- Token cost: `visual_token_estimate` is labelled `ESTIMATE_NOT_MEASURED` (assumes patch 16, merge 2, 32-px rounding → 11×14 = 154 tokens per 364×436 image; real Qwen3-VL/MLX processor settings `NOT_VERIFIED`). 3 slices ≈ 462; no VLM loaded.

## 3. Geometry rules implemented

`inspect_nifti` (shape, dtype, axcodes, units, qform/sform codes and conflict, header-vs-affine spacing, obliquity; flags `UNITS_UNKNOWN`, `QFORM_SFORM_CONFLICT`, `NO_VALID_XFORM`, `SPACING_HEADER_AFFINE_MISMATCH`, `OBLIQUE_OR_SINGULAR`); `canonicalize_ras` (reorients with `inv_ornt_aff`, checks ≥64 voxels map to the same world point and intensity, keeps the original affine in the report); `axial_display_plane` (orientation-invariant display; equals `np.rot90` on LAS; tested on flipped and permuted storage); `compare_grids` (**grid compatibility only**, `registration_proven: false`); `check_image_mask_geometry` (nearest-neighbour resample to image grid only if grids differ; label set must be preserved and within the declared ontology; image-support fraction flags `MASK_OUTSIDE_IMAGE_SUPPORT`; mask is never a model input; grid compatibility ≠ registration). Never RAS-reindex z045 without remapping physical coordinates.

## 4. Pilot API and private binding schema

```
python3 scripts/run_mri_data_prep_v1.py --manifest M.json --private-out DIR_PRIVATE --public-out DIR_PUBLIC
mri_rl.mri_data_prep_v1.run_pilot(manifest_dict, private_dir, public_dir)
```
Manifest (PRIVATE, schema `mri-data-prep-manifest-v1`):
```
{"schema":"mri-data-prep-manifest-v1","legacy_levels":[45,91,136],"mask_allowed_labels":[0,1,2,3,4],
 "rows":[{"opaque_id":"r001","modality":"flair|t1ce","nifti_path":"/abs/private","pair_key":"k01",
          "split":"train","expected_sha256":"optional","mask_path":"optional"}]}
```
`opaque_id`/`pair_key` must match `^[A-Za-z][A-Za-z0-9]{0,23}$` and not start with `sub`/`ses`; `split` must be `train`; ids unique. The pair_key is the within-source patient proxy for duplicate/disjointness checks. Intended root binding: 3 TRAIN patients × 2 modalities = 6 rows, 3 pairs.

Outputs: private `images/<id>_legacy_zNNN.png`, `<id>_cand_fNN.png`, `full_geometry_PRIVATE.json` (paths, source SHA256, affines, duplicate pairs); public `pilot_public_aggregate.json` with opaque-id rows only: geometry/flags, display axes & aspect, intensity stats, legacy-vs-candidate world z/support fraction/coverage/pixel-hash16 (first 16 hex)/mean abs pixel diff/difference reason, normalization controls at the centre slice, 2.5D offsets, token estimates, elapsed seconds, pair grid/NMI proxy, duplicate-guard counts. No paths, source SHAs, subject ids or images. Tested: public JSON contains no path/filename text.

## 5. Task candidates (`TASK_CANDIDATES`; none changes active Vero reward routes)

A sequence recognition — `ENGINEERING_BASELINE`; B lateral side (brain-support midline, margin, abstain; right=L display convention) — `NOT_ADMITTED`; C lesion presence by slice mask area (≥30 mm², abstain between) — `NOT_ADMITTED` (mask nonzero ≠ visible; `visibility: NOT_VERIFIED`); D cross-slice lesion pattern — `NOT_ADMITTED`. Each record carries source, required evidence, answer format, verifier, reward (existing binary exact-match), abstain rule, admission, and an independence note. `independent_mask_count` makes multiple questions on one mask count once. No expert adjudication is claimed.

## 6. Proposed variable-isolated experiments (PROPOSED, not run, not authorized)

1. **Data effect:** same parent checkpoint, same Vero-style GSPO recipe/budget; arm L = Legacy slices, arm P = `coverage_fraction_v1` slices (only change). Evaluate both on new-patient data under the same input contract of each arm plus a cross-input table; patient-clustered CIs.
2. **Vero RL effect:** fix the input contract; compare GSPO vs matched SFT vs unreinforced parent.
3. **Adaptation effect:** only after 1–2, with data and budget fixed.
Normalization and 2.5D controls enter as separate arms, never merged with P. final100 is not used for development; new16 outcomes stay sealed.

## 7. QC

`qc_flags_for_plane` (LOW_COVERAGE <20 % nonzero, LOW_CONTRAST std<8, SATURATED >20 % at 255) and slice-level `LOW_COVERAGE` are technical, predeclared heuristics; flagged items abstain. They are not clinical judgments.
