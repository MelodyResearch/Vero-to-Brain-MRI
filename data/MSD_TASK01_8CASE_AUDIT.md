# MSD BrainTumour: eight real paired volumes and image-only visual prompts

**Status:** `TESTED_CPU` for data decoding and prompt construction; **no model inference or training is claimed here**. Audit date 2026-10-15. Source: [Medical Segmentation Decathlon official AWS registry](https://registry.opendata.aws/msd/) and [challenge page](https://medicaldecathlon.com/dataaws/), CC-BY-SA 4.0 per the embedded dataset descriptor. The public `Task01_BrainTumour.tar` is 7,608,266,240 bytes. We fetched only eight image/mask pairs by exact HTTP byte ranges from the official bucket, into ignored `local_data/msd_task01/`, without retrieving the whole TAR. The ignored `range_index_private.json` and `visual_prompts/private_manifest.json` record TAR offsets, case filenames, hashes and file-level checks.

## Actual files verified

Eight **volumes**—not eight verified independent patients—have four MRI channels in the dataset's documented order: FLAIR, T1-weighted, post-contrast T1-weighted, T2-weighted. All eight image volumes decode as finite arrays of shape 240 × 240 × 155 × 4. Their paired mask volumes have shape 240 × 240 × 155, identical spatial affines, RAS+ orientation and 1 mm isotropic spacing. All contain the dataset's mask values 0 background, 1 edema, 2 non-enhancing tumor and 3 enhancing tumor. The checksum and geometry checks apply only to these eight downloaded pairs. The source-case/patient crosswalk and exact manual-versus-fused annotation provenance remain **UNVERIFIED**; the [BraTS 2017 custodian's data page](https://www.med.upenn.edu/sbia/brats2017/data.html) explains why older BraTS labels cannot all be assumed manually traced.

## Image-only prompt construction

The ignored `local_data/msd_task01/visual_prompts/` directory contains 16 four-panel PNGs and **72** prompts across five question types. Panels show only the four MRI image channels, with sequence titles. There are no mask overlays, target contours, source filenames or gold answers in the PNGs. Each channel is normalized from its own image volume's nonzero 0.5th and 99.5th intensity percentiles, then an axial slice is rotated consistently for display. The prompt file `prompts_image_only.jsonl` contains image paths, questions and anonymous prompt tokens; `gold_mask_derived_PRIVATE.jsonl` contains the corresponding hidden answers and mask-derived coordinates. A private manifest records the source mapping and hashes. The model loader must pass only PNG pixels and question text to Qwen3-VL; it must never pass the gold file or private source filenames.

| Slice protocol | Selection rule | Cases / images | Questions | Role |
| --- | --- | ---: | ---: | --- |
| **Fixed midpoint** | `z = floor(155/2) = 77`, specified from image shape before mask inspection. | 8 / 8 | 40 | Prespecified first model-comparison diagnostic, still only an eight-volume engineering sample. |
| **Exploratory balanced** | `z = 40`, chosen **after inspecting masks** because it yielded mixed tumor presence. | 8 / 8 | 32 | Debugging and error analysis only; **not** a sealed evaluation, validation or prompt-selection set. |

Prompt categories are: binary any-tumor presence, binary enhancing-tumor presence, exact subset of edema/non-enhancing/enhancing classes, 3 × 3 tumor-centroid grid cell, and bounding box of all mask-positive pixels within the 240 × 240 FLAIR panel. Centroid and box prompts are emitted only for slices with mask-positive tumor pixels. The slice used for the fixed-midpoint prompts was **not** chosen based on mask presence. The hidden mask is used solely to compute reference answers after rendering. Bounding-box coordinates are in the displayed FLAIR panel's pixel frame after the same orientation rotation, not full-canvas coordinates.

## Necessary majority controls

The fixed-midpoint sample has tumor present in all 8/8 cases, so an **always-yes** baseline scores 100% on that question; accuracy there has no discriminative value. Enhancing tumor is present in 6/8, so a majority-yes baseline scores 75%. The exact three-class set is identical in 6/8, so its majority baseline is also 75%. The most frequent centroid grid answer occurs in 4/8, or 50%. At exploratory `z=40`, any-tumor presence is 4/8 positive and 4/8 negative, but its balance was selected after reading labels and must not be treated as an unbiased test. These are descriptive prompt counts, not patient-level statistical estimates.

## Before any scientific cross-task claim

1. Use a predeclared, mask-independent slice schedule for evaluation and include more cases with source-patient lineage. A fixed midpoint is reproducible but does not solve task triviality or class imbalance.
2. Compare base Qwen3-VL and pretrained Vero on exactly the same 40 fixed-midpoint prompts and images, with answer parsing fixed in advance; include majority, no-image and shuffled-image baselines. Report per-category outcomes and raw model outputs. No model comparison was performed in this data audit.
3. Keep the eight volumes together with all derived prompts when splitting. Do not count 40 prompts as 40 independent samples. For patient-independent evaluation, obtain a source-case-to-patient crosswalk or a semantically matched external cohort.
4. Distinguish 2D frame localization from 3D tumor segmentation; the four-panel PNG is a multisequence *single slice*, not a native 3D model input.
