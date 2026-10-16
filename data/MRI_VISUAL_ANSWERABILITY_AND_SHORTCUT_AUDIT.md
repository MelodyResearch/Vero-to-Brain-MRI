# ds007045 visual answerability and shortcut audit

**Independent audit, 2026-10-15 (local).** This is a CPU-only check of the existing ignored `local_data/ds007045_bounded` payload, renderer, prompt/gold files, and patient split. I inspected rendered MRI panels directly and recomputed label geometry from all 40 downloaded masks. This is a benchmark-validity audit, **not** a neuroradiologist review, clinical ground-truth validation, or a new model experiment. No patient identifiers or MRI pixels are included in this repository file.

## What was checked

- Read the local `visual_prompts/private_manifest.json`, `prompts_image_only.jsonl`, separate `gold_mask_derived_PRIVATE.jsonl`, and the renderer in the project workspace `work/ds007045_prompts.py`. The pack contains 40 patient-keyed trios, 360 PNG files, and 543 questions. The 32/8 source split and 28/4 internal train/dev subdivision were cross-checked against the actual `vero_training_v1/{train,internal_dev}.verl.jsonl` records: 382/54 records, 28/4 distinct patients, zero train–dev patient overlap and zero overlap with the eight held-out patients. There is one held-out patient per site, not an unseen-site split.
- Re-rendered all **240 single-sequence PNGs** from the local NIfTI volumes using the recorded rotation, volume-wise nonzero 0.5/99.5 percentile scaling, and bilinear enlargement. Every saved single image and the corresponding crop of its paired image matched pixel-for-pixel. This checks slice indexing and renderer consistency, not whether a clinician would endorse every segmentation.
- Recomputed tumor-present (`mask > 0`), enhancing-present (`mask == 3`), and the tumor centroid for every fixed slice `z=45,91,136`. All 120 presence answers, all 120 enhancement answers, and all 63 stored grid coordinates matched the masks and the renderer equations. The sequence answers are tied to the selected modality files and separate source sidecars as documented in [the eligibility audit](PATIENT_KEYED_MRI_ELIGIBILITY.md).
- Visually inspected a contact sheet spanning held-out no/no, held-out yes/yes, held-out yes/no, and edge-case train slices. One held-out middle slice had a conspicuous mass/ring on both panels; another held-out middle slice had a smaller FLAIR abnormality without enhancing mask. A superior slice with a five-voxel positive mask had no lesion I could confidently distinguish in the unannotated two-panel rendering. These are qualitative observations by a non-clinical auditor, not image-level adjudications.

## Answerability and label fragility

| Finding | Measured evidence | Consequence |
| --- | --- | --- |
| Slice-positive does not always mean visibly answerable | The smallest tumor-positive slice contains **5 mask pixels** in the native 182×218 slice. One more positive slice contains 169 pixels; the 10th percentile of positive area is 474 pixels. | A zero-threshold `mask > 0` gold label may mark a small residual that is not distinguishable after PNG rendering. An apparent model error can reflect answerability rather than reasoning. |
| Enhancing-present can be set by tiny components | Of 43 enhancing-positive slices, **two contain one enhancing voxel**, one contains two voxels (held-out), and six contain at most 100 voxels. | `mask == 3` with any positive voxel is fragile for an image-only yes/no question. A prespecified minimum area and human review are needed for a clinically interpretable endpoint. |
| Grid labels are boundary-sensitive | Among 63 grid questions, 23 centroids lie within 10 display pixels of a 3×3 boundary; 45 lie within 20 pixels. For four questions the arithmetic centroid falls outside the positive mask. | The current grid route tests an exact mask-derived centroid, which can be unstable near a categorical boundary and may not correspond to a visible lesion center. It should not be treated as a clean localization benchmark without tolerance or adjudication. |
| The renderer itself aligns with masks | Re-rendered single and paired panels matched all 240 source slices; recomputed gold matches masks. | These problems are label/visual-answerability limits, not a detected off-by-one or rotation bug. |

The above area counts are **2D mask pixels on selected slices**, not lesion volumes, 3D Dice, or clinical lesion sizes. The mask is an expert-reviewed, image-derived reference according to the source documentation; this audit did not independently verify annotation quality.

## Shortcut controls that matter for model comparisons

1. **Axial level is a strong shortcut for both tumor tasks.** The fixed levels are recognizable from brain anatomy. A rule fitted only to the training split's per-level majority (`z45=no, z91=yes, z136=no`) gets **19/24 tumor-presence** and **20/24 enhancing-presence** answers correct on the eight held-out patients. It uses no lesion information. Training per-level counts are tumor `no/yes = 17/15, 3/29, 24/8` and enhancement `20/12, 11/21, 30/2` for `z45, z91, z136`; held-out counts are tumor `8/0, 1/7, 4/4` and enhancement `8/0, 2/6, 6/2`. This baseline is substantially higher than the global majority baselines in the eligibility audit. Any image-grounding claim needs to beat this prespecified anatomy-level control on independent patients.
2. **Tumor routes share the same exact paired image.** Presence and enhancement questions always reuse a paired PNG; the grid question reuses it when tumor is positive. Thus apparent transfer between these three routes may reflect shared image/answer correlation. A cross-task matrix should separate genuine held-out-patient transfer from within-image multi-label reuse and compare against a per-level shortcut baseline.
3. **All sampled patients have brain tumors at the case level.** A `no` presence or enhancement answer refers only to one selected axial slice. It is not a healthy-vs-tumor patient diagnosis task. The source cohort cannot support claims about generalization to normal brains.
4. **Grid is conditional on tumor presence.** The existence of a grid question itself conveys that the slice has positive tumor mask; it must never be used as a presence-evaluation prompt or mixed into a control that claims the tumor status was hidden.
5. **Sequence labels are balanced within each patient and level**, with one FLAIR and one contrast-enhanced T1 PNG. The saved single-image pixels do not contain a modality title. However, local PNG filenames and record IDs include `flair`/`t1ce`, and the paired tumor PNGs do contain panel titles. The inspected `frozen_eval.py` passes a decoded image plus prompt to the model; it does not concatenate the path or record ID into the text prompt. The filename is still an avoidable metadata-leak risk for any future data loader, export, or tool-mediated evaluation. Rename image files to opaque IDs and assert that serialized text prompts contain no answer-bearing path before a definitive benchmark.
6. **Patient isolation is local to this source.** The checked records have no overlap among internal train, internal dev, and held-out patient keys. This does not establish independence from the model's pretraining data, external MRI sources, or duplicated underlying images outside ds007045.

## Decision for the Vero transfer experiment

Keep the present 40-patient pack as a reproducible **engineering diagnostic**, with the existing held-out split frozen. Report the anatomy-level baseline and patient-clustered metrics alongside model results. For a primary image-grounded transfer claim, build an additional eligible evaluation set with varied levels and negative/positive cases, prespecified lesion-size rules or expert adjudication, and held-out patients; keep the current eight test patients out of prompt/reward selection. Sequence recognition remains the cleanest of the four current categories with respect to label independence from the tumor mask, although source/preprocessing style can still be exploited. Presence and enhancement are useful only with the stated slice-level and tiny-label caveats; the current exact-cell grid score is particularly fragile.

**Status:** renderer and mask-derived labels `TESTED_CPU`; human clinical answerability `NOT_VERIFIED`; independent-patient local split `TESTED_CPU`; external patient/pretraining nonoverlap `NOT_VERIFIED`. No new GPU inference or training was run for this audit.

## Metadata-only reserve for a prospective project test

I separately checked the official public [ds007045 metadata repository](https://github.com/OpenNeuroDatasets/ds007045) at the **same pinned commit `283007b96977b5b3285bd22cd67b7d97114b1eed`**, without opening any additional MRI or mask payload or prospective label. The metadata has 363 unique `participants.tsv` patient keys. For each key I checked the presence of valid Git-annex SHA256/byte-size pointers for the processed T1ce, processed FLAIR, and matching segmentation mask, plus raw `run-02_T1w` and `FLAIR` JSON sidecars. All 363 have these five metadata items. Excluding the existing 40-patient cohort leaves **323 metadata-eligible patients** across seven sites:

| Site prefix | All patient keys | Existing 40 | Metadata-eligible reserve |
| --- | ---: | ---: | ---: |
| BA | 3 | 3 | 0 |
| BO | 171 | 7 | 164 |
| FO | 8 | 5 | 3 |
| JS | 11 | 5 | 6 |
| KA | 110 | 7 | 103 |
| ME | 7 | 4 | 3 |
| NSK | 39 | 5 | 34 |
| TO | 14 | 4 | 10 |
| **Total** | **363** | **40** | **323** |

The reserve has 969 pointed-to MRI/mask files totaling **4,195,990,392 bytes (3.91 GiB)** by pinned annex metadata. No subject directory outside the existing 40 exists under this project's local `ds007045_bounded` payload. Thus these 323 are **untouched by this project's local image, mask, reward, and model inspection**, as far as the bounded workspace shows. `BA` is exhausted, so a new cohort cannot preserve all eight sites; `FO` and `ME` have only three reserve patients each. A site-stratified seven-site cohort is feasible, but its sample size and selection algorithm should be frozen from metadata before downloading or viewing masks, and it should remain independent of all reward, prompt, checkpoint, and threshold selection.

This is **metadata eligibility**, not 323 verified usable image trios. Their actual bytes, NIfTI geometry, hashes, lesion visibility, and annotation quality remain `NOT_VERIFIED`. The 3.91 GiB estimate is not a download receipt. “Prospective” means prospectively held out within this research workflow; these are existing retrospective scans, and model-pretraining or cross-source patient overlap is unknown. The reserve also remains an all-tumor source cohort; adding its patients alone does not solve the healthy-control or axial-level shortcut problems identified above.
