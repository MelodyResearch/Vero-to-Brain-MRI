# Direction C1: exact smaller-subset feasibility

Status: `TESTED_CPU`; task admission: `ENGINEERING_ONLY`. No VLM training or inference was run by this analysis. This new C1 version preserves `slice_label_candidate_v1` and the frozen B schedule.

## Executed design

`build_c1_matched_subsets.py` searched all integer pair quotas from largest to smallest, then solved patient-to-slot bipartite matching. Each slot is a level and binary label. One patient can occupy at most one slot per task. Positives retain the predeclared ≥1,000 native-pixel threshold; negatives require exactly zero source-mask pixels. Small positives are excluded, never relabeled. At least two levels and all levels having both eligible classes are required. Within every included level, yes/no counts are equal. Level imbalance is permitted; label imbalance within level is not.

This is a new, explicit selection intervention. It does not retrofit B, and it does not establish MRI visual answerability. All selection used the frozen 382 training records from 28 training patients; no development or historical-test predictions were read.

| Task | Largest feasible subset | Labels | Level45 | Level91 | Level136 |
|---|---:|---|---:|---:|---:|
| Tumor presence | 26 distinct patients | 13 yes / 13 no | 18 (9/9) | 6 (3/3) | 2 (1/1) |
| Enhancing presence | 10 distinct patients | 5 yes / 5 no | 4 (2/2) | 6 (3/3) | Excluded |

Enhancement has **zero** eligible positive examples at level136. Requiring positive/negative support at all three original levels is therefore impossible for any nonempty exact three-level subset. The new two-level scope is declared, rather than weakening lesion size or relabeling cases. The tumor maximum follows exhaustive rejection of all feasible-capacity 28-slot quotas before selecting 26; enhancement is bounded above by five eligible positive records, and a ten-patient matching attains that bound.

The earlier maximum partial-assignment counts (27 tumor / 19 enhancement) concern a different 28-slot problem and are not balanced usable subset sizes.

## Eligibility inventory before matching

| Task | Level | Eligible no | Eligible yes |
|---|---:|---:|---:|
| Tumor | 45 | 14 | 12 |
| Tumor | 91 | 3 | 23 |
| Tumor | 136 | 21 | 1 |
| Enhancement | 45 | 17 | 2 |
| Enhancement | 91 | 10 | 3 |
| Enhancement | 136 | 26 | 0 |

No source-label/mask-area contradictions occurred. This threshold is a geometric screening criterion, not a clinical visibility gold standard. Matched cohorts overlap across tasks; 26+10 must not be reported as 36 independent patients.

## Site and shortcut audit

Tumor site counts: BA1, BO4, FO3, JS3, KA6, ME3, NSK4, TO2. Enhancement: BO3, FO2, JS1, ME2, NSK2. Exact site×label×level counts remain in the private receipt. Site balance was not optimized after inspecting these results.

| Label prediction rule | Tumor | Enhancement |
|---|---:|---:|
| Fixed no | 13/26 | 5/10 |
| Level-conditioned majority, same-sample association | 13/26 | 5/10 |
| Site-conditioned majority, same-sample association | 17/26 | 7/10 |
| Site+level majority, same-sample association | 18/26 | 7/10 |
| Site majority, leave-one-patient-out | 7/26 | 4/10 |
| Site+level majority, leave-one-patient-out | 7/26 | 0/10 |

The same-sample results measure residual metadata association, **not out-of-sample performance**. Leave-one-patient-out results are descriptive and unstable: exact balancing and tiny strata can reverse the remaining majority when one sample is removed. Their low values do not establish absence of shortcuts. Constant/metadata rules are not VLM no-image inference, which remains `NOT_RUN`. Simple image baselines were subsequently executed as described below; these do not substitute for VLM no-image inference.

Within-level matching removes level alone as a class discriminator in the selected dataset. It does not remove site/protocol, head shape, image normalization, or subtler anatomy shortcuts. Confirmatory experiments need independently selected patient cohorts with the same eligibility rules and prespecified site-stratified endpoints.

## Prompt boundary

The script checks text components of all 382 structured training prompts for private patient/record identifiers and gold metadata fields. There are zero detected failures. Image transport paths contain patient tokens, slice indices and sequence names, but are excluded from textual content; the current SFT runner constructs chat text from `extra_info.question` and passes opened pixels separately. Any future runner that stringifies the whole record would violate this boundary and must be tested independently. This check is not an exhaustive information-flow proof or a substitute for rendered-image inspection.

An initial audit incorrectly treated structured image transport locators as visible text, producing 382 false positives. The corrected implementation has a regression covering structured image paths and still rejects identifiers/gold metadata in text. The superseded private receipt is preserved and is not scientific leakage evidence.

## C2: bounded additional training metadata

Pinned official source commit: `283007b96977b5b3285bd22cd67b7d97114b1eed`. Existing 40 patients (28 train, 4 development, 8 old test) and all 14 reserved prospective patients were excluded by source subject identity before selection. There are **309** remaining metadata-eligible patients:

BO162, FO1, JS4, KA101, ME1, NSK32, TO8.

A new deterministic SHA256 selection takes up to two per site: **12 patients** (2,1,2,2,1,2,2 respectively). Their checksum-bound annex references require **149,834,294 bytes** for FLAIR, T1ce and masks together; masks alone total **318,016 bytes**. No new payload bytes were downloaded or inspected by this stage. Metadata availability cannot establish lesion-size or visible-evidence eligibility.

A scientifically useful next fetch is the twelve small masks, each verified against its annex SHA256, processed one at a time to measure level-specific positive eligibility. Only then fetch matching MRI for selected feasible cases. Keep original split and reserved prospective lists untouched. Do not accept new agreements or transfer data externally.

## Reproduction and evidence

Run from the repository with one CPU thread:

```bash
OMP_NUM_THREADS=1 python3 scripts/build_c1_matched_subsets.py \
  --root local_data/ds007045_bounded \
  --output local_data/ds007045_bounded/overnight_c1_v2/new_receipt_PRIVATE.json \
  --metadata-repo ../../work/ds007045_metadata
```

Executed final receipt: `local_data/ds007045_bounded/overnight_c1_v2/matched_subsets_v2_PRIVATE.json`. It binds training references and training JSONL SHA256, script source, private selected records, geometry, site distributions and checksum-bound new candidate pointers. New output required; existing receipts cannot be overwritten. Five regression tests passed before review. Independent verification is recorded separately; this author does not self-approve C scientific admission.

## Executed real-pixel shortcut probes

A prespecified CPU-only leave-one-patient-out 1-nearest-neighbor diagnostic read the selected real PNGs one at a time (26+10=36 task rows), verifying their frozen SHA256. Feature sets were fixed before execution: (1) raw 16×16 bilinear grayscale thumbnail pixels scaled to [0,1], and (2) a normalized 16-bin histogram of the full grayscale image. Distance was Euclidean, with record-ID tie breaking. Every example from the query patient is excluded from its neighbor pool. No hyperparameter tuning, VLM loading, optimization, or held-out patient outcome inspection occurred.

| CPU real-image rule | Tumor presence | Enhancement presence |
|---|---:|---:|
| Thumbnail 1NN | 12/26 | 6/10 |
| Intensity-histogram 1NN | 13/26 | 4/10 |
| Fixed no | 13/26 | 5/10 |

These are exploratory training-cohort diagnostics with tiny sample sizes and selection-dependent distributions. Thumbnail features retain global anatomy and display layout; histograms retain brightness/protocol differences. Neither method isolates lesion evidence, and the one-example enhancement difference from a constant is not evidence of MRI understanding or statistical improvement. No scientific admission status changes.

Implementation: `scripts/c1_image_shortcut_baseline.py`; private predictions and hash-bound receipt: `local_data/ds007045_bounded/overnight_c1_v2/image_shortcuts_PRIVATE.json`. Two regression tests verify patient exclusion and label-independent deterministic ties. Combined C tests: seven passed.

## C2 executed follow-up: twelve real source masks

After the next model process exited, `scripts/probe_c2_masks.py` retrieved **12 actual public OpenNeuro mask files**, serially, totaling exactly **318,016 bytes**. Each byte count and SHA256 matched the frozen annex pointer. All masks passed finite-value, `{0,1,2,3}` label and same-source shape/affine checks. Header geometry and a64MiB array ceiling were checked before materializing one mask at a time. Existing40 and prospective14 exclusions were rechecked against source identities. Full MRI intensity images were not downloaded.

| New mask eligibility | Level45 | Level91 | Level136 |
|---|---:|---:|---:|
| Tumor positive≥1,000 pixels | 1 | 8 | 1 |
| Tumor zero-mask negative | 8 | 2 | 5 |
| Tumor small positive excluded | 3 | 2 | 6 |
| Enhancing positive≥1,000 pixels | 0 | 1 | 0 |
| Enhancing zero-mask negative | 11 | 5 | 8 |
| Enhancing small positive excluded | 1 | 6 | 4 |

The new masks add one upper-level tumor-positive case but **do not supply any upper-level enhancing-positive case**. Merely expanding the cohort does not resolve the enhancement task's three-level support gap.

Combining the original28 training geometry references with the12 new mask-only subjects yields a candidate pool of40 distinct training-source patients. Reapplying the exact same maximum-matching algorithm gives a mask-geometry maximum of **32 tumor patients** (16/16; per-level pair quotas10/4/2) and **12 enhancing patients** (6/6; pair quotas2/4 at levels45/91). These are potential future cohorts, not rendered examples or eligible training datasets. New geometry rows explicitly carry `NOT_DOWNLOADED_OR_RENDERED` image status. MRI retrieval is justified for a bounded subset that adds missing geometry support, followed by MRI/mask affine verification and visible-evidence review. No enhancement claim should extend to level136.

Final mask evidence: `local_data/ds007045_bounded/overnight_c2_masks_v2/receipt_PRIVATE.json`; derived combined geometry: `combined_geometry_PRIVATE.json` in the same directory. An initial sandbox DNS failure created only an empty v1 directory; the authorized network run used a new v2 directory and completed. No historical dataset or selection was overwritten. Two new tests verify checksum/size rejection and exclusion of reserved/existing subjects.

## C2 actual image-linked upper-level extension

`OBSERVED` / `C_TASK_ELIGIBILITY`: A targeted follow-up selected exactly one new training-source patient from the twelve downloaded masks whose level136 tumor area meets the unchanged≥1,000-pixel threshold. There was exactly one qualifying patient, with1,867 tumor pixels; enhancing area733 remains below its eligibility threshold. Deterministic SHA256 tie breaking and exclusion of the original40 and reserved14 are implemented, even though no tie occurred.

The official matched FLAIR/T1ce pair was actually downloaded: **2 MRI volumes, 1 patient, 10,317,786 bytes total**. Both payload sizes/SHA256 match the frozen annex references. Serial decoding verified both volumes have shape182×218×182, LAS orientation,1mm spacing, finite intensities and the same affine as the source mask. One volume array at a time used28,884,128 bytes. No model was loaded.

Private single-channel and paired level136 PNGs were rendered using the historical policy: nonzero-volume subsampling at stride16 for0.5/99.5 percentile intensity bounds, `np.rot90`,2× bilinear scaling, and modality titles only. There is no mask overlay, patient identifier or target text in the rendered pair. A visual implementation check confirmed both MRI panels are rendered and the expected modality headings are present; this was **not clinical or independent visual-answerability adjudication**.

Paired PNG SHA256: `45bec4d19bcbf5ce9309ef81d40d723403d9da1fc45be0a51dc4a0a6fa426a57`.

This advances a single geometry candidate from mask-only to actual image-linked evidence. It does not make the combined32/12 candidate cohorts image-ready, does not add a primary training task, and does not change any B schedule. Task admission remains `ENGINEERING_ONLY`; clinical visibility remains `NOT_VERIFIED`.

Source and tests: `scripts/c2_targeted_image.py`, `tests/test_c2_targeted_image.py`. Final private source URLs, payload hashes, mask linkage, six-entry Mission Alignment Check and rendering receipts reside in `local_data/ds007045_bounded/overnight_c2_upper_image_v1/`. All eleven focused C regression tests passed. Downloads and rendering were split into separate phases; serial decoding occurred only after the lead confirmed its model process had exited.
