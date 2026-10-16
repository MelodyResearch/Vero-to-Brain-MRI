# Frozen C2 cohorts now linked to real MRI images

Tag: `C_TASK_ELIGIBILITY`. Execution status: `FROZEN_COHORT_IMAGE_LINKED`; scientific admission: `ENGINEERING_ONLY`; clinical visibility: `NOT_VERIFIED`; C model training: `NOT_RUN`.

## Actual completed work

The frozen maximum-matching results selected32 tumor-presence patients and12 enhancing-mask patients. Their new-source union required exactly10 of the already selected12 mask-only patients. No additional subjects were searched for or selected, and the original40 plus reserved14 prospective subjects were excluded from the new cohort.

Exactly20 corresponding FLAIR/T1ce MRI volumes totaling **124,875,617 bytes** are now verified. Two volumes from the earlier upper-level pair were reused; **18 files /114,557,831 bytes** were newly downloaded from the official public OpenNeuro source. This stayed below the150MB retrieval scope. Every file matched its pinned annex size and SHA256.

After the lead confirmed the model process had exited and reserved an idle window, one image volume at a time was decoded with thread caps. All20 passed finite-intensity and source-mask shape/affine checks. Eleven needed paired frames were rendered using the historical intensity normalization, rotation, bilinear display and modality-only titles. The array process exited normally. Frozen original training images were also rehashed when assembling the cohorts.

The result is **44 image-linked task rows across35 distinct patients and38 unique image hashes**. Nine patients occur in both task cohorts. These are eligible source/geometry candidates awaiting visual-answerability adjudication, not44 independent patients or two independent clinical capabilities.

## Exact cohort distributions

| Cohort | Patients / rows | Binary labels | Level45 | Level91 | Level136 |
|---|---:|---|---:|---:|---:|
| Tumor presence | 32 | 16 yes /16 no | 20 | 8 | 4 |
| Enhancing-mask presence | 12 | 6 yes /6 no | 4 | 8 | Excluded |

Each included level has equal yes/no counts. The positive threshold remains≥1,000 native mask pixels; negatives require exactly zero. Small positives remain excluded. Enhancement still has no eligible positive support at level136.

Tumor sites: BO4, ME4, FO2, JS3, KA8, BA1, TO4, NSK6. Enhancement sites: FO2, KA1, ME2, JS2, NSK4, BO1.

## Residual shortcut evidence

| Nonimage rule | Tumor | Enhancing mask |
|---|---:|---:|
| Fixed no | 16/32 | 6/12 |
| Level-only majority association | 16/32 | 6/12 |
| Site-only majority association | 21/32 | 9/12 |
| Site+level majority association | 22/32 | 9/12 |

Site results are same-sample association diagnostics, not fitted-model generalization estimates. Within-level balancing removes level alone as a class discriminator in these selected cohorts, while substantial site associations remain. The earlier thumbnail/histogram1NN results concern the smaller C1 cohorts and must not be presented as results on these new C2 cohorts. No VLM, no-image inference or new image classifier was executed on the expanded cohorts.

## Scientific interpretation and next gate

This expansion solves the actual-image availability gap for the frozen geometry matches. It does not establish visible disease or clinically valid enhancement labels. Both routes still share source tumor-mask provenance; they must not be promoted as two independent clinical reasoning skills. The stronger task comparison remains sequence/contrast recognition versus independently validated spatial reasoning, with these binary routes usable as explicit engineering diagnostics.

Next, freeze blind visual-answerability checks and site-aware controls on these source-linked images. Any exclusion or change to the cohorts requires a new version and explicit intervention record. Do not train C or change the B schedule solely because all image bytes are available.

## Reproducibility and resource discipline

- Source commit: `283007b96977b5b3285bd22cd67b7d97114b1eed`.
- Script: `scripts/c2_frozen_cohort.py`, with separate `plan`, `fetch`, `render` phases.
- Two regression tests cover exact selected-union logic, existing/reserved patient exclusions and total byte limits.
- Scope was frozen before fetching, including all six Mission Alignment entries, source/matching/selection hashes, exact files and finite budgets.
- Streaming downloads and hashing used no NumPy/MLX imports. Array work ran only in the explicitly reserved GPU-idle window; one source volume array at a time was released before the next.
- Private evidence directory: `local_data/ds007045_bounded/overnight_c2_frozen_cohort_v1/`.
- `scope_PRIVATE.json` SHA256: `70a286932eb5b320cb37e63e0c4209084b7dad1c0247898d1b22aa300d2c508a`.
- `download_PRIVATE.json` SHA256: `541922c7e7de7a86e71e03d1c32bc918f267650415a5f28e4bf8965b81641139`.

All raw MRI, source identities, image-linked cohort rows and private receipts remain outside Git. Independent reviewer verification is requested against these primary artifacts; engineering evidence never substitutes for clinical adjudication.

## Independent verification

`REPRODUCED`: The isolated Codex methods/statistics reviewer independently checked all20 MRI hashes against the pinned official annex pointers, the exact10-patient union and exclusions, image/mask header geometry and affine, all44 image-linked rows, and11 PNG hashes/headers. It reproduced35 patients,38 image hashes,32/12 task counts and the site/site×level associations above. Verdict: `PASS_WITH_LIMITATIONS` for source integrity, linkage and aggregate scope. It did not materialize MRI arrays or rerender the full cohort during the concurrent training window. Clinical visibility and primary scientific task admission remain `NOT_VERIFIED`. Its evidence is recorded in `results/OVERNIGHT_METHOD_STATS_REVIEW.md`.
