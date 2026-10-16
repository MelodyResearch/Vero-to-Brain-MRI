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

Site results are same-sample association diagnostics, not fitted-model generalization estimates. Within-level balancing removes level alone as a class discriminator in these selected cohorts, while substantial site associations remain. The earlier thumbnail/histogram1NN results concern the smaller C1 cohorts and must not be presented as results on these new C2 cohorts. No VLM or no-image inference was executed on the expanded cohorts. The subsequent fixed PNG-only 1NN diagnostic is recorded separately below.

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


## Frozen C2 image shortcut diagnostic — actual CPU execution

Tag: `EXPLORATORY_DIAGNOSTIC`, evidence: `OBSERVED`, status: `TESTED_CPU`.
The exact C1 feature definitions and original prediction function were applied to the unchanged C2 cohort. This is a new result on **32 tumor /12 enhancement rows**, separate from the historical C1 **26 /10** results.

| Fixed diagnostic | C2 tumor | C2 enhancing mask |
|---|---:|---:|
| Fixed no | 16/32 (50%) | 6/12 (50%) |
| 16×16 grayscale thumbnail, 1NN | 18/32 (56.25%) | 5/12 (41.67%) |
| 16-bin grayscale histogram, 1NN | 18/32 (56.25%) | 6/12 (50%) |

Each query excludes every row from the same patient. Neighbors are selected **within the same task category**, so the nine patients appearing in both categories never supply cross-category neighbors. All four prediction sets have zero same-patient neighbors. The private receipt retains the query/neighbor patient and record IDs, image hashes, squared distances and predictions. Overall denominators remain44 task rows,35 unique patients; the two task results are not independent replications.

The thumbnail uses bilinear grayscale16×16 pixels divided by255; the histogram uses16 equal bins over[0,256), normalized by pixel count. Squared Euclidean distance and record-ID tie breaking are unchanged from C1. All images were rehashed against the frozen render receipt. No features, thresholds, labels, exclusions or hyperparameters were selected from these outcomes.

This modest tumor result does not isolate spatial lesion information: the intensity histogram gives the same aggregate accuracy, although individual predictions differ. Enhancement does not exceed the fixed-no baseline. These are exploratory diagnostics on a training-selected cohort, not evidence of held-out patient generalization, clinical visibility or failure of a capable MRI model. Site association remains a separate measured rival explanation; it was not removed by this diagnostic. There is no new task-admission or C-training approval.

### Mission Alignment Check

1. **Main-question link:** establish simple image baselines before C tasks enter controlled Vero-to-MRI recipe transfer.
2. **Claim/falsifier:** test whether fixed coarse spatial or intensity features predict the selected labels better than fixed no; preserve below-baseline outcomes. Either result is insufficient for lesion understanding.
3. **Isolated change:** the frozen C2 cohort replaces C1 input only; C1 algorithms, C2 labels/images, B training and historical controls remain fixed.
4. **Provenance:** immutable render receipt and each PNG hash verified; exact script, C1 source, scope and result hashes below. Per-patient data stay private.
5. **Shortcut/baseline:** per-category leave-one-patient-out1NN versus fixed-no, alongside earlier level/site associations. No pooling of paired patients or feature tuning.
6. **Resource/decision:** one finite44-PNG pass, one image at a time, three thread caps set to1. Actual process completed in0.22 seconds with MLX/MLX-VLM/Torch/nibabel import blocking. Continue visibility/site checks; retain `ENGINEERING_ONLY` admission.

### Reproducibility

- New adapter: `scripts/c2_image_shortcut_baseline.py`; original C1 implementation remains unchanged.
- Regression evidence: three new tests failed before implementation, then the new and existing C1 tests passed (5 total). They cover same-patient exclusion/ties, exact feature values, image-byte drift and duplicate-patient rejection. No model was imported.
- Private output: `local_data/ds007045_bounded/overnight_c2_shortcuts_v1/receipt_PRIVATE.json`.
- Render cohort SHA256: `5786681d14498f4ac2545d858e293a5f12e5ad0cd31234e8c2b6a79ef030b06b`.
- C1 source SHA256: `f88cc384e5d50a152826ee852418fc6b9a3ceb87fc5f9ce4f3204f9e2d03f283`.
- C2 shortcut source SHA256: `ea3fa5f5c564d83941f6d1e68e88d595bd5fbf50bbda6b7df4d3a59e1edb2254`.
- Pre-execution scope SHA256: `d2a50c99069c4aa6fedcd088290779842690d876ccc71f4fcfb0698ebfe08903`.
- Result receipt SHA256: `59d15c50d24537a066f720c311a81b22decc529499ffef99a63ebf643cff6a75`.
- `REPRODUCED`: isolated Codex methods/statistics reviewer independently reproduced all88 predictions, exact neighbor IDs, squared distances and confusion counts; verified hashes and no same-patient neighbors or within-category duplicate image hashes. It used integer grayscale counts and scalar distance summation without importing producer predictions. Verdict: `PASS_WITH_LIMITATIONS`; source/caption/site/anatomy shortcuts and training-cohort selection remain limitations. See `results/OVERNIGHT_METHOD_STATS_REVIEW.md`. Cross-model Claude review: `NOT_VERIFIED`.
