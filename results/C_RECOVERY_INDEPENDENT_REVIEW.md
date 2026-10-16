# C recovery independent engineering review

**Verdict: PASS_WITH_LIMITATIONS. Evidence: TESTED_CPU / OBSERVED.** An independent Codex agent replayed the new recovery analysis using a separate implementation, actual PNG bytes, and saved source/schedule/geometry receipts. Same model family; private cross-model review remains NOT_VERIFIED. No author analysis functions, MLX, model libraries or nibabel were imported. No model launch or training is approved by this review.

## Independently reproduced results

Every saved prediction, selected neighbor identity, held-out site/patient exclusion, pooled count, patient macro and site macro matched. Features were recomputed from actual hashed PNGs: bilinear 16×16 grayscale thumbnails and normalized 16-bin histograms; squared distances and record-ID tie breaking were independently calculated. All 168 sequence images and 44 C2 task rows were covered.

| Cohort | N | Thumbnail | Histogram | Fold majority | Fold level-majority |
|---|---:|---:|---:|---:|---:|
| C2 tumor | 32 | 23 | 18 | 11 | 10 |
| C2 enhancing mask | 12 | 4 | 4 | 3 | 3 |
| Sequence z45 | 56 | 53 | 53 | 28 | 28 |
| Sequence z91 | 56 | 49 | 48 | 28 | 28 |
| Sequence z136 | 56 | 52 | 51 | 28 | 28 |

Each patient maps to its source site. The ten C2 additions were checked against original-source identities and the existing reserved exclusion list; no reserved images or outcomes were accessed. The 28 original training-patient inventory, sequence labels, saved image hashes and A/B exact-record exposure flags reproduce.

All **84** claimed supervised-panel-exposed sequence frames were independently compared pixel-for-pixel with the matching actual panel crop. Exposure totals are 56/56 z45 frames, 28/56 z91 frames and 0/56 z136 frames for both A and B. Exact sequence-record supervision remains a different quantity: 28 at z45 and zero at the other slices. These remain previously selected training patients, including at z136.

## Geometry receipt checks and limits

The independently recomputed receipt aggregates match: 68 planes/38 patients; 38 C2 planes/35 patients; 21 preserved grid candidates. Positive counts are 40/20/21; boundary-under-10 counts 14/4/0; centroid-cell fractions below one-half occur 9/2/1 times. All saved grid centroids, pixel-center offsets, boundary-distance arithmetic, cell sums, source-reference labels and recorded tumor/enhancing areas checked consistently. The 21 candidates have 14 single-component and seven multicomponent masks; minimum largest-component fraction is 0.81492537. The five nested tumor/enhancing area-state counts reproduce as 6,10,4,1,17.

**Source mask arrays and affines were not reloaded in this independent pass.** Their geometry/orientation, segmentation component extraction and centroid-on-mask results remain supported by the author's hash-bound mask execution receipt, whose preserved executed-source hash was verified. They are not independently rerendered results. Clinical visibility, source annotation quality, scanner/protocol invariance, the title-band claim and expert interpretation were not newly adjudicated here.

## Mission and interpretation

This strengthens engineering evidence for task admission within the Vero-to-MRI transfer program. It does not admit C as a clinically validated task family, demonstrate MRI reasoning, or make these selected training-source folds an untouched prospective test. Strong histogram discrimination is compatible with acquisition/intensity shortcuts. Cohort eligibility used all selected masks before this diagnostic; no fresh holdout was created. Preserve ENGINEERING_ONLY and the separate frozen likelihood experiment.

Private independent evidence: `local_data/ds007045_bounded/c_recovery_independent_v1/verification_PRIVATE.json`, SHA256 `9e93e7bd7f4eafd41e29e186900270d73f1d105c3193cd00bb888bb9f66b5d43`. The independent replay script and per-query distances are preserved beside it. No patient-level evidence is copied into this public report.
