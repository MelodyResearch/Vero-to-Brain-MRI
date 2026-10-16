# Paired-change verifier challenge on real Dallas MRI

**Execution date:** 2026-10-15. **Status:** `IMPLEMENTED`, `TESTED_CPU`; **not** `INFERENCE_COMPLETED`, `TRAINING_COMPLETED`, or `SCIENTIFICALLY_VALIDATED`. The initial run covered 12 subjects. After a second eight-subject acquisition and registration batch, the **current run covers 20 subjects and 40 actual wave1/wave2 T1 NIfTI volumes** from pinned Dallas ds004856 v1.3.0; two of these subjects also have FLAIR wave pairs. Every source file was rehashed against its private download receipt. Per-subject identities, paths, hashes, image derivatives, transformations, and the private challenge receipt remain ignored under `local_data/`.

## What was tested

`src/mri_rl/paired_verifier.py` separates (1) **provenance**: BIDS subject, wave order, acquisition type and protocol, trusted file bytes and header geometry; (2) **image evidence**: the registration agent's actual 6-DOF T1 overlap/intensity QC, plus its explicit visual hold; and (3) **change claims**: measurement source, independent reference, and repeatability. Its comparator is a simpler rule that checks only subject, chronological order, and T1 sequence. Neither system is a learned model. A known-source SHA-256 rejection is file integrity, **not** learned recognition of a left/right flip or anatomy.

The registration input was bound to the exact image paths in the trusted cohort. Its `qc_pass` is a provisional numerical criterion based on foreground Dice and Pearson, not an expert registration verdict. All **20** registered T1 pairs passed this number-only flag; **19 remain uncertified** after visual review, and one is explicitly **held** due to persistent peripheral/field-of-view artifact visible in multiple planes. None is promoted to a scientifically eligible change-evaluation pair. The separate [registration audit](../data/LONGITUDINAL_QC_RESULTS.md) gives the image processing and metrics.

## Patient-level adversarial results

Each 20-pair condition uses the same 20 patients once, so rows are paired interventions and must not be treated as independent samples. FLAIR and pixel-flip rows have their own denominators. `ACCEPT` in the provenance column means only that scan linkage and source checks passed. `NOT_EVALUABLE` for biological change is deliberate abstention.

| Condition | Actual or controlled input | Simple baseline | Verifier observation | Biological change |
| --- | --- | ---: | --- | --- |
| Correct T1 pair and order | 20 real verified pairs, actual registration receipt | 20/20 provenance accept | 20/20 provenance accept; image QC 19 numerical pass but uncertified, 1 held artifact | 20/20 `NOT_EVALUABLE` |
| Reversed visits | Same real files, reversed argument order | 20/20 reject | 20/20 provenance reject | `NOT_EVALUABLE` |
| Wrong-person prior | Real wave1 from a different one of the 20 patients | 20/20 reject | 20/20 provenance reject | `NOT_EVALUABLE` |
| Wrong sequence | Real same-person FLAIR wave1 paired with T1 wave2 | 2/2 reject | 2/2 provenance reject | `NOT_EVALUABLE` |
| Left/right **header** flip | One affine axis sign inverted on each real scan record, bytes unchanged | 20/20 accept | 20/20 reject through trusted affine comparison | `NOT_EVALUABLE` |
| Coordinate/affine mismatch | +20 mm header translation on each real record, bytes unchanged | 20/20 accept | 20/20 reject through trusted affine comparison | `NOT_EVALUABLE` |
| Left/right **voxel array** flip | One true NIfTI array reflected; original header retained, derivative stays local | 1/1 accept | 1/1 reject because source SHA-256 no longer matches; this does not test visual asymmetry recognition | `NOT_EVALUABLE` |
| Unregistered physical grid substituted for registered evidence | Actual pre-registration image metrics from all 20 pairs | 20/20 accept | Numeric image gate rejects 18/20, **accepts 2/20** | 20/20 `NOT_EVALUABLE` |
| Acquisition protocol stress | Two real FLAIR wave pairs | Baseline has no spacing check | 1/2 incompatible under prespecified 10% voxel-spacing rule; not a T1 change test | `NOT_EVALUABLE` |
| Repeatability unknown | Real pairs with an engineering measurement field but no limit | 20/20 accept | 20/20 abstain on change due to missing limit | `NOT_EVALUABLE` |
| Below repeatability | Hypothetical −0.2 measurement and 0.5 limit **fixture** attached to each real pair | 20/20 accept | 20/20 abstain on change; no real patient measurement was tested | `NOT_EVALUABLE` |
| Fabricated/unsupported measurement | Hypothetical regional delta with no traceable source | 20/20 accept | 20/20 reject the **claim**, while provenance still accepts | Reject unsupported claim |

The unregistered-grid case shows that Dice ≥0.75 and Pearson ≥0.50 alone are insufficient: two same-person pairs satisfy those coarse thresholds even without the estimated rigid transform. In the initial 12-pair run this was **10 rejects / 2 accepts**; in the expanded 20-pair run it was **18 rejects / 2 accepts**. These image-derived QC observations are **not** sensitivity for disease or true anatomical change. The protocol stress uses actual FLAIR acquisitions and remains separate from the more comparable T1 cohort. The header and file-integrity attacks assess trust in known data provenance; they do not establish generalized 3D visual reasoning. The fabricated and repeatability examples are labeled engineering fixtures, never medical outcomes.

## Interpretation and next gate

**Observed result → supported conclusion:** A deterministic manifest plus geometry and provisional registration QC catches several controlled source and correspondence failures; it adds detection of header/file edits beyond the simple identity/order baseline. The simple baseline already solves the identity, visit, and sequence checks, so no RL benefit follows from those checks. Every biological change decision abstained or rejected an unsupported claim because the FreeSurfer workbook lacks a verified BIDS-to-row crosswalk, segmentation provenance, independent evaluation reference, and repeatability bound.

**Alternative explanations:** The strong rejection rates for header and pixel alterations are exact-release receipt checks. The 18/20 unregistered-grid result depends on coarse intensity similarity thresholds and scan positioning; 2/20 false acceptance shows why landmark or anatomical correspondence QC is needed. The visually held pair shows that a numerical pass can coexist with an obvious artifact. No model-generated text, reward learning, held-out evaluation, or clinical adjudication was involved.

**Scientific consequence → next experiment:** Preserve these gates as deterministic data validation and abstention controls. Obtain a defensible scan-to-measurement crosswalk, per-scan segmentation QC, scan-rescan or equivalent repeatability limits, and independent patient-level change review. Tighten registration QC with landmark/region correspondence and full three-plane review before constructing a verifier reward. A Vero-style RL comparison is still gated on answerable image-dependent tasks, valid rewards, and patient-disjoint evaluation.

## Reproduction

With the ignored licensed/public Dallas image payload and private receipts present:

```bash
PYTHONPATH=src python3 -m pytest -q tests/test_paired_verifier.py
PYTHONPATH=src python3 - <<'PY'
from pathlib import Path
from mri_rl.paired_verifier import load_verified_cohort, run_challenge
import json
root = Path('.')
result = run_challenge(load_verified_cohort(root), root / 'local_data/longitudinal_t1_registration_private.json')
(root / 'local_data/paired_verifier_challenge_private.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
PY
```

Observed: **8 verifier tests passed**; challenge ran first on 12 T1 subjects, then on the expanded **20 T1 subjects**, 2 FLAIR pairs, and 1 pixel-reflected derivative. The output in Git is aggregate only.
