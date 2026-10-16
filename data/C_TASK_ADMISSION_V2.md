# MRI task admission cards — C v2

These cards preserve Control B and candidate C-v1. They apply to a separate training-only C redesign. No candidate below currently has scientific transfer admission: source-valid target geometry alone does not establish perceptual answerability. Current status is `ENGINEERING_ONLY` or `NOT_VERIFIED`, not `ELIGIBLE`.

## Shared source and interpretation boundary

Inputs are real ds007045 MRI, MNI152 registered/skull-stripped/N4 corrected FLAIR and contrast-enhanced T1, with source tumor masks. Existing rendering takes voxel-axis2 indices45/91/136, `np.rot90`, and2× bilinear display. Source affine orientation is LAS; axis2 is superior. Image-space left/right and grid cells are display coordinates; they must not be described as patient anatomical left/right without explicit affine validation. A 2D frame does not prove 3D lesion extent or clinical diagnosis.

Train reference population is28 patients and382 records. Four development and eight old test patients were not used to choose C1 eligibility or quotas; the14 reserved prospective subjects remain excluded from training. A source patient may contribute to multiple tasks, so task counts are not independent sample counts. Patient grouping and site-stratified uncertainty are mandatory for later evaluation.

## C3-A — Sequence/contrast discrimination

- **Scientific question:** Can a model distinguish FLAIR from contrast-enhanced T1 using MRI intensity/contrast evidence across anatomy and patients?
- **Input:** One unlabelled real MRI frame; no filename, modality title, mask or metadata text. Current paired source supports both sequences at the same three fixed levels.
- **Reference provenance:** Pinned source modality records and paired derivative identity; exact image hashes. This verifies acquisition labels, not pathology.
- **Allowed evidence:** Tissue/CSF contrast and sequence-dependent appearance. Head crop or site artifacts are unwanted shortcuts.
- **Distribution:** Preserved C-v1 sequence candidate28 patients,14/14 labels; levels45=10,91=9,136=9. This candidate was not optimized for within-level class balance. A stronger future counterfactual design uses both sequences of each training patient at each selected level, grouped by patient in every split.
- **Patient/site/protocol:** Source-site distribution must accompany a frozen paired schedule; current28-patient candidate inherits the bounded training sites. Unknown scanner/acquisition details remain a protocol confound.
- **Verification:** Exact two-label parser plus image hash/modality mapping; swapped same-patient same-level image must be scored against the new image's label.
- **Baselines:** Fixed class50% for globally balanced candidate. Level/site cross-tabs, no-image VLM, blank-image VLM and simple intensity-feature model remain required for a frozen C3-A schedule. Historical B controls do not validate this new schedule.
- **Failure/abstention:** Corrupt/misaligned pair, sequence text visible, severe artifacts, unverified modality or unidentifiable appearance. Do not invent labels for missing acquisitions.
- **Admission:** `ENGINEERING_ONLY`. Need blind visible-evidence check and grouped shortcut baselines. This is an acquisition/contrast capability distinct in target provenance from mask-derived lesion tasks, but success alone is not clinical reasoning.

## C3-B — Lesion spatial localization

- **Scientific question:** Can a model locate the visible lesion spatially in paired MRI, beyond slice/site priors?
- **Input:** Matched FLAIR/T1ce frame panels with consistent coordinate convention; no mask overlay or lesion bounding box. Modality panel titles identify channels, not target locations.
- **Reference provenance:** Source-mask centroid projected through the identical display transform. Current C-v1 grid candidate21 patients with tumor area≥1,000 native pixels and centroid≥10 display pixels from a grid boundary.
- **Distribution:** Levels45=10,91=10,136=1. Labels left-middle8, right-middle7, center-middle3, center-top2, left-bottom1. Imbalanced and not suitable as a primary four-task benchmark without redesign.
- **Allowed evidence:** Visible lesion location in MRI. A segmentation centroid can lie outside a visible mass for multifocal/irregular lesions; eligibility needs image+mask consistency adjudication, with ambiguity exclusions declared before outcomes.
- **Patient/site/protocol:** One candidate frame per eligible training patient; site×grid×level distribution needs freezing and counterfactual matching. MRI registration/cropping may encode location or source.
- **Verification:** Deterministic displayed-grid answer; affine/transformation unit tests and boundary abstention. This is2D localization, not3D Dice or clinical laterality.
- **Baselines:** Global/level/site-conditioned grid priors; no-image VLM; simple image model and lesion-visible controls pending for C3-B. Existing binary C1 baselines do not cover grid localization.
- **Failure/abstention:** Small/ambiguous lesion, boundary centroid, frame mismatch, registration failure or no visible lesion. Do not infer visible disease solely from a mask.
- **Admission:** `ENGINEERING_ONLY`. Spatial mapping is distinct from sequence identification; three routes from one tumor mask are not three independent clinical abilities.

## C1-T — Matched tumor-presence diagnostic

- **Question/input:** Does the displayed paired MRI frame contain visible tumor-associated tissue? Source binary endpoint is any tumor-mask pixel, screened to require≥1,000 pixels for positives and exactly0 for negatives.
- **Geometry/reference:** Same fixed axial levels and transform; source mask hash and selected record/image hash bound in private receipt. Positive threshold is an engineering visibility proxy.
- **Counts:**26 patients,13/13 classes; levels18/6/2 at45/91/136, each level exactly balanced. Sites BA1/BO4/FO3/JS3/KA6/ME3/NSK4/TO2.
- **Visible evidence/limits:** Lesion appearance must be independently checked. Zero mask means mask absence on this frame, not absence of disease in this patient. No diagnosis, prognosis or longitudinal claim.
- **Verification:** Strict binary parser against unchanged source labels; patient-disjoint matching and area checks executed.
- **Shortcut controls:** Fixed-no13/26; level-conditioned association13/26; site17/26; site+level18/26. Resubstitution is descriptive. Actual VLM no-image `NOT_RUN`; executed CPU LOPO thumbnail1NN12/26 and intensity-histogram1NN13/26 (no tuning).
- **Failure/abstention:** Source inconsistency, small positives, frame/affine mismatch, ambiguous rendered evidence. Site leakage or a result not exceeding metadata baselines defeats an image-reasoning claim.
- **Admission:** `ENGINEERING_ONLY`, even though matching is feasible. Next: blind positive/negative evidence checks, then patient-grouped image versus no-image comparisons on a separately frozen evaluation cohort.

## C1-E — Matched enhancing-mask diagnostic

- **Question/input:** Does the displayed paired MRI frame contain the source mask's enhancing component? Endpoint is label3 pixels, with the same≥1,000/zero screening.
- **Counts:**10 patients,5/5; level45=4 and91=6 with class balance inside each. Level136 has no eligible positive and is explicitly outside this version. Sites BO3/FO2/JS1/ME2/NSK2.
- **Visible evidence/clinical limits:** Postcontrast T1 alone does not establish enhancement relative to precontrast acquisition. Mask labels are segmentation annotations; clinical enhancement assessment and visibility remain unverified. Shared source mask with tumor-presence means this is not an independent pathology capability.
- **Verification:** Source label3 count plus strict binary parser; no relabeling of small positive masks.
- **Shortcuts:** Fixed-no5/10; level association5/10; site and site+level associations7/10. VLM no-image `NOT_RUN`; executed CPU LOPO thumbnail1NN6/10 and intensity-histogram1NN4/10 (no tuning). Tiny sample precludes robust generalization claims.
- **Failure/abstention:** Ambiguous postcontrast evidence, source label disagreement, small regions, matching/geometry failure. No threshold relaxation to reach desired size.
- **Admission:** `ENGINEERING_ONLY`; not a primary transfer task until endpoint validity and visible evidence are established.

## Admission rule and next decisive experiment

No C task enters the primary Vero transfer matrix until its reference alignment, prompt boundary, visible-evidence audit, patient grouping and prespecified shortcut comparisons pass independent review. Freezing a larger source cohort without these checks does not repair answerability. C1 establishes that a balanced multilevel geometry candidate exists; it does not establish MRI reasoning.

The most informative next experiment is blind image-level answerability assessment plus patient-grouped image/no-image/level controls for sequence contrast and lesion spatial reasoning. Maintain same-patient same-level counterfactual pairs where available; score against visible targets. Run Vero task-mixture comparisons only after at least two task capabilities pass admission, from common starting weights and matched budgets.

## C2 mask-only expansion evidence

Twelve additional training-source masks were actually downloaded and checksum/geometry verified, totaling318,016 bytes, without touching the14 reserved prospective subjects or the existing development/test patients. They provide eligible tumor positives at levels45/91/136 in counts1/8/1, and enhancing positives0/1/0. Combining mask geometry with the original28 training patients supports prospective balanced maxima32 tumor and12 enhancing subjects under the unchanged eligibility rules. These counts are **not image-ready task sizes**: no corresponding new MRI intensity payloads have been retrieved or visually checked. C admission remains `ENGINEERING_ONLY`; enhancement at level136 still lacks a positive class.

### One upper-level real-image pair now available

`OBSERVED`: One of the twelve new mask-only subjects now has its two matching real MRI volumes downloaded and source-hash/affine verified (10,317,786 bytes;182×218×182;LAS;1mm). Its level136 frame contains1,867 source tumor-mask pixels and733 enhancing-mask pixels. Private FLAIR/T1ce panels and an unoverlaid paired image were rendered with the historical display policy. Thus one additional upper-level geometry candidate now has actual linked pixels, while clinical answerability remains unadjudicated. The733-pixel enhancing component is still excluded by the unchanged≥1,000 positive-eligibility rule. No C training admission or full-cohort image availability is implied.
