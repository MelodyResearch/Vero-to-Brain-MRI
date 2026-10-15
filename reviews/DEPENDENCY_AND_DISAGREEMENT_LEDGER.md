# Research dependency and disagreement ledger

| Claim or operation | Dependency | Current decision |
|---|---|---|
| Genuine longitudinal MRI access | Public license, pinned metadata, checksum-matched bytes, subject/session linkage, decoded 3D images | **Met for eight Dallas volumes only** |
| Visually faithful evidence | Orientation, affine and voxel spacing, deterministic selection, coordinate invertibility, image review | **Met for pilot rendering**; anatomical alignment and per-region visibility remain untested |
| Image-grounded task | Hidden target, image-visible signal, independent or well-characterized reference, no-image control, patient split | **Not met** |
| Model transfer | Eligible held-out cohort, matched base/Vero checkpoints, identical evidence and decoding | **Not run** |
| Balanced-RL benefit | Above plus matched updates, informative group diagnostics, normalization control, seeds, patient-level uncertainty | **Not run** |

**Reviewer disagreement / resolution:** The imaging reviewer allowed a conditional FreeSurfer-referenced change-direction task; the statistical reviewer considered the two-subject sample incapable of supporting a scientific reward or interval; Claude considered cross-wave consistency potentially novel but warned that small changes may be measurement noise. We retain the *measured numeric differences* as private exploratory data and do not score direction or claim biological progression until an external repeatability threshold and image-only control are fixed. All reviewers agreed that Dallas supplies no voxel boxes, segmentation ground truth, or radiology report targets for the existing generic verifier.

**Acquisition issue:** One real FLAIR pair changes through-plane spacing from 6 to 2.5 mm. Physical display scaling fixes the image presentation, but cannot remove this confound. T1 wave 1→2 is the more defensible first paired-image path. Wave 3 has a scanner change and is outside this pilot.
