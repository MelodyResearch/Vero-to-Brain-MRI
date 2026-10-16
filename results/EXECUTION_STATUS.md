# Execution record — 2026-10-15

**Historical first-pilot snapshot.** The two-subject counts and `MODEL_INFERENCE: NOT_RUN` statements below describe the original checkpoint and are superseded by the current [README](../README.md), [Vero-to-MRI research contract](../research/VERO_TO_MRI_PRIMARY_EXECUTION_CONTRACT.md), [multi-task data audits](../data/MULTITASK_MRI_DATA_TAXONOMY.md), and [real frozen-model diagnostic](FROZEN_MODEL_DIAGNOSTIC.md). Preserve this record as chronology, not current status.

## Verified execution

- **SOURCE_AUDITED:** A project-local Norman-style reading skill and protocol were validated; 11 specified medical/MRI papers plus Vero and BrainTRACE received individual primary-source audits. Three PDF-size gaps were resolved by exact-version local PDF inspection. Independent novelty, statistical and authenticated Claude Code reviews are archived; the Claude review's PDF/code limits are explicit. These are literature findings, not reproduced model outcomes.

- **REAL_IMAGE_PILOT_VERIFIED:** OpenNeuro Dallas ds004856 v1.3.0 metadata commit `077d0a4e4f32b7c18e3628893a96708ef24e1178`. Eight genuine NIfTI files from two subjects and two waves (T1 MPRAGE plus FLAIR each) matched their Git-annex byte counts and SHA-256 keys, decoded to finite 3D arrays, and produced deterministic anatomical evidence views. Four wave 1→2 subject/modality pairs passed identity and elapsed-time checks. One FLAIR pair had discordant 6 versus 2.5 mm slice spacing. Private exact hashes, affines, voxel statistics, and PNGs remain under ignored `local_data/`.
- **TESTED_CPU:** `PYTHONPATH=src pytest -q` passed 13 tests after adding physical-aspect and voxel↔display-coordinate checks. The Dallas pilot command returned `subjects=2`, `sessions=4`, `volumes=8`, `T1_pairs=2`, `FLAIR_pairs=2`, `spacing_discordant_pairs=1`. The coronal FLAIR render was visually inspected after the aspect correction.
- **SOURCE_AUDITED:** The FreeSurfer 5.3 brainsummary workbook was downloaded and checksum matched to its annex key; selected wave 1/2 global-volume rows exist. No repeatability threshold, diagnostic reference, or image-grounded region target was established from that workbook.
- **INDEPENDENT_REVIEWED:** Imaging, novelty, RL/statistics, and significance reviewers inspected distinct risks. A separately authenticated Claude Code session produced a source-limited independent critique. The shared conclusion is that the two-subject pilot proves access and rendering only; it cannot establish MRI reasoning, balanced-RL benefit, or clinical utility.
- **GITHUB_PRIVATE_PUSHED:** Initial commit `a80641307ce2fcd14511df27e1adfef3f1b08bea` was pushed to the private `MelodyResearch/Longitudinal-BrainMRI-Verifiable-RL` repository and remote parity verified. Later commit parity is checked separately after each push.

## Not executed or validated

`MODEL_INFERENCE`, `RL_TRAINING`, `TRANSFER_MATRIX`, `SCIENTIFIC_VALIDATION`: **NOT_RUN**. MPS and CUDA were unavailable in the current Python environment, and no authorized GPU endpoint was found. SGMRI-VQA image-linked QA, BrainTRACE, and NeuroQA remain `DATA_BLOCKED` on their separate upstream image/access requirements. Dallas provides real images but does not cure those benchmark-specific blocks. No generated fixtures, publisher counts, FreeSurfer deltas, or two-subject bootstrap should be presented as model or clinical results.

## Decision

**GO** for expanding a controlled real-image feasibility cohort. **REFORMULATE** longitudinal reward claims around image-visible, repeatability-qualified targets and no-image controls. **NO GO** for any claim that Vero-style RL has improved MRI reasoning, since no model was run and no eligible patient-held-out comparison exists.
