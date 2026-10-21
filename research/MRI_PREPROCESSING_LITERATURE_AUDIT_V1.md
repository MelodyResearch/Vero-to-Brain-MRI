# MRI preprocessing literature audit v1 (data-preparation comparison, not a research plan)

Branch `research/vero-mri-data-preparation-v1`. This document does not modify or compete with `CURRENT_RESEARCH_PLAN.md`; Vero-style GSPO remains the method line. No model, MLX, RL/SFT/GSPO, final100 or new16 outcome was touched.

## 1. Source access status — read this first

| Source | Requested | Actual access in this session | Status |
|---|---|---|---|
| arvindmvepa/mpLLM | code + paper | `gh api` / `curl` / WebFetch each required approval that was not granted; two delegated read-only agents were also denied Bash | `NOT_VERIFIED` (no commit SHA, license, paper version, or code line read) |
| gfnnnb/MM-NeuroOnco | same | same | `NOT_VERIFIED` |
| ChatMED/neurovlm-bench | same | same | `NOT_VERIFIED` |
| AdvancingEther/MedVol-R1 | same | same | `NOT_VERIFIED` |
| Yuxiang-Lai117/Med-R1 | same | same | `NOT_VERIFIED` |
| zlab-princeton/vero | same | same (an earlier full Vero paper audit exists in-repo: `research/paper_reading/2604.04917-vero.md`, not repeated) | `NOT_VERIFIED` this session; prior audit is the only evidence |
| BAAI-DCAI/M3D | small-3D/VQA interface reference | same | `NOT_VERIFIED` |

The earlier root-side observation that the initial six GitHub pages were reachable is the root's report; I could not reproduce it from this sandbox. **No commit pins, licenses, paper versions or file:line evidence are claimed here.** Nothing below attributes a specific preprocessing practice to a source paper. The sections that follow are therefore a *question list to resolve when access exists*, plus what is established from this repo's own code and receipts.

To unblock: grant approval for `gh api`/`curl` to `api.github.com`, `raw.githubusercontent.com`, `arxiv.org` (or run the fetch at root and drop text files into `local_data/mri_data_preparation_v1/sources/<repo>/`). Digest template per repo: pinned SHA/date, SPDX license + LICENSE presence, paper title/arXiv id/version/publication state, files read vs missing, then the questions in §2.

## 2. Per-source questions to answer from code (all currently NOT_VERIFIED)

Common questions: raw data type; NIfTI loader; reorientation to RAS or none; spacing resample or none; foreground/brain crop; fixed shape and interpolation (continuous vs nearest for masks); normalization scope (per-volume vs per-modality vs dataset, percentile vs z-score, train-only fit); sequence organization and registration assumptions; image–mask alignment; 2D/2.5D/3D representation and key-slice rule; mask→question rules and thresholds; patient-level split and dedup; verifier/metric.

Source-specific focus requested by the researcher: mpLLM `dataset/prepare_brats_3d_dataset.py`, `create_brats_3d_vqa_dataset.py`, `create_brats_imaging_dataset.py`, `data/vqa_dataset.py`; MM-NeuroOnco `data_processing/`, `benchmark/splits/`, `evaluation/`; NeuroVLM-Bench `map_images.py`, `split.py`, `evaluate.py` (modality vs plane vs diagnosis separation); MedVol-R1 `data_preparation/`, `evaluation/grounding/` (note: public experiments mainly CT — do not transfer parameters to brain MRI); Med-R1 data/RL/SFT interface; Vero `docs/DATA.md`, `vero-rl/`, `vero-eval/`; M3D loader/VQA interface as small-3D reference only.

`PROPOSED` use (not evidence): treat each source as a design reference for (a) mask→question families, (b) patient-level split practice, (c) grounding verifiers — to be adopted only after the file-level reading above.

## 3. What this repo already establishes (OBSERVED; local receipts)

From `results/SEQUENCE_QUALIFICATION_V1.json` and renderer code (`src/mri_rl/sequence_qualification_v1.py`):
- 56 TRAIN volumes, all LAS storage, MNI-like 182×218×182; 28/28 same-person FLAIR/T1ce pairs share shape and affine; 56/56 source SHA256 match the pinned annex.
- The historical PNG render (`np.rot90(volume[:,:,z])`, nonzero `[::16]` 0.5/99.5 percentile, uint8, 364×436 bilinear) was reproduced 168/168 pixel-exactly, max abs diff 0.
- Coverage review at z045/091/136: 28 adequate_inferior, 28 adequate, 27 adequate_superior, 1 pair insufficient (abstained).
- Leave-person-out fixed-feature 1-NN on intensity summaries: 152/168 = 0.905 with no fitted parameters; categorical site/scanner/patient resubstitution ceiling 0.5.
- Filenames encode the target for all 168 rows (`filename_target_lookup_if_exposed_as_text` 168/168); the inspected native message path does not serialize paths.

## 4. Limits

No claim of reproducing any published pipeline is made. No unrun code is described as reproduced. Cross-model review: `CROSS_MODEL_REVIEW=NOT_VERIFIED`.
