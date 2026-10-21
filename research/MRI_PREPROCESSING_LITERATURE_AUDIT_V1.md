# MRI preprocessing literature audit v1 (data-preparation comparison, not a research plan)

Branch `research/vero-mri-data-preparation-v1`. Does not modify `CURRENT_RESEARCH_PLAN.md`; Vero-style GSPO remains the method line. No model, MLX, RL/SFT/GSPO, final100 or new16 outcome was touched. No third-party code was copied; this repo's contract is implemented independently.

## 1. Access and reading scope (OBSERVED)

Root fetched public sources with its authorized GitHub CLI into `local_data/mri_data_preparation_v1/sources/` (private local path; not committed). I read them from there; I did not use the network myself.

**Reading scope is selective, not full-paper.** For each paper I grepped the fixed-version `*_html.txt` for preprocessing, mask/VQA construction, split, dedup/leakage and publication-status passages and read only those lines. I did **not** read any paper end to end. For code I read the listed files in full or by targeted ranges; files that exist upstream but were not fetched are listed as `NOT_FETCHED` and any claim depending on them is `NOT_VERIFIED`.

| Repo | Pinned commit (date) | License evidence | Paper (fixed version) / publication state |
|---|---|---|---|
| arvindmvepa/mpLLM | `88336c3c0f3a` (2026-02-18) [tree](https://github.com/arvindmvepa/mpLLM/tree/88336c3c0f3ad0aa014ea7836befaf6d24f3631d) | Apache-2.0 (GitHub metadata + `LICENSE` fetched) | arXiv 2509.25889v3 (revised 12 Feb 2026); arXiv comment "17 pages, 3 figures"; no venue stated in fetched abs page |
| gfnnnb/MM-NeuroOnco | `320537c15881` (2026-08-07) | **GitHub license = null**; README badge says "Research Only" and links `./LICENSE`, but no `LICENSE` blob is in the fetched tree (80 entries) | arXiv 2602.22955v1; no venue in abs page |
| ChatMED/neurovlm-bench | `6e87ba48ed7d` (2026-01-01) | GitHub license = null; no license file in tree | arXiv 2603.24846v1; abs comment: "Manuscript **submitted** to BMC Medical Informatics and Decision Making" (submitted, not shown accepted) |
| AdvancingEther/MedVol-R1 | `051b20f64e28` (2026-10-05) | GitHub license = null; README badge says MIT (`README.md:9`); no root `LICENSE` blob in tree (third_party dirs have their own) | arXiv 2605.26621v1; no venue in abs page |
| Yuxiang-Lai117/Med-R1 | `88d76c73e7d2` (2025-07-07) | GitHub license = null; no license badge found in the fetched `README.md` (the earlier `README.md:9` reference was a `<div>`, corrected); no `LICENSE` blob in tree | arXiv 2503.13939v5 (revised 27 Oct 2025); no venue in abs page |
| zlab-princeton/vero | `025ca3c800fc` (2026-08-29) | Apache-2.0 (`LICENSE` in tree) | arXiv 2604.04917v3 (revised 18 Jun 2026); abs comment "Project page" only |
| BAAI-DCAI/M3D (interface ref only) | `63e72005a230` (2025-04-13) | MIT (metadata + `LICENSE`) | arXiv 2404.00578v1; comment "MLLM, 3D medical image analysis" |

**License limits.** Where GitHub reports no license and no license file exists, default copyright applies: I treat all such code as read-only reference, never copyable. A README badge is a claim, not a license grant. Even Apache/MIT code was not copied. MM-NeuroOnco images are gated, BraTS 2021 is not redistributed (`README.md:80-96`, `docs/data_sources.md:7,48`).

"Peer-reviewed" is `NOT_VERIFIED` for every paper above: the fetched arXiv pages show preprint status only.

## 2. Source facts (file:line at the pinned commits; paths relative to repo root)

### mpLLM (brain mpMRI 3D VQA; closest MRI precedent)
- **Input**: BraTS-2024 GLI/MET/GoAT NIfTI, `dataset/prepare_brats_3d_dataset.py:14`. **Paper** says BraTS data are co-registered multiparametric, 1 mm³, skull-stripped, resampled to 32×256×256 (`2509.25889v3_html.txt:493`) — i.e. **upstream-preprocessed data**; the repo does no registration.
- **Preprocessing**: `nib.load().get_fdata()` (`:42`) with **no reorientation or spacing resample** (grep: no `as_closest_canonical`/`Spacing`); per-volume min-max (`:51-52`); MONAI `CropForeground` + `Resize([32,256,256], bilinear)` (`:21-24`). Same recipe as M3D (below).
- **2D PNG path**: `dataset/create_brats_imaging_dataset.py:259` slices **coronal** `data[:, y, :]` (not axial); modality slices scaled by *that slice's own* max (`:277-280`, `uint8` truncation, not rounding); segmentation `astype(np.uint8)` (`:263`) — the same silent-cast pattern fixed in this repo (fractional labels).
- **Labels**: masks → questions (volume/region/shape/spread) via `dataset/vqa_3d_utils.py`; relative volume divides by `t1n>1e-6` voxels with comment "all non-zero voxels are brain voxels" (`:1087-1100`, ×250 scaling at `:1100`). Nonzero≠brain-mask is an assumption this repo refuses (see contract).
- **Split**: paper: random 80/10/10 "from the imaging studies" (`2509.25889v3_html.txt:507`). Code: `create_brats_3d_vqa_dataset.py:117` seed 0, `:167` `question_key = "volume_file_id"`, `:182-183` calls `generate_train_val_test_splits` from `dataset/vqa_utils.py`, which is in the upstream tree but **NOT_FETCHED** → whether grouping is by patient (BraTS ids look like `BraTS-GLI-03064-100` = patient + timepoint, `create_brats_imaging_dataset.py:245`) is **NOT_VERIFIED**. `INFERRED` risk: split by volume id can put timepoints of one patient on both sides.
- **Dedup**: none found in fetched files.

### MM-NeuroOnco (2D slice benchmark/instruction data)
- **Input**: mostly already-2D Kaggle/Zenodo/Figshare images plus BraTS21 (`docs/data_sources.md`, dataset table); `data_processing/metadata_extraction/metadata_tumor_05.py:10,109-116` looks up a same-stem `.png` mask — **no NIfTI handling found in the fetched files** (`files/` text search for nifti/.nii/nibabel returned nothing); the full upstream tree contents were not all read.
- **Paper vs repo on dedup**: the paper states "Global pixel-level deduplication is performed prior to partitioning", one 2D slice per volume, stratified partition, and admits slice-only datasets lack patient ids (`2602.22955v1_html.txt:1638-1640`). **The repo tree has no dedup/hash script** (80 paths; none contain "dup"/"hash"); the only split artifact is `benchmark/splits/mm-neuroonco_test_ids_v1.0.json` (**NOT_FETCHED**). So the claimed global dedup is **a paper promise with no implementation observed in the public repo** (`OBSERVED` absence in tree; dedup may exist elsewhere, e.g. in the gated HF data — `NOT_VERIFIED`).
- **Labels**: silver labels produced by LLM pipeline (`pipeline/Step1_…`, `Step2_double_blind…`, `Step3_gemini_audit`; scripts not read beyond names) — label provenance is model-derived; `NOT_VERIFIED` beyond file names.

### NeuroVLM-Bench (2D, modality/plane/diagnosis separation)
- **Split code** `split.py:22` stratifies on `class, original_class, subclass, modality, modality_subtype, plane` (`random_state=42+i`, `:74,83`; `np.random.seed(42)` `:93`). **No patient/group column exists** (`map_images.py:300` fieldnames) → a *patient-disjoint* split is impossible from that CSV.
- **Paper vs code**: the paper says subsets are stratified (`2603.24846v1_html.txt:1133`) **and** that Phase-0 screening subsets are "patient-disjoint" (`:1142`). The fetched code implements stratification only; patient-disjointness is **not implemented in the fetched split.py** (`OBSERVED`; may be handled upstream of the CSV — `NOT_VERIFIED`).
- **Dedup**: path-string duplicate check only (`map_images.py:19,58-61`), no content hash.
- **Labels**: `plane`/`modality` come from dataset/folder names, e.g. Br35H is hard-coded `axial` (`map_images.py:73`), sclerosis plane from the folder name (`:190-202`). These are metadata assumptions, not image-derived.
- **Evaluation** separates `diagnosis_name`, `diagnosis_detailed`, `modality`, `plane` fields (`evaluate.py:49-54`) — a usable precedent for keeping sequence recognition separate from diagnosis.

### MedVol-R1 (CT only; do not transfer parameters to MRI)
- CT window WC 40 / WW 400 and 512×512 resize (`data_preparation/ctorg/convert_volumes.py:27-31`); linear (`order=1`) for CT, nearest (`order=0`) for labels (`:132-136`); 64 uniformly spaced slices (`:35`, `:142`); key slice = median of top-K non-zero-area slices (`select_slices.py:20,207`); `RANDOM_SEED = 42` (`generate_rl.py:47`).
- Reward: format + IoU>0.5 box on predicted slice (`third_party/EasyR1/examples/reward_function/ct_grounding.py:225-277`). No MRI, no reorientation/spacing code in the fetched scripts. Paper evaluates CT-ORG, AbdomenCT-1K, KiTS23 (`2605.26621v1_html.txt:109,281`).

### Med-R1 (2D, multi-modality RL interface)
- Consumes pre-rendered 2D images by path (`src/r1-v/local_scripts/prepare_hf_data.py:58-77`), `.shuffle(seed=42)` (`:103`); rewards: `<answer>` exact accuracy and `<think>…</think><answer>…</answer>` format (`src/r1-v/src/open_r1/grpo_vqa.py:53-100`). No volumetric or MRI preprocessing. `Splits/…json` exist upstream (**NOT_FETCHED**) so train/test overlap with OmniMedVQA is `NOT_VERIFIED`.

### Vero (RL recipe; general domain)
- Dataset curation = heuristic dataset filter (<1K examples, <200K px, binary-only), manual 50-sample inspection, LLM question filter (Qwen3-VL-235B), LLM answer normalization, per-sample `reward_type` routing (`docs/DATA.md:89-145,164-193`). **No dedup or benchmark-decontamination step appears in the fetched docs/scripts** (grep for dedup/decontam finds only skill-codebook dedup in the paper, `2604.04917v3_html.txt:1858,4458`). "Missing global dedup implementation" therefore applies to all seven repos' fetched content; Vero's overlap handling beyond the fetched files is `NOT_VERIFIED`.

### M3D (interface reference)
- Same CropForeground + Resize[32,256,256] + per-volume min-max recipe (`Data/process/m3d_seg_data_prepare.py:14-18,52-53`; `m3d_cap_data_prepare.py:22-24,107-108`). Bench training augmentation includes **random flips on all three spatial axes at p=0.1 and RandRotate90** (`Bench/dataset/multi_dataset.py:34-39`): left-right flips would corrupt a lateralised label **only if the target text is not also transformed**; whether the M3D code transforms any target text was not established here (see the cited `multi_dataset.py` lines). We do not adopt flips.

## 3. Cross-source synthesis (what this changes in our contract)

| Question | Finding | Action here |
|---|---|---|
| Reorient / resample? | In the inspected code, none of the fetched sources reorients or resamples spacing (mpLLM, M3D resize only); not a claim about other code | Ours records orientation and refuses non-axis-aligned input; no silent resample |
| Normalization | per-volume min-max (mpLLM/M3D), per-slice max (mpLLM PNG), CT HU window (MedVol); none train-fit | Ours: legacy per-volume window is the control; alternatives are named controls |
| Mask cast | `astype(np.uint8)` without integrality check (mpLLM `:263`) | Ours rejects non-integer/nonfinite labels |
| Split | volume-id (mpLLM, patient-grouping NOT_VERIFIED), image-level stratified (NeuroVLM), paper-only claims (MM-NeuroOnco dedup, NeuroVLM patient-disjoint) | Ours: patient-level disjointness check per source namespace + exact/near-duplicate guard, global lineage `NOT_VERIFIED` |
| Slice choice | uniform (MedVol), coronal all-slices (mpLLM PNG), single random slice per volume (MM-NeuroOnco paper) | Ours: one frozen coverage rule in physical mm |
| Flip augmentation | M3D Bench flips all axes | Not used |

## 4. Is our dataset already upstream MNI/preprocessed?

`NOT_VERIFIED` from public documentation: no ds007045 dataset documentation was supplied in the fetched sources. What is `OBSERVED` locally (`results/SEQUENCE_QUALIFICATION_V1.json`): 56 TRAIN volumes, LAS storage, 182×218×182 — the standard 1 mm MNI152 grid size — and same-person FLAIR/T1ce pairs share shape and affine in 28/28 pairs. That is **consistent with** MNI-space resampling but does not establish it (grid size alone is not registration; the pilot's NMI proxy also never proves registration). Cross-patient anatomical comparability of a fixed index depends on upstream registration quality, which stays `NOT_VERIFIED` until a public dataset document (or a root-supplied receipt) is read.

## 5. What this repo itself establishes (OBSERVED; local receipts)

From `results/SEQUENCE_QUALIFICATION_V1.json`, `research/ROOT_CAUSE_CPU_FIXES_2026-10-18.md:31`, and renderer code (`src/mri_rl/sequence_qualification_v1.py`):
- 56 TRAIN volumes, all LAS, 182×218×182; 28/28 same-person pairs share shape+affine; 56/56 SHA256 match.
- Historical PNG render (`np.rot90(volume[:,:,z])`, nonzero `[::16]` 0.5/99.5 percentile, uint8, 364×436 bilinear) reproduced 168/168 pixel-exactly.
- Coverage review (84 same-patient pairs at z045/091/136): 28 adequate_inferior, 28 adequate, 27 adequate_superior, 1 insufficient (`insufficient_tiny_disconnected_fragments`). **The single insufficient pair is the known sparse z136 case** (one patient's two z136 images had only ≈1.92 % and 2.13 % nonzero foreground, `ROOT_CAUSE_CPU_FIXES_2026-10-18.md:31`). It is **not** attributable to z045: root reports the 28 z045 records as `adequate_inferior` (the 28 `adequate_inferior` count matches, but the coverage record in this repo does not label levels, so that mapping is root-supplied, not independently re-derived here). The z136 attribution rests on the root-cause note.
- Sparse z045 (a low/inferior level) and one-modality-per-person are confirmed *design limitations*; that they cause lower accuracy is `NOT_PROVEN`.
- Filenames encode the target for all 168 rows, so filename leakage is a **latent risk only**; the inspected native message path does not serialize filenames, and actual leakage is **not** confirmed.

## 6. Limits

No claim of reproducing any published pipeline. No unrun code is described as reproduced. `CROSS_MODEL_REVIEW=NOT_VERIFIED` (a second Claude agent would not be an independent model family).

## 7. Audit 4 — selected-source processing facts (supersedes §4's `NOT_VERIFIED` for lineage)

Source: root's anonymous `LOCAL_SOURCE_PROCESSING_PUBLIC_FACTS_FOR_CLAUDE.json` (header-only; I did not read MRI or private paths). §4 above said no ds007045 documentation was available; root's pinned official README (commit `283007b9…`) now provides it:
- `OBSERVED` (source-author claim): the selected variant `processing_MNI152_skull-stripped_N4` is stated to be skull-stripped, registered to the MNI152 template and N4 bias-corrected. The z-scored variant is a different folder and is **not** the selected one.
- `OBSERVED` (headers, 6 volumes): qform = sform with code **1 (scanner)**, not 4 (MNI152); LAS, 1 mm, 182×218×182, one shared affine; z045 = −27 mm. These are consistent with, but are **not proof of**, MNI alignment: `NOT_VERIFIED` registration accuracy, skull-strip/N4 effectiveness, global duplicate/pretraining exposure.
- Consequence: no automatic second registration, crop, resample, N4 or z-score; "foreground" is nonzero support, not a validated brain mask.
