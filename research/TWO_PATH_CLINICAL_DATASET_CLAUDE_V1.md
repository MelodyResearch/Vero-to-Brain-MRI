# Two-path clinical dataset recovery — primary assessment (Claude, V1, final)

Date: 2026-10-22. Scientific lead: Claude Code (claude-opus-5-5). Independent source, metadata and image-QC review: Codex.
Codex integration notes below correct case-proxy semantics and record completed CPU checks.

**Scope.** Public metadata and five official public images only. No model, GPU, training, data expansion, expert contact, installs or
private local data. The frozen plan and lock are unchanged.

**Tiers:** ACCESSED (read by Claude), CODEX_OBSERVED (Codex public fetch or inspection, not re-fetched by Claude), INFERRED, NOT_VERIFIED.

## Decision: `PATH_B_READY_FOR_SMALL_PILOT` (feasibility only)

| Dimension | Status |
|---|---|
| **Task readiness** | **READY.** The task is VQA-RAD head-MRI midline-shift presence: original clinician-authored, evaluated yes/no QA; CC0; case-disjoint balanced split. All five official images match their OSF-published SHA-256, and technical QC passed. Remaining small CPU items are in §4. |
| **Method/evaluation readiness** | **NOT READY.** Source binding and yes/no verifier passed 10 CPU tests; exact native scope and fresh human permission remain required. No native training integration was executed. |
| **Method claims** | **None.** No adaptation arm exists: C was rejected, and all earlier failed methods are retained as negative evidence. No GSPO variant is proposed just to fill an arm. |
| **Accuracy / generalization** | **None.** The 2+2 design can show feasibility only. |

## 1. Source facts

| Source | Fact | Tier |
|---|---|---|
| SLAKE | Data license: author HF repo `BoKelvin/SLAKE` (pinned `a9083ce6…`, linked from the official page), CC BY 4.0. The paper's CC BY-NC-SA is the article license, not a data conflict. | CODEX_OBSERVED |
| SLAKE | Brain-MRI locations include both `Brain` and `Brain_Tissue`. English MRI rows train/val/test are 855/184/193 on 97/21/22 images, i.e. **140 distinct brain-MRI images**. Within English the image splits are disjoint; mixing English and Chinese rows makes images cross splits (130/142/26). No patient or source-volume field. | CODEX_OBSERVED |
| SLAKE | CT/MRI single slices were "randomly chosen from 3D volume cases"; questions were doctor-selected templates; the answer source is not stated. | ACCESSED (paper) |
| VQA-RAD | OSF `89kps` data license CC0 1.0. The public JSON has 2,248 rows (the paper's 3,515 includes framed questions). 103 head images; no modality field. | CODEX_OBSERVED |
| VQA-RAD | One image per MedPix teaching case. QA by clinical trainees from captions; about 75% double-annotated. The original question-level split is not used. | ACCESSED (paper) |
| BraTS 2021 (TCIA) | 1,480 public patients of the 2,040-patient benchmark, CC BY 4.0. Includes 299 UCSF-PDGM cases, so cross-collection overlap is documented. Expert masks, not QA. | CODEX_OBSERVED |
| BraTS-METS | Neuroradiologist-approved metastasis masks; no QA. | ACCESSED |

## 2. Task choice

**SLAKE brain MRI is not chosen,** even though it has 140 images.
- The blocker is structural: there are no patient or volume identifiers, and the slices come from 3D volumes, so a strict
  patient-disjoint split **cannot be rebuilt from public metadata**. Even English-only image disjointness does not exclude multiple
  slices of one patient landing in different splits.
- Its answer source is not stated, and it offers generic Organ/Position labels rather than one clinical endpoint.

**VQA-RAD midline-shift presence is chosen.**
- It is a single, clinically meaningful finding that can be judged visually on axial head MRI, with a binary verifiable answer and no
  masks.
- It is distinct from the FLAIR/T1ce sequence task.
- Clinical-personnel authorship is accepted, per the researcher.
- Five public items are used. This is **our predeclared sample limit**, not evidence about how many usable items the endpoint has in the
  full dataset.

| Role | Original qid | Source answer |
|---|---|---|
| TRAIN | 52 | no |
| TRAIN | 69 | yes |
| EVAL | 1037 | yes |
| EVAL | 2236 | no |
| Reserve (not used) | 1050 | yes |

Each selected item has one source case URL. The full image–case graph puts the five items in five disjoint components; one TRAIN component also includes another unselected image, which must stay excluded from EVAL. These are case proxies, not authenticated patient registry IDs. Original question text is preserved and selected items do not share paraphrases.

## 3. Image QC and burned-in content (recorded honestly)

**Codex technical, non-clinical contact-sheet QC** (CODEX_OBSERVED; this is **not** expert adjudication):
- all 5 are single-panel, axial, MRI-like head images;
- all 5 carry MedPix, source or uploader watermarks;
- one image also has scan-header and scale marks;
- none shows a printed yes/no answer or an answer-pointing arrow or circle.

**How this is recorded:**
- `NON_ANSWER_BURN_IN_PRESENT`: watermarks on 5 of 5 and header/scale marks on 1 of 5.
- `ANSWER_BURN_IN_NOT_OBSERVED` (technical inspection).
- The images are **not** described as overlay-free.

**Shortcut and domain implications**
- **Watermark presence on all 5:** presence alone is constant, but pattern, position, text and style vary. These remain potential source/label cues; the tiny sample cannot exclude shortcut learning. MedPix JPEG versus NIfTI panels is a domain confound.
- **Header/scale marks on 1 of 5:** these are class-specific within a 4-item design, so they are a disclosed possible shortcut. They are
  **not** removed by cropping or a new view, and they are reported per item.
- **Unchanged inputs:** the original source image and question are used as-is. Metadata never enters the model prompt.

**Expert review is not required for this feasibility pilot,** because its clinical labels are the source's evaluated QA and no clinical
claim is made.

**Additional review or independent image-dependence evidence is needed before claiming validated clinical visual reasoning**, for two reasons:
1. The trainee-authored answers were written from case captions, so midline shift may not be visible on this exact chosen slice. Our project has not independently re-adjudicated slice answerability beyond the original clinically generated/evaluated QA.
2. Burned-in content may interact with the finding.

## 4. Remaining small conditions (no model calls)

1. **Codex case-linkage review:** confirm one MedPix case per image and no case used in two roles.
2. **Verifier and binding tests:**
   - CPU tests for a strict yes/no route that reuses the existing Vero format and 0.8/0.2 reward weights;
   - SHA-bound source images and questions;
   - the reserve is never ingested.
3. **Execution scope:** an exact scope, independent review and fresh human permission. Nothing here authorizes execution.

**Disclosed limitations (not blockers):**
- Global patient overlap between MedPix and our cohort is NOT_VERIFIED.
- Domain mismatch.
- Scale: a 2+2 design.
- No patient-level generalization claim.

## 5. Prospective ≤1-update matched feasibility design (conditional; NOT authorized; no GPU scope)

**Arms**
- **Parent Adam52:** no update.
- **Direct Vero:** same-prompt GSPO, K=4, ONE joint update over the existing frozen sequence TRAIN unit(s) plus the 2 midline TRAIN
  items. Equal task weight; existing reward, LR, LoRA, F32 contract and concise prompt.
- **No third arm.**

**Evaluation (same contract for both arms)**
- Midline: 2 EVAL items, strict greedy yes/no, reported per item.
- Sequence: DEV24 as an engineering control only.

**Predeclared outcomes**
- **NO_SIGNAL:** the Parent's K=4 rollouts on the midline TRAIN items show no within-image reward variance. Record this and stop, with no
  resampling.
- **FEASIBLE:** the run completes, with no strict-format collapse and no FLAIR or T1 recall loss vs Parent on DEV24.
- **FAIL:** an invalid-output increase, a sequence regression vs Parent, or a pipeline or verifier error.

No accuracy, transfer or method-efficacy claim is possible. Exact budgets and thresholds are frozen only in the reviewed scope.

## Sources
- [SLAKE paper](https://arxiv.org/abs/2102.09542)
- [SLAKE official page](https://www.med-vqa.com/slake/)
- [BoKelvin/SLAKE](https://huggingface.co/datasets/BoKelvin/SLAKE)
- [VQA-RAD article](https://pmc.ncbi.nlm.nih.gov/articles/PMC6244189/)
- [OSF 89kps](https://osf.io/89kps/)
- [BraTS-METS](https://arxiv.org/abs/2306.00838v3)
- [BraTS 2021 benchmark](https://arxiv.org/abs/2107.02314)


## 6. Codex integration: actual bounded acquisition and CPU preparation

- Original OSF QA JSON: 2,248 rows, 314 referenced image names; directory has 315 images. HEAD: 715 QA / 103 images. MRI has no standalone acquisition-modality field. The independent review's **37 MRI candidates** are a conservative modality-QA rule, not the full MRI census. No CT-negative answer alone establishes MRI.
- Downloaded exactly **5 VQA-RAD images / 132,278 bytes**; each matches the original OSF SHA and size. SLAKE/BraTS/BraTS-METS: **0 sample images downloaded**; only source text/metadata. No whole dataset or model was downloaded.
- Five selected evaluated QA have original human source endpoints. A source-bound private candidate manifest is actually built: TRAIN2 (1 yes/1 no), EVAL2 (1 yes/1 no), RESERVE1 (never for training/evaluation). The complete image–case graph excludes linked unselected images from opposite roles. This is a source-case-proxy separation, not authenticated global patient independence.
- Offline reproduction entry: `scripts/prepare_vqarad_midline_task_v1.py`; module `src/mri_rl/vqarad_midline_task_v1.py`. It verifies source JSON and image hashes against saved official OSF metadata, rejects linked cases, duplicate payloads, non-reviewed endpoints and imbalance, preserves original question text, and exposes only image/question as model input. Its `EXTERNAL_REVIEW_REQUIRED` flag deliberately does not self-certify clinical or MRI identity.
- Reward preparation reuses the original Vero tag/box parser and 0.8 accuracy / 0.2 format components. `strict_accuracy` is separate: malformed partial-format answers cannot be counted as correct strict greedy outputs. Existing training loss, reward module, optimizer, group gate and trainer are unchanged. The callback is not wired into a running native driver.
- 10 focused CPU tests passed, including case-alias leakage, source corruption, endpoint rejection, duplicate questions/images, preservation of original `shifted` wording and integer qids, and format-versus-strict accuracy. Actual source binding also passed. This is **not** a native trainability or accuracy result.
- Private candidate SHA: `68fad7db7c68b18778469bc8c78d86ede61e15b2477f34ee8fc449fb0f2350af`. Separate reproducible offline candidate SHA: `0a50021e6cb693a17cdf6db4613fc2876030dc0c16acb8b93810287c3ef04c40`. The first additionally records technical QC; the second is minimal source binding. Neither is an execution authorization.

### Other screened sources: exact limits

**BraTS 2021:** [benchmark §2.1.4](https://arxiv.org/html/2107.02314) describes automated initial segmentation, expert manual refinement, and two senior board-certified neuroradiologists approving or returning masks. [TCIA](https://www.cancerimagingarchive.net/analysis-result/rsna-asnr-miccai-brats-2021/) lists 1,480 public subjects, CC-BY-4.0 and an ID crosswalk; it explicitly includes a 299-case UCSF-PDGM v1 excerpt. This is documented source overlap, not a clean independent cohort. Four MRI modalities and their segmentation masks support the original segmentation endpoint; no newly invented slice-level presence question or visible lesion centre was qualified here. No image/mask payload was downloaded or newly aligned.

**BraTS-METS:** [paper v3](https://arxiv.org/html/2306.00838v3) documents staged annotation, second board-certified review and final neuroradiologist review. It reports 402 challenge training, 31 validation and 59 test studies; optional UCSF/Stanford sets are separate. These are studies, not a newly verified patient census. Segmentation labels are not original QA; a source neuroradiologist mask does not validate arbitrary 2D QA. [Current organizer access guidance](https://www.synapse.org/Synapse:syn5637528/discussion/threadId=12058) requires Synapse registration and post-challenge terms. No terms were accepted and no data were downloaded. Current exact terms and full cross-source identity mapping remain unverified; do not transfer TCIA's license to this release.

### Source/domain controls for any later pilot

The unchanged MedPix JPEG inputs preserve the original clinical QA context. Burn-in and question wording may support shortcuts; all five watermarks do not prove all watermark patterns are identical. The four-item design cannot separate source, image style and label effects. Model-visible sidecars, filenames, patient IDs, answer keys and captions are excluded; existing image pixels still contain the disclosed scan/source text. Do not claim anatomical grounding from this alone. Any later expansion must retain full case components, inspect pixel duplicates across selected payloads, freeze source-specific balanced denominators and include suitable image-swap/no-image controls under a fresh bounded scope. No current model output has been obtained.

No MRI-specific method intervention is selected just to populate an adaptation arm. The direct-transfer feasibility check above is the only defensible immediate experiment design. A matched adaptation comparison remains **NOT_READY** until one independently justified single-variable intervention is specified; past failed Paired-Prompt/C/VJP experiments are not rerun. The pilot must report actual 0-update outcomes if no valid group exists. No resampling to force signal and no automatic scaling.

## 7. Path A: optional minimal consultation, not a prerequisite for Path B

Official contacts checked 2026-10-22; none contacted:

| Priority | USC expert and fit | Official profile / verified channel |
|---|---|---|
| 1 | Mark Shiroishi — brain tumour MRI; BraTS-METS coauthor | [Keck faculty](https://keck.usc.edu/faculty-search/mark-shiroishi/), listed phone **+1 323 865 9914** and official Email link |
| 2 | Saman Hazany — neuroradiology / structural brain MRI | [Keck faculty](https://keck.usc.edu/faculty-search/saman-hazany/), listed phone **+1 323 442 7672** and official Email link |
| 3 | Nasim Sheikh-Bahaei — Neuroradiology Section Chief, MRI research | [Keck faculty](https://keck.usc.edu/faculty-search/nasim-sheikh-bahaei/), official Email link; no personal email address inferred |

The official pages render email links through protection; no address is guessed. These are faculty directory contacts, not a promise of availability. [USC fellowship](https://keck.usc.edu/neuroradiology-division/training-education/) provides a potential departmental referral route; no trainee name or private contact is inferred.

**Unsent inquiry (for researcher approval):**

> Dear Dr. Shiroishi, I am working on an independent research project adapting verifiable reinforcement learning to brain MRI. Would you be open to a brief feasibility review of 1–3 de-identified, publicly licensed MRI images? We would only ask whether the finding is visible on the supplied slice, whether the question is answerable from that image, and whether the proposed answer definition is appropriate; “uncertain” is welcome. This is not a request to annotate a dataset. I can share a small review packet if you are interested. Thank you for considering it.

The existing local one-item packet can be reused only after its sharing rights and contents are checked. MRI-only review precedes revealing the hidden endpoint; no packet or patient-linked material was sent. A reply or one approved item cannot qualify an entire cohort. Existing clinical-source QA can support this bounded feasibility study without waiting for a new expert; independent adjudication remains valuable for ambiguous or newly generated endpoints.
