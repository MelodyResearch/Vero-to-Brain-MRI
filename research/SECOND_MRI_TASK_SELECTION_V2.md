# Second MRI task selection V2 — one-pass decision

Date: 2026-10-23. Base: `a32efc1e27b95ed4f0e5aa5de002c482cb082d8e`.

## Decision

**PAUSE_HETEROGENEOUS_GPU_PENDING_SOURCE_CASE_MAPPING.** Stop the current VQA-RAD Midline2+2/K4 configuration.
The strongest remaining **candidate** is one original SLAKE question, “Is the brain healthy?”, on26 distinct image entries
(11yes,15no). It is not yet a case-disjoint, pixel-qualified training/evaluation set. No runnable GPU scope or new GPU approval
request is issued. Do not reuse the consumed V2 permission. This is a finite source-selection decision, not a new gate/loss project.

The prior pilot established generation/scorer feasibility, not a successful RL update: its Midline K4s had reward variation from
format, but lacked the project's required within-image strict-valid correct/wrong contrast. Standard GSPO can have computable
format-driven gradients in that situation. No claim that standard GSPO has mathematically zero gradients is made.

## What was actually done

- One closed screening pass over already acquired original VQA-RAD JSON and pinned author SLAKE JSONs; no broader dataset search.
- Claude Code2.1.306/`claude-opus-5-5` was actually called and returned `Not logged in`/exit1. No Claude analysis is claimed this round.
  Codex performed the source selection; an isolated Codex reviewer independently checked the counts.
- The existing original QA answers were never changed or relabeled as new clinical truth. No model outputs were used to select rows.
- Minimum metadata screen: at least4 TRAIN and4 EVAL source-case components, at least2 of each of two labels in each split.
  This is a minimal pilot screen, not a statistical-power or publication standard. Image IDs do not satisfy the case criterion.
- Only719,425 new bytes of official ZIP directory/per-image JSON metadata were read (two fixed candidate images' metadata, no
  image pixels). The212MB archive was not downloaded. Existing source JSON hashes and official Git blob IDs matched exactly.
-0MRI image downloads,0model/GPU calls,0rollouts,0backward,0updates. The frozen plan/lock and all historical evidence remain unchanged.

## VQA-RAD: no balanced single endpoint in the verified MRI subset

Scope:183 evaluated HEAD QA on33 images with positive original source-QA MRI evidence, forming33 full image–MedPix case graph
components. This is a conservative lower bound, not a complete MRI census. Another48 evaluated HEAD image entries have unconfirmed
modality; they were not silently promoted to MRI. No global assertion that VQA-RAD has no usable task is justified.

| Single endpoint / screen | Verified source-case result |
|---|---|
| Edema presence |2yes /0no |
| Hemorrhage presence |0yes /1no |
| Mass effect |0yes /1no |
| Ventricular enlargement |1yes /0no |
| Lesion enhancement |0yes /1no |
| Direct anatomical plane |7components, allaxial |
| Laterality screen |7right /1left /2bilateral; no adequate two-class balance |
| Literal region answers |15components; resolved labels are singletons; one unresolved source variant |
| Literal pathology names |6components; no two literal classes each with4components |

These are endpoint-specific counts, not one generic abnormality pool. No repeat/paraphrase was counted as a new case. Broad disease
names and region answers were not merged into invented labels; unresolved source variants remain unresolved. No split was created.
The image–case graph provides source-case proxies, not certified global patient IDs.

## SLAKE: an adequately sized metadata candidate with a concrete case-linkage blocker

Current author revision: `a9083ce6c34ac3ffb17671a605962924d8a8f9e9`, data licenseCC-BY-4.0.
English `base_type=vqa`, source `modality=MRI`, `location=Brain` or `Brain_Tissue` gives140 distinct image entries/1,133 vision-only QA.
This is source metadata, not140 verified independent patients or a fresh pixel audit.

**One exact original question is preserved:** “Is the brain healthy?”

| Original source partition | Yes | No | Image entries |
|---|---:|---:|---:|
| TRAIN |8 |11 |19 |
| Validation |1 |2 |3 |
| Test |2 |2 |4 |
| Total |11 |15 |26 |

The exact candidate has26QA/26image IDs, no duplicate exact QA, and no image crossing its partitions. English-only official image
partitions are disjoint. The previously observed130/142/26 cross-partition intersections arise when English and Chinese partitions
are pooled, and are NOT a leakage finding against this English-only candidate. No SLAKE training has occurred in this project.

The original paper documents physician organ/disease annotation, doctor-proposed/amended question templates, and **image-level**
splitting. It does not provide a patient/source-volume-disjoint guarantee. The QA JSON fields have no patient, case or volume ID.
The ZIP directory has3,235 entries with no apparent grouping-map filename; two fixed candidate images' question/detection JSONs also
have no source-case fields. This is a bounded metadata check, not inspection of every archive file or proof that no mapping could
be supplied by the authors.

Clinical-source annotations are reusable evidence. Do not claim independently reviewed per-QA diagnoses or interpret “healthy” as
proof of entire-patient clinical health. Here it is only the original source QA label. Actual source-image binding, answerability,
duplicate checks and burn-in checks remain unperformed because the case split is not established; no mask-derived new QA is created.

Other source endpoints do not rescue the split problem and are less useful:

- Tumor presence:8image entries, allYes. Edema presence:8, allYes.
- Tumor enhancement:6, fiveYes/oneNo.
- Open plane questions:54, all“transverse plane”; binary plane questions must not manufacture apparent label diversity from wording.
- Disease questions:59image entries with original multi-label co-occurrence answers. These were not collapsed into an exclusive
  tumor/edema binary task or counted as multiple independent skills.

A private, SHA-bound manifest preserves **all26** exact-source candidate rows and source partitions, with no assigned experimental
TRAIN/EVAL split and `training_authorized:false`. No outcome-based selection has occurred.

## One candidate, one conditional next data/task change

If the source publisher supplies a de-identified image-to-source-case/volume mapping and a small pixel check confirms the original
QA is answerable, the only retained next candidate is to replace Midline with this exact original SLAKE endpoint. No Adam, GSPO,
reward, LR, LoRA, K4, format search or new auxiliary objective is proposed.

Only then freeze a genuine case-component split and exact image/TF budgets. Parent and Direct Vero must share inputs, concise
format contract, verifier and evaluation; at most1 joint update, once-only on-policy rollout, equal task weights, protected Sequence
DEV24 and strict per-task accuracy/recall/invalid/cost reporting. Compare Parent versus RL on the SAME new task; comparing SLAKE and
Midline accuracies would not identify a Vero effect. No automatic scaling follows from positive or negative small-sample results.

**Currently NO-GO for GPU.** There is no exact executable scope SHA: inventing one without qualified cases and bound image payloads
would misrepresent readiness. Stop this selection pass now. If source grouping is unavailable, pause this heterogeneous MRI route
rather than run another image-only smoke test. No expert annotation project or broader paper/dataset search is started.

## Sources and evidence

- [SLAKE official release](https://www.med-vqa.com/slake/).
- [Pinned author data card](https://huggingface.co/datasets/BoKelvin/SLAKE/blob/a9083ce6c34ac3ffb17671a605962924d8a8f9e9/README.md).
- [SLAKE paper§2.1/2.3/2.4](https://arxiv.org/html/2102.09542v1): annotation and splitting claims only; not a new literature survey.
- [Original VQA-RAD release](https://osf.io/89kps/).
- `results/SECOND_MRI_TASK_SELECTION_V2_ACTUAL.json`: source-bound aggregate decision.
- PRIVATE source-question/answer memberships, graph components, archive metadata and candidate manifest remain outside Git.

## Independent final review

[Independent aggregate review](../results/SECOND_MRI_TASK_SELECTION_V2_INDEPENDENT_REVIEW.json):
**SCIENTIFICALLY_BLOCKED_FOR_GPU_PENDING_SOURCE_CASE_MAPPING**. The reviewer independently reproduced all26 exact-source candidate
counts, zero exact duplicates/conflicts and English image-disjointness, and matched all three raw SHA/byte counts/Git blob IDs.
It confirmed that patient/source-case independence is unestablished rather than asserting data unavailability. The VQA-RAD independent
case-component/endpoint proof is retained. Review added0downloads,0model calls and0GPU operations.

PRIVATE review SHA256 `cabc5cd77f682b48197bf8e56526b176c086cc74a5969475f1ff48eb9375d89b`;
PUBLIC review SHA256 `e8c1ebe7e971d0c33731fe3e45ccc638389aa790f079d149fa5bb403d81b9ad8`.
The source-selection pass is closed. No new executable training/evaluation scope is being submitted for approval.
