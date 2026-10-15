# Three-layer reviews of the closest works

Status: source-based methodological review, not reproduction. Source commits and model revision hashes remain `UNVERIFIED`; the linked primary papers and official code/cards are the evidence. Other named works in the request remain unreviewed in depth and are not claimed as novelty clearances.

## [Vero, arXiv:2604.04917v3](https://arxiv.org/html/2604.04917) · [official code](https://github.com/zlab-princeton/vero)

1. **Research process.** Inherits group-relative vision-language RL and task-specific verifiers. Its gap was that single-category RL can narrow general visual capability. The authors curated 600K balanced across six general-visual categories, use online GSPO with eight rollouts, and route ten reward types; open-ended tasks needed an LLM judge. A plausible alternative was proportional sampling with the same 100K mixed budget, which they test.
2. **Evidence audit.** Their matched single-category and mixed runs support category-interference findings in their own taxonomy and base-model families. The larger 600K run also changes data exposure, so it alone cannot isolate balance. The report of answer-only captioning collapse and judge reward hacking is directly relevant to clinical explanations. The June release adds datasets/checkpoints; paper-era and newest checkpoints are not interchangeable baselines. Current code documentation's default judge differs from the paper, so reproduction needs a pinned commit/model.
3. **Next experiment.** Compare a matching base and Vero checkpoint on fixed, real MRI evidence; then train single-task and mixed MRI arms from one checkpoint at matched rollout/visual-token cost. **Kill:** no legitimate image-linked test or no patient separation. Even a positive baseline transfer is not proof of MRI task-balancing benefit.

## [MedGRPO, arXiv:2512.06581](https://arxiv.org/html/2512.06581)

1. **Research process.** Extends medical video RL to heterogeneous datasets/tasks and uses training-set SFT median/IQR calibration and a medical judge. The gap is unequal reward behavior across tasks; a simple common `[0,1]` range does not imply equal learning signal.
2. **Evidence audit.** Their normalization ablation supports improved results under the studied medical-video setup. It does not prove the same mechanism matters for brain MRI. In group-normalized GRPO/GSPO, positive affine reward rescaling largely cancels within a same-task group, so a proposed MRI benefit must explain nonlinear ranking, ties, filters or update effects rather than rely on raw scale alone.
3. **Next experiment.** Hold task sampling/rollouts fixed and compare raw verifiers, faithful MedGRPO calibration and any proposed alternative, with calibration fitted on training data only. **Kill:** no gain beyond the MedGRPO control or no measured change in effective updates.

## [LoV3D, arXiv:2603.12071v2](https://arxiv.org/html/2603.12071)

1. **Research process.** Combines a native 3D encoder, longitudinal T1 MRI, structured regional anatomy/progression labels and normative age/sex-adjusted volumetry verification; uses staged encoder/SFT and preference optimization with four sampled candidates. Its motivation is longitudinal, auditable anatomical reasoning rather than general visual task balance.
2. **Evidence audit.** ADNI and external MIRIAD/AIBL comparisons provide evidence for its setup, not for online GSPO or multi-sequence grounding transfer. Some prompts include demographics, APOE, cognitive scores or prior FreeSurfer labels; image-only comparisons must control that information. Normative category labels are not automatically visually answerable from every rendering.
3. **Next experiment.** If approved repeat scans exist, isolate an image-grounded longitudinal direction task and test paired visits with reference measurements hidden. Compare the method's longitudinal capability under matched evidence. **Kill:** no legitimate visit linkage, or model output can be predicted from exposed metadata/reference values.

## [SGMRI-VQA, arXiv:2604.15808](https://arxiv.org/html/2604.15808) · [official dataset](https://huggingface.co/datasets/SpatialGroundingVQA/SGMRI-VQA)

1. **Research process.** Creates frame-indexed MRI QA from fastMRI/fastMRI+ source images and annotations, providing detection/localization/classification/captioning and multi-frame tasks. It supplies a fast test of MRI visual grounding, but its multi-frame observations are slices within a volume, not longitudinal visits.
2. **Evidence audit.** The public card's brain QA counts can be cited as publisher counts, not locally usable images. Its current official file tree does not expose the promised image archives; the third-party repackage's embedded PNGs do not establish redistribution rights. Patient-disjoint split is not proven by the visible volume IDs. A model score without these checks could confound image mismatch, subject overlap and licensing.
3. **Next experiment.** Obtain source-approved images, reconstruct QA–volume–frame links, decode and bounds-check boxes, then freeze patient-separated test cases for paired Qwen3-VL/Vero inference. **Kill:** no legal image path or no verifiable patient split for independent-patient claims.

## Additional review queue

ViGoRL, MedVLM-R1, Med-R1, BrReMark, MedVol-R1, MedMO, NV-Reason-CT, NeuroQA and BrainTRACE require paper-level three-layer review before a method novelty claim. The closest prior-art boundaries for Med-R1, BrReMark and MedVol-R1 are summarized with links in `PRIOR_ART_AND_NOVELTY_MATRIX.md`; that summary is not a complete reproduction audit.
