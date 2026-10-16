> **Archival note:** This is the raw independent Claude Code CLI review, preserved without scientific edits. Claude read full-text HTML through a summarizing fetcher, but no PDFs or code for most papers. Its conclusions and source mismatches are assessed separately in the cross-paper synthesis; do not treat this file as a verified full-PDF audit.

The report is finished. I wrote it to the plan file because plan mode blocked other writes. I couldn't call ExitPlanMode because the tool isn't available here, so the report is reproduced below.

# Independent methodological review: Vero-style RL for MRI, with six reference papers

Date: 2026-10-15. I did not use Codex conclusions, touch local files or patient data, or run code.

## 0. Read depth and caveats

I fetched the six arXiv pages, the Vero GitHub README and `docs/TRAINING.md`.

- I read the arXiv **HTML** full texts through a summarizing fetcher. I did **not** read the PDFs, so **PDF page numbers are NOT_VERIFIED**. I cite section, table and figure numbers instead.
- **Vero:** The last ~6k characters of the HTML were unread. Appendix B hyperparameter tables and the judge prompt were not visible. I took hyperparameters from `docs/TRAINING.md`.
- **MedGRPO:** The last ~20k characters (appendices) were unread.
- **BrReMark, MedVol-R1, LoV3D:** I did not open the code repositories, so their code is **NOT_VERIFIED**.
- **Med-R1:** I found no code link. **NOT_VERIFIED.**
- Numbers were transcribed by the fetcher. Treat them as "read from HTML" until checked against the PDF.
- I was not given your proposal. I assume it is one RL policy over several MRI task categories, with uniform mixing, task-routed rule and LLM-judge rewards, and GRPO/GSPO. Correct me if that's wrong.

## 1. Identifier verification

| Paper | Verified title | Versions / date | Code |
|---|---|---|---|
| [2604.04917](https://arxiv.org/abs/2604.04917) | Vero: An Open RL Recipe for General Visual Reasoning (Sarch, Cai, Wang, Wu, Chen, Liu) | v1 6 Apr 2026, v3 18 Jun 2026. README says ECCV 2026. | [zlab-princeton/vero](https://github.com/zlab-princeton/vero) (Apache-2.0, veRL-based). Project page: vero-reasoning.github.io. |
| [2512.06581](https://arxiv.org/abs/2512.06581) | MedGRPO: Multi-Task RL for Heterogeneous Medical Video Understanding | v1 6 Dec 2025, v4 8 Apr 2026. CVPR 2026. | Project site uii-america.github.io/MedGRPO. No repo on the arXiv page, so **NOT_VERIFIED**. |
| [2503.13939](https://arxiv.org/abs/2503.13939) | Med-R1: RL for Generalizable Medical Reasoning in VLMs | v1 18 Mar 2025, v5 27 Oct 2025 | None found. |
| [2606.25894](https://arxiv.org/abs/2606.25894) | Enhancing Brain MRI Anomaly Detection and Reasoning with ROI Rethink and Synthetic Data (**BrReMark**) | v1 24 Jun 2026 | github.com/fdu-farm/BrReMark (not opened) |
| [2603.12071](https://arxiv.org/abs/2603.12071) | LoV3D: Grounding Cognitive Prognosis Reasoning in Longitudinal 3D Brain MRI via Regional Volume Assessments | v2 14 Mar 2026 | github.com/Anonymous-TEVC/LoV-3D (not opened) |
| [2605.26621](https://arxiv.org/abs/2605.26621) | MedVol-R1: Reward-Driven Evidence Grounding for Volumetric Reasoning Segmentation | v1 26 May 2026 | github.com/AdvancingEther/MedVol-R1 (not opened) |

**SOURCE_MISMATCH:** BrReMark's arXiv title does not contain "BrReMark". The name appears only in the abstract.

## 2. Per-paper audit

### 2.1 Vero (full text read, appendix partly unread)

- **Design:**
  - Vero-600K has 59 datasets and 100K samples in each of 6 categories, mixed at equal ratios.
  - It uses ten routed verifiers plus an LLM judge.
  - It trains with GSPO, no KL term, clip-higher and a soft overlong penalty.
  - Reward is R = 0.8·R_acc + 0.2·R_fmt + R_overlong (Eq. 3–4).
- **Config** (`docs/TRAINING.md`): lr 1e-6, batch 256, n=8 rollouts, 4,000 steps, clip 3e-4 / 4e-4, 8 GPUs.
- **Claims versus evidence:**
  - **Gain over base (Table 3):** +2.9 to +5.4 on the instruct starts.
  - **8B-Instruct vs Thinking:** 66.1 − 62.3 = 3.8, which matches the abstract.
  - **Routed reward vs math_verify (Table 4):** 57.2 vs 51.8.
  - **RL vs SFT on the same data (Table 4):** RL 57.2, SFT 52.8, base 52.4.
  - **Uniform mixing (Table 2):** Equal ratios give +5.8. Alternatives give +4.8 to +5.2, a gap of 0.6–1.0 points. I saw no seeds or CIs. The same model's gain is +5.8 in Table 2 but +5.4 in Table 3. This is **NOT_VERIFIED** and may be a budget difference.
  - **GSPO vs GRPO/DAPO:** 54.7 vs 54.3, a 0.4-point gap. The text says GSPO is "more stable", but it has the highest entropy in the table.
  - **Single vs joint (Sec. 5):** Single-category training hurts other categories by −2 to −35 points. Mixed training at the same 100K budget gives +0.3 to +4.2.
  - **SOURCE_MISMATCH:** The paper says the judge is Qwen3-32B with thinking off. The repo default is Qwen3.5-27B at temperature 0.7. This may be a version change.
- **Limitations (Sec. 8):** The taxonomy is not shown to be optimal. The behavioral analyses are descriptive, not causal. Only 7–9B models were tested.
- **Medical content:** PathVQA (1,108 samples) and VQA-RAD (678) sit under STEM with string match. There is no MRI, no 3D, and no patient-level split.
- **Smallest falsifier:** Same budget, 3 seeds, uniform vs proportional mixing vs best single category. If the spread is inside seed noise, "uniform mixing is better" fails.
- **Novelty:** Mostly engineering and scale. What transfers is routed verifiers and equal category budgets.

### 2.2 MedGRPO (full text read, appendices unread)

- **Design:**
  - G=8, DAPO-style clip, no KL.
  - **Cross-dataset reward normalization (Eq. 3):** r = σ(k·(x − p50)/IQR) with k=3. The statistics come from the SFT model on the training set, per dataset-task pair.
  - GPT-4.1 scores captions on 5 clinical dimensions.
  - lr 5e-7, 5,000 updates, 8×H100.
- **Evidence:**
  - Without normalization, training collapses (Table 3 row B: CVS 0.020, STG 0.010). This is the strongest reward-scale-imbalance evidence in the set.
  - SFT → MedGRPO (Table 2): TAG@0.3 0.142 → 0.216, STG 0.177 → 0.202, RC LLM score 2.757 → 3.442.
  - NAP was excluded from the reward and regressed from 0.442 to 0.405. Normalization does not protect tasks left out of the reward.
  - The judge ablation raises RC from 2.984 to 3.235. The judge family also appears in the metric, so part of this is circular.
  - The TAG+STG-only row roughly matches the full model on DVC and RC. This weakens the multi-task benefit claim.
- **SOURCE_MISMATCH:** ε_high is 0.3 in §3.2 and 0.28 in §4.1.
- **Gaps:** No seeds or CIs. The human study has 12 participants on one dataset. The Qwen3.5-4B result is deferred.
- **Smallest falsifier:** Normalized vs raw reward over several seeds, plus a per-task scalar weight instead of percentile normalization. If the weight fixes the collapse, the median/IQR machinery is unnecessary.

### 2.3 Med-R1 (full text read)

- **Design:** GRPO with format and accuracy rewards on OmniMedVQA (88,996 QA pairs). Base model Qwen2-VL-2B. The accuracy reward is a letter match.
- **Config:** Effective batch 4, 2×H100, one epoch. lr and step count are **not reported**.
- **Evidence (Table IV):**

  | Model | Overall |
  |---|---|
  | Base | 38.11 |
  | SFT | 54.07 |
  | Think | 69.91 |
  | No-Think | 72.49 |
  | 72B | 68.05 |

- **Inconsistencies:**
  - The abstract claims +29.94%. The table gives +31.80 points (or +83% relative). The 29.94 figure is **NOT_VERIFIED**.
  - Text and Table I per-modality numbers disagree.
  - Ablation tables use Qwen2.5-VL-3B but the headline uses Qwen2-VL-2B.
  - Text says 81.64 for the best cross-task row and the table says 81.16.
- **Reading of the result:** No-Think beats Think, so the gain comes from the reward on templated multiple-choice answers, not from reasoning.
- **Limitations (Sec. IV-G):** 2D only. CT/MRI volumes are not handled.
- **Leakage:** The 80/20 split is over pooled VQA pairs. Image and patient leakage control is **NOT_VERIFIED**.
- **Smallest falsifier:** Train and test on an image/patient-disjoint split. If the gain over SFT shrinks, the result reflects dataset regularity.
- **Novelty:** GRPO applied to medical VQA. Little else.

### 2.4 BrReMark (full text read, repo not opened)

- **Design:**
  - Turn 1 proposes a lesion box. Turn 2 re-checks the marked image and gives a diagnosis.
  - SFT (~3.7K cases), then GRPO on 12,391 samples with G=6, lr 1.5e-6, β=0.001.
  - Reward is r_fmt (≤0.4) + r_loc (min(IoU, 0.5)) + r_llm (Lingshu-32B, ≤1.0). There is a modality gate and a penalty for fabricated lesions on healthy images.
- **Evidence:**
  - Internal mAP50 goes from 0.74 to 37.54. The base barely emits boxes, so it is a weak baseline.
  - Qwen2.5-VL-72B beats it on NOVA: mAP30 37.02 vs 33.05, mAP50 21.25 vs 13.30. Diagnostic accuracy on NOVA is 10.43 vs 20.36.
  - The "45.7% fewer false positives" holds only against Qwen-72B at mAP30 (285 vs 525).
  - Internal false positives are higher than Qwen-72B (812 vs 624).
  - The ablation shows r_loc matters: mAP50 falls from 37.54 to 21.98 without it. Removing synthetic data costs about 2.5 points.
- **Missing ablations:** SFT-only vs RL, and removal of the ROI rethink step. Neither headline component is shown to help.
- **Circularity:** Lingshu-32B is the reward judge, a data filter and a baseline. GPT-4o scores the diagnosis.
- **Leakage:** No patient-level split control is described. **NOT_VERIFIED.**
- **Smallest falsifier:** SFT-only vs SFT+RL with identical data, and one-turn vs two-turn inference.

### 2.5 LoV3D (full text read, repo not opened)

- **Design:**
  - A 3D ResNet-50 encoder feeds Qwen-2.5-14B with LoRA.
  - A deterministic verifier scores outputs against FreeSurfer regional volumes, with age- and sex-normed Z-scores.
  - The verifier drives **DPO** (K=4, β=0.1), not GRPO.
- **Evidence:**
  - ADNI test accuracy is 93.7%, with 82.6% region accuracy. The split is patient-level (258 test subjects).
  - DPO adds +0.4 points of accuracy over Stage 1b (93.3 → 93.7). The larger gains are in text and fabrication metrics: BLEU-4 0.354 → 0.584, F.Sev 4.1 → 2.2.
  - The same-split classifier baseline (ResNet-50 linear head) is weak at 58.9%. Published binary classifiers are not comparable.
  - MIRIAD's 95.4% is plain accuracy. Its balanced accuracy is 93.2%, below the cited prior work (95.7% and 94.9%). AIBL is 82.9%, balanced 74.2%.
  - Verifier labels and training labels both derive from FreeSurfer, so the verifier is aligned with the labels rather than independent clinical truth.
- **Limitations (Sec. 4):** FreeSurfer reliance, T1 only, no amnestic vs non-amnestic MCI split.
- **Smallest falsifier:** Compare Stage 1b and Stage 2 over 3 seeds on accuracy and kappa. Add a plain classifier on the same encoder features. If it reaches ~93%, the LLM stack adds mainly text.
- **Relevance:** This is the best verifiable-reward idea in the set for MRI. The verifier uses quantities the model never sees.

### 2.6 MedVol-R1 (full text read, repo not opened)

- **Design:**
  - Qwen3-VL-4B picks a key slice and boxes. A frozen MedSAM2 propagates them to 3D.
  - LoRA SFT (r=128, lr 2e-5), then GRPO (G=4, β=0.01, lr 2e-6, 3 epochs).
  - Reward has four equal-weight terms: format, slice visibility, Hungarian IoU, neighborhood Dice.
- **Evidence (Table 1, DSC, SFT → SFT+RL):**
  - AbdomenCT-1K: 85.52 → 89.86
  - CT-ORG: 83.23 → 85.43
  - KiTS23: 36.21 → 45.46

  The gains (+4.34, +2.20, +9.25) are arithmetically correct. On CT-ORG, SFT-only is below M3D (83.49).
- **Ablations (Table 2):** The reward-component ablation runs on AbdomenCT-1K only and the Δ sweep on CT-ORG only. RL without SFT fails to converge. No seeds are reported.
- **Reward validity:** The reward depends on ground-truth masks. The frozen MedSAM2 does the 3D work, so part of the gain is slice selection.
- **Limitations:** CT only. MRI is listed as future work.
- **Smallest falsifier:** Compare the RL-chosen slice with the SFT-chosen slice and an oracle best slice. If the oracle gap is small, the four-term reward is over-engineered.

## 3. Across-paper disagreement ledger

| Issue | Position A | Position B | Resolution |
|---|---|---|---|
| Does reasoning help? | Med-R1: No-Think ≥ Think (72.49 vs 69.91) | Vero: reasoning patterns differ by category. BrReMark: verify step helps (no ablation). | Nobody isolates reasoning on MRI. Open. |
| Is uniform mixing right? | Vero: equal ratios win by 0.6–1.0 point (seeds not visible) | MedGRPO: raw mixed rewards collapse, so normalization matters, not mixing | Reward scale is first-order. Mixing weights are second-order. |
| Joint vs single training | Vero: single-category training hurts other categories by up to −35 | MedGRPO: NAP regressed in the joint model when left out of the reward | Joint training needs a check on every task. |
| Rule vs judge rewards | Vero: routed verifiers beat math_verify (57.2 vs 51.8) | BrReMark and MedGRPO: judges double as baselines or metrics | Judge independence is not established anywhere. |
| RL vs SFT gain | Vero and MedVol-R1: RL adds over SFT | BrReMark: no SFT-only ablation. LoV3D: DPO adds +0.4 accuracy. | Effect size ranges from +0.4 to +9. |
| Algorithm | Vero: GSPO ≥ GRPO ≥ DAPO by 0.4 | MedGRPO uses DAPO-style. LoV3D uses DPO. | No evidence the algorithm matters. |
| Evaluation validity | Med-R1: pooled random split | LoV3D: patient-level split | Only LoV3D states patient-level splitting. |
| Number consistency | Med-R1: 29.94% vs 31.80 points. MedGRPO: ε 0.3 vs 0.28. Vero: judge Qwen3-32B vs repo Qwen3.5-27B. | | All three are SOURCE_MISMATCH. |

## 4. Challenge to the proposed Vero-style MRI RL

1. **Is it necessary?** Not as stated. Vero's evidence is broad general-vision transfer over 30 benchmarks, not a narrow single-modality medical domain. The closest medical analogue, MedGRPO, shows the critical piece is reward normalization. A mixture of MRI tasks may not need Vero's 600K scale.
2. **Task categories.** Vero's six categories are a generic taxonomy that its authors say is not shown to be optimal. For MRI, derive categories from clinical decisions: detection, localization, sequence/plane identification, diagnosis, longitudinal change, quantification. Measure a train-A/test-B interference matrix first, as Vero does in Sec. 5.
3. **Uniform mixing.** The evidence is a ≤1-point difference in one model family. MRI task datasets differ by orders of magnitude in size, difficulty and label noise. Uniform mixing repeats small datasets many times and risks memorization. Use equal budgets as one arm against proportional and inverse-accuracy weighting, with seeds.
4. **Reward validity.**
   - Letter-match rewards (Med-R1) reward guessing on templated questions.
   - IoU rewards capped at 0.5 (BrReMark) can be met by large boxes on lesion-heavy images.
   - Dice and IoU rewards need ground-truth masks. Check that no ground truth leaks into the prompt.
   - Judge rewards (Lingshu-32B, GPT-4.1, Qwen) reward fluent clinical language.
5. **Reward hacking.** None of the six papers measures it directly. Plausible exploits in MRI:
   - Always emit a plausible lesion box.
   - Output the majority class.
   - Echo sequence names to pass a modality gate.
   - Pad rationales to please a judge.
   - Use sequence or plane cues instead of pathology.

   Audit by tracking reward against an independent held-out metric, response length and per-class rates, and run a shuffled-image control.
6. **Does prior medical RL already address it?** Partly.
   - Multi-task medical RL is in MedGRPO, on video.
   - MRI grounding RL is in BrReMark, without an SFT-only ablation.
   - Verifiable-reward brain MRI is in LoV3D, using DPO.
   - 3D evidence grounding is in MedVol-R1, on CT.

   I found no controlled MRI study of joint vs single-task training with matched compute and seeds. My literature search was a single web query (MedVLM-R1 and MediX-R1 appeared), so completeness is **NOT_VERIFIED**.
7. **What would be a meaningful outcome?**
   - Gains on a patient- and site-disjoint test set, against a matched-compute SFT baseline and a plain discriminative classifier.
   - False-positive rate on normal scans, and calibration at a clinically relevant operating point.
   - Rare critical errors (for example a missed hemorrhage or infarct) and abstention behavior.
   - Rationales and boxes that support the answer, rated by radiologists blind to model.
   - Seeds and confidence intervals reported.

## 5. Minimal falsifying experiment for the proposal

Pick 3–4 MRI tasks with patient-disjoint splits. Use one 4B–8B base, a matched-budget SFT baseline, and 3 seeds. Arms:

- (a) SFT only
- (b) per-task RL
- (c) joint RL with uniform mixing and raw rewards
- (d) joint RL with MedGRPO-style normalization
- (e) joint RL with proportional mixing

The proposal is falsified if (c) vs (e) is inside seed noise, or if (b) matches (d) on average. Add a hacking probe with shuffled images and normal-only scans.

## 6. Recommendation: **REFORMULATE**

- **Not GO.** Nothing in these six papers shows Vero-style uniform mixing helps for MRI. Vero's evidence is general-domain, ≤1 point, and I saw no seeds.
- **Not KILL.** There is a real gap: no controlled, patient-disjoint comparison of joint vs single-task training, mixing and reward design in MRI. LoV3D-style verifiable rewards and MedGRPO-style normalization are concrete building blocks.
- **Reformulate** as a controlled study of reward and mixture design for MRI RL, with a reward-hacking audit. Don't frame it as a Vero replication. The novelty has to be methodological (normalization, verifiable rewards and a hacking audit), not "Vero applied to MRI".

Sources: [Vero arXiv](https://arxiv.org/abs/2604.04917), [Vero repo](https://github.com/zlab-princeton/vero), [MedGRPO](https://arxiv.org/abs/2512.06581), [Med-R1](https://arxiv.org/abs/2503.13939), [BrReMark](https://arxiv.org/abs/2606.25894), [LoV3D](https://arxiv.org/abs/2603.12071), [MedVol-R1](https://arxiv.org/abs/2605.26621).
