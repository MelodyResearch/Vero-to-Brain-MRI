# When verifiable reinforcement learning does not improve brain MRI sequence recognition

## Abstract

We investigate a local adaptation of Princeton Vero-style heterogeneous verifiable reinforcement learning to brain MRI, separating execution correctness from measured visual accuracy. Historical experiments completed 18 RL arms, 3 SFT controls and 22 frozen 100-question evaluations; subsequent matched small-development experiments compared Direct same-prompt GSPO and Paired-Prompt grouping from one saved parent. Both GSPO endpoints scored 16/24 against the common Parent's 18/24, while a newly executed patient-paired supervised continuation scored 15/24 after 28 updates and strengthened the FLAIR prediction bias. These results support a bounded negative finding for this implementation and explored development population; they do not establish a general failure of Vero, independent-patient improvement, or clinical reasoning ability.

## 1. Introduction

A verifiable answer format provides a computable reward, but it does not ensure that group-relative updates improve image-conditioned classification. We ask whether a small language-LoRA vision-language model adaptation benefits from direct Vero-style GSPO on MRI, whether pairing prompts improves the signal, and whether dense balanced supervision resolves a known modality-exposure defect. The final supervised continuation is a performance-oriented control, not a new RL method.

The project is independent of Princeton and does not use released Vero weights. Its MLX/Qwen3-VL-8B 4-bit recipe, frozen vision encoder, language LoRA, K4 groups, short responses and finite F32 qualification differ from the official distributed Vero implementation. We make no implementation-equivalence claim. This draft is an evidence report, not a validated clinical method or a submission-ready generalization study.

## 2. Data and evaluation populations

The original frozen formal evaluation contains 84 sequence images from 14 people and 16 Dallas whole-head correspondence questions. All 22 endpoints were evaluated under that original contract. Subsequent method development did not reopen these predictions for new selection. Dallas remains an engineering task; whole-head correspondence is not established as brain-specific reasoning.

Existing sequence TRAIN data contain 28 people with paired FLAIR/contrast-enhanced T1 images at three levels. The historical 28-row main sequence pool exposed one modality per person. The new supervised experiment uses 56 existing images at z045, one of each modality for every TRAIN person, without rerendering or adding patients. Source labels are acquisition metadata, not radiological diagnoses.

DEV24 contains four TRAIN-disjoint but repeatedly explored people, with three levels and both modalities per person. It is a development population with 24 correlated images, not 24 independent patients and not an untouched generalization test. The 16-new-person / 32-image collection remains sealed. Any final independent evaluation requires an endpoint and once-only protocol confirmed by the researcher before outputs are opened.

## Related Work

Princeton Vero supplies the heterogeneous-task, task-routed verifiable-RL/GSPO research framework ([official source](https://github.com/zlab-princeton/vero)). This independent MLX transfer uses different hardware, model adaptation and response limits; it is not an official implementation-equivalence study. The supervised continuation is a conventional full-response cross-entropy control and is not claimed as a novel objective. MRI sequence labels come from acquisition metadata, not clinical expert diagnoses; the task measures sequence recognition. The present evidence supports an implementation-specific empirical comparison, not a new clinical benchmark or broad prior-art superiority claim.

## 3. Methods and controls

### 3.1 Verifiable RL comparisons

Direct Vero-style GSPO uses real on-policy same-prompt K4 responses, task-routed verifiable rewards and persistent Adam. Paired-Prompt is an explicit adaptation of group construction and relative normalization across paired MRI prompts. The matched earlier pilot kept common parent, data exposure, reward, learning rate, LoRA, numerical policy and update cap, with separate on-policy responses after policy divergence. Each arm completed two updates. The GSPO formula and numerical traces did not show a sign error in the tested cases. Numerical agreement in limited contexts does not prove visual learning or official parity.

### 3.2 Dense balanced supervised continuation

The final candidate tests whether complete paired exposure and direct targets are sufficient under the existing model and state. For patient p and modality k, let CE(p,k) be mean negative conditional log probability over the fixed target response and EOS. Each update uses

`L(p) = 0.5 CE(p,FLAIR) + 0.5 CE(p,T1ce)`.

The two own-image gradient graphs are evaluated serially, averaged in float32, and applied once. The target shares a generic reasoning prefix and formatting tokens across classes. Labels have different token lengths: 17 and 20 response tokens including EOS. This full-response objective can improve template probability without improving class decisions. No margin loss, contrastive term, synthetic rollout or reward change is introduced.

The model restores Adam52 / full moments / RNG, retains 1e-5 LR and existing LoRA, and takes one 28-patient pass to Adam80. A seeded fixed patient order is the only planned replication variation. Inherited moments confound attribution: this is supervised gradients plus historical optimizer dynamics, not a reset-optimizer SFT experiment. It also uses more people/updates and different task allocation than the earlier GSPO pilot; it is not a budget-matched SFT-versus-GSPO causal contrast.

### 3.3 Readout and stopping rule

All matched DEV results use the same concise prompt, F32 scoring/generation path, processor, strict class parser, greedy temperature 0 and 256-token cap. Invalid answers remain in denominators. Only the end of the fixed pass is evaluated. The stopping rule admitted two additional order seeds only if the first seed reached 19/24 correct, FLAIR>=10/12, T1>=8/12, invalid=0 and >=7 fully correct pairs. That candidate cap did not require additional runs after the first seed failed; training and method search are closed.

## Experiments

The primary final comparison uses the four existing endpoints under one concise-prompt, F32, strict-greedy DEV24 readout. We retain all 24 images and every invalid answer in the denominator; the descriptive cluster unit is the patient. The original 22-model final100 cohort is reported separately, without post-hoc best-seed selection or pooling with development performance. Independent16 evaluation has not run.

## 4. Results

| Model | Accuracy / balanced accuracy | FLAIR recall | T1 recall | Invalid | Fully correct pairs |
|---|---:|---:|---:|---:|---:|
| Parent Adam52 | 18/24 (75.00%) | 10/12 | 8/12 | 0 | 7/12 |
| Direct GSPO A54 | 16/24 (66.67%) | 9/12 | 7/12 | 0 | 5/12 |
| Paired-Prompt GSPO B54 | 16/24 (66.67%) | 9/12 | 7/12 | 0 | 5/12 |
| Paired balanced SFT Adam80 | 15/24 (62.50%) | 12/12 | 3/12 | 0 | 3/12 |

Table 1 reports the same explored DEV24 readout. The first three rows are authenticated saved results; the last row is the new native experiment. No historical formal84 percentage is mixed into this table.

The new SFT model gains two FLAIR answers but loses five T1 answers. Its label counts change from Parent 14 FLAIR / 10 T1 to 21 FLAIR / 3 T1. Nine paired inputs now both predict FLAIR. The frozen improvement gate fails; seeds 1 and 2 are not run. The results show increased class preference despite balanced paired exposure, and do not demonstrate improved image discrimination.

Exact 4^4 patient-cluster resampling gives descriptive 95% ranges 58.33–91.67% for Parent and 54.17–66.67% for SFT. The paired accuracy-difference range is −37.50 to +8.33 percentage points. With four explored clusters these are descriptive uncertainty summaries, not a population significance claim.

The historical formal test is reported separately: B28 scored 52/84, with FLAIR 42/42 and T1 10/42. The three historical SFT seeds scored 54/84, 57/84 and 52/84. All seeds remain visible. A post-hoc best formal-test seed cannot be called a new independent winner, and no same-contract DEV24 evaluation of these historical SFT controls was run here.

## 5. Execution evidence and resource cost

The new native run completed 56 backwards, 14,028 TF positions, 28 optimizer updates and 24 DEV generations totaling 417 returned tokens. Training took 372.913 s, evaluation 119.222 s, and the child process 557.305 s. Independent review hashed all 28 full checkpoints and receipt chains, verified 10 Parent files unchanged, and checked the saved exact terminal reload. The process exited normally with model references released, 1,050 active MLX bytes and a free global lock.

Peak RSS was 5.8243 GiB; native reported MLX peak 7.9008 GiB. Swap increased 17.0425→21.2609 GiB, peaking 21.2999 GiB. The dynamic safety policy did not observe CRITICAL or trigger termination; temperature telemetry was unavailable. These measurements describe this run and cannot guarantee future system stability. No other application was closed or system protection changed.

## Discussion

Sparse correctness signal, incomplete historical modality exposure and strict generation contracts are directly observed difficulties. Correcting paired exposure with dense full-response supervision was insufficient in the final candidate. Historical Adam dynamics, shared-template token weighting, label length, frozen vision/language-LoRA capacity and distribution shift remain plausible contributors; this experiment does not isolate their causal effects.

The paired readout is a behavioral image-sensitivity diagnostic. It is not an independent image intervention or evidence of anatomical reasoning. No additional no-image probe, canonical margin analysis, Dallas retention test or clinical diagnosis evaluation was performed for the SFT endpoint. Invalid=0 does not imply correct visual decisions.

## Limitations

Repeated DEV use, one failed order seed, four patients, nonmatched SFT/GSPO budgets and absent sealed evaluation restrict conclusions. The study cannot show cross-center generalization, a second clinically qualified task, statistical superiority, or reliable replication. Neither executing more updates nor finding valid gradients is a substitute for those outcomes.

## Final selected model and independent evaluation

The sole principal deliverable is Parent Adam52 (`P_calibrated13 / Adam52 / boundary114`), adapter SHA256 `c70b426b58191446545ba6333f7595e96f4fdb4cc689ddbaf5206e276b329db6`. It has the best verified accuracy within the four matched, repeatedly explored DEV24 endpoints. Historical SFT has not been evaluated under that exact DEV24 contract, so this is not a global-best checkpoint claim.

The independent 16-person evaluation is **NOT_RUN_INPUT_METADATA_NOT_LOCATED_AND_NOT_UNSEALED**. Historical collection/seal/exclusion hashes exist, but the current private row manifest and per-pair quality map were not found after the bounded project/private-run/work and historical-reference recovery check. No replacement patients or model outputs were used. A fixed Parent-only protocol includes all 32 images, including all four historically flagged pairs, with strict greedy/F32, no training and no output-based exclusions. Restoration and verification of the original metadata, independent executable-scope review and explicit researcher unsealing confirmation are required. The missing independent result does not change the reported negative DEV result.

## 7. Conclusion

For the completed comparisons, retain the unchanged Adam52 Parent as the best same-contract DEV endpoint. Both the tested GSPO grouping adaptation and the final balanced SFT continuation failed to improve it. The appropriate output is a reproducible negative development study with preserved checkpoints and explicit unresolved mechanisms, rather than an accuracy or clinical claim. Further training is stopped; training and method search are now closed. The private handoff is indexed; the missing independent evaluation remains unrun pending original metadata and researcher confirmation.

## Evidence references

1. [Original22 formal matrix](../results/ORIGINAL22_FINAL100_ACTUAL_MATRIX_V1.json) and [class/format/cost supplement](../results/ORIGINAL22_FINAL100_CLASS_FORMAT_COST_SUPPLEMENT_V1.json).
2. [Matched GSPO pilot](../results/ACCURACY_PILOT_V2_ACTUAL_RESULTS_V1.json).
3. [Actual SFT independent verification](../results/FINAL_BALANCED_SFT_SEED0_ACTUAL_REVIEW_V1.json).
4. [Methods, limitations and exact scope](FINAL_ACCURACY_RESULTS_V1.md), [source design](https://github.com/MelodyResearch/Vero-to-Brain-MRI/blob/a71bb39a5b75d4cb65b7a958fa79548ff0805254/research/FINAL_BALANCED_SFT_DESIGN_V1.md).
5. [Princeton Vero source](https://github.com/zlab-princeton/vero): primary method provenance. No new literature review or third-party reproduction is claimed in this sprint.

## Internal claim review / draft status

The principal contribution is evidence and bounded failure analysis, not an established new method. Major missing evidence: untouched patient evaluation, matched historical SFT readout, replication of an actually positive candidate, and clinically qualified heterogeneous transfer. No venue acceptance score or statistical significance is asserted. This is an internal English paper draft, not submitted or publication-ready.
