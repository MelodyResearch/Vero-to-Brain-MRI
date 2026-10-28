> Historical preparation snapshot. Native execution subsequently completed with a negative result; see [actual results](FINAL_ACCURACY_RESULTS_V1.md). Status claims below describe preparation time.

# Accuracy-first sprint: evidence, candidate and evaluation boundary

This operational experiment follows the researcher's newest accuracy-first request. It does not modify the frozen research plan,
declare research closed, reactivate the old auxiliary objectives, or authorize sealed16 evaluation. Repository cleanup is deferred.

## What has failed, and what the evidence actually establishes

| Question | Evidence and conclusion | Status |
|---|---|---|
| Class preference | Formal B28 sequence52/84 comprises FLAIR42/42 and enhancedT1 10/42. This is a real class failure, not a new model-selection criterion. Adam60's separately saved single-pair candidate scores also favored FLAIR on both images. | CONFIRMED outcomes; global image-ignorance causality NOT_PROVEN |
| TRAIN exposure | The old28-row sequence pool represents28people with one modality per person. The existing paired inventory contains168PNGs/28people/3levels/2modalities. Original finite pilots did not train on all of it. | CONFIRMED_DATA_DESIGN_LIMITATION |
| Visual path | Prior native computations showed image-conditioned score/gradient changes. Frozen vision plus language-only LoRA is the actual local adaptation. These tests do not prove class-correct GSPO gradients or sufficient adaptation capacity. | LIMITED_NATIVE_EVIDENCE; capacity bottleneck UNVERIFIED |
| GSPO signal | Original18RL:362 accepted updates/2286 attempted groups. The latest heterogeneous pilot sampled16answers but admitted no Midline correctness-contrast group and made0updates. Gate refusal is not a theorem that all standard GSPO gradients are zero. | CONFIRMED sparse effective signal; accuracy causality unresolved |
| Prompt/readout | At the same Adam52 weights, original-prompt DEV24 was10/24 with9invalids; the frozen concise condition was18/24 with0invalids. This improvement is prompt/readout, not RL weight learning. | CONFIRMED matched saved-output evidence |
| Paired-Prompt | SamePrompt and PairedPrompt both ended16/24 versus Parent18/24. They gained0 and lost2 correct predictions each; all24predicted labels agree between A/B. B admitted more groups without improving accuracy. | CONFIRMED negative internal pilot |
| Prior auxiliaries | v1/v2 native local class-margin/Adam directions conflicted; v3 did not establish general protection or accuracy gains. None is being retried here. | ARCHIVED_EXPLORATORY |
| Historical optimizer | Saved update directions correlate strongly with inherited first moments. No accuracy-causal momentum ablation is established. | HYPOTHESIS; retain original state and disclose it |
| Historical SFT | Sequence counts across all three same-data SFT seeds are54/84,57/84,52/84. A post-hoc best seed is not independent evidence, and those formal84 outcomes are not comparable to a new DEV24 percentage. Dense supervised labels provide a different signal mechanism, not a proven explanation of the seed results. | CONFIRMED results; mechanism INFERRED |

The current failure is not accurately summarized as a disconnected image input or a GSPO-sign bug. The strongest actionable
combination is incomplete paired label exposure and weak/uneven sampled correctness signal, alongside a real generation-contract
effect. Their separate causal contributions remain unknown.

## One candidate, not a search over two methods by default

The selected candidate is **patient-paired, class-balanced sequence SFT** on the existing56z045images. It is not GSPO, a new RL
loss, a visual feature classifier, the archived v1/v2/v3 margin auxiliary, or a larger architecture. Every step averages two
own-image mean response CE gradients, one per modality of the same person, then applies the restored persistent Adam once.
The vision encoder, base model, LoRA, optimizer formula and1e-5LR stay unchanged. Correct targets are available even when an
on-policy K4 group would contain0/4correct; no sampled response or advantage is invented.

The fixed target is the already-defined generic prefix plus the acquisition label, closing tags and EOS. Identical template
tokens can lower loss without improving classification. FLAIR/T1 targets have17/20tokens, so equal image weighting does not mean
equal weight for every label token. Strict greedy classification and class protection—not loss decrease—decide whether the
first run merits two order-seed replications. No new label-margin, contrastive term, scaling coefficient, reward, sampling search
or vision-adapter experiment is added merely to fill the two-candidate allowance.

This candidate changes data exposure, supervision and sequence-only task allocation relative to the old mixed-task GSPO pilot.
Its comparison to the unchanged Parent measures a supervised-stage change under matched evaluation; it cannot identify an SFT
versus GSPO method effect at equal budget. Inherited Adam52 moments are preserved, so residual historical momentum also contributes.

## Frozen development and stopping rules

- Parent chosen by the already-declared Adam52/boundary114 continuity, not by formal100 ranking.
- Existing56image hashes and source labels verified; no rerender or new data. One explicit non-site scheduling stratum.
- One complete pass:28patient-pair updates/56serial backwards/14028TFpositions. No sampled TRAIN rollout.
- Evaluate only the end of the fixed pass on all frozen DEV24:4TRAIN-disjoint development people, three levels, both modalities.
- Same concise prompt, processor, F32 numerical route, strict parser, greedy temperature0 and256-token cap for all comparisons.
- Replicate the single method only if seed0 has>=19/24correct, FLAIR>=10/12,T1>=8/12,invalid0 and>=7/12fully correct pairs.
  Seeds1/2 change only patient order. All results, including failures, are retained; at most3newtrainingruns, below the user's4run cap.
- Report all seed scores, confusion matrices, patient accuracy and descriptive patient-cluster bootstrap intervals. Four clusters
  and repeated development use prevent an independent/generalization or statistically robust improvement claim.
- Pair analysis reuses both real image evaluations and checks non-pixel input equality before any image-substitution interpretation.
  It is not proof of anatomical reasoning. No extra model calls are added for it.
- New16 outputs remain sealed. A final endpoint, protocol and selection rule must be sent to the researcher for confirmation
  before a once-only independent evaluation. No original final100 reuse for selection or candidate evaluation.

## Implementation and genuine readiness

Claude Code actually implemented the candidate helper and37NumPy/CPUtests; Codex implemented the exact entry and counters.
Independent Codex rechecked source binding, saved accuracy arithmetic and code. See
[design](FINAL_BALANCED_SFT_DESIGN_V1.md) and [preparation](../results/FINAL_ACCURACY_SPRINT_PREPARATION_V1.json).

The latest broad user grant authorizes bounded native training, but the dated AGENTS substantial-training power rule is not
silently removed. Default admission is AC with the retained color/RSS/MLX/disk/emergency protections. A fresh exception for this
new SFT must be explicit if battery operation is desired. At the preparation snapshot, the machine was onBATTERY59%,YELLOW,
available7.34–7.38GiB/free41%, stable16.23GiBswap and0otherheavyprocesses. This is a **power HOLD, not a model OOM or numerical
failure**. No native run is claimed and no run budget has been consumed by that snapshot.

## Source evidence

- [Original22 matrix](../results/ORIGINAL22_FINAL100_ACTUAL_MATRIX_V1.json)
- [Formal class/format/cost counts](../results/ORIGINAL22_FINAL100_CLASS_FORMAT_COST_SUPPLEMENT_V1.json)
- [Original21 training completion](../results/ORIGINAL21_TRAINING_ACTUAL_COMPLETION_V1.json)
- [V2 actual matched results](../results/ACCURACY_PILOT_V2_ACTUAL_RESULTS_V1.json)
- [V2 regression diagnosis](../results/V2_REGRESSION_DIAGNOSIS_ACTUAL_CPU_V1.json)
- [Heterogeneous pilot actual no-signal](../results/HETEROGENEOUS_MRI_PILOT_V2_NATIVE_ACTUAL.json)
- [Existing z045 preparation](../results/PAIRED_Z045_DATA_V1.json)

All original result JSONs, images, parent/Adam/RNG states, negative results and sealed data remain unchanged.

## Fresh scoped power authorization

After that preparation snapshot, the researcher explicitly replied in this chat: “仅为本次SFT允许电池≥20%，保留全部其他保护”. The new SFT scope alone permits this battery floor; global/historical policies are unchanged. This is not a reuse of a consumed micro-pilot exception. All other true-danger/color/RSS/MLX/disk protections remain, and the changed local admission code receives a fresh independent review before any native load.
