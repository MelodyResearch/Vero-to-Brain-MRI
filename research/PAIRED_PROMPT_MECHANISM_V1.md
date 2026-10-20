# Paired-Prompt Group-Relative GSPO — CPU mechanism and limits

**Status:** `PROPOSED VERO-STYLE ADAPTATION`, CPU mechanism checked; no model loaded, no new responses, no native gradients or optimizer updates. This document records a bounded mechanism check. It does not modify the authoritative research plan.

## Isolated change and matched block

At one fixed checkpoint and numerical policy, take four genuine on-policy responses from each of a same-person/same-level FLAIR/T1 pair. Give every response its original prompt/image, complete tokens including applicable EOS, own generation logps, immutable original reward/format/accuracy/overlong receipt, and checkpoint/policy identity.

- Same-input control: `(F0,F1,F2,F3)` and `(T0,T1,T2,T3)`.
- Proposed paired-input grouping: `(F0,F1,T0,T1)` and `(F2,F3,T2,T3)`.

Both contain the same eight response objects, four exposures per class, and two K4 groups. The CPU block objective is the mean of those two group objectives, hence weight **1/8 once per admitted response**. Group mean, sample standard deviation (`ddof=1`, epsilon `1e-6`), the inherited Adam60 valid-format-subset accuracy/format/overlong informative predicate, original response mean-logp ratio and original clip constants are retained. Rewards are carried through without recomputation or reassignment. Both contexts carry the same exact task-routed reward/calibration SHA; the future Adam60 calibrated arm must remain unchanged. Raw binary values in the CPU fixtures are explicitly synthetic. A common positive affine transformation preserves their ordering and rescales the epsilon term, so raw binary normalized numbers must not be claimed numerically identical to calibrated rewards. This changes the conditioning scope of the group baseline and normalization. It is an adaptation, not official same-prompt Vero GSPO.

`src/mri_rl/paired_prompt_groups_v1.py` provides immutable response/context records and a typed CPU scoring request/receipt. A scorer must use each response's own conditioning and return matching response, context and token identities. This is a routing contract; it cannot authenticate fabricated upstream receipts. Future native code requires an explicit mixed-context scorer because the historical single-`self.active` prompt update cannot safely consume another prompt's tokens unchanged.

## Inherited admission gate attribution

Both future arms preserve `uniform_balance_continuation_gate_v1.accuracy_group_is_informative`, inherited from the Adam60 continuation. It requires correctness variation among format-valid responses but can admit a group containing another invalid or half-format response. This is different from historical18's all-format-strict `mlx_gspo_smoke.task_group_is_informative`, which requires every response to have format1. Neither predicate is labeled official Vero here. A descriptive CPU fixture `[accuracy,format]=[(1,1),(0,1),(1,.5),(0,1)]` is admitted by the inherited Adam60 predicate and rejected by the historical18 strict predicate. The fixture documents the difference; the current predicate remains unchanged in both proposed arms. Receipt routing accepts finite calibrated accuracy in `[0,1]` and carries explicit binary `raw_accuracy`; a binary synthetic receipt may default raw correctness to its accuracy. Nonbinary calibrated accuracy requires the original raw correctness field. Diagnostics report within-condition raw correctness separately from calibrated accuracy. Both accuracy and caller-supplied total are preserved without regrading. A CPU receipt with calibrated accuracies `.7/.3`, raw correctness `1/0`, and supplied totals `9.25/8.75` demonstrates representation only; those totals are not a proposed calibration formula.

## Local scalar surrogate

For response `i`, preserve its original context `x_i`, own-old conditional token logps `o_it`, and current conditional token logps `l_it`. With its response length `T_i`, define

\[
 d_i=T_i^{-1}\sum_t(l_{it}-o_{it}),\qquad r_i=\exp(\min(d_i,10)).
\]

For the group-relative advantage `A_i`, the original scalar loss is

\[
 \ell_i=\max[-A_i r_i, -A_i\operatorname{clip}(r_i,1-.0003,1+.0004)].
\]

The block loss is `sum_i admitted_i * ell_i / 8`. A CPU dual-number oracle compiles the unchanged historical native `loss_fn` AST and respects its `stop_gradient` operations. Its forward values and token derivatives agree with this reference away from clip/cap ties. This establishes the pinned port's local convention; it is not a full standard-GSPO or official-Vero theory claim. At the own-old point, `r_i=1` lies inside the clipping interval, so

\[
 -\nabla L=\frac18\sum_i \operatorname{admitted}_i A_i\nabla\operatorname{meanlogp}(y_i\mid x_i).
\]

This finite-sample conditional surrogate is well defined. The conclusion requires actual own-context probabilities and genuine upstream rewards. It does **not** establish an unbiased estimator of the original balanced expected reward, improved class recall, or an on-policy multi-update training implementation.

## Why a conditional baseline needs care

For a normalized complete-response policy under the usual differentiability/support assumptions,

\[
 \mathbb E_{y\sim\pi(\cdot|x)}[\nabla\log\pi(y|x)] = \nabla\sum_y\pi(y|x)=0.
\]

Therefore an action-independent conditional baseline contributes zero **in expectation** to this joint-score policy gradient. Mixing prompts does not by itself invalidate that identity. If reward is truly constant within each prompt and each complete response has fixed length, the positive F/negative T advantages can have zero expected signal while finite-sample gradients remain nonzero.

The actual group advantage has additional dependencies: its group mean includes the response's own reward, and its sample-standard-deviation denominator depends on the actions/rewards. They cannot simply be treated as a universally action-independent baseline and deterministic common scale. Same-prompt normalization already changes expected weighting; mixed-prompt normalization changes which difficulties and outcomes determine that weighting. The exact enumeration below demonstrates a relative direction change, not a claim that only the proposed grouping has bias.

There is a further response-length qualification. The GSPO local score uses complete-response logp divided by `T(y)`. For two complete outcomes with probabilities `p` and `1-p` and lengths `1` and `2`,

\[
 \mathbb E[\nabla\log\pi]=0,\qquad
 \mathbb E[\nabla\log\pi/T]=p(1-p)/2.
\]

At `p=.5`, this is `.125`. A nonzero cross-prompt constant advantage can activate this response-length weighting even while the conditional reward gradient is zero. No change to the historical mean-logp loss is proposed here; this is an explicit failure boundary.

## Reproducible synthetic counterexamples

The deterministic CLI writes `results/PAIRED_PROMPT_CPU_MECHANISM_V1.json`. All following probabilities, rewards and Jacobians are synthetic mathematical fixtures, not MRI model estimates.

| Case | CPU finding | Limit on interpretation |
|---|---|---|
| F four correct, T four wrong, format valid/equal overlong | Same-input admitted groups `0/2`; paired admitted groups `2/2`. Mixed advantages are approximately `+.866023904` for F and `-.866023904` for T. | T still has zero correct sampled responses. Each condition's accuracy variation remains zero; admission changes only at the mixed group scope. More admitted groups is not accuracy evidence. |
| Same reward pattern, every complete-response mean-logp Jacobian equal | Paired local descent is exactly zero by cancellation. | No guaranteed visual learning signal despite nonzero advantages. |
| Same observed pattern, shared FLAIR-prior score derivatives F=`2`, T=`1` | Paired descent in that parameter is `+.4330119519`; a valid binary-policy construction gives T correctness directional derivative `-.2165059759`. The sampling event has probability `.019775390625`. | A possible genuine on-policy event can strengthen the wrong T prior; T improvement is not guaranteed. |
| Total reward varies only with format while accuracy is constant | Reward-variable groups `2/2`, inherited Adam60 gate admitted `0/2`. | Reward variance must not replace the retained informative predicate. |
| Prompt-constant reward, one-token Bernoulli outputs | Exact enumeration gives zero expected conditional-score direction despite admitted groups. | A baseline difference does not create a new expected correctness signal. |
| Prompt-constant reward with the variable lengths above | Mixed F logit drift is `+.05412649399`, true reward gradient is zero. | Mean-token weighting and finite samples need explicit interpretation. |

For difficulty/std weighting, enumerate all **256** outcomes of eight independent one-token Bernoulli responses with correctness probabilities `p_F=.3`, `p_T=.9`. Use the same response outcomes in both grouping plans and reuse the unchanged inherited Adam60 group gate. The true equally weighted expected-reward logit gradient is `[.105,.045]`.

- Same-input normalized expected direction is approximately `[.1486372985,.06587207973]`.
- Paired normalized expected direction is approximately `[.1469492249,.06225477913]`.
- Paired component scales relative to the true gradient are different: about `[1.399516428,1.383439536]`.

Let one shared parameter have F/T correctness-logit slopes `[1,-2.346]`. The true balanced-reward gradient is `-.00057`; same-input normalized descent is `-.005898600514`; paired normalized descent is `+.000899513077`. The paired direction decreases T correctness by about `.000189923191` per unit local descent. This is a concrete shared-parameter direction reversal under the stated model; it does not assign these probabilities to actual FLAIR/T1 inputs.

## Evidence and future on-policy comparison

This worker read no actual response tokens or model outputs and constructed no real paired block. The separate saved-evidence audit reported that the named sources do not provide an authenticated matched F4/T4 eight-response block: the original76 pool has one sequence modality per person, the completed stochastic F32 qualification has sequence K4 plus Dallas K4, and the candidate/margin probes generated no responses. Actual paired replay is therefore unavailable pending an authenticated contrary receipt. Its detailed audit belongs to the separate availability artifact.

The exact-same-response CPU contrast fixes one policy. A future real A/B experiment that updates after every group has different policy trajectories: A generates F4 then T4; B generates F2/T2 twice. Their later responses cannot be promised bitwise identical. Match the parent/full Adam/RNG, images, labels, numerical/reward/optimizer policy, total per-class exposures and finite generation/compute budgets; report realized trajectories, attempted/admitted groups and actual updates. Deferring eight responses and treating stale second-group probabilities as fresh on-policy updates would require another declared algorithm, not a silent implementation shortcut.

## Decision

The candidate is a mathematically definable, attribution-safe finite-sample surrogate with known reweighting and failure cases. CPU feasibility supports preparing an independently reviewed controlled comparison. It does **not** provide an automatic training GO or a guarantee of T1, MRI accuracy, Dallas retention, patient generalization, native numerical parity, or safe parameter updates. No CE, margin auxiliary, resampling until informative, advantage fabrication, loss adjustment or native launch is included in this mechanism work.
