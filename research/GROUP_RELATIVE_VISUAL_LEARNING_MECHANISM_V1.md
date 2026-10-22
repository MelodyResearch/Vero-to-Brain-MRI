# Group-relative visual learning mechanism (V1)

Date: 2026-10-22. Scientific lead and author: Claude Code (claude-opus-5-5). Independent verification: Codex (`/root/v2_actual_verifier`).
`CROSS_MODEL_REVIEW=NOT_VERIFIED` beyond the named Codex review.

**Final verdict: INCONCLUSIVE_CPU_COEFFICIENT_ONLY / INCONCLUSIVE — NOT_READY_FOR_TRAINING.**

This CPU experiment does **not** answer whether candidate C helps MRI accuracy. MRI accuracy is unchanged: Parent Adam52 18/24, Direct A54 16/24,
Paired B54 16/24 (DEV24, repeatedly explored, engineering only). This round produced no new accuracy result. Canonical class margins, task-separated
gradients, C's gradient and any visual or anatomical grounding are NOT_MEASURED. The existing actual MRI images were not accessed by the author here.
The frozen research plan and lock are unchanged.

Boundaries:
- 0 model loads, 0 generation, 0 teacher forcing, 0 backward/VJP, 0 optimizer calls, 0 new rollouts or rewards.
- No DEV/final100/new16 data used for selection.
- CPU counterfactual advantages **were** computed for C on saved responses. Zero historical advantages were rewritten or faked.

Mission alignment: PRIMARY_VERO_TRANSFER. The object of study is Vero group-relative advantage construction on one scientifically qualified
real MRI task (FLAIR vs contrast-enhanced T1); Dallas is an engineering control. This is not an Adam study, a prompt study or a supervised loss.

## 1. Evidence

| Evidence | Tier | Artifact |
|---|---|---|
| Mechanism diagnosis of the authenticated V2 run (re-derived plans, production regrade, inputs unchanged) | OBSERVED / REPRODUCED | `results/GROUP_RELATIVE_VISUAL_SIGNAL_DIAGNOSIS_ACTUAL_CPU_V1.json` (`57792250…`) |
| Independent scalar recomputation of A/B mechanism | REPRODUCED (Codex) | `results/GROUP_RELATIVE_VISUAL_SIGNAL_INDEPENDENT_FACTS_V1.json`, `…_VARIANCE_LENGTH_INDEPENDENT_V1.json` |
| C matched A/B/C on exact Common32, plus segment-1 observational | OBSERVED (actual CPU run: exit 0, 2.257 s, peak child RSS 115.6 MB, 27 tests PASS, inputs unchanged) | `results/GROUP_RELATIVE_C_MATCHED_ACTUAL_CPU_V1.json` (source `answer_matched_baseline_v1.py` `d3998ac4…`, CLI `ee716220…`) |
| Independent stdlib C recomputation | REPRODUCED (Codex; matches the actual run) | `results/GROUP_RELATIVE_C_INDEPENDENT_SCALAR_RECOMPUTE_V1.json` |
| Saved joint gradients gA0/gB0 | REUSED existing evidence, no new tensors | `results/GROUP_RELATIVE_SAVED_JOINT_GRADIENT_CONTEXT_V1.json` |
| Score-normalizer and action-dependent-baseline counterexamples | SYNTHETIC (tests) | `results/GROUP_RELATIVE_SYNTHETIC_SCORE_NORMALIZER_COUNTEREXAMPLE_V1.json`; `tests/test_answer_matched_baseline_v1.py` |
| Bounded primary method notes (GSPO, GRPO, DAPO, Dr.GRPO, Vero, mDPO/CoPO, SignBalance) | primary-source read, bounded | `results/GROUP_RELATIVE_PRIMARY_SOURCE_NOTES_V1.json` |
| Prior confounds: realized step ≈ inherited momentum (cos .992 A / .983 B), A53≡B53 on 4 selected DEV images, clipped surrogate worse after the step | OBSERVED / REPRODUCED, confound only | `research/V2_NATIVE_READONLY_ACTUAL_DIAGNOSIS_2026-10-21.md` |

Only segment 0 (Common32) is an exact matched comparison: same Parent Adam52 policy, and identical 32 responses, 1465 tokens, images, rewards
and exposure. Segment-1 rollouts come from different A and B post-update policies, so they are an observational audit only.

## 2. What the historical groups contain (actual MRI)

| Plan | Admitted / MRI groups | Same-image format-valid correct/wrong | Between-image only | MRI share of abs joint advantage mass |
|---|---|---|---|---|
| A0 Direct (Common32) | 2/4 | 2 | 0 | .392 |
| B0 Paired (Common32) | 4/4 | 2 | 2 | .572 |
| A1 Direct (own policy) | 0/4 | 0 | 0 | 0 (new gradient input is Dallas-only; Adam also retains history) |
| B1 Paired (own policy) | 4/4 | 0 | 4 | .583 |

**Common32.** FLAIR images 5/8 correct, T1 images 1/8 correct; 13/16 format-valid. Exactly 2 of 4 MRI images have same-image format-valid
correct/wrong contrast. Among format-valid responses, 53% of the correctness sum of squares lies between images.

**B1.**
- FLAIR: 6 valid-correct / 0 valid-wrong / 2 invalid. T1: 0 valid-correct / 6 valid-wrong / 2 invalid.
- Valid-only reward SSE is 100% between-image.
- 2 wrong answers on T1 images get positive advantage purely through format (.28% of MRI absolute mass).

**Sign quality.** In segment 0, both A0 and B0 are fully sign-correct.

**Length coefficient.** The GSPO per-token coefficient max/min ratio is 4.33 (A0), 12.38 (B0) and 388 (B1, from a 3-token invalid response).
These are scalar weights, not gradients.

**Existing saved joint gradients (reused, not new).** ‖gA0‖ = .2751, ‖gB0‖ = .3667, cos(gA0, gB0) = .6826, ‖gB0 − gA0‖ = .2691. Grouping
therefore changed the joint update input substantially. This does not show whether MRI class margins improved. The difference isolates MRI
grouping changes only under the assumption that the shared Dallas gradients are deterministic. Task-separated vectors were not saved.

**Score-normalizer counterexample (synthetic).** A zero answer-indicator projection can coexist with a nonzero score gradient that raises both
correct log-odds. Between-image-only groups can therefore genuinely help discrimination, so lack of within-image contrast alone cannot justify
rejecting them.

## 3. Candidate C and its actual matched CPU result

**C = answer-matched cross-image advantage plan** (`src/mri_rl/answer_matched_baseline_v1.py`). For each same-patient F/T pair (K=4 per image)
with unchanged rewards:
- Each format-valid response whose parsed canonical class label also occurs on the other image gets the mean reward of that label set as its
  baseline (ANSWER_MATCHED).
- Every other response, including invalid ones, gets its own-image mean (SAME_IMAGE_FALLBACK, i.e. Direct).
- A_i = (r_i − b_i)/(SD of the pair's 8 rewards + 1e-6).
- Admission is one inherited gate per pair. Weights stay .25 × 1/8.
- Dallas, reward, GSPO ratio/clip/length mean, LR, LoRA, optimizer and prompt are unchanged.

### Qualifications (all retained)

- **One combined advantage-plan intervention.** Relative to B's two K4 groups, C jointly changes the baseline set, the scale (pair-of-8 SD) and
  the admission unit (pair gate). It is **not** an isolated baseline-only causal test.
- **Action-dependent, changed estimator.** Label-conditioned baselines are policy-dependent, not the midpoint. In the synthetic case (uniform
  images, P(FLAIR|F)=.8, P(FLAIR|T)=.6):

  | Quantity | Value |
  |---|---|
  | b(FLAIR) | .43691 |
  | b(T1) | .47315 |
  | True β-gradient | −.015222 |
  | Baseline term | +.007249 |
  | Result | −.007973 |

  C changes the expected score-function gradient and is not unbiased. The real score normalizer still applies.
- **Label matching only.** Matching the parsed class label does not hold the full response text, reasoning prefix or length fixed. It removes one
  class-label contrast, not all nuisance, and identifies no visual causality.
- **Sign by construction.** For label-matched responses, sign = correctness *by construction*, so Q1 is largely definitional. Format share and
  point-biserial are coefficient descriptors, partly shaped by the construction itself.
- **No global centering.** Matched sets are centered, but overlapping same-image fallbacks are not, so the pair total is nonzero. It is reported
  below and deliberately not recentered.

### Actual matched Common32 result

Independent recomputation agrees across 64 checks: counts match exactly and the maximum absolute numerical difference is
1.11 × 10⁻¹⁶. This is not a claim of bitwise floating-point identity.

| Plan (same 32 responses) | Admitted MRI responses | Sign errors (wrong-valid+ / correct−) | Format share of abs baseline-split mass | Valid point-biserial (correct, A) | Credited Direct-inactive images (valid sign-correct credits) | Signed weighted MRI sum | Per-token coeff max/min | Backward calls if trained |
|---|---|---|---|---|---|---|---|---|
| A Direct (actual) | 8 | 0 / 0 | 0 | .8660 | 0 of 2 | ≈0 (−3.5e−17) | 4.33 | 20 |
| B Paired (actual) | 16 | 0 / 0 | .23774 | .95546 | 2 of 2 (5/5) | ≈0 | 12.38 | 28 |
| C (COUNTERFACTUAL) | 16 | 0 / 0 | .10384 | .97981 | 2 of 2 (5/5) | **−.09625** (F +.02379, T −.12004) | 5.20 | 28 |

C's admitted labels: 11 ANSWER_MATCHED and 5 SAME_IMAGE_FALLBACK. Signs: 6 correct+, 7 valid-wrong−, 3 invalid−. Both matched sets sum to
≈0 (max |sum| 6.7e−16). MRI share of absolute joint advantage mass: C .601 vs B .572.

Reading of the predeclared criteria: Q1 sign, Q2 format, Q3 hard modality and Q4 alignment **all pass**, so the decision is
**INCONCLUSIVE_CPU_COEFFICIENT_ONLY**. Specifically:
- **Not more groups:** C admits the same 16 MRI responses as B and covers the same 2 Direct-inactive images with the same 5 sign-correct credits.
- **What differs from B:** lower format mass (23.8% → 10.4%), higher point-biserial (.955 → .980), a smaller length spread, and loss of zero-sum
  centering. The first two are partly by construction.
- **The negative signed sum (−.09625, mostly from T) is not a parameter or bias direction.** It is a label-indicator-level coefficient total, and
  the normalizer counterexample forbids reading it as one.
- **A and B were already sign-correct on Common32,** so C fixes no sign error there. Its only sign-error advantage over a historical plan appears
  in segment 1.

**Segment 1 (OBSERVATIONAL counterfactual, different policies, never mixed):**
- **A1:** the actual Direct plan admits 0 MRI; C would admit 16, credit all 4 inactive images with 10/10 sign-correct, at format share .214 and
  signed sum −.16876.
- **B1:** the actual Paired plan has 2 wrong-valid-positives, format .208, point-biserial .920 and length ratio 388. C has 0 sign errors, format
  .174, point-biserial 1.000, ratio 44.5 and signed sum −.14500.

These describe scalar credit on post-update rollouts only.

## 4. Prior art (bounded notes only; no novelty claim)

- **mDPO/CoPO** (arXiv:2406.11839v2 §3.1) contrasts the **same full response** under a preferred vs less-compatible image. That is a stronger
  control than C, which matches only a parsed label across independently sampled responses. The same-answer/different-image idea is a known
  conceptual precedent.
- **SignBalance** (arXiv:2609.04063v1 §3.1/3.4) keeps verifier signs, replaces composition-dependent magnitudes with a global scale, and balances
  reward classes.
- **GRPO** (§4.1) and **GSPO** (§4.1/4.2) normalize within same-query groups, and GSPO adds a 1/length sequence ratio.
- **Dr.GRPO** (§3.1/3.2) analyzes and removes SD and length factors.
- **DAPO** (§3.2/3.3) filters all-0/all-1 prompts and resamples.
- **Vero** (`025ca3c…`) inherits same-query normalization.

C keeps GSPO length normalization, does not resample, and changes the baseline set, scale and admission unit together. It also belongs to the
action-dependent control-variate family. C is a PROPOSED VERO-STYLE ADAPTATION, distinct from the historical Paired-Prompt control.

## 5. Next single check: prospective protocol only

This is **not** a reviewed executable scope and **not** a current authorization request. Exact native teacher-forcing positions, source and input
bindings, the runner, resources and call budget must still be prepared and independently reviewed before any human approval. No call counts
here are audited.

**Question.** On Parent Adam52, do the exact Common32 plans A, B and C move canonical correct-minus-wrong class margins differently?

**Prospective design.**
1. **Load.** One read-only Parent Adam52 load, with no update.
2. **Per-response VJPs.** One mean-log-prob VJP for each saved Common32 response that has nonzero weight under any plan. These are needed
   because C's joint gradient cannot be reconstructed from the saved gA0/gB0.
3. **Plan directions.** Form g_A, g_B and g_C using the exact frozen GSPO response loss, own-old log probabilities and actual sequence-ratio/
   clipping derivatives. Compare newly scored conditional log probabilities with the authenticated saved own-old values; do not substitute
   `old = current` or assume the ratio is exactly 1. A weighted mean-log-prob VJP sum is exact only with its actual ratio/clip coefficients.
   The Dallas contribution is identical across plans and is included. Numerical agreement with saved A/B joint gradients is a prerequisite.
4. **Canonical margins.** At a fixed common prefix, compute correct-minus-wrong canonical label margins and their gradients for the 4 Common32
   MRI images. Because Dallas contributes to the joint update, also compute Dallas canonical margins for the Common32 Dallas contexts as a
   protection measurement. These are diagnostic only and never a loss.
5. **Two directions per plan.** Report each first-order margin change ⟨d, ∇margin⟩ for:
   - the negative gradient, d = −g_X;
   - the actual original Adam52-preconditioned step, d = −lr·(β1·m52 + (1−β1)·g_X)/(√(β2·v52 + (1−β2)·g_X²) + ε), using the saved
     read-only m52/v52 with bias_correction=False. This is neither an Adam ablation nor a reset; the m52 term is common to all plans and
     may swamp their differences.

**Falsification.** C is falsified if any of the following holds:
- under the Adam52-preconditioned direction, its first-order T1 margin change is not actually positive, or does not exceed both A and B
  (the per-image/aggregate rule and numerical tolerance must be frozen before any VJP; merely being less negative cannot pass);
- it lowers the FLAIR margin where A or B does not;
- its Dallas margin change is worse than both A and B beyond a tolerance frozen in the reviewed scope.

If C beats A and B only under −g but not under the preconditioned direction, then plan differences are swamped and C is not ready.

Even a favourable result would be first-order and local. It is not a finite update, not accuracy and not grounding. Any training would need a
separate decision; there is no automatic training, sampling or coefficient search.

## 6. Answers to the five research questions

**Q1. Why did Vero group-relative rewards fail to produce correct MRI discrimination?**
The strongest observed mechanism is that a strongly FLAIR-biased parent (T1 images 1/8 correct on Common32) leaves Direct same-image groups
with signal only where one image already yields both answers. For the hard class that signal disappears: A1 has 0 MRI groups and its new
gradient input is Dallas-only; the realized Adam step also inherits historical moments. Paired supplies cross-image signal, but compares different
labels across an arbitrary half-split. That lets format and image
difficulty into credit: B1 has 2 format-driven wrong-positives, a 21% format share and a 388× length spread.

Ruled out: reward/advantage replay errors, and sign errors in segment 0.

Not established: that either mechanism caused the observed regressions. A and B had different allocations and a joint-gradient cosine of
only .68, yet A53≡B53 on the selected DEV images and the realized step was ~collinear with inherited momentum (a confound). Margins and
task-separated gradients remain unmeasured.

**Q2. What was wrong or insufficient about Paired-Prompt grouping?**
It added groups and sometimes useful cross-image signal; the counterexample shows between-image contrast is not intrinsically invalid. What it
did not add was evidence quality. Its comparison sets depend on response order, mix different labels, and let peers' format validity shift
credit. On Common32 it was nonetheless sign-correct (format share .238). Its failures are the segment-1 format-driven wrong-positives and the
extra groups' lack of any accuracy gain (B54 16/24 = A54).

**Q3. What is the new Vero-method improvement?**
Candidate C, the answer-matched cross-image advantage plan defined in §3. It is one combined plan intervention (baseline set, pair scale and
pair admission), an action-dependent changed estimator, and not globally centered. It keeps cross-image comparisons only between responses
sharing a parsed class label. It differs from Paired in which comparisons it keeps and in its scale and admission. CoPO is a stronger
full-response precedent and SignBalance a related sign-preserving rescaling. No novelty claim.

**Q4. Does it work in the smallest real MRI experiment?**
Unknown. Only CPU scalar evidence exists. On the exact Common32, C passes all four predeclared descriptors without admitting more responses
than B. It has lower format mass, higher correctness alignment and the same hard-modality coverage, but these are partly by construction, and
its signed MRI total is −.096 rather than zero. No native gradient, margin or accuracy measurement exists.

**Q5. Should we expand?**
**INCONCLUSIVE; NOT_READY_FOR_TRAINING; no expansion.** Independent review agrees that at most one tightly bounded read-only falsification
check (§5) may be justified next, after an exact native scope is prepared, independently reviewed and freshly approved by a human. If that check
falsifies C, stop C. Do not change more variables.
