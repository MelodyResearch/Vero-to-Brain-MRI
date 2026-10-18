# Paired MRI prior-cancelled accuracy experiment (V3)

**Status:** IMPLEMENTED / NOT NATIVE-RUN / ACCURACY BENEFIT UNKNOWN.
**Research objective:** improve *MRI image-conditioned class discrimination*, not training reward or formatting. The original 18 GSPO arms and three SFT controls remain unchanged. The experimental code is isolated from `main` and must not be merged during an active frozen arm.

## Evidence motivating this test

- Actual native B28 internal-development results: sequence **15/24 (62.5%)**, Dallas same/different **8/16 (50.0%)**. U_raw seed7: **13/24 (54.2%)** and **8/16 (50.0%)**.
- An **independently numerically checked CPU-only image classifier** using *actual rendered spatial pixel features* scored **22/24 (91.7%)** sequence and **14/16 (87.5%)** Dallas on the **repeatedly inspected** internal-development examples. This does **not** mean Qwen or GSPO improved. See `results/V2_MATCHED_PIXEL_LINEAR_PROBE.json`. A histogram control scored **22/24** and **11/16**; scanner/intensity shortcuts remain plausible.
- The frozen Qwen vision feature diagnostic has not produced a native model result. The preexisting V2 separate-label SFT `paired_discrimination_pilot_v2.py` was CPU-prepared, not proven to train or improve MRI accuracy.
- The initial GSPO U7 result showed a class-preference shift and no improvement on C40. An auxiliary preselected **two-person** image subset later favored U7 on sequence, but is exploratory and not sufficient to reverse the full-C40 result.

## Method: eliminate shared answer bias from the *paired* learning signal

For the **same person's same-slice** FLAIR image `x_F` and T1ce image `x_T`, let:

`m(x) = mean_logp("contrast-enhanced T1" | x, fixed prefix) - mean_logp("FLAIR" | x, fixed prefix)`.

The V2 loss applies a binary cross-entropy independently to each image. The independent V3 pilot instead optimizes

`L = 0.5[softplus(m(x_F)) + softplus(-m(x_T))] + softplus(m(x_F) - m(x_T))`.

The **pair-ranking term is invariant** to an additive shared prior `m(x) → m(x)+c`; the first term keeps the two absolute image labels anchored. Therefore the objective directly rewards image-pair separation *as well as* correct independent answers, without extra labels or clinical supervision. This is a testable method hypothesis, **not a new-algorithm or performance claim**.

The existing paired source and frozen model/processor/target token strings are reused. The pilot uses two 4-image (two matched-pair) updates from the **same four selected pairs as V2**. It never selects patient/image examples by development accuracy. These four pairs are **not necessarily four independent patients**, and neither pilot can establish external validity.

### Implementation and memory

- `src/mri_rl/paired_prior_contrast_v3.py`: exact paired objective, analytic partial derivatives for each image margin, same patient/level validation, native two-pass serial-image backend and bounded entry point.
- `tests/test_paired_prior_contrast_v3.py`: regression cases for bias cancellation, finite-difference derivatives, correct optimization direction, source pairing and invalid inputs.
- Native pass 1 records each image margin **without retaining gradients**. Pass 2 streams the individual margin gradients and weights them using independently calculated derivatives before the single Adam update. This avoids holding both MRI autograd graphs at once, at the cost of additional native forwards.
- Reuse the V2 verified candidate/source pack, measured native input counts, frozen system prompt, common B28 4-bit base and LoRA, and the existing atomic SFT model/Adam/RNG/cursor checkpoints. New V3 source and objective hashes enter a **separate** run binding.
- Preserve the existing single-heavyweight-process lock, power/thermal/memory checks and no unrelated-app termination policy. **Do not run V2 and V3 at the same time**.
- Accounting: the inherited SFT ledger counts each distinct candidate token sequence once, **not** actual doubled native forward FLOPs. Compare wall time and forward counts before making any compute-matched causal claim.

## Exact decision, rather than chasing score

1. Finish/record original model evaluations without selecting favorable seeds or changing original arms.
2. CPU-check V3 analytic versus finite-difference gradients, native prefix/label-token alignment, and the inherited two-step bounded journal. Revalidate actual MLX gradients and the *real* checkpoint reload during the first native pilot. **No actual native training is claimed yet.**
3. Compare **B28**, existing **V2 two-candidate CE**, and **V3 CE + prior-cancelled paired ranking** from the **same exact B28 initialization and training MRI pairs**. Record each true-class recall, image pairwise direction, mean/raw candidate margins, no-image score shift, and greedy answer accuracy; do not relabel a forced-choice score as greedy correctness.
4. Before changing hyperparameters or extending to other patients, decide the expansion and independent patient-disjoint evaluation **on training-only evidence**. Do not use the repeatedly inspected four-patient internal dev or sealed final100 for iterative V3 selection.
5. If V3 increases pairwise separation without improving true-image class accuracy, it is **not** a successful accuracy result; if it only exploits acquisition cues, it is not brain-anatomical reasoning.
6. If the smallest pilot does not give a meaningful signal, preserve the negative result and prioritize stronger vision-adapted features rather than arbitrarily scaling GSPO updates.

**Paper boundary:** Vero/GSPO remains prior art. This is an exploratory MRI-specific paired objective built to test whether visual discrimination can improve under fixed training labels. A publishable contribution requires genuine full-model results, strong spatial/histogram and SFT/RL baselines, meaningful independent-patient evaluation, uncertainty, controls for acquisition shortcuts, and an honest novelty comparison.
