# Complete A0/A28/B28 training-only likelihood diagnosis

Mission: `B_LABEL_CONTROL / EXPLORATORY_DIAGNOSTIC`. Evidence: `OBSERVED` and `REPRODUCED` by independent Codex numerical reviewer. All 171 frozen records are durable; each checkpoint has 56 images from the same 28 training patients plus one shared no-image reference. No new training updates occurred.

| Measure | A0 | A28 | B28 |
|---|---:|---:|---:|
| Strict free-generation format | 0/56 | 56/56 | 56/56 |
| Strict free-generation correct | 0/56 | 34/56 | 36/56 |
| Mean-token candidate preference correct | 34/56 | 34/56 | 37/56 |
| Mean-token same-patient separation | 1.190332 | 0.832248 | 0.874533 |
| Raw-logp same-patient separation | 4.758709 | 3.177639 | 2.830817 |
| First-token same-patient separation | 4.763393 | 3.196429 | 2.834821 |
| Mean-token no-image T1-minus-FLAIR margin | -2.957787 | -1.941104 | -1.281400 |
| True-label real-minus-no-image mean-token logp | 0.660560 | 0.478992 | 0.367134 |

**Interpretation.** A0's zero strict score is a format failure; conditional candidate likelihood shows existing image sensitivity. SFT substantially changes output format and class preference. B28 versus A28 changes mean-token margins by +0.624651 on images and +0.659703 without images, with only two greedy predictions changing. This supports a dominant shared answer-prior shift. Mean-token paired separation rises +0.042285, while raw separation falls -0.346822 and first-token separation falls -0.361607. None of these separation means exceeds A0. Balanced exposure has not demonstrated improved general visual discrimination in this frozen cohort.

**Cross-model disagreement preserved.** Authentic Claude independently read the sanitized aggregate before Codex's conclusion; see `CLAUDE_LIKELIHOOD_INDEPENDENT_INTERPRETATION.md`. Claude describes the A28-to-B28 effect as a prior shift “not grounding.” Codex qualifies that wording: the dominant common shift is supported, but the small image interaction is score-dependent and not zero. The experiment does not establish that only priors changed. Claude inspected aggregates, whereas the isolated Codex reviewer independently recomputed raw token and patient-paired values; do not conflate those scopes.

**Limits.** All subjects are training patients, with earlier paired-panel exposure. Fixed-prefix conditional likelihood is distinct from unconstrained generation. FLAIR and T1 labels have different token lengths; raw, mean and first-token scores are separate prespecified estimands, not calibrated clinical probabilities. Swaps are exact visible-input permutations, not new forwards or independent patients. No gradient-dominance inference follows from prefix NLL. No independent-patient generalization, clinical competence, RL improvement or full Vero transfer is established.

**Evidence.** `A0_A28_B28_LIKELIHOOD_COMPLETE171_AGGREGATE_V1.json` SHA256 `dad7f1895a81f5e1b8b9c240ede6b6a3def046541b78a24a0e0161e7c91a0723`; `COMPLETE171_INDEPENDENT_NUMERICAL_VERIFICATION.json` SHA256 `03dcda9a808e0791f9d7258eae47013ad17bd3f2838ab5638400be8ba22bc39d`. A0: 1,539 token fields independently recomputed; final primary/private comparison: 6,529 numerical values matched. Raw patient-linked outputs remain local.

**Next executed research lane.** Proceed to the source-bound genuine GSPO pilot, then eligible multi-task specialist/proportional/uniform comparisons with same-data SFT and reward controls. Likelihood diagnosis is complete; GSPO training and the project remain incomplete.
