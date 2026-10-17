# Claude's independent reading of the complete 3-checkpoint likelihood aggregate (written before reading any Codex conclusion)

Source: `A0_A28_B28_LIKELIHOOD_COMPLETE171_AGGREGATE_V1.json` (28 training patients, 56 images + 1 shared no-image input per checkpoint, level z45, training-only, exploratory, no held-out data, `cross_model_review` NOT_VERIFIED). Mean-token margin = T1 minus FLAIR score (negative favors FLAIR).

| | A0 | A28 | B28 |
|---|---|---|---|
| greedy full format | 0/56 | 56/56 | 56/56 |
| greedy strict correct | 0/56 | 34/56 (FLAIR 28/28, T1 6/28) | 36/56 (FLAIR 28/28, T1 8/28) |
| mean-token margin, images | -1.834 | -1.068 | -0.444 |
| no-image margin | -2.958 | -1.941 | -1.281 |
| pair separation (T1-image minus FLAIR-image margin), mean | 1.190 | 0.832 | 0.875 |

Paired shifts (mean-token): A28-A0 images +0.766 vs no-image +1.017; B28-A28 images +0.625 vs no-image +0.660. Signed (correct-direction) margin change: A28-A0 -0.179, B28-A28 +0.021. Greedy label changed for only 2 of 56 inputs between A28 and B28 while the margin changed for 54.

Reading (descriptive, not causal):
1. **SFT mostly bought answer format, not visual discrimination.** A0 never produced a valid strict answer; both SFT checkpoints always do. Image-specific separation did not rise (1.19 -> 0.83 -> 0.88): the largest separation is in the checkpoint that cannot answer.
2. **The A28 -> B28 change is a shift of the class prior, not grounding.** B28 moves the margin toward T1 by 0.625 with images and 0.660 with no image; the signed (correct-direction) change is ~0 (+0.021). The two extra T1 greedy successes (8 vs 6 of 28) are consistent with a few high-T1-margin inputs crossing the decision threshold under a prior shift.
3. **A real but small image-dependent component exists in every checkpoint** (positive mean separation; 12 to 18 of 56 swap directions flip correctly) but it is small next to the FLAIR prior offset (class means remain negative for both image types) and was not improved by balanced T1 exposure.
4. **Not established:** independent-patient generalization (all inputs are training patients; all z45 frames had prior paired-panel exposure), clinical meaning, or that the shift reflects learning rather than label-frequency calibration. One slice level, descriptive spreads only.
Implication for GSPO: a verifiable-reward signal on this sequence task would at best reshape a prior the model already holds; any claim of visual improvement needs image-swap / no-image controls, which the dev-eval and likelihood tooling already provide.
