# Answer-likelihood analysis: independent scientific and arithmetic review

**Current Step D verdict: PASS_WITH_LIMITATIONS for the corrected analyzer source, synthetic arithmetic and provenance checks. Actual MRI likelihood comparisons remain NOT_VERIFIED because no completed matched outputs were available to this reviewer.**

**Initial verdict: FAIL for the affected preparation/profile checks and patient aggregation.** Reviewer: isolated Codex C/data agent, independent of the analyzer author. `CROSS_MODEL_REVIEW=NOT_VERIFIED`. This is a CPU/source review with MLX, MLX-VLM, Torch and Transformers imports blocked. No MRI likelihood scores were available or inferred during the initial review.

## Scope and initial evidence

The reviewer read the frozen likelihood contract, producer's scoring/index validation, analyzer tests and actual analyzer source. Initial analyzer SHA256: `ae18c3f21c6cc5c5071025c1f45fd52b8d413860332b05c00291e4e654ada1b0`.

Three synthetic counterexamples were independently reproduced against this exact version. They use invented patient IDs and token scores, not patient data or model outputs. Private findings are preserved in `local_data/ds007045_bounded/likelihood_analysis_independent_v1/initial_counterexamples_PRIVATE.json`.

| Finding | Reproduction | Consequence and minimal correction |
|---|---|---|
| LA-1: partial replacement preparation not checked | Analyze the first fixed patient pair, with all fixture outputs present, and alter B's prepared-pixel metadata for the next patient's replacement input. The analyzer accepts and labels the other-patient control an exact visible-input permutation. | A consumed replacement outside the selected origin cohort can have a mismatched checkpoint input. Validate cross-checkpoint tokenization, prompt and prepared-pixel metadata for every consumed replacement, or all common available inputs. |
| LA-2: equal but wrong scope profiles accepted | Supply three matching scopes with temperature 1, cap 512, rank 8, alpha 16, 8-bit quantization and group size 128. `load_scoped` accepts despite the frozen temperature 0, cap 256, rank 4, alpha 8, 4-bit/group-64 design. | Equality across checkpoints does not establish adherence to the frozen experiment. Require the declared fixed profile values in addition to cross-checkpoint equality. |
| LA-3: patient summaries assume adjacent rows | Interleave otherwise valid patient pairs, with B's T1 score shifts 0.3 and 0.9 and unchanged FLAIR scores. The analyzer reports patient means 0 and 0.6, descriptive SD 0.3. Correct per-patient means are 0.15 and 0.45, SD 0.15. | The grand mean happens to match while patient-level variation is wrong. Group by actual frozen patient identity, not adjacent row positions. Production's current ordering does not justify an unvalidated general assumption. |

The author independently identified two further hardening needs: proving that derived control prompts differ only by image payload, and binding input snapshots against concurrent append/change before publishing hashes. These were not represented as reviewer-discovered counterexamples. Corrections and fresh independent verification are required before interpreting an analyzer output.

## Scientific interpretation boundaries

Mean label-token log probability is the prespecified length-aware diagnostic. Raw sums retain label-length effects; both must remain separately reported. A softmax over either pair of candidate scores describes relative candidate preference, not a calibrated posterior over all possible answers. First-token scores test a separate boundary question and do not measure the probability of the complete answer.

The primary repeated-measures unit is one patient's same-slice FLAIR/T1 pair. Image-level descriptive counts are useful, but two image rows and two swap directions are not independent patients. A single common no-image prompt supplies one reference per checkpoint, not 28 independently sampled controls. Same-patient and other-patient swaps are derived re-indexings of already scored visible inputs under an identical question; they add no forward pass or new independent patient evidence. Partial-cohort results must identify the prespecified prefix and missing controls, and must not be presented as all 28 pairs.

These are training-source patients. At z45 all frames had prior paired-panel exposure, regardless of direct sequence supervision. A trained checkpoint's score shift on them cannot establish held-out MRI generalization. Fixed-format, answer and suffix losses are distinct measured quantities; token counts and scalar losses do not establish gradient dominance, capacity limits or a causal explanation for the prior development result.

## Mission Alignment Check

1. **Main question:** test whether controlled supervision changed image-conditioned answer likelihood before further Vero-to-MRI training.
2. **Falsifier:** mismatched inputs, profiles or patient units invalidate the corresponding comparison; targeted counterexamples test these directly.
3. **Isolation:** no labels, checkpoints, training code, reward, selected patients or generation settings were changed by this review.
4. **Provenance:** exact analyzer hash and synthetic failure receipts identify executed checks; actual MRI scores remain absent at the initial stage.
5. **Controls:** preserve raw versus normalized scores, unique no-image reference, paired images and explicitly derived substitutions without inflated sample sizes.
6. **Resources/decision:** bounded CPU-only checks; hold the affected analyzer interpretation pending correction, while unrelated authorized research continues. This review grants no model launch or training approval.


## Step D — corrected analyzer independently verified

**Verdict: PASS_WITH_LIMITATIONS**, evidence tier **REPRODUCED** for synthetic arithmetic and rejection checks. Final analyzer SHA256: `eecf622803635dde133e677a485149c083a841908b7187ac94f028183b467993`; regression-test SHA256: `18c31a75ee7bcac24ea1561d9227e010d527886f3d910a1360093510135ae1ca`.

The reviewer reran the three original scenarios against the corrected source using a separate saved replay script. LA-1 now rejects with `Cross-checkpoint prompt/pixel configuration mismatch`; LA-2 rejects with `Execution profile differs from frozen diagnostic`; LA-3 now gives the correct patient means 0.15/0.45 and descriptive SD 0.15 despite interleaved rows. These numbers are synthetic fixture values, not MRI findings.

A fresh CPU-only suite run produced **29 passed in 0.19 seconds** across the analyzer, answer-likelihood and paced-likelihood tests. Heavyweight runtime imports were explicitly blocked. This includes the author's added prompt-equality and snapshot-mutation regressions. Source inspection confirms common completed replacement inputs receive cross-checkpoint preparation checks, actual patient IDs define paired summaries, the fixed decoding/precision profile is enforced, and input bytes are snapshotted with change detection before publication. The identical-question check removes only the declared image payload before comparing prompts. Exact source snapshots and immutable approval receipts remain prerequisites supplied by the producer; an analyzer PASS cannot substitute for producer execution approval.

Additional independent arithmetic checks used a deliberately length-confounded fixture: the short answer has log probability −2 per token, the long answer −1 per token. The analyzer reports raw preference for the short label and mean-token preference for the long label, counts all four ranking disagreements, and reports zero patient-pair image separation when these scores are identical across images. It does not turn this length effect into evidence of image use. Exact equal scores retain the `TIE` category.

A separate incomplete-prefix check supplied only the first complete patient pair and the unique no-image row at all three checkpoints. Explicit partial analysis identifies **one** analyzed pair and **one** no-image input, marks unavailable other-patient replacements `NOT_READY`, and rejects the default full-cohort analysis. Derived same-patient controls retain zero additional forwards and zero additional patients. A partial fixed-prefix diagnostic must still be declared before examining outcomes; the program's option does not itself prove prospective selection.

Independent Step D script and receipt are private under `likelihood_analysis_independent_v1/replay_step_d_PRIVATE.py` and `step_d_PRIVATE.json`. The original failing evidence is preserved separately.

### Remaining scientific and execution limits

- No actual A0/A28/B28 likelihood aggregate was produced or independently checked in this review. A resource-stopped load supplies no likelihood observation. Do not substitute synthetic examples for absent scores.
- Production full analysis requires all 28 matched training-patient pairs and one common no-image reference at each checkpoint. If only a fixed prefix completes, report its exact patient denominator and partial status; do not extrapolate to all 28.
- Primary mean-token, raw-sum and first-token comparisons remain different estimands. Their disagreement is informative; none can be retrospectively chosen to rescue a preferred conclusion. Candidate entropy is not general model uncertainty.
- Patient pairing supports descriptive repeated-measures comparisons. There are no independent 56-patient or 56-direction samples, and this code supplies no population confidence interval or clinical claim.
- Same-patient and deterministic other-patient substitutions reuse already evaluated input/response pairs. Their correct-label scoring and directional shifts are descriptive controls, not an additional randomized image intervention or new inference experiment.
- Training-cohort likelihood changes cannot establish independent-patient generalization, explain the B development result causally, prove a capacity limit, or justify another training run without a separately falsifiable design.

The review clears the identified analyzer defects within the frozen source scope. It grants neither a model launch nor training approval and does not establish full Vero-to-MRI method transfer.

## Step D — immutable pilot-prefix continuation analysis

**Verdict: PASS_WITH_LIMITATIONS** for the updated analyzer's adoption integration, reviewed independently by the Codex methods/statistics agent against the root agent's implementation. Analyzer SHA256: `eb38b85154005114d8d6b984599c663148506a737a778fdc3396791c64aac41f`; analyzer-test SHA256: `6b24ad7f17fa18e81739500640bf5e2dc72f3c3458671223eee53416972cb743`. This review concerns provenance validation, not new MRI results, launch approval or cross-family review.

The source validates the immutable prefix bridge using the original independent approval, actual-result audit, original prefix bytes, unchanged numerical-loop AST and matching numeric/model/input profile. `completed_prefix` binds historical identities to exact prefix positions and permits only the current identity afterward. The analyzer receives a per-position identity sequence, not an unrestricted set of acceptable identities. Original rows and their original identities are retained.

Four synthetic end-to-end cases exercised the real adoption validator with a temporary private filesystem root: three historical rows followed by two current rows are accepted; a historical identity reused in the new tail, a changed prefix byte and a changed bridge digest are each rejected. These fixtures contain no MRI outputs. Their artificial numeric values are not experimental findings.

The reviewer identified a limited publication-time provenance gap: the new dependency files were checked during loading, but initially were absent from `main`'s final pre-publication evidence checks. The root author added bridge, prefix, approval and audit hashes to that final check. Four additional publication-path fixtures mutate these dependencies individually after validation and before publication. All four now reject without producing either aggregate or private output. The publication fixtures substitute synthetic scoring/validation callbacks only to isolate the final boundary; full real-validator checks are covered by the preceding integration tests. Public provenance lists dependency digests without exposing private paths.

Fresh CPU-only verification: **35 passed in 0.41 seconds** across `test_analyze_answer_likelihood.py` and `test_likelihood_adoption.py`. No model or native MLX work was performed. Actual checkpoint results, prospective cohort selection and resource admission remain separate review scopes. This change prevents repeated completed inference through byte-preserving continuation; it does not turn partial outputs into a completed cohort or increase the number of independent patients.
