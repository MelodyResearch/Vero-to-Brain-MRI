# First matched likelihood pair: independent review

**PASS_WITH_LIMITATIONS — one fixed training-patient pair, three input conditions per checkpoint.** Actual A0/A28/B28 outputs were independently checked without model imports. All numerical token partitions, scores, oracle values and resource decisions reproduce; native prompt IDs, mask/grid fingerprints, source pixels and prepared pixel fingerprints match across checkpoints. All nine oracle disagreements are zero. The repeated A0 image pair also matches its earlier successful run exactly. Prepared pixel tensors were not independently rerendered; their fingerprints are native recorded evidence.

The public aggregate `ANSWER_LIKELIHOOD_Z045_FIRST_PAIR.json` (SHA256 `9bfae899830d3aceb06a221e7b795915d419d802bc6f858b8cf304a3f0a0afaa`) was separately checked for raw/mean/first-token margin distributions, signed margins, no-image shifts, patient-pair separation, checkpoint differences and strict greedy counts. Those checked results agree. This is a partial diagnostic, not the full28-patient result, an independent-patient test, or clinical validation. All selected z45 pixels were previously exposed through supervised panels.

## Verified observations

Mean label-token log-probability margin is T1 minus FLAIR. Negative values favor FLAIR; mean scores are candidate-relative heuristics, not calibrated class posteriors.

| Checkpoint | FLAIR image | T1 image | No image | Pair separation: T1 image minus FLAIR image |
|---|---:|---:|---:|---:|
| A0 | −2.678809 | −2.079753 | −2.957787 | 0.599056 |
| A28 | −1.712485 | −1.162522 | −1.941104 | 0.549963 |
| B28 | −1.104852 | −0.562694 | −1.281400 | 0.542158 |

Raw-sum and first-token metrics likewise prefer FLAIR for all nine inputs. Positive pair separation shows an image-dependent numerical response in the expected ordering for this pair, but neither candidate metric correctly classifies both images. Separation does not improve from A28 to B28: mean separation changes by−0.007805, raw separation2.742568→2.625590, first-token separation2.750000→2.625000.

A28→B28 mean margins shift toward T1 by0.607633 on the FLAIR image,0.599828 on the T1 image, and0.659703 with no image. Thus the no-image shift is at least as large as either real-image shift in this example. This constrains an image-specific learning interpretation even though the margins changed.

A0's actual response text names FLAIR but lacks the required full structure: strict format0/2 and strict accuracy0/2 on images. A28 and B28 each have full format2/2 and strict accuracy1/2; both choose FLAIR for both real images and no-image. Their final labels are unchanged, although one B28 reasoning phrase differs. All generations finish with `stop`; none hit the token limit.

## Competing hypotheses

| Hypothesis | What this pilot establishes |
|---|---|
| H1: supervision did not change answer-specific learning | The strict claim of unchanged answer likelihoods is contradicted for this pair: A28→B28 margins change materially relative to the exact numerical replay error. This does not establish improved discrimination or a cohort-level learning effect; “similar” requires a prespecified scientific criterion. |
| H2: greedy output hides changed answer preferences | The descriptive signature is present: label margins change while A28/B28 final labels remain the same. Because the no-image margin shifts at least as much, this does not isolate useful image-grounded learning or justify threshold tuning. |
| H3: training fit improves but generalization fails | Not established. This training pair still has an incorrect T1 preference and no separation improvement. A broader training-cohort result and matched development likelihood evidence are needed; older development greedy results alone do not complete this test. |
| H4: label priors dominate pixels | Compatible with the common FLAIR preference and similar checkpoint shifts including no-image. The nonzero paired separation and real-minus-no-image shifts contradict the stronger claim that pixels have no numerical effect. Their causal contribution or dominance is not identified by one pair. |
| H5: informative pixels but unsuitable task/prompt | The separate fixed-feature site-held-out sequence result supports information in the selected pixels. This pilot does not distinguish prompt mismatch, representation limits, acquisition shortcuts or adaptation budget, and does not provide a stronger visually supervised VLM comparator. |
| H6: remaining implementation defect | No prompt/index/visual-token/adapter-consistency/oracle defect was observed in these completed records. Independent exact arithmetic and repeated-pair agreement reduce those specific concerns, but do not prove every backend or scientific implementation correct. The two earlier resource stops remain genuine failed attempts. |

Do not choose one hypothesis as settled. The smallest scientifically useful expansion is the remaining **fixed** training pairs with unchanged metrics and source settings, through independently released bounded chunks when resources permit. No additional SFT/GSPO, threshold selection or reserved-patient access follows from this pilot.

## Engineering evidence

B28's54-event stage passed resource replay: battery81→81%, minimum available6.498GiB, minimum pressure-free36%, swap8.284180→8.260742GiB, MLX peak5.977558GiB; release active allocation approximately0.000001GiB. Earlier completed stages and failures are recorded in `LIKELIHOOD_INDEPENDENT_ENGINEERING_REVIEW.md`. Near-threshold memory remains a limitation, not a stability guarantee.

Private evidence includes `B28_hold_v3_003_audit_PRIVATE.json`, `matched_first_pair_independent_PRIVATE.json`, and `first_pair_aggregate_independent_PRIVATE.json` under `likelihood_independent_review_v1/`. The latter SHA256 is `7073c4eeb4f2c638822ea0d04aac1df7fe8f04837f5f7cc9f578fb170c5e0264`. Review is by an independent Codex agent of the same model family; private cross-model review remains NOT_VERIFIED. This report grants no new model-launch scope.
