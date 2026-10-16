# A28 answer likelihood: saved 42-record partial stage

**Status: actual inference outputs summarized; `PARTIAL_FIXED_PREFIX`.** This is a single-checkpoint diagnostic on training-source patients. It is not a full 28-patient result, a matched checkpoint comparison, an RL result or evidence of independent-patient generalization. The model stage stopped at its battery resource boundary; the saved prefix was preserved. Independent arithmetic and scope verification returned **PASS_WITH_LIMITATIONS** for this exact partial diagnostic; see [the engineering review](LIKELIHOOD_INDEPENDENT_ENGINEERING_REVIEW.md).

## Evidence and accounting

The final aggregate is [A28_LIKELIHOOD_PARTIAL42_AGGREGATE_V3.json](A28_LIKELIHOOD_PARTIAL42_AGGREGATE_V3.json), SHA256 `4ed55f4d4081e1c75f46f89f0c05ddebc1031177df5c0a2ab4ddffed757b45a7`. V1 and V2 are retained as provenance history. V3 uses the final scope-bounded loader and shared-metric source; their scientific values are unchanged. There were no repeated model forwards to produce any version.

The 42 saved records contain **three preserved pilot records and 39 newly computed resident records**. No completed pilot generation was repeated. Their scientific accounting is:

- **20 complete patient pairs:** 20 FLAIR and 20 contrast-enhanced T1 images.
- **One additional FLAIR image:** reported separately, with its unrun T1 partner not imputed.
- **One unique no-image reference:** not 20 independent reference observations.

The valid final approval permitted progression from 35 to 43 records; stopping at 42 is within that scope. The loader verifies the approved existing-prefix hash, exact next frozen record IDs and bounded cursor, plus the byte-preserving historical-prefix adoption and current-tail identities. It accepts partial completion of the final approved chunk, but rejects an earlier approval that permits fewer than the observed records.

## Actual greedy outputs

| Saved image subset | FLAIR correct | T1 correct | Strict correct | Full format | Termination |
|---|---:|---:|---:|---:|---|
| Complete patient pairs | 20/20 | 4/20 | **24/40 (60%)** | 40/40 | 40 stop; 0 length; 0 unknown |
| Additional unpaired image | 1/1 | Not run | 1/1 | 1/1 | 1 stop |
| All saved images | 21/21 | 4/20 | **25/41** | 41/41 | 41 stop; 0 length; 0 unknown |

All 16 incorrect T1 predictions were FLAIR. There were no invalid or unparsed image outputs. The unique no-image output was also strict-format FLAIR with stop termination; it has no image-specific gold label and is not included in accuracy.

## Paired likelihood diagnostic

For each input, the margin is the contrast-enhanced-T1 score minus the FLAIR score. For each complete patient, **separation** is this margin on the T1 image minus the margin on the same patient's FLAIR image. Positive separation means the relative T1 preference is larger on the T1 image; it does not require either image to be correctly classified.

| Score definition | Mean separation, 20 pairs | Median separation | Range | Positive / negative pairs |
|---|---:|---:|---:|---:|
| Mean label-token log probability | **0.836744** | 0.212748 | −0.326747 to 4.107669 | 15 / 5 |
| Raw label log-probability sum | 3.118168 | 1.059285 | −1.605400 to 12.076334 | 15 / 5 |
| First label-token log probability | 3.131250 | 1.062500 | −1.625000 to 12.125000 | 15 / 5 |

The mean-token margin averaged **−1.460737 on FLAIR** and **−0.623993 on T1** across complete pairs. Both class means still favor FLAIR. All three score definitions give the correct candidate preference on 24/40 paired images. Positive paired separation coexists with substantial T1 misclassification and five reversed pair directions.

The unique no-image margins were −1.941104 for mean-token scores, −9.705869 for raw sums and −9.625000 for first-token scores. Across the 40 paired images, mean true-label-signed real-minus-no-image margin was 0.418372, 1.559084 and 1.565625 respectively. **These balanced signed averages equal half the corresponding mean pair separation algebraically:** the common no-image margin cancels between the two labels. They must not be presented as independent corroborating experiments. Per-class and true-label log-probability differences are retained in the aggregate for inspection.

## Interpretation and remaining experiments

The saved A28 checkpoint exhibits image-associated relative candidate-score variation within this training-source prefix, alongside a strong FLAIR preference. This is a descriptive observation about the evaluated inputs. The data do not establish that SFT caused the variation, that balanced exposure improves it, or that it generalizes to independent patients. That requires matched A0/A28/B28 outputs and later independent-patient evaluation. The uncompleted records are explicitly absent.

Raw sums, mean-token scores and first-token scores are different estimands. Candidate preference is not a calibrated disease posterior. The resource-stopped prefix is not a random patient sample; no p-values, population intervals or outcome-based selection were used. Prior paired-panel exposure means absence of direct sequence supervision does not make an image unseen.

## Engineering verification scope

The new single-checkpoint preparer is `scripts/analyze_single_checkpoint_likelihood.py` (SHA256 `dc778988b20e0979c1354908fb5c67662e123ff67085b434946ab86affe8f7df`). Its 13 tests use synthetic inputs. An independent reviewer found that the first version accepted a stale approval with an insufficient record budget. Three failing-first regressions reproduced missing cursor, existing-prefix hash and next-ID checks; the corrected loader rejects all three. The same gap was reproduced and corrected in the paired offline loader, SHA256 `c368064de200b464c452d0dd0c5c01fb2a0210ba138634800f9fcd1a237e9f88`.

The combined single-checkpoint, paired-checkpoint and adoption suite passed **51 tests in 0.45 seconds**, with no model/native runtime work. Independent Step D verification replayed the actual stale approval (rejected) and valid final approval (accepted). The reviewer separately recomputed **391 scalar aggregate fields** from raw token vectors and directly parsed responses without importing the analyzer or scorer; all matched V3 within 1e-12. Independent audit SHA256: `7b42537e73a5c5f62184832370d25226dc18d07653f9cc512daa75cde70c0277`. This supports the arithmetic and provenance of the partial result, not scientific generalization. No inference/training runner was modified by this analysis work.
