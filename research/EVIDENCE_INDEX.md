# Final evidence index

## Authoritative current entries

| Entry | Evidence and role |
|---|---|
| Final Results | [Research report](FINAL_ACCURACY_RESULTS_V1.md): actual negative comparison, costs and limitations |
| Final Metrics | [Machine table](../results/FINAL_METRICS.json), independently recomputed by `scripts/reproduce_results.py` from six immutable JSON inputs |
| Model Handoff | [Exact Parent and private access boundary](FINAL_MODEL_HANDOFF_V1.md), [state/config qualification](../results/FINAL_MODEL_ELIGIBILITY_REVIEW_V1.json) |
| Paper | [Internal empirical draft](FINAL_ACCURACY_PAPER_DRAFT_V1.md) |
| Reproducibility | [Current entry points and historical source dependency](../REPRODUCIBILITY.md) |
| Historical Summary | [Condensed methods, failures and negative results](HISTORICAL_SUMMARY.md) |
| Simplification | [One per-file deletion and dependency manifest](../results/SIMPLIFICATION_MANIFEST.json), fixed base commit and hashes |

## Immutable numerical sources retained in the current tree

- [Original 22-model formal matrix](../results/ORIGINAL22_FINAL100_ACTUAL_MATRIX_V1.json), [class/format/cost supplement](../results/ORIGINAL22_FINAL100_CLASS_FORMAT_COST_SUPPLEMENT_V1.json), [18 RL + three SFT completion](../results/ORIGINAL21_TRAINING_ACTUAL_COMPLETION_V1.json).
- [Matched Direct/Paired-Prompt GSPO actual results](../results/ACCURACY_PILOT_V2_ACTUAL_RESULTS_V1.json).
- [Balanced SFT actual checkpoint/answer verification](../results/FINAL_BALANCED_SFT_SEED0_ACTUAL_REVIEW_V1.json).
- [Parent state/config qualification](../results/FINAL_MODEL_ELIGIBILITY_REVIEW_V1.json).

These six source payloads, their pinned hashes and the published metric table are unchanged. Current CPU tests verify counts, strict class/format/cost behavior and the original matrix's nine historical source pins against Git objects. The original raw private answer records, receipts and model state remain private; a saved-aggregate recount does not regenerate inference.

## Fixed historical locations

- [Pre-second-round source and all 138 removed files](https://github.com/MelodyResearch/Vero-to-Brain-MRI/tree/1e0d7be1bb099b42e717eb4eb081facc12607ed3). The manifest gives each exact path/SHA. This contains the complete prior training/analysis wrappers and tests.
- [Previous cleanup manifest and its 573 removals](https://github.com/MelodyResearch/Vero-to-Brain-MRI/blob/1e0d7be1bb099b42e717eb4eb081facc12607ed3/results/CLEANUP_MANIFEST.json), [previous semantic cleanup report](https://github.com/MelodyResearch/Vero-to-Brain-MRI/blob/1e0d7be1bb099b42e717eb4eb081facc12607ed3/research/REPOSITORY_CLEANUP.md), [verification](https://github.com/MelodyResearch/Vero-to-Brain-MRI/blob/1e0d7be1bb099b42e717eb4eb081facc12607ed3/results/CLEANUP_VERIFICATION.json).
- [Historical independent original-22 numerical review](https://github.com/MelodyResearch/Vero-to-Brain-MRI/blob/1e0d7be1bb099b42e717eb4eb081facc12607ed3/results/ORIGINAL22_FINAL100_ACTUAL_MATRIX_NUMERICAL_INDEPENDENT_REVIEW_V1.json), [final delivery verification](https://github.com/MelodyResearch/Vero-to-Brain-MRI/blob/1e0d7be1bb099b42e717eb4eb081facc12607ed3/results/FINAL_REMAINING_VERIFICATION_V1.json). Other redundant review/process records are indexed per path in the manifest, without duplicating them.
- [Original scientific implementation and chronology](https://github.com/MelodyResearch/Vero-to-Brain-MRI/tree/a71bb39a5b75d4cb65b7a958fa79548ff0805254); [pre-first-cleanup main](https://github.com/MelodyResearch/Vero-to-Brain-MRI/tree/b8b0493754451befaf5e61285ca8bb05570ca36d).

The final read-only engine's 90 exact source files are an explicit dependency exported from the fixed pre-second-round commit. They are not maintained implementations on current main and their source-only export is not a full training environment. Current packaging is CPU-tested, NOT_NATIVE_QUALIFIED.

## Protected boundaries

Frozen plan/lock remain byte-identical. Their dated next-step instructions and old relative citations describe historical state; resolve removed paths at the fixed commits above, not as current execution instructions. The experiment log is append-only. No model/GPU call, output unsealing, private-asset cleanup, email sending, branch deletion or main merge occurs in this round.

Independent16 is **NOT_RUN** under the [fixed protocol](../configs/final_evaluation_v1.json). Exact original inputs and explicit researcher confirmation remain missing. Git hashes authenticate files, not expert judgment, patient independence or human permission. Four explored DEV patients do not establish clinical generalization. Authorized payload handoff is required for any native recipient.
