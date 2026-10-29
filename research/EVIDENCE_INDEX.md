# Final evidence index

## Principal results and reproducibility

| Item | Current evidence |
|---|---|
| One selected model | [Parent identity/config/state qualification](../results/FINAL_MODEL_ELIGIBILITY_REVIEW_V1.json) and [handoff](FINAL_MODEL_HANDOFF_V1.md) |
| Final comparison and selection limits | [Final results/research report](FINAL_ACCURACY_RESULTS_V1.md), [machine metrics](../results/FINAL_METRICS.json) |
| Independent numerical reproduction | `python scripts/reproduce_results.py --check results/FINAL_METRICS.json`; six source hashes embedded in `src/mri_rl/final_results.py` |
| Original formal22 endpoints | [Original matrix](../results/ORIGINAL22_FINAL100_ACTUAL_MATRIX_V1.json), [class/format/cost](../results/ORIGINAL22_FINAL100_CLASS_FORMAT_COST_SUPPLEMENT_V1.json), [training completion](../results/ORIGINAL21_TRAINING_ACTUAL_COMPLETION_V1.json) |
| Matched GSPO endpoints | [Actual V2 comparison](../results/ACCURACY_PILOT_V2_ACTUAL_RESULTS_V1.json) |
| Actual new SFT, rejected | [Independent28checkpoint and24answer verification](../results/FINAL_BALANCED_SFT_SEED0_ACTUAL_REVIEW_V1.json) |
| Independent16 status | [Fixed protocol](../configs/final_evaluation_v1.json): NOT_RUN, current original inputmetadata not located, no researcher unsealing confirmation |
| Paper | [Internal draft](FINAL_ACCURACY_PAPER_DRAFT_V1.md), empirical negative development study |
| Current runtime source | Retained `src/mri_rl/` compatibility closure, `scripts/evaluate.py`; [CPU and native boundaries](../REPRODUCIBILITY.md) |
| Historical methods and failures | [Condensed history](HISTORICAL_SUMMARY.md); original immutable source commit below |

## Immutable historical locations

- Full source, raw scientific chronology and historic protocols: [a71bb39](https://github.com/MelodyResearch/Vero-to-Brain-MRI/tree/a71bb39a5b75d4cb65b7a958fa79548ff0805254).
- Pre-cleanup main: [b8b0493](https://github.com/MelodyResearch/Vero-to-Brain-MRI/tree/b8b0493754451befaf5e61285ca8bb05570ca36d).
- Previous cleanup/results tree: [b30dace](https://github.com/MelodyResearch/Vero-to-Brain-MRI/tree/b30dace1e96dc86bd88e6bc1c230946c541495cf).

Frozen plan/lock are copied byte-for-byte from the scientific branch, not rewritten as a new plan. Their dated future instructions are historical context; the final researcher decision ends training. The original full append-only log remains at the immutable scientific commit. The main publication log is a short final-sprint record and points to that original history.

## Privacy, evidence tiers and handoff

Only safe aggregates and opaque hashes are committed. MRI pixels, patient identifiers/row lists, modelweights, fullAdam/RNG and generated answer text stay in private local storage. The repository is PRIVATE, not a published model release. Parent payload/config hashing is independently verified; the final wrapper is not newly native-qualified. Four exploredDEVpeople do not establish independent clinical/generalization claims. Git hashes authenticate files, not expert judgment or human permission.

A collaborator with repository access can audit the numerical conclusions immediately. Full inference requires an authorized payload handoff and exact environment, plus a fresh execution contract. No private weight/data transfer or external mailing occurs in closeout.
