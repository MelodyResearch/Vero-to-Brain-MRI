# Vero-to-MRI accuracy readout checkpoint (2026-10-18)

**Mission Alignment Check — `PRIMARY_VERO_TRANSFER`, `OBSERVED` for saved outputs and receipts.** This reads the original two-task internal-development cohort to test whether the Vero-style RL recipes improve actual MRI task accuracy over common B28. The falsifier is no improvement on strict greedy MRI candidate discrimination or a material Dallas decline. We change no training schedule, weights, prompt, parser or sealed-test selection. This is a readout checkpoint, not a new evaluation or a claim of clinical generalization. Units are four sequence patients and four Dallas paired cycles (eight people), not 40 independent patients.

## Actually available matched C40 accuracy

| Checkpoint | Lineage | Sequence strict greedy | Dallas strict greedy | Equal-task macro | Sequence candidate | Interpretation |
|---|---|---:|---:|---:|---:|---|
| B28 | shared starting adapter | 15/24 = 62.50% | 8/16 = 50.00% | 56.25% | NOT_RUN under the later candidate protocol | Best observed *greedy macro* among these saved evaluations; common start. |
| U_raw seed7 | independent original RL arm from B28 | 13/24 = 54.17% | 8/16 = 50.00% | 52.08% | NOT_RUN under the later candidate protocol | No observed improvement over B28. |
| P_calibrated seed13, Adam52 | independent original RL arm from B28 | 9/24 = 37.50% | 8/16 = 50.00% | 43.75% | 15/24 = 62.50% | Seven sequence generations invalid at the 256-token cap. |
| Adaptive continuation, Adam56 | same Adam52 lineage, four additional real updates | 11/24 = 45.83% | 7/16 = 43.75% | 44.79% | 15/24 = 62.50% | Greedy gain accompanies Dallas decline; no candidate improvement. |
| Uniform continuation, Adam60 | same Adam56 lineage, four additional real updates | **NOT_EVALUATED** | **NOT_EVALUATED** | **NOT_EVALUATED** | **NOT_EVALUATED** | Do not rank by update count. |

FLAIR/T1 recall on B28 is 12/12 and 3/12; on U_raw7 it is 7/12 and 6/12; on Adam52 it is 5/12 and 4/12; on Adam56 it is 6/12 and 5/12. Dallas same/different recall on B28 and U_raw7 is 0/8 and 8/8, respectively; on Adam52 it is 8/8 and 0/8; on Adam56 it is 6/8 and 1/8. The B28/U_raw7 class ordering comes from the frozen published confusion tables; these patterns caution against equating macro score with balanced visual discrimination. Adam52/56 each has seven invalid sequence generations; B28/U_raw7 have none in their saved C40 readouts. These are repeatedly examined development data.

The original 18 RL arms are all `TRAINING_COMPLETED` with independently checked final receipts. Across them: 2,286 completed groups, 362 accepted optimizer updates and 3,012,459 charged tokens. These are **training-cost totals, not accuracies**. Of those 18, only U_raw seed7 and P_calibrated seed13 currently have saved full C40 accuracy. The other 16 require native inference and remain `NOT_EVALUATED`; the three same-C-data SFT arms are separate controls, not RL arms. The live queue must finish naturally before any new heavyweight inference. Patient-cluster uncertainty for the full 18+3 matrix is therefore `NOT_AVAILABLE`; no confidence interval can be inferred from reward traces.

| Original RL method | Seed 7 updates / C40 | Seed 11 updates / C40 | Seed 13 updates / C40 | Three-seed charged tokens |
|---|---|---|---|---:|
| Sequence specialist | 42 / NOT_EVALUATED | 40 / NOT_EVALUATED | 12 / NOT_EVALUATED | 503,309 |
| Dallas specialist | 2 / NOT_EVALUATED | 3 / NOT_EVALUATED | 10 / NOT_EVALUATED | 501,753 |
| Proportional raw | 47 / NOT_EVALUATED | 8 / NOT_EVALUATED | 48 / NOT_EVALUATED | 501,629 |
| Uniform raw | 39 / 13/24 sequence, 8/16 Dallas | 5 / NOT_EVALUATED | 8 / NOT_EVALUATED | 501,452 |
| Proportional calibrated | 9 / NOT_EVALUATED | 7 / NOT_EVALUATED | 52 / 9/24 sequence, 8/16 Dallas | 502,475 |
| Uniform calibrated | 15 / NOT_EVALUATED | 8 / NOT_EVALUATED | 7 / NOT_EVALUATED | 501,841 |

Adam60's saved adapter and complete optimizer/RNG continuation have an independent CPU proof, but its accuracy is `NOT_EVALUATED`. An exact read-only evaluation scope has been prepared, with proof SHA-256 `9dcad56048539502c09b3f0307db2f4412405a045da0fe4912b9596d31eed6e0` and scope SHA-256 `860fde061f6dfe106878b1c5bfb50850ccae541460afbf4bb17c2e8fec060a6f`. An isolated Codex reviewer independently checked all 40 input-image hashes, six prior Adam56 evidence hashes, the no-update evaluator and queue lock, then recorded `PASS_WITH_LIMITATIONS` in an ignored private scope review (SHA-256 `dfbb01bb3e2632ea679655a606a46bbf47cfae34aec1cc0a9183667557c2913f`). The evaluator's own `verify_review` accepted that exact review. Model execution still awaits natural queue idle. The scope reuses byte-identical Adam56 output and binds 40 greedy questions plus candidate readout. Sealed final100 remains unopened.

## Evidence identities and limitations

- `results/COMPLETED_RL18_INDEPENDENT_AGGREGATE_REVIEW_V1.json` SHA-256 `d80c8c53e5e6130169847ba15a870cfaca6a4bfc6f93f9a63171ea39bf537a03`.
- `results/C40_B28_COMMON_START_BASELINE.json` SHA-256 `9e838d469736e373b44626baba8acba388e8dbe44bcaa0bed97018bfad561c4a`.
- `results/U_RAW_SEED7_FINAL_C40_PAIRED_EVALUATION.json` SHA-256 `09d65834f4e61db5ecdc846af61bf5c0af16639720d0b51887998601e1bce5dd`.
- `results/ADAPTIVE_ACCURACY_GATE_PARENT_CHILD_ACTUAL_OUTCOMES_V1.json` SHA-256 `6d5313e8da010ec4acc5392851faf18726132676de4b4d592e1e57b0a2c6ab14`.
- `results/UNIFORM_COMPLETION_ORIGINAL_RESUME_AND_UPDATED_ETA_V1.json` SHA-256 `6ef597062b92d7caf1a3a9dff9a47b2d0d88b5ab6295df3869ea6a90aae6d41e`.

No output or image from sealed final100 was read. This report contains only aggregate results and hashes, with no MRI pixels, patient identifiers, private generations or weights. Underlying Qwen3-VL-8B + B28 training uses local Vero-style/GSPO recipe transfer; it is not an inference result from the released Princeton Vero checkpoint.
