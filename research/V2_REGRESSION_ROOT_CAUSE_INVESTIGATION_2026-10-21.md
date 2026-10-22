# Accuracy Pilot V2 negative result: root-cause investigation (2026-10-21)

Author: Claude Code (claude-opus-5-5), primary scientific author and author of the phase-1 analysis module, CLI and tests.
Scope: CPU-only, read-only interpretation of saved V2 evidence. No model, native code, gradient, optimizer, new generation, final100 or sealed new16 output was run for this investigation.
Public file: aggregate-only. It has no case/pair indexes, identifiers, texts, token arrays or private paths. Detailed evidence stays PRIVATE.

Evidence tiers: `OBSERVED` (read from saved artifacts), `REPRODUCED` (recomputed independently by root and/or an isolated Codex reviewer), `DERIVED` (algebra on saved tensors), `INFERRED`, `PROPOSED`, `NOT_VERIFIED`.

## 0. Provenance and review status

Codex executed the analysis (`cpu_attempt2` of `scripts/analyze_v2_regression_v1.py`). I did not.
- Source pins: module `11b0ce6292ce57e023f8315152ea89c820fb29f0641529407d3939f11b5d6a8f`, CLI `e16080dba8f9c4cfaa9d395297a0db7511f92290c566fa61181c9b40eadef6ce`, tests `9e68d151d5dda0e595156c47a7e63d874185f57bdcfe8502d95ed28dd6e10ae8`.
- Run facts (reported by root): exit 0, 5.772637624992058 s, peak child RSS 1,038,712,832 bytes. Source and inputs were unchanged, all 13 source pins MATCHed, 7 stage anchors and 77 journal commits were verified, and no heavy imports were loaded.
- Targeted pytest (reported by root): 35 PASS in 0.52 s, exit 0 in 0.765 s, source unchanged.
- Aggregate result: `results/V2_REGRESSION_DIAGNOSIS_ACTUAL_CPU_V1.json`, SHA-256 `fcd3028fb0226a21db7cb54ef20bec50a00047b4dc416781a68e5e58edef8bfa`.
- Isolated Codex numeric/source/privacy review: `results/V2_REGRESSION_DIAGNOSIS_INDEPENDENT_REVIEW_V1.json`.
- I first read the numbers below from the earlier `cpu_attempt1` output. In this documentation-only pass I did not re-read the attempt-2 JSON, so any discrepancy against `fcd3028…` falls to the final report review.
- Workflow: Claude wrote the code, and Codex independently reviewed and executed it. That is a genuine cross-family check of the code and numbers.
- The final independent interpretation verdict is recorded separately in `results/V2_REGRESSION_DIAGNOSIS_INDEPENDENT_REVIEW_V1.json`. This report does not self-assign PASS.

## 1. Primary scientific judgment

**The root cause is INCONCLUSIVE.** The two regressions are real. Their population effect and their mechanism are not established.

1. **Two local, deterministic regressions were observed** (`OBSERVED`).
   - Each arm lost 2 of the 24 frozen DEV cases relative to Parent (one FLAIR, one T1) and gained none.
   - Arms A and B make identical strict predictions on 24/24 cases.
   - These are actual changes in the greedy outputs of the updated models. They are not noise in the measurement.
   - Whether this reflects a population or generalization effect is unresolved: there are 24 images from 4 repeatedly explored patients, one seed, and two updates per arm.
   - An unclustered case-level sign test on the 2 discordant cases gives p = 0.5. It is **descriptive only**: cases are clustered within 4 patients, and the test is not equivalence or no-effect evidence. I do not use it to rank causes, and I make no zero-effect claim.
2. **Both arms moved mostly along a shared direction, aligned with the inherited Adam52 first moment** (`REPRODUCED`/`DERIVED`, joint quantities).
   - The fresh joint gradients differ substantially:
     - cos(gA0, gB0) = 0.68, even though the segment-0 Dallas units are identical;
     - cos(gA1, gB1) = 0.19.
   - The parameter moves are nonetheless highly aligned:

     | Step | cos(ΔA, ΔB) |
     |---|---|
     | 53 | 0.990 |
     | 54 | 0.897 |
     | total | 0.968 |

     The effective LoRA-weight cosine is 0.954.
   - The final A−B parameter difference is ≈25% of either arm's total delta norm.
   - Scale context only: the effective LoRA change is ≈1.7% of ‖scale·A_parent·B_parent‖. That is the **parent LoRA contribution**, not the full base+LoRA weight. I draw no inference from this ratio.
   - Momentum-only (g = 0) baseline cosines with the actual step:

     | Step | Cosine |
     |---|---|
     | A0 | 0.992 |
     | A1 | 0.910 |
     | B0 | 0.983 |
     | B1 | 0.943 |

   - Two-step saved-denominator projection shares onto the total delta (signed projections, not norm shares, not causal damage fractions):

     | Arm | Parent history m52 | Segment-0 gradient | Segment-1 gradient | Residual / ‖Δ‖ |
     |---|---|---|---|---|
     | A | 0.957 | 0.002 | 0.041 | 1.1e-5 |
     | B | 0.946 | 0.031 | 0.023 | 1.1e-5 |

     The split holds to float32 rounding, not bitwise.
   - Interpretation:
     - The common Adam-aligned displacement is **consistent with** the arms reaching similar predictions. It does not explain or prove the two particular flips.
     - A small fresh-gradient projection share does **not** bound that gradient's influence on accuracy.
3. **The changed labels follow early greedy-trajectory divergence** (`OBSERVED`).
   - In the two changed cases the A/B greedy path first departs from Parent's at zero-based positions 2 and 4. Those are the 3rd and 5th response tokens: within the first 5, and before the `\boxed{}` decision.
   - Parent's own-prefix selected label-token probability in those cases is > 0.9998. That is a high selected-token probability along Parent's own prefix. It is not a class margin, and it is not confidence in the full classifier.
   - Saved evidence contains no top-2 margin at the divergence position.
   - The earliest differing token is an observation. It does not prove a specific causal chain.
   - This rejects my earlier provisional prediction that the changed cases would diverge at or near the decision token with low selected-token probability.

**Joint-update macro table** (`REPRODUCED`). These are JOINT MRI+Dallas quantities, not MRI class margins.

| Update | Admitted groups (MRI / Dallas) | Joint ‖g‖ | g·Δ (actual) |
|---|---|---|---|
| A0 | 2 / 3 | 0.27506973305744936 | −0.0002526768782361217 |
| A1 | 0 / 4 | 0.8034194127748593 | −0.00494433138955216 |
| B0 | 4 / 3 | 0.36668142558626504 | −0.0007683723238761073 |
| B1 | 4 / 3 | 0.6208485797552189 | −0.0029642378482333644 |

What the evidence rules out or limits (`REPRODUCED`):
- **A global ascent/sign or Adam-formula bug: REJECTED.**
  - g·Δ < 0 on all 4 updates, which is first-order descent of the joint surrogate.
  - Every Adam step recomputes with m and θ bitwise equal on all 10,911,744 elements per step. v differs only by ~1e-38 subnormals, with no tolerance violations.
- **Reward/advantage misalignment: a limited scalar counterexample only. Its primary causal role is NOT_VERIFIED.**
  - The scalar-only raw-reward counterfactual holds all 32 plans, admission and weights fixed.
  - Result: 2 of 92 admitted plan occurrences change advantage sign (2 of 128 plan items overall).
  - Both are B1 MRI answers that are wrong but format-valid (17 and 29 tokens), moving +0.0178 → −0.210.
  - The 37 raw-correct (stay positive) and 55 raw-wrong (negative under the counterfactual) items are **admitted plan occurrences**. Shared Common items are counted in both arms. They are not unique generated responses.
  - Totals: 128 plan items = 32 K4 group instances × 4 items, across four macro-update plans, versus 96 unique TRAIN generated responses.
  - The actual wrong-positive share is 0.28% of B1's absolute **scalar advantage** mass. It is not per-response gradient influence, which was not saved.
  - This gives no basis for a reward change.
- **Changed global predicted-class counts as the sole explanation: ruled out.**
  - Parent and both arms predict 14 FLAIR and 10 T1.
  - This does not rule out more general text-side or image-conditioned class-prior changes (H5).

## 2. Hypotheses (unranked by p-value; explicit counterevidence)

| ID | Hypothesis | Supporting evidence | Counterevidence / gap | Verdict |
|---|---|---|---|---|
| H1 | A common inherited-momentum displacement produces a shared early-token greedy-path change | Momentum shares ≈0.95; cos(ΔA,ΔB)=0.968 despite cos(gA1,gB1)=0.19; identical A/B predictions | Alignment ≠ cause of the specific flips; no A53/B53 DEV; no early-token margins | Consistent with the evidence; causal link NOT_VERIFIED |
| H2 | MRI–Dallas gradient interference | A1's fresh joint gradient is Dallas-only by the gate (0 MRI groups admitted) | Mixed historical moments remain; B (4 MRI groups in B1) reaches the same predictions; no per-task vectors saved | INCONCLUSIVE |
| H3 | MRI signal starvation / between-image-only contrast (admitted MRI groups / all-response within-image / format-valid within-image) | A0: 2/2/2 · A1: 0/0/0 · B0: 4/3/2 · B1: 4/2/0; T1 images were rarely sampled correctly in TRAIN | Between-image contrast is the intended Paired-Prompt mechanism and yields a nonzero MRI gradient | Plausible contributor; not shown to cause the losses |
| H4 | Reward/advantage misalignment | 2 wrong-positive scalar items in B1 | Scalar advantage only, not gradient influence; A has none and the same outcome | Limited counterexample; primary role NOT_VERIFIED |
| H5 | Text-side or image-conditioned class-prior change (not a global count shift) | One loss per modality; early pre-decision divergence | Global counts unchanged, which excludes only a changed global count as the sole cause; no conditional measurement exists | UNVERIFIED |
| H6 | Overfitting to the 4 training patients | — | Not testable from saved evidence; segments 0/1 use different patients (not a before/after) | INCONCLUSIVE |
| H7 | Frozen-vision limitation | — | No vision-side measurement exists | NOT_VERIFIED |
| — | Ascent/sign or optimizer-implementation bug | — | g·Δ < 0 on all 4; m/θ bitwise reproduced | REJECTED (global sign/formula) |

## 3. One conditional intervention (PROPOSED; supported, not causally proved; no training authority)

Decision: **MORE_READONLY_EVIDENCE_REQUIRED**, not GO. The cause is inconclusive, and section 4 comes first.

**Single factor: optimizer-state ablation.** Reset the Adam first and second moments to zero at the start of the RL update. Everything else stays unchanged:
- `bias_correction=False` (as in the pilot);
- LR, betas, eps and the step counter;
- the GSPO objective, reward, prompts, grouping, LoRA configuration and RNG.

Constraints:
- The full historical Adam52 state stays preserved, read-only, on disk. Any reset state is a separately versioned experimental copy.
- This is an optimizer-state ablation. It is **not** an exact Adam continuation, not an official-Vero-parity claim, and not a new main Adam optimization project. I make no claim about the public Vero optimizer convention.
- Why this factor: H1 is the hypothesis with the most saved support, and the inherited state is the variable it implicates.
- Risks:
  - Without bias correction, the zero-moment update magnitude differs from the inherited-state update magnitude. Under the fixed LR, the effective step size changes along with the state, so a result will not isolate "history direction" from "step scale".
  - Any useful second-moment curvature information is also lost.

Falsifier and matched design (bounded, one step):
- **Candidate:** ONE update from the same Parent on the same Common32 segment-0 rollout, with the Paired-Prompt grouping B fixed in advance and only the moments reset.
- **Control:** the existing V2 B53 checkpoint from the same Paired-Prompt grouping, used after a checkpoint-identity verification. There is no duplicate historical control update.
- **Manipulation check, not benefit evidence:** a reduced history projection after the reset is tautological.
- **Outcome measures:** matched strict DEV accuracy, FLAIR and T1 recall, paired-image correctness, and Dallas retention.
- **Success bar:** the candidate must **beat Parent**, not merely do less badly than 16/24.
- **Scope of the result:** DEV24 is reused here as an engineering check only, with no claim of future generalization.
- **No second step by default.** A second step would need that arm's own new on-policy rollout. Existing A1/B1 responses cannot be reused after a different first update.
- **Not ready:**
  - no executable native scope or source/input/budget-bound independent review exists;
  - no fresh human authorization exists;
  - no final100/new16 is involved.

## 4. Next read-only measurement: `PROPOSED_NOT_EXECUTABLE`

This is a design, not a permission request. It needs an exact source/input/budget-bound independent review and fresh explicit human authorization before any operation.

Design:
- **Models:** 2 serial loads, A53 and B53, with one heavyweight MLX model at a time under the existing resource guards.
- **TRAIN rescoring:** 64 teacher-forced F32 scorings of the same saved Common32 TRAIN response tokens (32 × 2 models).
  - The expected 23,282 positions are not yet externally pinned. Contexts must be derived and hashed before any run.
  - Report:
    - full-response mean logp change;
    - selected answer-token change under fixed prefixes;
    - correctness and format;
    - unclipped ratio and clipped surrogate.
  - This is post-update fixed-response rescoring, not old = current parity.
- **DEV:** at most 8 greedy generations in total (the 2 regressed cases plus their same-patient opposite-modality partners, per model), at most 2048 returned tokens in total.
- **Budget:** 0 backward passes, 0 updates, no new TRAIN sampling, no final100/new16. The frozen plan is unchanged.
- **Can tell us:**
  - whether the two errors are already present at step 53;
  - whether the TRAIN objective moves as intended while DEV diverges.
- **Cannot tell us:** full Adam53 DEV24 accuracy, MRI–Dallas task vectors, class margins, or causation.

## 5. Code changes in this pass (Claude-authored; executed by Codex)

Files: `src/mri_rl/v2_regression_diagnosis_v1.py`, `tests/test_v2_regression_diagnosis_v1.py`. Both were executed by Codex: attempt 2 exited 0, and 35 targeted tests PASSed (section 0). These are CPU software tests, not MRI accuracy evidence.

1. **`Check.add` false PASS fixed.**
   - An element passes only if prediction, saved value and bound are all finite, the bound is ≥ 0, and |pred − saved| ≤ bound.
   - Nonfinite counts are reported.
   - Negative saved v is rejected before `sqrt`.
   - The two-step split is labelled float32-rounding, with a residual tolerance flag.
2. **Public output is aggregate-only.** Per-case records, changed-case identities and ranks, and per-group lists were removed. `public_schema_check` is a structural forbidden-key test, not an unlinkability certificate. `pair` was renamed `unit_pair`.
3. **External authentication anchors.**
   - `verify_stage_anchors` binds the 7 stage terminals and exit records to the committed public result.
   - `verify_journal_commit` re-checks the `safe_micro_journal_v1` fields, the self-hash, the reservation and every file-map entry.
   - `link_terminals` ties the A/B segment receipts and DEV rows to the anchored terminals.
   - A coherent local re-hash is rejected by the committed anchor (tested).
4. **Within-image contrast flags** (`all_response_within`, `format_valid_within`), with admitted counts per segment.
5. **Scalar raw-reward counterfactual.** It reports plan occurrences vs unique responses, sign changes by kind, and the wrong-positive scalar mass share. It is labelled not-gradient and not-accuracy.

## 6. Remaining limitations

- No per-task, per-unit or per-response gradient vectors were saved. MRI–Dallas cosines and MRI-only projections are NOT_VERIFIED, except that A1 is Dallas-only by the gate.
- No A53/B53 DEV outputs exist, so the timing of the regression is NOT_VERIFIED.
- No class margins and no top-2 margins at the divergence position exist.
- DEV24 comes from 4 repeatedly explored patients, with one seed and two updates per arm. It supports no generalization claim.
- Projection shares are algebraic, on joint quantities. They are not causal attributions.
- `CROSS_MODEL_REVIEW` of the code and numbers: Claude implemented and Codex independently reviewed/recomputed them. The separate independent-review JSON records the final interpretation verdict and its limitations.
