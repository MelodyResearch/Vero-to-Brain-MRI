# SHARED_ROLLOUT_BATCHED_ONE_STEP_GSPO_MECHANISM_CONTROL — implementation note

Status: `IMPLEMENTED_CPU_NOT_NATIVE_QUALIFIED`. Nothing here ran a model, generated a response, or performed an optimizer update. This note does not modify `CURRENT_RESEARCH_PLAN.md` or its lock; it is a new execution subtype inside the fixed plan, distinct from a future `SEQUENTIAL_ON_POLICY_GSPO` (which updates after every K4).

## Design

One existing TRAIN same-person z045 F/T pair is rolled out once from preserved Adam60 (F4 then T4, 8 responses, ≤256 tokens each). The immutable bundle is consumed by both arms:

| arm | groups | gate |
|---|---|---|
| A | F0–F3 ; T0–T3 | inherited Adam60 valid-format-subset gate (same in both arms) |
| B | F0F1T0T1 ; F2F3T2T3 | same |

Each gate-admitted response has loss weight exactly 1/8 (mean of the two group losses). Each arm restores the identical full parent state (all 1942 parameters, persistent Adam moments/step, Python+MLX RNG), computes per-response gradients of the **unchanged native GSPO `loss_fn` AST** under the response's own prompt/image/positions, sums them with weight 1/8, and performs **at most one** `optimizer.update`. No CE or margin auxiliary; reward/calibration, ratio/clip, LR, LoRA and F32 policy are unchanged. A no-admitted-group or zero-gradient arm makes 0 updates (negative evidence); there is no resample and no momentum-only step.

## Files (all new)

| file | role |
|---|---|
| `src/mri_rl/shared_rollout_gspo_core_v1.py` | pure CPU: immutable bundle (identity, token/logp/EOS/reward-receipt binding), arm plans (via `paired_prompt_groups_v1.make_plan`), derived budgets, strict atomic counters, weighted gradient sum, update decision, `.002` parity, fail-closed authorization |
| `src/mri_rl/shared_rollout_gspo_driver_v1.py` | crash-consistent orchestration; reuses `paired_z045_continuation_driver_v1._publish/inspect_boundaries` |
| `src/mri_rl/shared_rollout_gspo_backend_v1.py` | lazy-runtime mixed-context F32 rollout/scorer/gradient backend over the pinned `F32NativeGSPOBackend` helpers; reuses `ClassMarginAuxBackend.transaction_snapshot/validate/restore` (auxiliary loss never invoked) |
| `src/mri_rl/shared_rollout_gspo_entry_v1.py`, `scripts/run_shared_rollout_gspo_v1.py` | CPU `prepare`; gated `qualify`/`train` |
| `src/mri_rl/shared_rollout_gspo_eval_v1.py` | matched F32 parent/A/B evaluation entry reusing `paired_z045_matched_f32_eval_v1.run_checkpoint` |
| `tests/test_shared_rollout_gspo_v1.py`, `tests/test_shared_rollout_gspo_backend_v1.py` | CPU tests (fake numpy backend; dual-number mock-MLX for the gradient path) |

## Frozen call budget (derived, not tuned)

Rollout once: ≤8 responses / 2048 returned tokens / 2056 sampler draws / 2080 generation LM calls / 8 generation vision embeddings. Own-old + rollout parity: 8 forwards once on the shared parent. Gradient: ≤8 backward (each with its own fresh forward) per arm only if the gate admits that group, ≤16 total. Scorer vision/LM calls 24; teacher-forced positions 24·(213+255)=11232 (prompt 213 sequence cost, response ≤256). Optimizer updates: train ≤2 (≤1/arm), qualify 0. One model, one image graph, 1800 s operation-start deadline. No retry, no transfer of unused budget, no generation per arm. Counters are persisted on every charge and restored on resume. These are API-endpoint counts, not kernel counts; AR lookahead and checkpoint recomputation are additional work.

## Crash consistency

- Rollout: durable reservation → generation → bundle file + commit marker. A reservation without a committed bundle stops every later run (`INCOMPLETE_ROLLOUT`), never regenerates.
- Own-old logps are committed once and reused on resume (never recharged).
- Per arm: reservation → pending staging → existing atomic `boundary_0000`/`boundary_0001` publication → once-only `arm_X_COMMITTED.json`. An uncommitted attempt stops (`INCOMPLETE_ARM_X`), with the model/Adam/RNG rolled back to the parent in-process. A rename that completed without its marker is recorded and never replayed.
- After each arm, and again at the end, the full parent identity (1942 parameter fingerprints, Adam, RNG) is re-verified against the original snapshot.

## Admission (fail-closed)

`qualify` and `train` need an exact launch scope bound to the current source closure and input hash, a fresh human authorization whose `action` is specific to the mode (`shared_rollout_gspo_readonly_qualification` vs `shared_rollout_gspo_one_step_train`), and an independent source/input/scope review, all digest-bound on the command line, consumed once. `train` additionally requires an independently reviewed actual same-source read-only qualification proof. A local `approved`/`authorized` flag in any scope is rejected. All gates precede any MLX import. None of these artifacts exist; this work did not create them.

## Revision 2 (response to preliminary independent review)

- Qualification proof is built only from run-written artifacts (`run_identity_PRIVATE.json` with the actual source/input, per-arm gradient evidence, and the `release_SAFE.json` written by `run_with_release`). Without an admitted group, nonzero finite gradient and backward coverage in BOTH arms it is `PARTIAL_...` with `blockers`, and `authorize_launch('train')` denies it.
- Post-generation RNG R1 is persisted (`rng_R1_PRIVATE.json` + safetensors via `pack_tree`/`save_arrays`); both arms restart from the exact parent R0, recorded in each arm receipt.
- Qualify→train adoption (`adopt=`, only after the fresh train authorization): the reviewed qualification's committed bundle and own-old are copied, bound to the proof (bundle/old/run-identity/release hashes) and to the same source/input/parent; no generation, no rescoring. New counters exclude inherited work (`adopted_limits`); inherited counters are reported separately. An interrupted adoption stops (`INCOMPLETE_ADOPTION`).
- The `boundary_0001` rename is the commit: an exception after it is reported committed, the marker is reconstructed on resume, leftover `.pending-*` dirs are ignored, nothing is replayed.
- Consumer entry revalidates response ID/modality/index, bundle parent receipt/adapter against the external anchor, and context rows/images/pair/reward; z045 rows must carry certified `paired_z045_data_v1` lineage (no `None==None`).
- Evaluator: `eval_budget(1..3)` (parent-only when both arms are aliases), executable CLI `scripts/run_shared_rollout_gspo_eval_v1.py`, admission requires the NEW shared qualification (effective, reviewed, same source) instead of the old F32 status, source closure includes `paired_z045_evaluation_v1.py`, `answer_likelihood.py`, `vero_mri_reward.py` and the shared files.

## Revision 3

- Pre-restore state audit: after own-old scoring, each arm's backward, the qualification arm, and every optimizer update, the driver compares full `optimizer_digest`/`rng_digest` against the parent BEFORE `restore_parent` (the inherited `transaction_validate` is parameter-only and restoration would hide a fault). Rollout must leave Adam unchanged and must consume RNG (R1 != R0); an updated arm may change Adam but never RNG and must actually change the moments.
- `input_sha256` now covers immutable scientific/runtime/parent inputs only (`immutable_spec_binding` excludes `output`, `resume`, `adopt_qualification_run`). Per-invocation metadata is bound in `prepared['invocation']` and therefore in the launch-scope digest that humans/reviewers sign; source, image, parent and calibration hashes are untouched and any scientific change alters the hash.
- Adoption and the qualification proof share `core.qualification_blockers(run)`, which re-derives completion, committed arm evidence and effective backward coverage from the qualification run's own files; own-old shapes and `.002` parity are recomputed on CPU for adopted and resumed files. Booleans in a proof or prepared dict never substitute.

## Limitations (not verified)

- Mixed-context F32 rollout/scorer/backward are **not** native-measured. The pinned F32 helpers are reused, but `_generate_context` is a new count-aware copy of the pinned rollout loop; only CPU doubles exercise it.
- `prepare`/`qualification-proof`/`_authorized` need the frozen real inputs and were exercised only through unit-level pieces (`launch_scope`, `authorize_launch`, `qualification_proof`), not end to end.
- Typed bundles do not authenticate real outputs; native qualification plus independent review do.
- The arm boundary uses the existing receipt format with `completed_groups=1` representing the shared two-K4 batch (`attempted_K4_groups=2` recorded); absolute counters therefore read 130 groups.
- Eval: `prepare()` delegates to `matched.prepare` (needs the frozen evaluation spec); a non-updated arm is aliased to the parent. Sealed new16 outputs are never read; no accuracy claim.
