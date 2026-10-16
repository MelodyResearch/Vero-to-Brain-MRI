# Continuous RL driver: independent engineering review

**Current Step D verdict: PASS_WITH_LIMITATIONS for the corrected source/CPU safeguards and independently inspected synthetic checkpoint evidence. Real Qwen integration and MRI RL launch remain NOT_VERIFIED / NOT_APPROVED. See the dated Step D section below.**

**Initial verdict: FAIL for the affected input/state publication checks.** Positive fixture evidence remains valid within its scope. Native integration and every MRI RL launch are **NOT_VERIFIED / NOT_APPROVED**. Reviewer: isolated Codex C/data agent, independent of the RL implementation author; no claim of distinct-model-family review.

The reviewer read the actual driver, native adapter, checkpoint serializer, tests and `research/RL_CONTINUITY_MINIMAL_CORRECTION.md` before receiving an implementation diagnosis. This was source/CPU work only, with MLX, MLX-VLM, Torch and Transformers imports blocked. No model or native MLX execution occurred.

## Inspected source and executed checks

| File | SHA256 at initial counterexample execution |
|---|---|
| `src/mri_rl/continuous_rl.py` | `925ded371dcde4f8909da35e45f6ccf1ed456fab3c15aa66239091e76ac004a8` |
| `src/mri_rl/mlx_continuous_gspo.py` | `2946128c6bef1307303316c1009ea43d8ebb1c77e5d6617eaaa65ab3787aaddf` |
| `src/mri_rl/optimizer_checkpoint.py` | `572535defbc3191cc29954f61bf0124cd1e78db4b4996d7aa67fed3382d1766d` |

The associated driver, native-boundary and optimizer-checkpoint tests produced **24 passed, one skipped** in0.22s. The skipped native fixture was not executed. Source changed during early inspection; the hashes above identify the actual version used for the decisive counterexamples, after the author's completed-receipt hardening had appeared.

## Confirmed counterexamples

The synthetic NumPy backend exercises the real continuous driver and real serialization, without MRI data. Every case below reached `BOUNDED_STAGE_COMPLETE`, one completed group and one optimizer update, when rejection was required.

| ID | Reproducible failure | Scientific/engineering consequence | Minimal correction |
|---|---|---|---|
| RLC-1 / P1 | Replace the image file's bytes inside the group authorization callback, after `_binding` hashes the schedule. The driver executes and publishes the group without rechecking the expected image hash. `_grades` does not verify the raw image identity either. | Approved data identity can diverge from actual rollout pixels, invalidating a claimed fixed-input experiment. No real MRI drift was observed; this is a tested missing guard. | Rehash the selected image at the authorized group boundary and verify the raw rollout image hash against the frozen binding. Add a failing-then-passing regression. Preserve exact prompt/config/model checks in the eventual native launcher. |
| RLC-2 / P1 | A backend update reports finite loss, gradient and old/current evidence, but the resulting adapter contains NaN. The driver checks changed bytes, not finite parameter values, and commits the adapter. The native backend likewise lacks an explicit post-update finite-parameter check. | A changed checkpoint can be unusable despite apparently valid gradient telemetry. Finite gradient norm alone does not prove a finite update. | Require backend-native finite trainable-parameter validation after the update and before publication, plus a driver contract that refuses absent/failed resulting-state validation. Check native fixture overflow/nonfinite outputs independently. |
| RLC-3 / P1 | Insert NaN into an Adam first moment after a nominal finite update. The boundary is published. The serializer validates scalar values and state inventory but not tensor finiteness. | Resume may later reject the boundary or propagate invalid state; complete-state hashes alone do not establish valid numerical state. | Validate complete finite optimizer tensors and nonnegative second moments before committing, without changing Adam recurrence or precision. Add regression for nonfinite and invalid second moments. |

Exact reproducer and immutable initial findings are private under `local_data/ds007045_bounded/continuous_rl_independent_review_v1/`. These are controlled synthetic failure injections, not observations of NaN or changed MRI bytes in an actual run.

## Native execution concerns requiring correction or explicit gate

The inspected native constructor calls eager `load(model_path)` before4-bit adaptation. It does not apply the current paced memory/cache limits, lazy-load-before-quantization sequence or language-layer checkpointing used by the reviewed SFT implementation. A generic pre-load callback does not prevent the subsequent eager weight materialization. This source is not suitable for inheriting the prior bounded Mac approval; the minimal integration should reuse the proven loading/resource discipline and receive a new scoped review.

The native rollout checks resources before each response and before active-group gradients, but lacks a final post-generation boundary for the last response, especially when the group is skipped. Completed raw evidence must be preserved before a post-generation HOLD/STOP, with no update performed after a failed gate. The current code can discard a partially completed group if generation raises before returning; that group should remain uncommitted, but retaining incremental raw rollout evidence would improve diagnosis.

The driver requires an injected independent authorizer but does not implement a real MRI execution-contract launcher. That future caller must bind base/processor/template/backend-source identities, initial adapter, training-only admission, resource policy and finite group budget. Merely supplying a callable is not independent approval. These are explicit integration limitations; the CPU fixture callback is intentionally synthetic.

## Checks with positive evidence

- One backend owns one optimizer across the live stage; effective update count and attempted-group cursor are separate.
- Continuous versus interrupted/restored NumPy runs agree on weights, Adam moments and RNG, including an inactive middle group. Skips do not call the optimizer and preserve weight/moment state in the tested path.
- Resume restores optimizer and RNG after backend initialization, then advances from the saved attempted-group cursor.
- Saved adapter and optimizer payloads are digest-bound; state layout preserves array shape/dtype; complete moment inventory is checked against the trainable signature.
- Completed-group bundles are staged in private temporary directories, then renamed. An injected interrupted save leaves the earlier boundary resumable and the partial directory uncommitted. A writer lock prevents simultaneous runs; completed schedules do not load a backend or repeat updates.
- Current source validates boundary schema, boolean update flag, sequential cursor, preceding receipt, schedule record and required rollout/state evidence. Configuration/schedule drift is rejected in tested cases.
- The strict reward remains unchanged and recomputed from raw text. Diagnostic active-group filtering remains an explicit departure from unfiltered Vero sampling. Advantages use the existing declared convention.
- Old/current response log probabilities are compared before the sole update. The source retains sequence-ratio GSPO-token algebra; this source inspection does not establish native autograd or clipping parity. A single initially on-policy update is not a test of clipping under a changed policy.
- The native loss closure references the same model object used by `mlx.nn.value_and_grad`; installed library source updates that object before invoking the loss. Capturing the instance is not by itself evidence that gradient flow is broken. Actual native gradient behavior remains unverified.

## Additional durability consideration

The adapter, rollout JSON and optimizer bundle have explicit fsync calls. The before/after optimizer fingerprint payloads are saved by the backend and included in the required committed inventory, but are not explicitly fsynced by the driver before the parent directory publication. Fsync all included payload files before publishing if the completed-boundary guarantee is intended to cover sudden process/system interruption. This is a durability concern; no power-failure test was attempted.

## Required next review cycle

The implementation author has received the three reproductions and native-integration concerns. Correct only the affected boundaries, add regressions, preserve historical optimizer-reset semantics, and supply final source hashes for independent Step D. Passing CPU tests can justify a separately reviewed tiny native fixture; it cannot approve a real MRI RL run. Do not issue a full-driver PASS before checking corrected image/state publication and native resource integration.

Mission alignment: this review removes specific data/state confounds before Vero specialist/proportional/uniform comparisons; it changes no MRI labels, rewards or scientific objective, uses finite CPU fixtures, preserves failed evidence and leaves the independent C task-admission work separate.


## Step D — independent correction verification, 2026-10-16

**Verdict: PASS_WITH_LIMITATIONS.** Evidence tier: **REPRODUCED** for the three original counterexamples and CPU tests; **REPRODUCED from saved artifacts** for synthetic native checkpoint equivalence. This supersedes the initial FAIL only for the corrected guards listed below. It does not approve an MRI RL launch, establish full Vero recipe fidelity, or validate MRI understanding. Reviewer remains an isolated Codex agent; `CROSS_MODEL_REVIEW=NOT_VERIFIED`.

| Corrected source | Independently checked SHA256 |
|---|---|
| `src/mri_rl/continuous_rl.py` | `401f6de30e2c230e7f75f6bc2f7787d072fa6a87a0358f03a95a50da085a6497` |
| `src/mri_rl/mlx_continuous_gspo.py` | `32ecdf37a897e38becac029e1de2090b5c9647f36786ee53a1c95df29c29df14` |
| `src/mri_rl/optimizer_checkpoint.py` | `572535defbc3191cc29954f61bf0124cd1e78db4b4996d7aa67fed3382d1766d` |

### Replayed failures and regression evidence

The reviewer reran the original private reproducer against these versions, changing only its output filename so the original failure evidence stays immutable. RLC-1 now raises `Frozen image changed before rollout`; RLC-2 and RLC-3 raise `Invalid resulting model/optimizer state`. Thus all three previously accepted injected failures now reject. These are synthetic injections, not newly discovered invalid MRI runs.

A fresh run of `test_continuous_rl_driver.py`, `test_mlx_continuous_gspo.py`, `test_optimizer_checkpoint.py` and `test_continuous_rl_native.py` produced **33 passed, 2 skipped in 0.28 seconds**, with MLX, MLX-VLM, Torch and Transformers imports explicitly blocked and the native opt-in variable removed. The two native tests were not executed by this reviewer. The suite checks negative second moments, post-generation HOLD retaining raw evidence without an update, invalid native state through NumPy array operations, continuity, skipped groups, corrupted boundaries, interrupted publication and writer exclusion.

Source inspection confirms that the driver rehashes the frozen image after group authorization, verifies the rollout image identity, checks it again after the post-generation authorization, and requires finite trainable parameters/optimizer tensors with nonnegative second moments before publication. All included payload files receive an fsync before the completed directory is renamed. Complete raw rollout evidence is written before the post-generation resource gate; incremental native responses are also saved. These checks address the reported publication and final-generation boundary defects. No sudden-power-loss test was run.

### Native synthetic evidence inspected independently

The implementation author executed the frozen revision's exact 3×3 synthetic native test in a separately reserved CPU window. The reviewer did **not** rerun MLX. Instead, the reviewer checked the actual saved artifacts in `continuous_rl_native_final_v1`, verified all source hashes in its receipt, all **42 payload hashes across 8 completed boundaries**, parent receipt links, attempt/update counters, and resulting-state validation records. A small independent NumPy/stdlib safetensors reader compared actual tensor contents: the final continuous and interrupted/restored adapter, Adam state and MLX RNG arrays are exactly equal and finite. Both runs have 3 attempted groups and 2 optimizer updates. All three corresponding rollout records, including both RNG draws, are identical; the inactive middle boundary preserves adapter bytes.

Author execution receipt SHA256: `b8e08a567bc5d3fcef36853560d60a2cfd9bc1408c2e99ed8e68e0a226b45e9f`. Independent saved-artifact checks and counterexample replay are retained under `continuous_rl_independent_review_v1` without overwriting the initial failures. This is evidence for the actual tiny native gradient/Adam/resume integration, with execution attributed to the author and checkpoint comparison attributed to the independent reviewer.

The tiny subclass deliberately bypasses the full native constructor and generates synthetic responses. It therefore does **not** test real Qwen loading, image preprocessing, chat-template/tokenizer identity, native VLM generation, generation/likelihood alignment, full-model gradient memory, or resource behavior. The source now uses lazy loading, quantization before parameter materialization, paced memory/cache limits and language-layer checkpointing; these changes have source evidence and denied-constructor CPU coverage, but their full Qwen execution remains **NOT_VERIFIED** in this review.

### Remaining scope limits and next gate

- The eventual launcher must bind exact base model, processor/template, backend source, adapter, prompt/reward contract, admitted training data, finite attempted-group budget and independent approval. An injected callable alone does not provide that authorization.
- The diagnostic retains persistent historical Adam at learning rate 1e-5, active-group filtering and one update per freshly generated group. Upstream optimizer/minibatch/sampler/open-ended-retention fidelity remains separate. The synthetic test does not establish clipping behavior after policy drift or an MRI cross-task transfer result.
- Two qualifying MRI updates and reviewed state transition are still **NOT_RUN** under this corrected driver. Missing qualifying groups must not trigger unbounded replacement sampling or reward relaxation.
- The new permission for scoped read-only battery inference does not authorize training on battery. This review changes no resource policy and grants no training launch approval.

### Mission Alignment Check

1. **Main question:** remove a confirmed optimizer-continuity and state-integrity confound before Vero-to-MRI specialist/proportional/uniform comparisons.
2. **Falsifiable check:** the original invalid image/NaN cases must reject, and resumed synthetic weights/moments/RNG must equal uninterrupted execution; both were checked in the scopes above.
3. **Isolation:** corrections concern input identity, state publication and resource boundaries; MRI labels, strict reward and historical runs remain unchanged.
4. **Provenance:** exact source hashes, immutable initial failures, new replay outputs and actual native synthetic artifacts identify the evidence.
5. **Alternative explanations:** checkpoint correctness does not exclude format learning, class priors, image shortcuts or task inadmissibility; those require the separate B/C studies.
6. **Resource/decision:** bounded CPU verification only in this independent review; no model imports or MRI training. GO only to a separately scoped integration review, not to an unapproved training run.
