# Optimizer continuity audit — 2026-10-15

Status: **IMPLEMENTED / TESTED_CPU** for a new, separate checkpoint utility. **NOT_VERIFIED** for integrated VLM training, Metal continuation, memory peak with a real model, and official Vero numerical parity. This report is an implementation audit by the isolated Codex RL specialist, not independent permission for a training launch. Existing trainers, contracts, checkpoint files and historical receipts were not changed by this work.

## Verified defect and its consequence

`src/mri_rl/mlx_gspo_smoke.py` constructs `optim.Adam(learning_rate=1e-5)` immediately before its single update. `mlx_multiarm_train.py` starts a fresh child process per group and explicitly records `optimizer_reset_each_group=True` / `exact_optimizer_resume=False`. Reloading only the latest adapter continues the weights but loses optimizer history.

Installed MLX is 0.32.4. In `mlx/optimizers/optimizers.py`, `Optimizer.__init__` creates a uint64 `step`; `apply_gradients` increments it; `state` assignment marks initialization pending; initialization preserves already populated state. Adam has per-parameter `m` and `v`, and `bias_correction=False` by default (Adam class starts at line 476). Its recurrence is:

- `m = 0.9*m + 0.1*g`
- `v = 0.999*v + 0.001*g*g`
- `parameter -= lr*m/(sqrt(v)+1e-8)` when bias correction is false.

Resetting moments is material even though this branch does not use the step for bias correction. For a first update with a gradient sufficiently larger than epsilon, its magnitude is approximately `sqrt(10)*lr` per nonzero coordinate. Recreating Adam every group repeats this first-step behavior; continuous Adam combines previous gradients and squared gradients. Our three-update NumPy regression produces different final parameters when moments are reset, while checkpoint/resume gives bitwise identical results to uninterrupted computation. Do not silently change `bias_correction` to True: that is another experimental intervention.

## Pinned official-source comparison

Local official checkout: `work/vero-official`, commit `025ca3c800fc95224c86323c1f7fd33dceed3090`. Source hashes are in `optimizer_continuity_cpu_receipt.json`.

| Source under `vero-rl/verl` | Observed behavior | Local implication |
| --- | --- | --- |
| `workers/fsdp_workers.py:565` | Creates a persistent PyTorch AdamW actor optimizer with configured LR/betas/weight decay and a separate scheduler. | MLX Adam without bias correction is not the same optimizer. Preserve its historical settings in an explicitly named local control. |
| `workers/actor/dp_actor.py:505` and `:593` | Zeroes gradients at minibatch boundaries, scales microbatch losses, accumulates backward passes, then calls optimizer step on the same actor optimizer. | Reusing model weights while resetting optimizer state every group does not reproduce this lifecycle. |
| `workers/actor/dp_actor.py:532` | For its on-policy case uses detached current log probabilities; otherwise reads stored old log probabilities. | Old probabilities must remain frozen for each rollout batch; after an update the next on-policy rollout must use the new weights. |
| `trainer/ppo/core_algos.py:1036` | Uses response-mask-normalized mean log ratio, token stop-gradient construction, maximum log-ratio clamp 10, asymmetric clipping, and sequence-mean/token-mean aggregation. | Current local single-response unpadded implementation has this formula structure; it is not a cross-framework gradient comparison. |
| `utils/checkpoint/fsdp_checkpoint_manager.py:233` and `:333` | Loads/saves optimizer state dictionaries. Extra state restores/saves RNG and LR scheduler state. | Adapter-only resume omits required state. Callable scheduler state would need a separate tested integration. |

The local smoke computes old/current parity before update, averages per-response loss gradients by group size, and applies one update per active group. It has no multi-epoch rollout reuse. A continuation implementation must preserve this one-group boundary or explicitly redesign and review accumulation and policy-staleness semantics. Persisting moments alone does not equate one-group local updates, LoRA/quantization, model precision, clipping, sampling and scheduler behavior to official distributed Vero.

## Implemented utility and evidence

New files: `src/mri_rl/optimizer_checkpoint.py`, `tests/test_optimizer_checkpoint.py`.

The utility uses a JSON structural manifest and injected tensor codec; it never imports MLX, pickles objects, or converts native tensors through NumPy. It preserves list/tuple/dict shape, primitive values, tensor dtype and shape. It validates scalar optimizer step against effective updates, completed-group count, presence and consistency of Adam moments, nonempty explicit RNG state, caller-supplied exact resume bindings, tensor file digest and tensor inventory. New directories publish after payload/manifest synchronization; existing destinations fail rather than overwrite. Failed tensor writes clean the unpublished temporary directory.

This is an immutable optimizer/RNG **sidecar**, not a complete model checkpoint. The caller must bind and verify actual adapter bytes, model identity, trainable signature, all optimizer hyperparameters (including bias correction), source/contract/configuration identities and backend version. These are not inferred or enforced as a fixed bindings schema by the generic utility. The tensor digest detects accidental tensor corruption; it is not authentication of the manifest or reviewer identity. Runtime finite-gradient checks remain the trainer's responsibility. Only completed groups with no pending gradient accumulation are supported. Mid-rollout/mid-gradient resume is unsupported.

CPU validation command (existing Python pytest plus installed MLX package):

```sh
MRI_TEST_MLX_CPU=1 PYTHONPATH=src:/Users/vv/Documents/Codex/2026-10-15/files-pasted-by-the-user-codex-2/work/mlx-venv/lib/python3.13/site-packages python3 -m pytest -q tests/test_optimizer_checkpoint.py
```

Observed: **9 passed in 1.16s**, exit 0. The initial sandbox attempt could not initialize the Metal device; the authorized unsandboxed rerun set `mx.set_default_device(mx.cpu)` and succeeded. No model or MRI data was loaded.

Tests establish:

1. NumPy continuous/resumed equality and deliberate reset divergence across three updates.
2. Dict/list/tuple and native integer tensor preservation; Python RNG next-draw equality and NumPy RNG-state structure preservation.
3. Wrong adapter/configuration binding, corrupt tensor file, wrong dtype, extra tensors, incomplete moments and mismatched effective update count rejection.
4. Existing-checkpoint protection and failure-before-publication cleanup.
5. Actual MLX Adam safetensors roundtrip, exact native step/LR/m/v equality after the next update, float32 and bfloat16 parameter equality versus uninterrupted updates, and nonzero parameter changes.
6. Actual MLX global RNG restoration reproduces its next seven-value draw exactly. Installed `mlx_vlm/speculative/common.py:47`–`:53` confirms the library itself copies `mx.random.state` and restores elements in place.

Running the default suite without `MRI_TEST_MLX_CPU=1` skips the native test, so default CPU pass alone must not be reported as fresh native validation. The opt-in test does not benchmark real-model memory or establish device-to-device reproducibility.

## Minimal integration proposed to the implementation lead

1. Add an explicitly versioned continuity mode, preserving historical reset-mode artifacts. Bind that mode, this utility's source, optimizer settings and full state signatures into the review contract. Obtain the required independent review before any model run.
2. Construct Adam once before generation. On continuation, verify previous adapter bytes and expected bindings, assign `optimizer.state = restored_optimizer_state`, and restore RNG **after** all model loading/initialization and immediately before the next stochastic training action. Use the same native safetensors codec as the native CPU test.
3. Keep existing per-group old/current parity checks, finite/nonzero-gradient gate and one active-group update semantics. Advance optimizer step only on an actual update. Advance completed-group/attempted-compute cursor after every completed group, including zero-variance groups.
4. After the update or inactive-group decision, evaluate native state. Save adapter and optimizer/RNG sidecar with a shared binding and cursor. Capture training RNG before any stochastic checkpoint-reload diagnostic and restore it after that diagnostic, or clearly define and test the diagnostic's RNG consumption. Never reseed resumed training unconditionally.
5. Publish a runner receipt only when both the adapter and sidecar are present and hashes match. In a crash between the two writes, use the prior complete boundary. A zero-variance group must still persist its consumed RNG and completed-group count while retaining moments and effective step.
6. If child processes remain necessary for memory, pass the sidecar between children along with adapter weights; reconstruct the identical optimizer and restore all used RNGs. A single long-lived process avoids repeated initialization but requires separate real-model resource evidence. Only one model process may run at a time.
7. Record exact state hashes, update count, group cursor, reload checks and peak memory in the receipt. A bounded reviewed dynamic run must compare the next real update against uninterrupted execution before claiming integrated continuity.

The utility is suitable for reviewing that integration. It does not itself change the current runner, authorize heavier training, or establish Vero transfer, task learning, cross-task preservation or clinical generalization.

## Post-panic corrections and renewed bounded validation

The initial implementation and nine-test native receipt above are historical. Independent review subsequently found missing-moment inventory, seed-only RNG, publication-race and directory-durability defects; corrected code and regression evidence are documented in `SAFE_RESUME_INDEPENDENT_REVIEW.md`. The helper now requires complete trainable signatures, validates concrete Python/NumPy/MLX RNG schemas, reserves a destination exclusively, and publishes its manifest last after synchronization. No whole-project local-model prohibition remains.

Under the researcher's renewed tiny-local authorization, the lead reran **one native MLX CPU Adam/RNG continuation test: 1 passed in 0.09 seconds**, with both `MRI_TEST_MLX_CPU=1` and `MRI_REAUTHORIZED_TINY_CPU=1`. A separate four-dimensional linear-layer fixture verified exact deterministic loss and gradient equality with/without the installed activation-checkpoint wrapper on MLX CPU 0.32.4. Neither loaded MRI or pretrained model weights.

The newly reviewed SFT runner saves optimizer moments, explicit MLX/Python RNG, trainable signatures and a frozen sampler cursor after each completed update. It still declares `exact_resume_supported=false`: there is no integrated cross-process SFT restore entry point or full-model next-update equivalence result. The one-process prefix uses continuous Adam. GSPO multiarm continuity has not yet been integrated; prior reset-mode runs remain exploratory. A synthetic native CPU equality result is not evidence of MRI learning, Metal peak-memory safety or official Vero parity.
