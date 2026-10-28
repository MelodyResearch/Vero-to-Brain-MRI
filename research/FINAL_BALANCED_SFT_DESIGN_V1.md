> Historical preparation snapshot. Native execution subsequently completed with a negative result; see [actual results](FINAL_ACCURACY_RESULTS_V1.md). Status claims below describe preparation time.

# Final accuracy sprint — balanced patient-paired sequence SFT candidate (design V1)

Author: Claude Code (`claude-opus-5-5`), 2026-10-27/28. Status: **SOURCE_IMPLEMENTED_CPU_ONLY_NOT_NATIVE_RUN**.
Model loads, native forwards/backwards, optimizer updates, measured accuracy: **0**. No commit and no launcher. No root
authorization exists, and this work does not issue one. `CROSS_MODEL_REVIEW=NOT_VERIFIED`. Root independent review pending.

Files: `src/mri_rl/final_balanced_sft_v1.py`, `tests/test_final_balanced_sft_v1.py` (37 CPU/NumPy tests, system python3:
**37 passed**). Evidence read: public source APIs and aggregate reports only. Nothing read from `local_data/`,
`local_adapters_PRIVATE/` or MRI weights.

## What this is and is not

This is ONE explicit supervised comparison (track B), and it is supervised cross-entropy. It is **not** Vero GSPO and it is not the
auxiliary v1/v2/v3 objectives. It cannot answer the main RL question (genuine verifiable-RL Vero→MRI transfer). It asks a narrower
question: under the identical parent, prompt, LoRA and evaluation, does direct label supervision with a balanced, patient-paired
design move image-conditioned sequence discrimination where two GSPO updates did not?

## Frozen design

| Item | Choice | Evidence |
|---|---|---|
| Parent | P_calibrated13 / Adam52 / boundary114 adapter + full Adam + RNG, restored via existing `restore_parent` | OBSERVED source (`accuracy_stage_native_v1.restore_parent` requires step 52, 504 trainable) |
| Backend | Lazy subclass of existing `MicroNativeBackend` (12 GiB / 256 MiB capped unchanged `__init__` AST). Only change: the unchanged `SharedRolloutBackend._prepare_context` AST recompiled with the concise prompt-length ledger, exactly the Pilot V2 mechanism (`accuracy_pilot_v2_native._context_node/exact_prompt_cost`) | REPRODUCED on CPU: the class builds without MLX, the AST hash equals the source, 233 is accepted, 213 is refused |
| Prompt | Unchanged concise evaluation prompt (`accuracy_pilot_v2_core.condition_row(row,'concise')`), 233 expanded tokens | 233 = 213 + 20 is INFERRED from the V2 report text. The native run fails closed if it differs |
| Data | Existing 28 TRAIN people, z045, FLAIR **and** contrast-enhanced T1 of the same person in every update. No new people. DEV/holdout16/final100 refused | CPU-tested refusals: split, lineage, level, source id/modality, label, question, data source, explicit exclusion sets |
| Order | Salted SHA-256 (`final-balanced-sft-v1:seed{s}`) in a single explicitly non-site TRAIN stratum, fixed before any output; seeds {0,1,2} change only the order | CPU-tested |
| Target | `<think>Visible MRI evidence.</think><answer>\boxed{<label>}</answer>` + `<|im_end|>` | Strict Vero parser format=1 and correct class (CPU-tested). Native EOS id membership is NOT_VERIFIED until load |
| Objective | Per image: mean token CE over **response positions only**, from the existing F32 full-TF scorer (`_context_logps` → `f32_response_logps_observed`) under `nn.value_and_grad`. Pair: 0.5·CE_F + 0.5·CE_T. Two serial image backwards (one graph resident), float32 0.5-scaled sum, then **one** `apply_update` | Masking equals the scorer slice (CPU reference, stable at logits ×1e4). Adam-on-mean reproduced exactly in a NumPy fixture |
| Optimizer | **Continue** restored Adam52: lr 1e-5, β .9/.999, ε 1e-8, no bias correction, no reset, constant LR | Same optimizer as the historical SFT/GSPO source |
| Size | One pass: ≤28 updates, ≤56 image backwards, 56 scorer forwards, 0 generations; TF positions = 28·Σ_label(233 + len(target) − 1) | `run_budget` |
| Durability | After every pair: adapter + full Adam + RNG + seed/order cursor in a pending dir → fsync → atomic rename → hash-chained receipt. Resume only from the last committed receipt after a fresh-process `load_checkpoint`. Stray/pending/receipt-less/tampered state is refused as INDETERMINATE. No in-memory rollback (see below). Counters are never refunded; a resume that cannot finish within 56 backwards is refused up front | CPU-tested: boundary-stop resume reproduces the uninterrupted pass bit-exactly; a mid-pair failure is refused; a failed save leaves no partial commit |
| Checkpoint schema | Same files/bindings as `StageNativeBackend.save_checkpoint` (`{run, adapter_sha256, trainable_signature}`), groups = 114 + pairs, updates = Adam step = 52 + pairs, so the inherited `load_checkpoint` restores it | CPU round trip through the real `optimizer_checkpoint` |
| Safety | Caller `before(name)` (authorization + dynamic resource gate) at run start, every pair, every image backward, before update, before checkpoint and after commit. The backend `before_unit` is also invoked inside the scorer and update. The native factory raises `PendingAuthorization`: entry module `mri_rl.final_balanced_sft_entry_v1` is absent | CPU-tested fail-closed |

Why there is no in-memory rollback: `safe_micro_restore_v1` documents that MLX's optimizer state setter aliases the caller's
containers, so a snapshot handed to `restore` can be corrupted by a later update. `begin_run` takes one `snapshot()` only so that the
inherited `apply_update` can validate the frozen 1438-parameter base. That snapshot is never used as a restore target. The last
durable checkpoint is the only rollback point.

## Optimizer choice and causal limits (explicit)

1. **Inherited GSPO momentum.** With no bias correction, update k uses `m_k = .9^k·m_52 + Σ…g`. The GSPO-era first moment weight is
   .90 (k=1), .59 (k=5), .35 (k=10), .12 (k=20) and .05 (k=28). The second moment keeps .999^28 ≈ .97 of v_52. Early SFT steps are
   therefore partly a continuation of GSPO momentum, and their per-coordinate scale is set by GSPO-era v. A momentum-only drift
   control was **not** run. The SFT-vs-Parent difference is "SFT gradients + residual GSPO momentum", not pure SFT.
2. **Not matched to the GSPO arms.** SFT sees 28 people / 56 images. Pilot V2 GSPO saw 4 people / 8 images, plus Dallas. SFT also
   has 28 vs 2 updates and a teacher target instead of sampled rewards. SFT−GSPO is not a controlled method contrast. Only SFT vs
   the same Parent under the same concise F32 evaluation is interpretable, and only as an engineering signal.
3. **Format vs label.** The think/tag/EOS tokens are identical across classes and dominate token count. Only the label span
   discriminates, effectively its first token. Per-image mean CE weights label tokens by 1/len(target), and the FLAIR and T1
   targets differ in length. Concise-prompt Parent already had 0/24 invalid outputs. A DEV gain therefore cannot come from format
   repair, which is useful, but the trained model will emit a fixed template. That erases free-text "reasoning" by design, and no
   reasoning claim is possible.
4. **Candidate readout is no longer independent.** The target think text is the evaluator's frozen candidate `PREFIX`. Candidate
   margins then measure the trained conditional directly. Strict greedy balanced accuracy / T1 recall / FLAIR non-drop must be
   primary; candidate scores are diagnostic only.
5. **Seeds vary only order.** The loss is deterministic: the RNG must be unconsumed per pair, or the run stops. Seed spread is
   order sensitivity, not sampling noise.
6. **Evaluation population.** DEV24 has 4 previously explored people. No independent-generalization, clinical or
   anatomical-grounding claim is possible. Holdout16 and final100 are untouched and must stay so.

## Scientific judgment

**Appropriate as a bounded B-track control; implement as specified.** It targets the observed failure directly. Parent's 12 DEV
pairs contain 3 constant-FLAIR pairs, 1 constant-T1 pair and 1 reversed pair. Same-person, same-prompt FLAIR/T1 pairs averaged into
one update push the image-independent template identically and the label span in opposite directions. Balanced labels prevent an always-one-class predictor from classifying both modalities correctly, but template learning can still lower CE without improving classification. Strict greedy paired accuracy is therefore required. It is minimal (one pass, one objective,
the existing optimizer) and needs no new data.

Expected risks: 28 LoRA rank-4 steps at 1e-5 may be too small to move greedy labels (NOT_MEASURED), or template collapse may
cost some correct-but-verbose outputs. Neither justifies changing the method before one run. Recommended (root decides):
- Run seed 0 once.
- Pre-register the V2 decision rule against the same-condition Parent: BA up, T1 recall up, FLAIR not down, invalid not up.
- Only on that rule, run seeds 1–2 for this single winner (≤3 runs, inside the ≤4 cap).
- No loss/LR/target search.

## Open items for root integration (not done here)

- Entry `Authorization` + exact scope + matching dynamic resource gate (`safe_micro_resources_v2` pattern). Launcher/child
  process, global flock, guardian (`safe_micro_watchdog_v1`).
- Supply rows, sites map and explicit DEV/holdout16/final100 key sets from private data. Image SHA pins are passed to
  `configure_stage`.
- Native-only verifications (NOT_VERIFIED):
  - actual concise length 233;
  - `<|im_end|>` is a single effective EOS id;
  - F32 value_and_grad memory over ~233 + target tokens is below the V2-observed 10.36 GiB MLX peak (INFERRED, since V2 GSPO
    backwards covered up to 256 response tokens);
  - RNG unconsumed;
  - wall time.
- Evaluation: reuse the V2 concise F32 DEV24 path for the terminal adapter (and the saved concise Parent if the contract matches).
  Sanitized aggregates only to the private GitHub repo.

## Codex integration and evidence boundaries

The source-qualified56-row pool is newly derived from the original immutable TRAIN JSONL and a fresh independent hash/header audit of the existing56PNGs. The lost original private pool is not represented as recovered. Only the obsolete project-root segment in image paths is rebound; no pixels or labels are changed. All28 source people belong to the historical TRAIN partition, so a fold within them is not an unseen-patient validation for Adam52. DEV24 remains the four previously explored people, at levels45/91/136. The source-level original TRAIN membership and prior disjointness evidence are inherited; new16 inputs and formal100 predictions are not reopened.

No authenticated acquisition-site join is available at this preparation point. The runner passes one explicit NON_SITE scheduling stratum; it does not claim or implement site balancing. Seeds0/1/2 change only patient order. One candidate and at most3runs are within the researcher's newest four-run maximum. Additional seeds require seed0>=19/24, FLAIR>=10/12, T1>=8/12, invalid0 and >=7/12fully correct modality pairs, with completed native exit/release evidence. Only the end of one fixed28-update pass is evaluated; no checkpoint or target search.

Per run:56scorer forwards/backwards,14028TFpositions (17FLAIR/20T1response tokens including EOS;233expanded prompt),28updates,24greedy calls capped at6144returned tokens, one serial model load,3600s operation-start window. Maximum3runs:168backwards,42084TFpositions,84updates,72greedy/18432returned tokens,3loads,10800s total. The retained color resource policy is RSS14GiB/MLX12GiB/cache256MiB/disk26GiB/battery20%, with actual critical/severe-risk stops, own-process guardian and no other application termination. Atomic GPU calls retain monitoring blind spots. No cache regeneration or sealed16 evaluation is authorized by this scope.

This is a performance-oriented supervised control against its unchanged Parent, not a budget-matched SFT-versus-GSPO causal comparison. Historical SFT final84 and GSPO final100 cannot be numerically ranked against new DEV24 results. A matched new evaluation of historical SFT remains NOT_RUN unless separately frozen within this sprint.
