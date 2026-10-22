# Native group-signal VJP (A/B/C on Parent Adam52 + Common32): implemented design

Author: Claude Code (claude-opus-5-5), 2026-10-22. Independent review: Codex (`/root/v2_actual_verifier`).

**Status: ACTUAL_IMPLEMENTED_CPU (synthetic tests by the root; RED captured before implementation). NOT_NATIVE_RUN.** There is no
native, GPU or model result and no PASS claim. Execution needs an independently reviewed exact scope plus a fresh human permission file.
Nothing here is an authorization request.

## Files
- `src/mri_rl/native_group_signal_vjp_v1.py` — CPU core: jobs, GSPO coefficients, joint gradients, reconstruction, canonical token
  plans/positions/budget, margins, read-only Adam52 directions, frozen screen/classification, `prepare`, `analyze`, public aggregate.
- `src/mri_rl/native_group_signal_vjp_entry_v1.py` — authority, the one-shot journaled controller (`run_readonly`), the resource-gate
  adapter, and CLI modes `--prepare/--launch/--child/--analyze`.
- `src/mri_rl/native_group_signal_vjp_backend_v1.py` — lazy native ops; importing it loads no MLX.
- `scripts/run_native_group_signal_vjp_v1.py`; `tests/test_native_group_signal_vjp_v1.py` (synthetic only).

## Inputs (exact, Common32 only)
- **Authentication and plans:** `v2_regression_diagnosis_v1.authenticate` and `rederive_arm` (segment 0, common parent, Parent Adam52).
  A and B are the actual plans; C is the frozen `answer_matched_baseline_v1.c_plan`.
- **Frozen C source:** the committed final source `a1a54b9a…`, linked by `results/GROUP_RELATIVE_C_DOCSTRING_PRECISION_V1.json` to the
  executed source `d3998ac4…` (docstring-only change; identical executable AST).
- **Never used:** DEV24, final100 and new16.
- **Response jobs:** the union of responses with nonzero weight under A, B or C, derived rather than forced: 28 = 16 MRI + 12 Dallas.
- **Margin contexts:** all 8 original Common contexts (2 FLAIR, 2 T1, 4 Dallas), validated per unit from the original plan.
- **Bound before any model construction:**
  - parent, A53 and B53 snapshot files;
  - saved gA0/gB0 against their journal commits;
  - the cache directory, including both historical SHARDS;
  - the model's non-weight files, the certificate and the tokenizer;
  - the runtime pin, and the source closure (own three modules, the CLI, the tests and this design note).
- **Not bound:** the mutable experiment log.

## Budget
Derived at prepare from the actual tokenizer and recorded prompt lengths, then cross-checked (not overridden) against
`results/NATIVE_GROUP_SIGNAL_VJP_INPUT_BUDGET_INDEPENDENT_V1.json`:

| Item | Count | TF positions |
|---|---|---|
| Response VJPs | 28 | 9679 |
| MRI margin VJPs (4 contexts × [F label mean 245 + T label mean 248 + first 244]) | 12 | 2948 |
| Dallas margin VJPs (4 contexts × first 416, single-token labels: mean = sum = first) | 4 | 1664 |
| **Total** | **44 forwards / 44 VJPs** | **14291** |

Also: 1 Parent load, 0 generation, 0 optimizer updates, 0 parameter, checkpoint or adapter writes. Positions are P + |prefix| + |label| − 1
for label means and P + |prefix| for the first token, with prefix = `answer_likelihood.PREFIX` (11 tokens).

## Mathematics (frozen; `ngs.TOLERANCES`)
- **GSPO derivative.** Per response, one native F32 mean-log-prob VJP u_i. Inside the same call the native loss's own F32 sequence
  log-ratio stop_gradient(mean(v − saved_old)) and ratio exp(min(·, 10)) are captured at no extra cost. Then κ = −A·r on the active
  unclipped branch, else 0.
  - An independent float64 recomputation cross-checks the captured log-ratio (≤ 1e-5).
  - A ratio within 1e-6 of a clip bound or of the cap gives AMBIGUOUS, hence INCONCLUSIVE.
  - Old == current is never assumed.
- **Plan gradients.** g_X = Σ_units .25·Σ_i (1/8·κ_i) u_i, accumulated in two-level float32 with Dallas included. Joint gradient-norm
  clipping is NONE, as in the original path.
- **Gate inside the child, before any margin call.** Own-old/current ≤ 1e-5 and A/B vector reconstruction relative L2 ≤ 1e-4, reported
  with max-abs, per-leaf, cosine and the 504 inventory. On failure: PARTIAL is committed, the model is released, and there are no margin
  calls and no C science.
- **Canonical margins.**
  - The prefix and suffix are identical literals for both labels, with the full boundary verified both on CPU and against the loaded
    tokenizer.
  - For MRI, mean and sum come from the two label-mean VJPs; sum = n_c·g_c − n_w·g_w.
  - The primary first value is the shared-prefix oracle contrast, i.e. the same forward that produces the first-token VJP. Each
    oracle logp must match the first label token of its own label-mean forward within the frozen .002 F32 scorer alignment, which is
    separate from the 1e-5 own-old gate; otherwise INCONCLUSIVE.
  - These are diagnostics only, never a loss.
- **Directions (read-only).** From the saved m52/v52/lr with betas (.9, .999), eps 1e-8, bias_correction False and step 52:
  - formula delta = −lr·m′/(√v′ + eps);
  - **primary:** the actual F32 parameter displacement float32(θ52 + delta) − θ52;
  - also momentum-only and incremental (formula and rounded), and −g.

  The saved-reference A/B rounded deltas must reproduce the historical θA53 − θ52 and θB53 − θ52 under the existing F32 parameter rule.
- **Screen.** Per context, metric and family, with −g and rounded Adam kept separate:
  - τ = max(1e-6, 5e), where e = max(|h·(dA_rec − dA_saved)|, |h·(dB_rec − dB_saved)|);
  - this is an empirical engineering screen, not a CI or a formal error bound.
- **Classification** (primary MEAN margin, Adam52-rounded direction, every context individually):
  - **T1, each image:** PASS if C > τ, C−A > 2τ and C−B > 2τ. FAIL if C < −τ, C−A < −2τ or C−B < −2τ. Otherwise AMBIGUOUS.
  - **FLAIR, each image:** FAIL unless C ≥ −τ, C−A ≥ −2τ and C−B ≥ −2τ.
  - **Dallas, all 4 contexts:** FAIL unless C−A ≥ −2τ and C−B ≥ −2τ; absolute values are reported.
  - **INCONCLUSIVE** whenever any of these holds:
    - the context inventory is missing, empty or differs from 2/2/4;
    - any plan × family × metric × context value or τ is missing or nonfinite;
    - A/B reconstruction or own-old parity fails;
    - the first-token alignment fails;
    - the historical-53 reproduction fails;
    - any value is AMBIGUOUS.
  - **Decision:** any FAIL gives FAIL_REJECT_C. Otherwise all-PASS gives PASS_FOR_SMALL_PILOT, which is local first-order evidence
    only; a separate training scope and fresh approval would still be required.
  - **Metrics:** sum and first are secondary; they are flagged and never rescue a failure.

## Execution path (actual code)
1. **Admission, before any native import:** scope-file SHA, a separately supplied review-file SHA (non-Claude isolated reviewer), and
   a fresh human file issued ≤30 min earlier, consumed once. The scope must byte-equal a fresh CPU `prepare`, and synthetic scopes are
   refused.
2. **Controller (CPU):** non-resident admission samples, then ONE guarded child (the existing `safe_micro_watchdog_v2` on its own child
   only). The child is verified (exit 0, reaped, global lock released, terminal committed with release), then the CPU analysis runs
   after the model is gone.
3. **Child:**
   - The global lock is held, with the unchanged `safe_micro_dynamic_resources_v1.POLICY` (GREEN/YELLOW/RED, RSS 14 GiB, MLX 12 GiB,
     disk 26 GiB, battery 20%; no fixed 9/10 GiB rule, no standalone swap STOP) in the B_MICRO gate with a one-load ledger.
   - The installed runtime pin and cache SHARDS are checked.
   - Load: `PilotV2Backend`, registered as resident before `__init__`, then `restore_parent` with the original binding. The full
     identity (1942 parameter fingerprints, full Adam, Python/MLX RNG, step 52) must equal the Common32 parent state.
   - **Integrity:** there is no in-RAM snapshot/restore wrapper (`pilot._readonly` is NOT used). The same full identity is compared
     after the last VJP, so a mutation is detected and reported, not hidden.
   - **Write paths:** the backend's write and optimizer paths (apply_update, save_checkpoint/adapter/rng, load_checkpoint, generate,
     optimizer.update) are replaced by refusing stubs. The probe performs no optimizer step and no finite parameter perturbation. This
     does not claim every internal model.update path is unreachable.
   - **Per call:**
     - The positions are checked before the forward.
     - Then the journal reservation is made and the ledger charged with the frozen positions, never refunded.
     - Then the native call runs, followed by the observed-position check and a float32 dtype check before any cast.
     - The finite 504-leaf vector is saved, fsynced, hashed and committed.
   - **Memory:** one image graph at a time, with references dropped and the cache cleared after each VJP.
   - **On failure:** the traceback is recorded, unwound exception frames are cleared, and the full or partial resident is released
     (<128 MiB measured). A failure/indeterminate record is kept, with no retry, replay or reload.
4. **Window and cost:** 3600 s of operation-start window including waits. ~90 s load plus ~10–15 s per VJP is a PLANNING ASSUMPTION
   from historical V2 rates, not measured timing. An in-flight atomic GPU call cannot be interrupted safely (acknowledged risk).

## Public aggregate
- **Public:** only class-level summaries (MRI_FLAIR / MRI_T1 / DALLAS counts and min/max/mean, C−A and C−B, τ ranges), the decision
  with failure COUNTS and flag KINDS, reconstruction/parity/alignment summaries, and norms/cosines.
- **PRIVATE, inspect locally:** per-image tables, context IDs, failure/ambiguous ID lists and per-context flags.
- **Checks applied:** private keys are stripped, private identifier literals are screened, and the recursive case/group-key and
  per-record-list contract applies. There are no pixel, token, response, gradient-vector or path outputs.

## Commands (root, CPU)
- **Prepare:** `ROOT/work/accuracy-recovery-venv-v1/bin/python scripts/run_native_group_signal_vjp_v1.py --prepare --root ROOT --out ROOT/local_data/native_group_signal_vjp_v1/attemptN_PRIVATE`
  - Use a NEW `--out` for every preparation.
  - It writes `scope_PRIVATE.json` and `prepare_summary_SAFE.json`, the latter including the independent budget cross-check.
- **Launch:** only after the independent review and a fresh human permission:
  `--launch --scope … --expected-scope-sha256 … --human … --expected-human-sha256 … --review … --expected-review-sha256 …`
