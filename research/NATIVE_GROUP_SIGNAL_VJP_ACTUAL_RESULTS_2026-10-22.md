# Native read-only A/B/C VJP on Parent Adam52 + Common32: actual results (2026-10-22)

Author and execution lead: Claude Code (claude-opus-5-5); final numerical-claim precision edits: Codex. Independent actual recomputation: isolated Codex, **PASS_WITH_LIMITATIONS**.
See [`NATIVE_GROUP_SIGNAL_VJP_ACTUAL_INDEPENDENT_REVIEW_V1.json`](../results/NATIVE_GROUP_SIGNAL_VJP_ACTUAL_INDEPENDENT_REVIEW_V1.json). Claude used approved sanitized aggregates; no Claude review of raw PRIVATE patient/model-state data is claimed.

**Frozen verdict: FAIL_REJECT_C.** The rule produced 2 FLAIR failures and 2 Dallas failures, with 0 ambiguous contexts and no flags.

- **Decision:** reject candidate C. There will be no one-update C pilot, no training GO, no coefficient search and no expansion.
- **Scope of the rejection:** this rejects one specific advantage plan. It does not show that Vero-style group-relative RL is ineffective
  for MRI.
- **No new accuracy result:** the run had 0 generation and 0 updates. Parent 18/24 and A54/B54 16/24 are prior DEV24 results, reused
  only as context.

Evidence: [`results/NATIVE_GROUP_SIGNAL_VJP_ACTUAL_NATIVE_V1.json`](../results/NATIVE_GROUP_SIGNAL_VJP_ACTUAL_NATIVE_V1.json). Design and
frozen rules: [`NATIVE_GROUP_SIGNAL_VJP_DESIGN_V1.md`](NATIVE_GROUP_SIGNAL_VJP_DESIGN_V1.md). CPU mechanism background:
[`GROUP_RELATIVE_VISUAL_LEARNING_MECHANISM_V1.md`](GROUP_RELATIVE_VISUAL_LEARNING_MECHANISM_V1.md).

## 1. Execution (OBSERVED)

**Authorization and integrity**
- The exact scope file `069520b2…` received a fresh human approval and an exact pre-execution review (`0801dd55…`); it was launched
  once and never repeated.
- **Native child:** PASS, exit 0, reaped, 310.2 s. Counters were exactly 44 forwards, 44 VJPs and 14,291 teacher-forced positions,
  with 1 model load, 0 generation, 0 optimizer updates and 0 parameter or checkpoint writes.
- **State and release:** the full weights, full Adam and RNG identity were unchanged before vs after. Release was verified at 1,032
  active MLX bytes, and the lock was released. There was no SIGTERM, SIGKILL, guardian stop or RED request.

**Resources**

| Measure | Value |
|---|---|
| Peak RSS (wait4) | 5.85 GiB |
| Peak MLX reported | 10.19 GiB |
| Minimum available memory | 6.27 GiB |
| Minimum pressure-free | 33% |
| Colors observed | GREEN and YELLOW |
| Swap | 12.51 → peak 14.10 GiB (+1.60 GiB), 14.04 GiB after the run |
| Minimum battery | 84% |
| Thermal sensor | unavailable |

Monitoring blind spots during atomic native calls remain.

**Controller failure (separate from the native run)**
- **Failure:** the original controller exited 1 after the native success, at the CPU public-export stage. The unchanged strict
  `public_case_key_check` correctly refused a scalar count published under the key `contexts`.
- **Fix:** the CPU-only fix renamed two public count keys to `count`. Maths, native, backend and entry code were unchanged; the
  executed core was `204727d4…` and the postprocess core `08d6b249…`.
- **Recovery:** the saved-vector CPU postprocess then exited 0 with 0 new native calls, and the root's 77 focused tests passed.

## 2. Preconditions (all PASS)

**A/B full-vector reconstruction** against the saved historical joint gradients, 504 leaves:

| Arm | Relative L2 (≤ 1e-4) | Cosine | Max abs error |
|---|---|---|---|
| A | 1.74e-5 | 0.99999999986 | 2.44e-7 |
| B | 2.77e-5 | 0.99999999962 | 3.84e-7 |

| Check | Result |
|---|---|
| Own-old/current max \|Δ\| | 0.0 (bound 1e-5) |
| Native F32 vs CPU log-ratio | 0.0 difference |
| Clip branches | every admitted response UNCLIPPED in A, B and C |
| First-token oracle vs label-mean forward | max \|Δ\| 0.0 (tolerance .002), 4/4 MRI contexts |
| Saved-reference rounded Adam vs historical θA53/θB53 | PASS, 0 violations, max abs error 1.8e-12 |
| Public screens | case keys PASS; 128 private identifiers screened, 0 hits |

## 3. Are C's native gradients different from A and B? (OBSERVED)

**Gradient norms**

| Direction | A | B | C |
|---|---|---|---|
| ‖−g‖ | 0.2751 | 0.3667 | 0.3314 |
| ‖incremental rounded Adam‖ | 0.00377 | 0.00555 | 0.00494 |
| ‖total rounded Adam‖ | 0.03000 | 0.03033 | 0.03036 |

The common momentum-only displacement has norm 0.03015, so it dominates every total.

**Cosines**

| Direction | A–B | A–C | B–C |
|---|---|---|---|
| Raw gradient | 0.683 | 0.820 | 0.936 |
| Incremental | 0.623 | 0.782 | 0.926 |
| Total Adam | 0.990 | 0.995 | 0.998 |

C's gradient is genuinely different from A's and moderately different from B's in raw-gradient and incremental space. The total
Adam52 directions are nearly collinear across all three plans. The preserved-state-only reference is much larger than the new-gradient
Adam difference; this arithmetic comparison supports a substantial inherited-state contribution, without isolating a causal mechanism.

## 4. Canonical margin changes (first-order, ⟨∇M, d⟩)

**Metric definitions**
- **mean:** the correct-minus-wrong mean label log-prob. This is primary.
- **sum:** the correct-minus-wrong summed label log-prob.
- **first:** the shared-prefix first-token contrast.

For MRI these differ, because the labels are 2 vs 5 tokens and the canonical prefix is shared. For Dallas the labels are single
tokens, so mean = sum = first. The primary direction is the original Adam52-preconditioned F32 rounded displacement (τ = 1e-6, the floor,
in every primary cell).

All entries below are [min, max] over the class.

**MRI, mean margin**

| Class (n) | Parent margin | Plan | −g | Adam52 rounded (primary) | Momentum-only (common) | Incremental Adam difference |
|---|---|---|---|---|---|---|
| FLAIR (2) | [1.10, 1.12] | A | [0.214, 0.246] | [−0.00193, −0.00055] | [−0.00340, −0.00232] | [0.00147, 0.00177] |
| | | B | [0.267, 0.322] | [−0.00165, −0.00006] | (same) | [0.00175, 0.00225] |
| | | C | [0.192, 0.218] | [−0.00195, −0.00054] | (same) | [0.00145, 0.00178] |
| T1 (2) | [−1.01, −0.65] | A | [−0.161, −0.159] | [0.00196, 0.00197] | [0.00306, 0.00320] | [−0.00123, −0.00110] |
| | | B | [−0.237, −0.199] | [0.00148, 0.00176] | (same) | [−0.00158, −0.00144] |
| | | C | [−0.143, −0.083] | [0.00196, 0.00236] | (same) | [−0.00110, −0.00084] |

**Dallas (4, engineering only), mean margin**
- Under Adam52 rounded, A, B and C each span about [−0.088, +0.085]. Momentum-only alone spans [−0.088, +0.086].
- C−A ranges over [−1.26e-4, +1.01e-3] and C−B over [−2.51e-4, +4.37e-4].

## 5. Reading under the frozen rule (primary: MEAN, Adam52 rounded, every context individually)

**FLAIR — both images FAIL**
- C is negative on both images, beyond −τ, and is also worse than B by 3.0–4.7e-4.
- A and B are negative under the total step too. The common zero-new-gradient reference is more negative; the new-gradient Adam difference
  partly offsets it, without producing a protected total direction on either image.
- C's incremental Adam difference relative to the zero-new-gradient reference is positive for FLAIR, but smaller than B's.
- Under −g, C raises FLAIR less than both A and B.

**Dallas — 2 of 4 contexts FAIL**
- These contexts are worse than at least one control beyond 2τ; that is not necessarily worse than both.
- The other two Dallas contexts pass.

**T1 — both images pass the primary rule, but the full joint −g direction is still negative**
- **What passed:** the total Adam52-rounded T1 change is positive, and C beats B by approximately 4.77e-4 or more. C beats A on one image by 3.9e-4,
  but on the other image by only 2.44e-6, barely above 2τ = 2e-6. τ is an engineering screen, not a confidence interval.
- **The inherited-state reference already has a positive T1 direction:**
  - Momentum-only arithmetic alone moves T1 by about +0.0031.
  - The new-gradient increment is negative for all three plans (C [−0.00110, −0.00084]).
  - C's full joint −g T1 change is still negative ([−0.143, −0.083]), although less negative than A's or B's.
- **Accuracy:** no accuracy causality can be attributed to momentum or to this positive total.

**Rescue rules:** no average, secondary metric or T1 result rescues the FLAIR and Dallas failures. First-token values are reported
separately and differ from mean; for example, B's FLAIR first-token change can be positive while its mean is negative. They do not
change the verdict.

**PRIVATE appendix:** the full per-image table (every context × plan × family × metric, τ and parent margins) exists only in the local
private analysis file (`cpu_analysis_private_file_sha256 bbd195bb…`) for human inspection. No public case identifiers are given here.

## 6. Scientific conclusion

1. **Scalar coefficient improvements did not satisfy the native class-protection criterion.** C reduced the CPU format-coefficient
   mass (23.8% → 10.4%) and improved scalar sign alignment; those descriptive statistics remain valid. The actual native joint gradient
   did NOT provide a class-protected learning direction:
   - its raw gradient still lowers the canonical T1 margin;
   - it raises FLAIR less than A or B;
   - under the real Adam52 step it fails FLAIR protection on both images and Dallas protection on 2 of 4 contexts.

   Better-looking scalar credit allocation is not the same as a better parameter-space direction.
2. **The implemented numerical checks passed here.** A/B reconstruction (relative error ≤ 2.8e-5), parity, branch capture, oracle
   alignment and historical-53 reproduction all passed. The observed own-old/reconstruction mismatches do not explain the rejection
   under the frozen engineering contracts; this is not a formal error bound for C or proof against every numerical limitation.
3. **Inherited state is a substantial component of every total direction:** total pairwise cosines range from 0.98966 to 0.99760,
   and the zero-new-gradient reference norm is about 5.4–8.0× the incremental Adam-difference norm. The latter is `Adam(g) − Adam(0)`;
   it includes nonlinear second-moment/preconditioning changes and is not a separate linear gradient component or an isolated
   first-momentum causal effect. Group-relative changes here are much less distinct in total-displacement space; no Adam experiment follows.
4. **What remains is Vero visual credit assignment.** None of A, B or C raises the canonical T1 margin along its complete joint `−g` direction on
   these two training pairs. MRI-only and Dallas-only contributions have not been isolated here, so task-interference causality remains
   unverified. No new loss, model, prompt or Adam ablation is proposed here; the next decision belongs to the researcher.

## 7. Limitations

- **Small sample:** two TRAIN patient pairs (2 FLAIR + 2 T1 images), plus 4 Dallas engineering contexts; Dallas is not a qualified
  medical task.
- **Local first-order evidence only:** no finite update was evaluated, and margins are canonical-prefix diagnostics, not sampled behaviour.
- **No generalization or grounding claim:** there is no anatomical-grounding claim, no independent-patient evidence and no new
  accuracy.
- **τ is a heuristic screen:** an empirical engineering screen frozen before C's values, not a statistical bound.
- **Monitoring gaps:** the thermal sensor was unavailable, and there are monitoring blind spots during atomic native calls.
- **Independent review:** saved-vector recomputation matched all production effects, margin values and τ exactly under the independent
  CPU implementation. This validates the recorded calculation within its definitions; it does not establish finite-update accuracy.
