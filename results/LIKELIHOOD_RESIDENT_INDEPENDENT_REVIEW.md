# Resident likelihood helper: independent engineering review

**Current Step D verdict: PASS_WITH_LIMITATIONS for the corrected standalone helper source and CPU control-flow fixtures. Numerical-runner integration, actual resident memory behavior and launch approval remain NOT_VERIFIED.**

**Initial verdict: FAIL for deadline and approval revalidation safeguards.** Reviewer: isolated Codex C/data agent, independent of the helper implementer. `CROSS_MODEL_REVIEW=NOT_VERIFIED`. This report concerns the new control helper, not the already executed numerical likelihood runner or approval of a resident model launch.

## Initial review and counterexamples

Source read: `src/mri_rl/likelihood_resident.py`, SHA256 `20fadcee3bf8573f0e821f6d1e8d1f171aec9fc8e23eafb8593fef9534f4345d`. Eight existing helper tests passed in 0.07 seconds under an explicit MLX/MLX-VLM/Torch/Transformers import blocker. The reviewer then injected independent synthetic fake-clock and filesystem failures. No model, patient image or actual inference output was used.

| Finding | Independent reproduction | Required correction |
|---|---|---|
| LR-1: approval can release work after the deadline | After the first chunk, let the second approval-verification callback advance the clock by 181 seconds. The helper yields the next two records at time 211 although the published-request deadline is 195. Its final deadline check occurs before that second verification. | Check the deadline after all final approval/output checks and immediately before yielding the authorized continuation. No late approval may release model work. |
| LR-2: initial approval changes during a resource gate | Replace the initial approval file with `{}` inside the initial resource gate, after it was verified. The helper still yields the first chunk. | Revalidate the exact initial approval digest/scope after the gate and plan/prefix recheck, before the first yield. |
| LR-3: resource history truncation inside the gate is accepted | Clear the same history list during `before_unit`. Object identity is unchanged, and the helper resets its recorded history length to zero. | Check history length after the resource callback as well as before it, preserving the prior lower bound. This is a guard counterexample, not an observation that the actual battery gate truncates history. |

An independent nine-record pending-chunk injection correctly raises the finite 1–8-record budget error. The original counterexamples are preserved privately in `local_data/ds007045_bounded/likelihood_resident_independent_v1/initial_counterexamples_PRIVATE.json`.

## Positive source and fixture evidence

The helper yields only frozen manifest slices and binds each continuation to the current persisted output prefix hash, record cursor, fixed run/configuration scope and a request digest containing a session nonce. It rejects preexisting response files, requests from another session, changed output prefixes, wrong approval digests, incomplete chunks and repeated iteration. Completed rows are preserved on stop. The 57-input manifest bounds total work, and each chunk is limited separately.

Waiting invokes resource checks, file/approval validation and bounded sleeps, with no adapter verifier or model callback inside the wait loop. The adapter verifier runs after a completed chunk and before the request is published. The caller owns the model and its final release; this module cannot prove that future caller cleanup is correct. Source/CPU evidence alone also does not establish actual resident memory behavior.

The declared timing is a 15-second inter-chunk cooldown, followed by request publication and a maximum 180-second approval-admission window. A synchronous resource callback can itself return after that deadline; the required guarantee is that it then releases no new computation. The helper is not a preemptive watchdog and cannot force an external callback to finish within 180 seconds.

The existing resource gate object, initial battery/swap baseline and history must persist across chunks. The resident helper must not manufacture a fresh resource allowance after an approval or cooldown. Training remains unauthorized on battery; this proposed helper controls read-only likelihood inference only.

## Mission Alignment Check

1. **Question:** reduce repeated loading while obtaining the same frozen training-only likelihood diagnostic needed before further Vero-to-MRI training.
2. **Falsifiable requirements:** fresh request-bound approval, exact persisted prefix, finite chunk, no late release and unchanged cumulative resource baseline.
3. **Isolation:** numerical likelihoods, targets, prompts, model weights and checkpoint comparisons are unchanged; the helper is not yet proof of runner integration.
4. **Evidence:** exact source hash, CPU tests and synthetic failure receipts; no resident model run is claimed here.
5. **Controls:** continued work without approval, stale approval, malformed prefix, oversized chunk and reset history are explicit failure cases.
6. **Decision/resources:** hold affected resident integration until corrections and independent Step D pass. No MLX imports, training, model launch or approval was performed in this review.


## Step D — corrected helper independently verified

**Verdict: PASS_WITH_LIMITATIONS**, evidence tier **REPRODUCED** for the synthetic control-flow checks. Final helper SHA256: `bf3c511124bcb7c33716bc4e048c2fc88f47dc76f37ec15835046b3a596aacce`.

The reviewer reran the original injected cases against this source. LR-1 now ends with `APPROVAL_TIMEOUT` and yields no continuation. LR-2 rejects the changed initial approval digest before yielding. LR-3 rejects in-place history truncation. The independent nine-record injection continues to reject. All stopped cases preserve the existing output prefix; no recovery loop launches extra work. The original failure receipt remains immutable alongside the new saved replay script and `step_d_PRIVATE.json`.

A fresh import-blocked run of `tests/test_likelihood_resident.py` produced **12 passed in 0.10 seconds**. Source inspection confirms the deadline is checked after final approval and prefix validation and again immediately before the next yield. Initial approval is reverified after the initial resource gate. The same initial resource sample and history objects are retained, and each gate must append to the history rather than shorten it. Other tests cover delayed independent approval, wrong request/session digest, output changes, approval-file changes, incomplete chunks, no work after a timeout, no duplicate iteration, and full finite traversal of the synthetic 57-input manifest.

The `verify_adapter` callback runs only at completed-chunk boundaries before review-request publication. No model/adapter callback is called inside `_wait`; only resource sampling, immutable request/approval/output checks and sleeps occur there. The CPU timeout fixture reaches 195 seconds because it includes the separately declared 15-second cooldown plus the 180-second request window. If a synchronous resource or verification callback returns late, the helper rejects continuation; it cannot preempt that callback or promise process release at precisely 180 elapsed seconds.

### Scope that remains unverified

The numerical likelihood runner was not integrated or executed by this reviewer. Its future caller must instantiate this helper with the already initialized process-wide resource gate, include resident mode and this exact source in the independently approved run identity, preserve exact numerical scoring, fsync rows, and release the model in `finally` on timeout, exception or early close. These obligations need an integration review of the actual diff and CPU tests; this standalone helper verdict does not establish them.

Actual cumulative battery/swap behavior, full-model retention during a review wait, actual model cleanup and performance benefit remain unmeasured here. Each next chunk still requires its own exact scope/approval and contains at most eight frozen records. No training is authorized, no resource threshold is relaxed, and a helper PASS does not approve an MRI launch or validate scientific interpretation.

Regression-test SHA256: `0e300181fe57c9dfd05989844b2c4b79700b7f83d62285e46326cc988303eadc`.


## Resident runner integration and immutable-prefix adoption

Independent source/CPU verdict: **PASS_WITH_LIMITATIONS** for the integrated runner `894cab501b5cce6d29b404ab283a21099b46afd66d21f3e70565c42d9559e384`, adoption helper `6fba1884422be1c0544ba3d5f8f1ecf480989833b22f95ef976ce8a93c8b08d5`, and previously reviewed resident helper. Actual bridge files, resource policy version and launch scopes require the separate execution review; this is not launch approval.

The earlier proposal's instruction to repeat the first three records was rejected as inconsistent with the latest researcher direction. The integrated approach adopts only exact original prefix bytes and retains their original row identities. Its sidecar binds the original output hash, independent approval and actual-output audit, checkpoint, contract, backend, scoring sources and numerical-loop fingerprint. Every later record must use the current identity. The runner refuses an absent adopted output, modified prefix bytes, or old identities extending beyond the adopted region. No completed A0 record needs to be generated again merely because execution control changed.

The reviewer reconstructed the original pre-integration source from the frozen patch and verified its exact SHA256 `ac5cc67e1a0b0e81d14dbb9ec413ab3a6c3a9845eeaaee1e7cbc7e1dac5860a0`, then independently compared it with the integrated source. The complete per-record loop is AST-identical, fingerprint `fc51dfd7fe5998b05e723e27f57e703f0f1912ffd6dc836f6614f5be22b0cb31`. The first attempted comparison had observed the already-applied proposal at the original path; the corrected proof uses the reconstructed, hash-verified original, and both receipts preserve this distinction. Base/adapter loading, quantization arguments, serial materialization and exact adapter checking were also inspected as unchanged in the integration. AST equality is a source claim, not a newly executed numerical parity test.

A fresh import-blocked run of the resident, adoption and paced-likelihood tests produced **30 passed in 0.21 seconds**. Tests reject source/backend/precision/checkpoint changes, wrong prefix/approval/audit digests, modified loop fingerprints, nonprivate paths, changed prefix bytes and historical identities outside the adopted region. They verify preservation of original bytes and identity.

Source inspection confirms that the resident controller receives the already initialized live gate and the existing adapter checker; it does not construct a new gate per chunk. The chunk iterator is outside the output-file context, so completed rows are flushed and the handle closed before approval waiting. The `finally` block closes the iterator, deletes model/processor references, collects garbage, clears cache and records a release boundary. No model cleanup or resident resource behavior was executed by this reviewer.

Adoption intentionally permits execution-control source changes only with a new exact approval. The sidecar's numerical-loop fingerprint must correspond to independently inspected original and current sources; the helper's check of the current fingerprint alone is not a general proof that arbitrary excluded runner changes preserve numerics. The exact inspected integration has the separate source proof above. Actual historical approval/audit files and prefix selection remain the execution reviewer's responsibility.

The aggregate analyzer must likewise understand verified adopted-prefix lineage before consuming mixed-identity outputs; it must not solve this by rewriting historical identities or disabling provenance checks. No aggregate result is inferred from this integration review.


## Battery v3 duration-based warning policy — independent review

**Verdict: PASS_WITH_LIMITATIONS for source/CPU policy behavior.** Reviewed battery source SHA256: `403387fddd5ce67c0b97155185f86735ae6264a05bc34dc6532e1e4d1d9ad854`; battery-test SHA256: `985d70c59f3fb74f958389f0a417619412043a08d08eb954f1b79df5771a6b15`. This prospective change implements the latest researcher direction not to pause solely for an isolated loading-time rate warning. It does not relabel historical stopped runs or infer stable AC from an ambiguous power reading.

The independently inspected rule admits a sole swap-rate warning when other gates are healthy and logs `WARNING` with `admitted=true`. A same-phase episode needs at least two warning observations spanning at least 30 seconds to enter HOLD. Three rapid rate-only observations may raise the generic supervisor to WARNING; the gate permits that state only when its history attributes the episode solely to rate warnings. An unattributed WARNING or CRITICAL state, other warning reasons, or any hard gate remains a stop. Loading-phase classification for materialization and quantized-model readiness does not replace the existing unit headroom threshold with a different threshold. Phase changes reset only the warning streak, not the initial swap/battery baseline or accumulated history.

A fresh run produced **25 passed in 0.04 seconds**, with MLX, MLX-VLM, Torch and Transformers imports blocked. The reviewer also independently executed fake-sensor cases without importing the producer tests:

- Three rapid rate-only observations are admitted without additional sleeps, including the third observation's attributable supervisor WARNING.
- A sustained rate episode enters recovery, then stops after exactly three 15-second waits when it remains unresolved. The original initial sample remains the same object, and all seven samples remain recorded.
- Critical pressure, pressure warning, thermal throttling, another heavy process, battery at 65%, insufficient memory headroom and unknown swap readings all stop.
- Explicitly unattributed supervisor WARNING and CRITICAL states stop even when their current reported reason is a swap-rate warning.

Existing tests also verify the unchanged cumulative swap/battery limits, rapid battery drain, two clean samples for recovery, clean samples breaking the warning streak, and phase changes retaining the cumulative budget. No numerical scoring code or model weights changed in this policy revision. The admitted-warning condition is a deliberate authorization change; hard thresholds have not been raised.

Independent replay source and receipt are private under `likelihood_resident_independent_v1/battery_v3_independent_PRIVATE.py` and `battery_v3_independent_PRIVATE.json`. These are synthetic sensor tests, not a claim of observed safe runtime behavior or prevention of a kernel panic. The verdict was sent directly to the execution reviewer; actual bridge files, exact launch scope, fresh resource readings and bounded resident execution remain that reviewer's responsibility. This review adds no per-batch researcher confirmation requirement.
