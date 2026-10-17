# Continuous GSPO pilot: narrow independent review

**Verdict: FAIL for the new generic pilot interface at commit `2158dde`.** Reviewer: independent Codex agent `/root/likelihood_finish_reviewer`, not Claude. This review covers the new launcher and reload-generation addition only; it does not repeat the prior continuous-driver audit or authorize model training.

The new pilot/backend focused tests pass: **9 passed**, CPU only with no model run. Passing those tests did not establish the two missing invariants below. The preserved likelihood counts remain42/3/3; no RL result is claimed.

## Confirmed defects

1. **The declared one-update budget is not enforced.** Pilot preflight advertises `optimizer_steps:1`, while `train` passes `max_new_groups=len(schedule)` to the existing driver. That driver bounds attempted groups and can update once per informative group. A valid two/three-record schedule can therefore execute more than one update under a one-update Tier B approval. Source inspection independently identified this before receiving the coordinator's matching reproduction. The coordinator's synthetic driver counterexample is saved in `GSPO_PILOT_CODEX_COUNTEREXAMPLE.json`. Restrict this pilot to one attempted group or implement an explicit total-update cap that still preserves inactive-group semantics; claiming a budget in preflight is insufficient.

2. **Reload generation does not protect the frozen image after its resource gate.** An independent synthetic CPU reproduction calls the actual new `greedy_generate` method with a resource callback that changes the image bytes. Generation reads the changed bytes and returns success. The new method checks prompt placeholders but never compares the image with the expected boundary-bound hash. Evidence: `GSPO_RELOAD_IMAGE_CODEX_COUNTEREXAMPLE.json`; no native imports or real image payload are involved. Bind the expected image hash from the validated training boundary and reject changes after the resource callback, immediately before generation and afterward.

## Incomplete reload evidence

The reload branch receives decoded text and token IDs but discards them. Its saved receipt contains only generation-token count and prompt hash. That receipt cannot support subsequent verification of the actual answer, strict format, final-answer extraction or truncation. Save the actual private response/token IDs and finish metadata, with image/prompt/boundary bindings; do not infer successful MRI answering from an exact adapter reload assertion.

## Resource and scope limits

The new launcher correctly performs contract/review preflight before constructing the backend and retains the training AC resource gate; the20% likelihood exception is not applied to GSPO. Output is confined to ignored local_data and source hashes bind the pilot/driver/backend. The driver persists optimizer/RNG/sampler lineage through its previously reviewed atomic boundaries.

The native backend still uses bulk `mx.eval(self.model.parameters())`; the current real-MRI likelihood attempt already stopped on its unchanged stage swap cap under lower available memory. No native constructor/update/reload success was executed for this new launcher. Even after the functional defects are corrected, launch admission and real numerical gradient/update/reload evidence remain **NOT_VERIFIED**. This engineering review cannot establish Vero parity, MRI learning or patient generalization.


## Final Step D — corrected source `4a1bf3e414b75163af6a6b6dd88fcce46c1ef50e`

**Verdict: PASS_WITH_LIMITATIONS for the reviewed source and CPU interfaces only. Native launch: NOT_APPROVED; actual Qwen GSPO update/reload/dev evaluation: NOT_RUN.** Later commit `c462f2a` changes documentation only. The original FAIL findings above remain preserved as historical evidence.

The actual focused command completed with **43 passed in 0.28s**, without a native model import or 8B launch:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_continuous_rl_pilot.py tests/test_mlx_continuous_gspo.py tests/test_continuous_rl_driver.py tests/test_continuous_rl_dev_eval.py
```

The total optimizer-update cap now applies before the next rollout and before constructing a resumed backend. Inactive groups do not consume update allowance. The original image-mutation counterexample is rejected after resource callbacks; cleanup executes when a post-constructor guard fails. Reload retains private decoded text/token IDs and binds the approved updated boundary. Native initialization actually calls serial materialization. The pilot source inventory includes that helper (9 source files); the dev evaluator inventory contains 8. Both inventories were compared with the final private candidate and remain unchanged. `GSPO_DEV_BINDING_STEPD_4A1BF3E.json` records exact hashes and test scope.

The corrected dev evaluator rejects the original wrong-model, changed-image and changed-starting-output counterexamples before backend construction. Model artifacts/config/template, seed, resume mode, existing output and reviewed pixel hashes are bound to review. These checks used tiny synthetic fixtures and wrong-directory rejection; this review did not hash the real base weights or establish actual model numerical parity.

The first candidate remains one prospectively selected FLAIR record from the frozen pair, initialized from B28, two rollouts, at most one total update. T1 remains a separate future stage. The frozen CPU preflight validates 382 train rows/28 patients, 54 internal-dev rows/4 patients, and disjoint patients. Exact reconstruction of both selected saved PNGs from their NIfTI sources, renderer caption placement, empty PNG metadata and actual native prompt tokens support a limited engineering acquisition-label diagnostic. Clinical expert blind visibility remains NOT_VERIFIED; this is not clinical task validation. See `GSPO_FROZEN_PAIR_TECHNICAL_VISIBILITY.json`.

The private export helper is an executable, syntax-checked template using the exact B28 adapter configuration and a future independently hash-verified changed one-update boundary. It writes a separate export and does not alter the atomic training boundary. Guarded train/dev shell templates pass `sh -n`; they require actual independent verdict paths/digests and the dev template also requires a real export receipt. No updated weights exist, so export and model evaluation are NOT_RUN. The preserved matched54 dev output has a different adapter fingerprint from this candidate's selected B28 initializer; baseline reuse is explicitly NOT_VERIFIED and NOT_APPROVED. Source prompt flags align (`enable_thinking=False`, native assistant generation prefix), but no new numerical equivalence is inferred.

`GSPO_FROZEN_FLAIR_CANDIDATE_READINESS_FINAL.json` records final readiness. Heavy execution remains blocked by actual memory headroom: the coordinator's latest sample was about 5.879GiB available, below the existing 8GiB floor, with pressure-free33%, AC100% and swap4.910GiB. This is an observed blocked state, not a new gate. No launch verdict was issued. The earlier A28 loading attempt stopped on the retained stage swap budget and added no outputs; likelihood progress remains A28 42/57, B28 3/57, A0 3/57. Prior partial comparisons remain preserved; no new or complete57-row comparison exists.

This PASS_WITH_LIMITATIONS does not prove a real gradient/update, valid native checkpoint reload, improved MRI understanding, patient generalization, or completion of Vero-to-MRI transfer.
