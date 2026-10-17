# Claude Code execution report (FINAL for this session)

**Mission alignment (covenant):** genuine verifiable-RL Vero-to-MRI transfer is the main question. This session produced software readiness only. Likelihood diagnosis (A0/A28/B28) is the current priority and is incomplete; B (supervised control) and C (task/data design) remain separate tracks; the task used for engineering is acquisition-label sequence recognition, ENGINEERING_ONLY.
**Evidence labels:** OBSERVED = measured; REPRODUCED = independently recomputed by Codex; CPU-TESTED = stub/synthetic tests; NOT_RUN = native Qwen never executed.

**Author/role:** authentic Claude Code 2.1.302, session 94ff5023 (claude-sonnet-5-5), implementation/execution lead. Counterexamples, Step D reviews and readiness files under `results/` were authored by Codex and its reviewer, not Claude. Independent review is Codex only; a second Claude agent is not cross-model.

## What did not happen (OBSERVED)
- **Zero new model forwards, zero optimizer updates this session.** Likelihood saved records: **A28 42/57, B28 3/57, A0 3/57**. Next cursors: **A28 record 43, B28 record 4, A0 record 4**. Nothing completed was regenerated.
- My A28 launch stopped while loading at `materialized:960`: swap +1.487666 GiB over the unchanged 1 GiB stage cap (MLX active ~1.94 GiB, available 6.47 GiB, pressure-free 36%, AC 100%); no rows written, model released. Global telemetry does not identify the cause (another process or eviction is only a hypothesis). Allocated swap is not a fixed maximum.
- Battery policy is reserve-20-only for read-only likelihood inference (already implemented and focused-reviewed before this session); it grants no training exception.
- Physical gate: 8 GiB initial load floor and 1 GiB stage cap unchanged. Fresh samples this turn: ~5.9 GiB available (computed from vm_stat), swap 5.03 of 6.14 GiB allocated, AC 100%, charged. Floor unmet; heavy work paused. No new threshold and no blind retry.

## Partial scientific results (prior data, not produced this session; 391 aggregate fields REPRODUCED by Codex from raw vectors, not by Claude)
- **A28, 20 complete training-patient pairs (partial 42-record prefix):** strict 24/40 (FLAIR 20/20, T1ce 4/20; all 16 T1 errors are FLAIR); full format 40/40, all stop. Mean-token pair separation 0.836744 (15 positive / 5 negative; median 0.212748). Both class mean margins still favor FLAIR (-1.460737 FLAIR image, -0.623993 T1 image). The balanced real-minus-no-image average equals half the separation algebraically, so it is not independent corroboration.
- **Only one pair has all three checkpoints (first z45 pair):** separation A0 0.599056, A28 0.549963, B28 0.542158. A28->B28 mean margin shifts toward T1 by +0.607633 (FLAIR image), +0.599828 (T1 image), +0.659703 (no image); greedy labels unchanged (both FLAIR). A0 failed strict format (0/2).
- **Diagnosis, no extrapolation:** conditional candidate preferences changed between A28 and B28, but the no-image shift is at least as large as either real-image shift and separation did not improve, which is compatible with a shared class-prior shift. It does NOT establish improved visual discrimination, and one pair supports no cohort claim. All inputs are training-source patients with prior panel exposure: no independent-patient generalization is claimed. The 57-row three-checkpoint comparison does not exist yet.

## Software delivered (IMPLEMENTED + CPU-TESTED; native Qwen NOT_RUN)
Frozen at source commit `4a1bf3e` (Codex Step D: **PASS_WITH_LIMITATIONS, source/CPU only; 43 focused tests passed**; command in `GSPO_FROZEN_FLAIR_CANDIDATE_READINESS_FINAL.json`). The historical FAIL at `2158dde` (update cap) is preserved as history.
- `continuous_rl_pilot.py`: Tier B GSPO pilot (`train`, `reload`) with total optimizer-update cap (Codex counterexample: 2 informative groups gave 2 updates; now 1, checked before rollouts and before backend construction, including resume), persisted Adam/RNG/cursor resume, reload bound to the exact updated boundary digest, guards+generation+close in one try/finally, image re-hash after the resource callback and after generation, full reload text/token ids saved privately.
- `mlx_continuous_gspo.py`: serial parameter materialization (shared helper with the successful read-only loader; fixture-verified only), `greedy_generate`, `generate_record`.
- `continuous_rl_dev_eval.py`: adapter-agnostic serial dev evaluation; scope binds whole model-file inventory, reviewed per-record image hashes, starting output hash, seed, manifest and adapter; Codex's wrong-model / changed-image / changed-output counterexamples reject before backend construction.
- Codex reviewer prepared (not Claude-written, not read by Claude) an adapter export script and guarded launch shells in the private candidate dir. The export copies the future updated adapter plus the exact B28 config without mutating the boundary. py_compile/sh -n only; NOT_RUN.

## Remaining limits and blockers
- **A source verdict is not a launch approval.** No fresh exact launch digest is issued; no GSPO, reload or dev-eval job is authorized.
- The candidate (one frozen FLAIR training record, 2 rollouts, <=1 update, B28 initializer) needs a nonzero reward-variance group; no gradient will be forced. Clinical blind visibility NOT_VERIFIED (technical reconstruction/prompt controls only).
- Saved 54-question baseline has a different adapter fingerprint from the B28 initializer: reuse NOT_VERIFIED / NOT_APPROVED; no rerun or search done.
- Residual window between model-inventory hash and native load (no file lock). Local MLX 4-bit LoRA profile, not veRL parity. Dev set: 54 questions / 4 previously used patients; reserved patients excluded.

## Reproducible continuation (private paths stay in `local_data`, not in Git)
1. If fresh headroom reaches the existing floor and Codex releases the scoped launcher: `sh local_data/ds007045_bounded/likelihood_resident_v4/reserve20_review/launch_A28_050.sh` (record 43), then `launch_B28_011.sh` (record 4), then `launch_A0_011.sh` (record 4).
2. Then paired analysis of the full 57-row comparison; only afterwards the GSPO pilot, reload and dev evaluations, each with its own exact approval.
3. Nothing continues in the background after this session ends.
