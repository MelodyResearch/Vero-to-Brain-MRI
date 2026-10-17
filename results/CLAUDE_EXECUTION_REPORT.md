# Claude Code execution report (DRAFT pending the final Codex Step D verdict)

**Author/role:** authentic Claude Code session 94ff5023 (model claude-sonnet-5-5), implementation/execution lead. Codex coordinated and independently reviewed; counterexample and review files under `results/GSPO_*` were authored by Codex/its reviewer, not by Claude. `CROSS_MODEL_REVIEW` applies only to what Codex's final verdict states.

## State (OBSERVED)
- Likelihood saved records: **A28 42/57, B28 3/57, A0 3/57**. Unchanged by this session; nothing completed was regenerated.
- Battery policy: reserve-20-only (commit 252ba1a) and mixed-prefix adoption (dc53506) were already in place and focused-reviewed before this session; I did not reimplement them.
- **My A28 launch (from record 43) stopped during model loading** at `materialized:960`: swap +1.487666 GiB against the unchanged 1 GiB stage cap (MLX active ~1.94 GiB, available 6.47 GiB, pressure-free 36%, AC 100%). No new rows were written; the model released cleanly. Global telemetry does not identify the cause of the swap growth (another process or page eviction is only a plausible hypothesis). Allocated swap is not a fixed maximum. Existing 8 GiB load floor and 1 GiB cap are unchanged; the 13.68 GiB earlier start is history, not a threshold. Last measured headroom ~5.8-6.9 GiB, below the floor.
- No likelihood comparison, no GSPO update, and no dev score exist. Nothing here is a scientific result.

## Software delivered (IMPLEMENTED, CPU/stub-tested; native Qwen NOT_RUN)
- `continuous_rl_pilot.py`: finite Tier B GSPO pilot, `train` and `reload`. Fixes driven by Codex counterexamples: total optimizer-update cap (2 informative groups previously gave 2 updates; now 1, enforced before rollouts and before backend construction on resume); `--resume` using persisted Adam/RNG/cursor; reload approval bound to the exact updated boundary digest; guards, generation and `close()` inside one try/finally; image re-hash after the resource callback before generation and after; full reload text/token ids saved privately.
- `mlx_continuous_gspo.py`: serial parameter materialization shared with the successful read-only loader (fixture-tested only), `greedy_generate`, `generate_record`.
- `continuous_rl_dev_eval.py`: adapter-agnostic serial greedy dev evaluation; scope binds the full model-file inventory (weights included), reviewed per-record image hashes, starting output hash, seed, manifest, subset and adapter; mutations after review are rejected before backend construction.
- Tests at 4a1bf3e: dev-eval 9 passed; dev-eval + pilot 19 passed; an earlier driver+pilot+dev-eval replay gave 34 passed. Opt-in native tiny test not rerun after edits.

## Limits and open items
- Adapter export handoff (boundary has no `adapter_config.json`): distinct private export dir with byte-exact original config; documented in the request file, NOT_RUN until an updated boundary exists.
- Residual window between the model-inventory hash and the native load (no file lock).
- Baseline reuse for B28 dev rows: NOT_VERIFIED pending Codex's check of prompt/format/precision/adapter/model equivalence.
- Task is acquisition-label sequence recognition (ENGINEERING_ONLY). Codex's CPU technical visibility check is PASS_WITH_LIMITATIONS only; clinical blind assessment NOT_VERIFIED. Dev set = 54 questions / 4 previously used patients; reserved patients excluded. No generalization or visual-grounding claim.
- Local MLX 4-bit LoRA diagnostic profile, not official veRL parity.

## Next (reproducible continuation)
1. When headroom meets the existing floor and Codex releases the scoped launcher: `sh local_data/ds007045_bounded/likelihood_resident_v4/reserve20_review/launch_A28_050.sh` (cursor: A28 record 43), then `launch_B28_011.sh` (record 4), then `launch_A0_011.sh` (record 4). Do not rerun completed records.
2. Only after likelihood: paired analysis, then the GSPO pilot and dev evaluations under separate scoped approvals.
