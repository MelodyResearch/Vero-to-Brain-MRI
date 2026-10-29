# Reproducibility

## Current maintained entry points — CPU first

```bash
python -m pip install -e '.[test]'
python scripts/reproduce_results.py --check results/FINAL_METRICS.json
python -m pytest -q
python scripts/evaluate.py --validate-only
python scripts/check_research_plan_guard.py --help
```

Use Python 3.11+ with the declared test dependencies. The recount computes the four same-contract DEV24 results, confusion/recall/invalids, paired outcomes, descriptive four-patient cluster intervals and separate original 22-model formal evaluation. Six immutable aggregate SHA256 inputs are checked; changed counts or nonfinite metrics are rejected. This is saved-evidence reproduction, not regenerated model inference. `final_results.recount_saved_responses` retains the original pure class/format/cost recount for saved response records. The six source JSON files and published `FINAL_METRICS.json` are unchanged by simplification.

The maintained package contains seven files: `final_evaluation`, `final_results`, `historical_runtime`, `research_plan_guard`, `rollout_precision_v2`, `vero_mri_reward`, and `__init__`. Default pytest collection stays inside `tests/` and excludes private historical scripts and hidden worktrees, including explicit `.` collection. Tests never need private MRI or model assets.

## Explicit historical native-engine dependency

The final evaluator's native call graph imports 15 engine modules directly, with an 89-module historical dependency closure. Rather than maintain their retired training/debug implementations on current main, the evaluator now requires an explicitly exported **90-file source dependency** (including package initializer), pinned to commit `1e0d7be1bb099b42e717eb4eb081facc12607ed3` and per-file SHA256 in the [single manifest](results/SIMPLIFICATION_MANIFEST.json).

```bash
# Choose a NEW directory outside private asset directories. CPU Git export only.
python scripts/evaluate.py --export-runtime /absolute/new/readonly-engine
```

This exports only reviewed Python blobs from existing local Git objects. It does not fetch, load MLX, open patient data or replace existing files. Missing commit objects, changed/extra files and symbolic/special files fail closed. A shallow clone lacking the commit must obtain that exact historical Git object explicitly; no automatic alternative revision is accepted.

A fresh native execution scope must name `historical_runtime_root` and pin **every actual exported source path** in `file_sha256`, as well as all maintained source files, the simplification manifest, evaluation script, protocol and environment configuration. The evaluator verifies them before activating the source dependency. Its `inspect`/AST-based loader, exact F32 generation/scoring, detached Adam/RNG restore and resource controller use unchanged historical source bytes. The two maintained reward/precision files that overlap this dependency are byte-identical. Import caches from another engine checkout are rejected.

**This reduces maintained source; it does not make native inference a standalone seven-file implementation.** Source packaging, imports, dynamic source reads and CPU numerical fixtures are verified. The changed packaging is **CPU_VERIFIED_ONLY / NOT_NATIVE_QUALIFIED**; no native model, inference or numerical-equivalence run was performed for this cleanup. A new scope cannot reuse an old authorization or claim a new native PASS from these tests.

## Historical functionality and restoration

Current entry points do **not** support training, cache reconstruction, optimizer ablation, historical experiment launch, raw original-22 analysis orchestration or figure generation. The source-only dependency above is limited to the read-only evaluation graph; it is not a complete historical training checkout.

Every removed file has its original path, size, SHA and reason in the manifest. All 138 removals are recoverable from the [pre-simplification source tree](https://github.com/MelodyResearch/Vero-to-Brain-MRI/tree/1e0d7be1bb099b42e717eb4eb081facc12607ed3). For one file:

```bash
git show 1e0d7be1bb099b42e717eb4eb081facc12607ed3:src/mri_rl/safe_micro_restore_v1.py
```

For original raw analysis/plot CLIs and their tests, use that fixed full checkout in a separate authorized workspace, not the current package. Actual final SFT implementation/protocol is pinned separately in the [results report](research/FINAL_ACCURACY_RESULTS_V1.md). Historical files describe what ran at the time; current refactoring is never represented as the original implementation. No Git history or branch ref is deleted.

## Model, environment and private inputs

The principal model is P_calibrated13 / Adam52 / boundary114. [Model handoff](research/FINAL_MODEL_HANDOFF_V1.md) and [state/config hashes](results/FINAL_MODEL_ELIGIBILITY_REVIEW_V1.json) identify adapter, full Adam/RNG, exact base/cache and processor. A Git checkout alone cannot run a model. Private assets were not read, moved, rehashed or cleaned in this round.

The metadata-only runtime snapshot is `configs/final_native_environment_v1.json`: Python 3.13.6, MLX 0.32.4, MLX-VLM 0.7.8. Future native execution must validate exact package/interpreter and payload identities. No dependency upgrade, alternate quantization or cache rebuild can silently substitute for them.

## Once-only independent evaluation — blocked, not authorized

Default `evaluate.py --validate-only` returns exit 2 with `NOT_RUN_INPUT_METADATA_NOT_LOCATED_AND_CONFIRMATION_REQUIRED`, before native imports. Original sealed metadata and four-pair quality membership were not located in the previously recorded bounded search. That is not proof of permanent loss. See the handoff for the existing private location-check receipt; no recovery search or model output was opened during cleanup.

After original metadata is recovered, an authorized manifest must include all 16 people/32 images, original 54-person exclusion inventory and all four quality flags. Freeze exact Parent, Concise Prompt/F32/strict greedy/256-token protocol, source/runtime/input hashes and independent review. Obtain the researcher's explicit sealed16 confirmation; a CLI flag or local JSON is not human permission. No replacement patients, later checkpoint selection or output-based exclusions are allowed.

Only then may `evaluate.py --execute` receive verified `--inputs`, `--runtime`, `--scope`, `--review`, `--approval` and once-only `--output` paths. Existing limits remain: one serial model load, 32 greedy responses/8,192 returned tokens, zero TF scoring/backward/updates; 1,800 seconds including initial validation/waits, deadline guardian with at most 20 seconds signal grace plus sampling latency. AC, global lock, GREEN/YELLOW/RED, RSS 14 GiB, MLX 12 GiB/cache 256 MiB, disk 26 GiB, battery 20% and serious-risk protections remain. Only the experiment's own child may be terminated. No incomplete-operation retries, other-app closure, shutdown/reboot or system-protection changes. Atomic calls retain monitoring blind spots.

The historical formal100, explored DEV24 and sealed16 are separate estimands. Seeds do not add patients. Independent16 remains NOT_RUN; cleanup does not constitute independent generalization or a new accuracy result.
