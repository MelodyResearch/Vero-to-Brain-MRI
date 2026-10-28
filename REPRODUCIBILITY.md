# Reproducibility

## Verified result reproduction — CPU only

Use Python3.11+ and install the package/test dependencies:

```bash
python -m pip install -e '.[test]'
python scripts/reproduce_results.py --check results/FINAL_METRICS.json
pytest -q
```

The recount independently computes the four same-contract DEV24 results, class confusion/recall/invalids, paired outcomes, descriptive four-patient cluster intervals and the separate original22-model formal evaluation. It validates six immutable aggregate SHA256 inputs and rejects changed counts or nonfinite metrics. It does not load MRI, MLX, models or a GPU and does not reproduce inference by replaying an aggregate. `--output PATH` writes the same deterministic machine table; do not overwrite an existing published table without checking its diff.

The historical formal100 and repeatedly exploredDEV24 are separate estimands. Seeds do not add patients. The original formal test is not reopened for final model selection. No final independent16 result exists.

## Model and private runtime

The sole principal model is P_calibrated13 / Adam52 / boundary114. [Handoff](research/FINAL_MODEL_HANDOFF_V1.md) and [state/config hashes](results/FINAL_MODEL_ELIGIBILITY_REVIEW_V1.json) identify the exact adapter, optimizer/RNG, model/cache and processor. Private assets are not distributed in Git. An authorized recipient needs those files and legal access to source data; a checkout alone cannot run a model.

The retained native compatibility modules preserve the tested historical F32 preprocessing/generation and detached full-state restore code. Some import dependencies include old training helpers: they remain because whole-module imports and source hashes are part of the verified backend. They are not current training entry points or permission to rerun experiments. Obsolete external launch scripts/configurations/tests are removed by the cleanup PR. The package exports only the final result recount and final read-only evaluator.

The actual historical MLX runtime lives in private local storage. The metadata-only runtime snapshot is `configs/final_native_environment_v1.json` (Python3.13.6, MLX0.32.4, MLX-VLM0.7.8). Before future inference, the exact interpreter/package versions, complete source closure and all parent/cache/processor payloads are checked against the reviewed scope. No MLX module was imported to collect this metadata. No dependency upgrade, cache regeneration or alternate quantization is permitted as a silent substitution. CPU tests of a wrapper are not a new native probability-parity PASS.

## Once-only independent evaluation — currently blocked, no GPU authorization

```bash
python scripts/evaluate.py --validate-only
```

The expected current outcome is exit2 with `NOT_RUN_INPUT_METADATA_NOT_LOCATED_AND_CONFIRMATION_REQUIRED`, before any MLX import. The historical sealed plan, rowmanifest and seal hashes are known; current execution metadata and per-pair quality map were not located at bounded known paths. Do not create substitute patients, replace quality-flagged cases or regenerate data.

After the original metadata is restored and verified, prepare a private input manifest containing all16people/32images, original54-person exclusion inventory and allfour quality flags. Freeze the Parent, originalConcise/F32/strictgreedy/256-token contract, source/runtime/inputs hashes and the exact resource scope. Obtain independent review and the researcher's explicit sealed16 confirmation. A local JSON or CLI flag is not human authorization.

Only then can `scripts/evaluate.py --execute` receive the exact private `--inputs`, `--runtime`, `--scope`, `--review`, `--approval` and once-only `--output` paths. The scope reserves1serialload,32greedy/8192returnedtokens,0TFscores,0backwards and0updates, with1800seconds including initial validation/waits; the guardian requests release at the deadline with at most20seconds signal grace and sampling latency. AC is required by the proposed final read-only scope; an old SFT battery exception is not reused. GREEN/YELLOW/RED, RSS14GiB, MLX12GiB/cache256MiB, disk26GiB, battery20% and serious-risk protections remain. The guardian handles only its own child process. No retry of an incomplete/STOP operation, other-app termination, shutdown/reboot or system protection change is allowed. Atomic GPU calls retain monitoring blind spots.

The new final wrapper is CPU-tested but **NOT_NATIVE_QUALIFIED**. Missing input metadata means its final scope is not execution-ready. If it cannot be run compliantly, publish NOT_RUN and retain the completed development result. Do not delay project delivery indefinitely.

## Verification after cleanup

Run the full retained CPU suite, import/dependency checks, saved-result recount, paper/table check, source-hash validation, Markdown links and deleted-file reference checks. Exact old code/results remain at immutable Git commits in the deletion manifest; no branch ref or history is deleted. Raw local weights, data and receipts are outside cleanup scope. See [evidence index](research/EVIDENCE_INDEX.md).
