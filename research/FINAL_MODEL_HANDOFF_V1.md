# Final model handoff and independent-evaluation boundary

Status: **PARENT_RETAINED; NO_NEW_ACCURACY_GAIN; SEALED16_NOT_OPENED** (2026-10-28).

## Model identities

- Retained same-contract DEV winner: P_calibrated13 / Adam52 / boundary114; adapter SHA256 `c70b426b58191446545ba6333f7595e96f4fdb4cc689ddbaf5206e276b329db6`.
- Preserved rejected SFT endpoint: seed0 / Adam80; adapter SHA256 `7a6c414bae6790f88fd002bbcdc68a887aac5ecda29108bddb7971bbe4f683e8`.
- Base historical GPU-cache shard SHAs: `6a332d4977c95fd6b51ac7155e039016d90041da1bf3cdd3e500192ea00fa626`, `2c94dc6761ab4641a4e2c612eb3ff5fd62d939c5704f8301025e741650b9e1e8`.
- The new supervised endpoint is not a GSPO improvement, not a clinical model, and not recommended instead of the Parent.

## Reproduction

Preserve the pinned source checkout, existing exact cached base, processor/model metadata, adapter, full Adam and Python/MLX RNG, manifest, input hashes, per-pair receipts, final cursor, evaluation rows, guardian logs and release/exit receipts. Private assets remain local; Git contains safe aggregates/hashes only. The private runtime index is under `local_data/final_accuracy_sprint_v1/`; the completed run is its `technical_entry_repair_v2/seed_0_PRIVATE/` subtree. Do not copy an adapter without its base/config/processor or represent it as standalone weights.

The exact runner is intentionally one-shot. An old consumed scope cannot authorize rerunning it. Reproducing on another device requires legal data access, a new resource review and an explicit new execution scope, not deleting the existing claim/output directory. Historical SFT training reproduction is pinned to the [original research source checkout](https://github.com/MelodyResearch/Vero-to-Brain-MRI/tree/b03d0b30a6f0de59f23ae2eb76a5b9563364fa06); this is historical provenance, not permission to train again. The current results delivery includes the final CPU recount and gated read-only evaluator described in REPRODUCIBILITY.md. Historical CPU checks in the pinned checkout can run with `PYTHONPATH=src python3 -m pytest -q tests/test_final_balanced_sft_v1.py tests/test_final_accuracy_sprint_v1.py`.

Model outputs and patient records must not be shared externally without permission. An independent recipient currently needs the researcher's authorized private payload handoff; the Git checkout alone is insufficient for native reproduction. No backup/transfer is implied by this index.

## Final model and processor configuration

Qwen3-VL-8B, historical exact 4-bit base cache, rank4/alpha8 language LoRA, frozen vision, original processor/template and Concise Prompt. Full adapter, Adam52 and Python/MLX RNG are present and independently authenticated as serialized state; no native restoration was repeated for this closeout. Model/cache config, adapter config, tokenizer, chat-template, preprocessor and shard identities are pinned in [final model eligibility](../results/FINAL_MODEL_ELIGIBILITY_REVIEW_V1.json). Do not substitute an upstream revision or independently quantized cache.

- Adapter config SHA: `032becabb7cbb6d2af6f3451eb127e5f944551d42eb51eaee3e8d75b839b2f6a`.
- Tokenizer SHA: `a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7`.
- Preprocessor SHA: `27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516`.

## Proposed once-only sealed16 evaluation — NOT AUTHORIZED / NOT RUN

For researcher confirmation only: freeze the retained Parent adapter above as the sole final performance endpoint. Include all16 previously sealed people and all32 existing paired images, including the four quality-flagged pairs; no output-based exclusions. Use the exact existing concise sequence prompt, F32, processor, strict parser, greedy temperature0 and256-token cap. Report strict/balanced accuracy, each class recall, invalids, all16 patient paired scores and patient-cluster uncertainty. Preserve quality flags and report their predefined stratum descriptively; primary denominator remains32.

Selection rule: no model, prompt, normalization, target, seed or training change after seeing the sealed outputs. This evaluation estimates the retained Parent's performance; it cannot demonstrate that the failed SFT improved it. Maximum proposed cost: one serial model load,32 greedy answers/8192returned-token cap,0backward/0updates. Exact token/source binding, resource budget and independent execution review must be frozen after the researcher confirms this final endpoint/protocol. This document is neither a runnable scope nor execution permission.

No independent result exists yet. Training and cleanup cannot be called a successful accuracy-improvement study while this remains unmeasured. The negative DEV result is complete and independently verified; further training is stopped.

## Current readiness and handoff boundary

The original sealed plan/rowmanifest/seal/quality map are not currently located at the bounded known private paths. Published historical hashes are preserved in the eligibility review; they do not establish present executable availability. No replacement data is permitted. Final status is NOT_RUN_INPUT_METADATA_NOT_LOCATED_AND_NOT_UNSEALED. The CPU-validatable final protocol is `configs/final_evaluation_v1.json`; `scripts/evaluate.py` cannot execute without verified original inputs, exact independent scope review and a fresh confirmation tied to the selected Parent. Its new wrapper is CPU tested, NOT_NATIVE_QUALIFIED. The reused native dependencies remain byte-identical to the preserved implementation.

The main deliverable is Parent. The rejected Adam80 checkpoint and historical GSPO/SFT endpoints remain private baselines/negative evidence, not alternate current candidates. This repository provides source/evidence and a private handoff index; it is not a public weight release or a completed independent clinical validation.

## Actual storage and recovery follow-up — 2026-10-29

All36authenticated Parent/base/config/processor/certificate/contract files were rehashed from current private storage:36/36match, including both base shards, adapter, complete optimizer state and receipt. The optimizer manifest still binds Adam52/boundary114 with504trainable arrays; its serialized layout contains both Python and MLX RNG. No model or MLX module was loaded.

Private paths are relative to the canonical repository root:

- Parent/full Adam/RNG: `local_data/c_two_task_pool_v1/tier_c_v1/calibrated_bundle_v1/P_calibrated_seed13/run/boundary_0114/`.
- Certified base shards/cache: `work/accuracy-recovery-historical-gpu-cache-v4/cache/`; certificate is its sibling `certificate.json`.
- Original processor/model metadata: `local_data/models/qwen3vl8b-base-recovered-v1/`.
- Existing runtime index: `local_data/final_accuracy_sprint_v1/RUNTIME_PRIVATE.json`.
- Actual per-file paths, bytes and expected/observed SHAs: `local_data/final_accuracy_sprint_v1/HANDOFF_ASSET_MANIFEST_20261029_PRIVATE.json`, SHA256 `544320420bbb3e6208898c6d769e733ea0764d2b3c435ae180b9efac9cdef773`. No payload is uploaded or copied to GitHub.

The original holdout plan, row manifest, seal, exclusion list and four-pair QA membership were not found after enumerating current project local_data, shared research local_data and the two existing runtime/work roots, following historical collector/receipt references, checking all-ref Git object paths, and hashing5994candidate metadata files (including possible renamed files) against original evidence hashes. No errors or target matches were observed. The only archive encountered was an unrelated hippocampus source archive; it was not substituted. The private location-check receipt SHA256 is `f9c895ba95091ef0a3ddce53fb056a206761910d8c6d47de1f0dfc3fc00fbbf1`. This is a bounded unsuccessful recovery attempt, not a declaration of permanent loss. All independent outputs remain NOT_RUN; no replacement/download/rerender or selection occurred.

Results PR #6 and cleanup PR #7 have already merged. Remaining wording updates and this handoff recheck are a small delivery correction; they do not change model selection, metrics, frozen research plan or native authorization.
