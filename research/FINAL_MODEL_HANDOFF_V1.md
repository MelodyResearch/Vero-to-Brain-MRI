# Final model handoff and independent-evaluation boundary

Status: **PARENT_RETAINED; NO_NEW_ACCURACY_GAIN; SEALED16_NOT_OPENED** (2026-10-28).

## Model identities

- Retained same-contract DEV winner: P_calibrated13 / Adam52 / boundary114; adapter SHA256 `c70b426b58191446545ba6333f7595e96f4fdb4cc689ddbaf5206e276b329db6`.
- Preserved rejected SFT endpoint: seed0 / Adam80; adapter SHA256 `7a6c414bae6790f88fd002bbcdc68a887aac5ecda29108bddb7971bbe4f683e8`.
- Base historical GPU-cache shard SHAs: `6a332d4977c95fd6b51ac7155e039016d90041da1bf3cdd3e500192ea00fa626`, `2c94dc6761ab4641a4e2c612eb3ff5fd62d939c5704f8301025e741650b9e1e8`.
- The new supervised endpoint is not a GSPO improvement, not a clinical model, and not recommended instead of the Parent.

## Reproduction

Preserve the pinned source checkout, existing exact cached base, processor/model metadata, adapter, full Adam and Python/MLX RNG, manifest, input hashes, per-pair receipts, final cursor, evaluation rows, guardian logs and release/exit receipts. Private assets remain local; Git contains safe aggregates/hashes only. The private runtime index is under `local_data/final_accuracy_sprint_v1/`; the completed run is its `technical_entry_repair_v2/seed_0_PRIVATE/` subtree. Do not copy an adapter without its base/config/processor or represent it as standalone weights.

The exact runner is intentionally one-shot. An old consumed scope cannot authorize rerunning it. Reproducing on another device requires legal data access, a new resource review and an explicit new execution scope, not deleting the existing claim/output directory. CPU checks can run with `PYTHONPATH=src python3 -m pytest -q tests/test_final_balanced_sft_v1.py tests/test_final_accuracy_sprint_v1.py`.

Model outputs and patient records must not be shared externally without permission. An independent recipient currently needs the researcher's authorized private payload handoff; the Git checkout alone is insufficient for native reproduction. No backup/transfer is implied by this index.

## Proposed once-only sealed16 evaluation — NOT AUTHORIZED / NOT RUN

For researcher confirmation only: freeze the retained Parent adapter above as the sole final performance endpoint. Include all16 previously sealed people and all32 existing paired images, including the four quality-flagged pairs; no output-based exclusions. Use the exact existing concise sequence prompt, F32, processor, strict parser, greedy temperature0 and256-token cap. Report strict/balanced accuracy, each class recall, invalids, all16 patient paired scores and patient-cluster uncertainty. Preserve quality flags and report their predefined stratum descriptively; primary denominator remains32.

Selection rule: no model, prompt, normalization, target, seed or training change after seeing the sealed outputs. This evaluation estimates the retained Parent's performance; it cannot demonstrate that the failed SFT improved it. Maximum proposed cost: one serial model load,32 greedy answers/8192returned-token cap,0backward/0updates. Exact token/source binding, resource budget and independent execution review must be frozen after the researcher confirms this final endpoint/protocol. This document is neither a runnable scope nor execution permission.

No independent result exists yet. Training and cleanup cannot be called a successful accuracy-improvement study while this remains unmeasured. The negative DEV result is complete and independently verified; further training is stopped.
