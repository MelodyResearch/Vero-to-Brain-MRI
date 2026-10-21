# Historical GPU cache v2 — prospective runtime scope (NOT executed, NOT authorized)

Runtime-dependency reconstruction only: no new model, no scientific experiment, no Adam/MRI. A/B stays blocked until the
historical cache is reproduced exactly (both shard SHA-256, config, tokenizer, index equal to the pinned cache certificate).
Draft 054ed61 was NOT_APPROVED by independent review (admission failures); this revision fixes them. It has still never run on MLX.

## Why
The CPU rebuild (`cpu_failure_001.json`) kept 406/406 retained units byte-exact but 0/344 quantized payloads matched the pinned
historical 750-unit proof. CPU-vs-historical-path mismatch is confirmed; a CPU-vs-GPU `mx.quantize` arithmetic cause is only INFERRED.
The failure message `Cache index differs` came from a separate, already-fixed checker bug (18c8e06: sorted `biases,scales,weight`
header order, which moved the 4 GiB shard boundary). That fix alone cannot certify the CPU cache. The CPU root is preserved and only READ.

## Exact scope
| item | value |
|---|---|
| units | 750 = 406 retained (copied + authenticated, 0 GPU operations) + 344 quantized on GPU |
| GPU unit attempts | max 344 |
| direct `mx.quantize` | max 344 |
| module-path `nn.quantize` verifications | max 344 (temporary single nn.Linear/nn.Embedding lookup check, not a model) |
| **underlying quantizations** | **max 688 = 344 + 344** (backend-enforced combined cap) |
| model forward / backward / generation / optimizer | 0 / 0 / 0 / 0; no full model is built or loaded |
| device | MLX default device must be an available GPU |
| MLX limits | memory 8 GiB, cache 64 MiB (original `NativeConvertBackend.__init__`, test-pinned) |
| gates | AC strictly; headroom/pressure/other-heavy via `paced_resources.decisions`; swap net growth ≤ 1 GiB over the whole run (unknown sensor stops); RSS ≤ 8 GiB (sampled); disk need + 20 GiB reserve; thermal throttle |
| window | every quantized unit must START within 1800 s, else STOP partial |
| payload | each part's SHA-256 must equal the pinned proof BEFORE `save`; first mismatch STOP, no second attempt |
| one shot | after consumption every outcome is final: no resume, no re-quantizing accepted units, no retry on a new root |

## CPU preparation commands (never import MLX)
`plan` (read-only) · `adopt-retained` (copies ONLY the 406 authenticated files + receipt lines into the NEW canonical work root) ·
`prepare-scope` (stream-hashes every source shard against the plan's pinned SHA-256, authenticates the 406 retained outputs, fingerprints the
installed interpreter/MLX without importing it, and publishes an IMMUTABLE private execution scope). The scope binds: canonical roots
(R from the git common dir), source dir, plan/qmap/historical-certificate/proof paths **and** SHA-256, per-unit inventory digest, source-shard
SHA-256, all limits above, installed python/numpy/MLX version + MLX tree digest, and the SHA-256 of the **whole transitive local source closure**
(static import scan incl. lazy imports; includes cc, qc, quant_cache, quant_probe, resource_supervisor, paced_resources, the shared admission core and their imports).
Any change to the code, an input file, the installed MLX, a root or a path (including symlinks) after preparation/review ⇒ refuse.

## `execute` admission (the only command that can create MLX arrays)
Three SEPARATE files with exact SHA-256 pins on the command line: the prepared **scope**, a **human authorization**
(`authorized`, `action`, `scope_sha256`, `human_instruction_sha256`, `approved_by`, issue/expiry ≤ 24 h) and an **independent review**
(`PASS`/`PASS_WITH_LIMITATIONS`, `reviewer_id`, `reviewer_kind` ∈ {isolated_codex, claude_code} and ≠ the implementer `claude_code`,
bound to the scope digest, the exact source-closure map and the input digest). Same contract as the project's other admissions
(`shared_rollout_gspo_core_v1` helpers). The code cannot prove who wrote a file; it requires the files, hashes and bindings and consumes once.
Order: validate (consumes nothing) → **non-blocking `flock` on the canonical remaining20 global lock** (before any MLX import; never created,
forced, modified or unlinked; released on exit) → re-hash sources, retained units, resources, disk → **consume scope + human instruction in the
global ledger** → work-root ledger/lock → backend (MLX) with the closure re-checked before every unit. Any refusal before consumption consumes nothing.

Canonical lock path used: `R/local_data/c_two_task_pool_v1/tier_c_v1/remaining20_bundle_v1/remaining20.lock` — the existing file every driver and
`original22_final_eval` use. (A path without `remaining20_bundle_v1/` does not exist; creating one would exclude nothing.)

## Limitations (honest)
- `flock` is advisory: it excludes cooperating drivers only. RSS/swap/disk/pressure are SAMPLED between operations, not enforced mid-call; the MLX
  8 GiB / 64 MiB limits are allocator guidelines not proven to be hard caps. Thermal `SENSOR_UNAVAILABLE` cannot refuse. No promise against OS failure.
- Only the fake `mx`/`nn` has been exercised. Whether GPU `mx.quantize` on this MLX build reproduces the historical bytes is `NOT_VERIFIED`; a mismatch
  stops at unit 0 with nothing saved and the scope is spent (a new scope + fresh human + review would be required).
- The scope binds the closure by absolute path; moving/renaming the worktree invalidates it. Editing any closure file (including this module) requires a new scope and review.
