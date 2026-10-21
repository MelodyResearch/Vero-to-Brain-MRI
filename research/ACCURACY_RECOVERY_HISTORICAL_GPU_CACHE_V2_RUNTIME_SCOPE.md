# Historical GPU cache v2 — prospective runtime scope (NOT executed, NOT authorized yet)

Runtime-dependency reconstruction only: no new model, no scientific experiment, no Adam/MRI. A/B stays blocked until the
historical cache is reproduced exactly (both shard SHA-256, config, tokenizer, index equal to the pinned cache certificate).

## Why
The CPU rebuild (`cpu_failure_001.json`) kept 406/406 retained units byte-exact but 0/344 quantized payloads matched the
pinned historical 750-unit proof (Codex streamed all 750). CPU-vs-historical-path mismatch is confirmed; a hardware
arithmetic (CPU vs GPU `mx.quantize`) cause is only INFERRED. The failure receipt's `Cache index differs` message came from a
separate, now-fixed checker bug (commit 18c8e06): the plan-derived layout predicted `weight,scales,biases` but the installed
MLX saves sorted names (`biases,scales,weight`), moving the 4 GiB shard boundary. The corrected plan-to-header layout reproduces
the historical shard sizes exactly (4207002792 / 920772600). That fix alone cannot certify the CPU cache.

## Exact scope (fixed in `SCOPE`, hash `SCOPE_SHA256`; no config/env/flag can widen it)
| item | value |
|---|---|
| plan | `71e2ca2709ae295d2b7aa3860701fd6ddc8ab3cd04e1a1cd011c07b420f2a68d` (relative filenames, unchanged) |
| units | 750 = 406 retained (copied, authenticated, read-only from failed CPU root) + 344 quantized (GPU) |
| GPU unit attempts / quantize ops / module-path checks | each max **344** (distinct counters; temporary single nn.Linear/nn.Embedding per unit) |
| device | MLX default device must be an available GPU, else refuse before any array |
| MLX limits | memory 8 GiB, cache 64 MiB (set by the original `NativeConvertBackend.__init__`; test-pinned) |
| RSS | 8 GiB envelope, sampled at operation boundaries |
| swap | net growth over the whole run ≤ 1 GiB from the first sample; unknown swap sensor stops; absolute swap not required to be 0 |
| gate | `paced_resources.decisions` + `ResourcePolicy`: AC strictly, known pressure/available, no throttle, no other heavy process |
| disk | free ≥ estimate (≈9.3 GiB incl. assembled cache) + 20 GiB reserve; per-unit check too |
| window | every quantized unit must START within 1800 s of execution start, else STOP with partial, no rerun |
| payload | each saved-to-be part's SHA-256 must equal the pinned proof BEFORE `save`; first mismatch STOP, no second attempt |
| work root | NEW `.../Vero-to-Brain-MRI/work/accuracy-recovery-historical-gpu-cache-v2` (failed CPU root never written) |

## Commands (`scripts/rebuild_accuracy_recovery_historical_gpu_cache_v2.py`)
- `plan` — read-only, never imports MLX. Authenticates the 406 retained CPU outputs (receipt, source bytes, whole-file SHA, payload SHA)
  against the pinned proof and reports scope/disk. Actual-metadata run: 406/406 authenticated, 344 pending GPU, 0 CPU-quantized adopted.
- `adopt-retained` — CPU file copy of ONLY the authenticated retained files + their receipt lines into the new work root (exclusive create).
- `execute` — the only command that can create MLX arrays. Requires `--source-dir`, `--expected-source-sha256`, a fresh
  `--authorization-artifact` + `--authorization-artifact-sha256`, an unused once-ledger, no lock, no prior failure receipt, passing resource/disk gates.
  Revalidates source shard hashes, plan, old receipts, and retained file SHAs first.

Pins required on every command: `--plan`, `--expected-plan-sha256`, `--expected-cache-certificate(+-sha256)` (historical certificate
`work/exact_quant_cache/cache_certificate.json`), `--historical-proof(+-sha256)` (`independent750verification_PRIVATE.json`), `--quant-map`, `--work-root`.

## Authorization artifact (written by the human/reviewer, never by this code)
JSON with: `schema`=`accuracy-recovery-historical-gpu-cache-v2-authorization`, `scope_sha256`, `plan_sha256`, `module_sha256` and `script_sha256`
(of the reviewed files), `historical_certificate_sha256`, `proof_sha256`, `independent_review_artifact_sha256`, `execution_nonce` (≥16 chars),
`human_authorization{approved_by,issued_at_utc,expires_at_utc}` (current, ≤ 24 h). Its own SHA-256 is passed on the command line. The code checks
hashes/expiry only; it cannot prove who wrote the file. Editing the module or script invalidates a previously issued artifact.

## Limitations (honest)
- RSS, swap, disk, pressure are SAMPLED between operations; nothing hard-caps a single MLX call. The MLX 8 GiB / 64 MiB limits are allocator
  guidelines inherited from the original converter and were not independently proven to be hard caps. Peak/active MLX counters are informational.
- Thermal `SENSOR_UNAVAILABLE` cannot refuse. No promise against an OS-level failure (a watchdog panic occurred on this Mac earlier).
- Only the fake `mx`/`nn` has been exercised. Real MLX behavior (GPU quantize equality with the historical bytes, `mx.metal.is_available`) is `NOT_VERIFIED`.
  A GPU re-quantization may still mismatch (e.g. different MLX/OS version than the original run); that would STOP at unit 0, nothing saved.
- Source shard hashing (~16 GiB) happens inside `execute`; the actual-metadata `plan` run did not do it (`source_checks: NOT_RUN`).
- If all 344 units cannot start within 1800 s the run stops partial and is NOT rerun automatically; a new scope/authorization would be needed.
