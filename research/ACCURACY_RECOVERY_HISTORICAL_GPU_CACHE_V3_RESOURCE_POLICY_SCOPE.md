# Historical GPU cache v3 — local resource policy + evidence-bound continuation (prospective; NOT executed, NOT authorized)

Runtime-dependency reconstruction only (no new model/experiment/Adam change). Cache PASS is not claimed; A/B stays blocked until the historical cache is
reproduced exactly. Old v2 source, logs, consumed ledger, the global training/resource gates and `cc`/`qc`/`quant_cache`/`paced_resources`/`resource_supervisor`
are untouched. v2 scope `1f2bd2db…` / canonical `3edf0701…` is fully consumed and is never retried.

## What happened (OBSERVED, from the closed v2 logs; they carry no timestamps, so no swap *rate* is derived)
Unit 0 ran 1 direct + 1 module-path quantization on the GPU (RSS peak 2.023 GiB); its payloads and file equal the pinned proof. The post-unit gate then saw
available 5.9315 GiB (< the old 6 GiB per-unit floor), pressure-free 35%, swap +0.5784 GiB (< the 1 GiB cap) and raised `sustained_swap_growth` (a supervisor
rate warning) + `insufficient_memory_headroom`. The file was saved, **no receipt was written**, 0 of 343 other units started. No payload mismatch occurred.

## Local policy (`ResourceController`; the inherited supervisor reasons are logged, never decide)
| | rule |
|---|---|
| CONTINUE | available ≥ max(5.5 GiB, unit forecast + 2 GiB reserve) **and** pressure-free ≥ 30%, AC, 0 other heavy processes, RSS known ≤ 8 GiB, swap known with cumulative net growth ≤ 1.5 GiB, no thermal throttle |
| HOLD | any transient shortage or unknown essential sensor (available/pressure/swap/RSS/power/heavy-process/OS level): **no operation starts**; resample every 5 s; ≤ 120 s per boundary; the 1800 s operation-start window includes holds; every sample logged with UTC + monotonic time |
| STOP | pressure-free ≤ 15%; real OS memory-pressure CRITICAL (`kern.memorystatus_vm_pressure_level & 4`, no 120 s wait); available < 2 GiB; available < 4 GiB for 30 s; RSS > 8 GiB; swap growth > 1.5 GiB; hold timeout; window exceeded |
| swap | cap = **net growth ≤ 1.5 GiB from the first known sample of the whole run, never reset** (old source cap was 1.0 GiB from the first sample, the same semantics; the old source had no other absolute-swap ceiling and absolute OS swap need not be 0 — this is my reading of "retain the old absolute swap cap"; please confirm). Rate is trend-only. |
| unchanged | AC strictly, 0 other heavy, RSS 8 GiB (sampled), MLX 8 GiB / cache 64 MiB (original backend), disk need + 20 GiB reserve, thermal throttle, canonical remaining20 `flock`, per-unit forecast/reserve (now a HOLD, floor never lowered; largest unit needs 5.30 GiB < 5.5) |
| limitation | thermal `SENSOR_UNAVAILABLE` cannot refuse; flock advisory; RSS/swap/pressure/disk are sampled at boundaries, not mid-call; MLX limits not proven hard caps |

## Seam (no monkeypatch, no global change)
`qc.convert` cannot be reused: it would re-process the receipt-less unit 0, its start gate raises `HeadroomBlock` instead of holding, and its end-of-unit gate raises
**before** the receipt is written (how v2 lost unit 0's receipt). `convert_v3` / `assemble_v3` / `certify_v3` are small local copies differing only there, each
differentially tested against the original; the final cache must still equal the pinned historical certificate (both shard SHA-256, config, tokenizer, index).
At a unit's end the controller only observes: the receipt is written first, then a STOP is raised.

## Saved unit 0 — explicit adoption, not a fabricated receipt
Verified on CPU (no MLX): whole-file SHA, all three payload SHAs, header order/dtypes/shapes (plan-derived) and the **source bytes** equal the pinned proof;
old failure opcounts (1 direct, 1 module, 2 underlying) and every old evidence file are hash-bound. **Present:** direct-quantize payload == proof (checked in memory
before the old save), file == proof. **Derived, not measured:** reload equality. **Missing and stated, not claimed:** module-path equality result, reload result,
MLX peak, post-release active memory, post-unit resource receipt. A passing receipt is *not* written; the unit is accepted only by the v3 adoption record and only
if the human authorization and the independent review of this scope accept those listed limits. Unit 0 is never re-quantized.

## Budgets
406 retained copied (0 GPU ops) + 1 adopted + 343 new quantized units: ≤ 343 attempts, ≤ 343 direct, ≤ 343 module-path ⇒ ≤ 686 new + 2 old = **≤ 688** cumulative.
No full model, forward, backward, generation, optimizer update, MRI or Adam. One shot: nothing is resumed or retried; the v2 and v3 consumed ledgers are both checked,
so neither the old instruction nor a fresh root can retry a consumed scope.

## Commands (`scripts/rebuild_accuracy_recovery_historical_gpu_cache_v3.py`)
`plan` · `prepare-scope` (stream-hashes all source shards, authenticates the closed v2 evidence, the 406 retained and unit 0, fingerprints the source closure + installed MLX,
writes an immutable private scope) · `adopt-continuation` (CPU copy into the NEW root `work/accuracy-recovery-historical-gpu-cache-v3`) never import MLX.
`execute` needs scope + human authorization + independent review (exact sha256 each), the free canonical lock, and is **not** run by this change.
