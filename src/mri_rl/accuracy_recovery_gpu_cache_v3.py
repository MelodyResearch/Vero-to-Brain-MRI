"""Cache v3: evidence-bound CONTINUATION of the closed v2 historical GPU cache attempt, with a local, explicit resource policy.

Why: the single authorized v2 run quantized unit 0 on the GPU (payload + file identical to the pinned historical proof), then its post-unit
resource gate refused (available 5.93 GiB < 6 GiB, plus a supervisor swap-rate warning on a 0.58 GiB rise that was far below the 1 GiB cap), so the
unit was saved without a receipt. v2 is closed and consumed; v3 never retries it. v3 is a NEW scope that binds the v2 evidence by hash, adopts
the saved exact unit 0 through an explicit adoption record (never re-quantizing it), and runs the remaining 343 quantized units on the GPU under a
local policy. Nothing in cc / qc / quant_cache / paced_resources / resource_supervisor / v2 or any global training gate is edited or patched.

Local resource policy (ResourceController; the inherited supervisor/paced decisions are only logged, never decide):
  CONTINUE  available >= max(5.5 GiB, unit forecast + 2 GiB reserve) AND pressure-free >= 30%, AC, 0 other heavy processes, RSS known and <= 8 GiB,
            swap known with cumulative net growth <= 1.5 GiB from the FIRST known sample of the whole run (never reset), no thermal throttle.
  HOLD      any transient shortage/unknown essential sensor: no operation starts; resample every 5 s; at most 120 s per boundary; the 1800 s
            operation-start window includes holds; every sample is logged with UTC and monotonic time.
  STOP      pressure-free <= 15%, real OS memory-pressure CRITICAL, available < 2 GiB, available < 4 GiB for 30 s, RSS > 8 GiB, swap growth > 1.5 GiB,
            hold timeout, window exceeded. Swap RATE is trend-only. Thermal SENSOR_UNAVAILABLE cannot refuse (documented current limitation).

Seam: qc.convert cannot be reused because (a) it would re-process the saved unit 0 (it has no receipt and a passing receipt must not be fabricated),
(b) its start gate raises HeadroomBlock instead of holding, and (c) its end-of-unit gate raises BEFORE the receipt is written (which is exactly how v2 lost
unit 0's receipt). convert_v3 / assemble_v3 / certify_v3 are small local copies that differ only in those points; each is differentially tested against
the original on fixtures, and the final cache must still equal the pinned historical certificate (both shard SHA-256, config, tokenizer, index).

Admission (execute): same scope + human + independent-review contract and canonical remaining20 flock as v2, plus the previous-attempt evidence, the
adoption record and the policy bound into the scope. One shot: every outcome after consumption is final; the v2 consumed ledger is also checked so the old
instruction cannot be reused, and a fresh root cannot retry a consumed scope. Honest limits: flock is advisory; RSS/swap/pressure/disk are SAMPLED, not
enforced mid-call; MLX 8 GiB / 64 MiB limits are allocator guidelines; the old process's module-path result, reload result and MLX peak/residual for unit 0
were not persisted and are NOT claimed. NEVER executed from this module's preparation commands (plan / prepare-scope / adopt-continuation import no MLX).
"""
from __future__ import annotations
import fcntl
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from mri_rl import accuracy_recovery_cpu_cache_v1 as cc, accuracy_recovery_gpu_cache_v2 as v2, quant_convert as qc, resource_supervisor, shared_rollout_gspo_core_v1 as shared

WORK_ROOT_NAME = 'accuracy-recovery-historical-gpu-cache-v3'
CONSUMED_REL = 'local_data/accuracy_recovery_v1/historical_gpu_cache_v3_consumed'
SCOPE_DIR_REL = v2.SCOPE_DIR_REL
ADOPTED_UNIT = 0
OLD_COUNTS = {'gpu_unit_attempts': 1, 'gpu_direct_quantize_ops': 1, 'gpu_module_path_quantizations': 1, 'gpu_underlying_quantizations': 2}
MAX_CUMULATIVE_UNDERLYING = v2.MAX_UNDERLYING_QUANTIZATIONS
POLICY = {
    'continue_available_gib_min': 5.5, 'continue_pressure_free_percent_min': 30.0, 'stop_pressure_free_percent_max': 15.0, 'stop_available_gib_below': 2.0,
    'sustained_low_available_gib': 4.0, 'sustained_low_seconds': 30.0, 'sample_interval_seconds': 5.0, 'hold_seconds_per_boundary_max': 120.0,
    'operation_start_window_seconds': v2.WINDOW_SECONDS, 'swap_cumulative_net_growth_cap_gib': 1.5,
    'swap_baseline': 'first known sample of the whole run; never reset', 'swap_rate': 'trend-only (logged, never decides)',
    'rss_envelope_gib': cc.RSS_ENVELOPE_GIB, 'unit_reserve_gib': qc.RESERVE_GIB, 'disk_reserve_gib': v2.DISK_RESERVE_GIB,
    'mlx_memory_limit_bytes': v2.MLX_MEMORY_LIMIT_BYTES, 'mlx_cache_limit_bytes': v2.MLX_CACHE_LIMIT_BYTES, 'ac_required': True, 'other_heavy_processes_allowed': 0,
    'os_critical_mask': 4, 'warning_only_inherited_supervisor_reasons_terminate': False,
    'unknown_essential_sensors': 'HOLD (no operation starts) then STOP at the per-boundary bound',
    'thermal_limitation': 'THROTTLED holds; SENSOR_UNAVAILABLE cannot refuse (documented current limitation)'}
SCOPE = {
    'name': 'VERO_TO_MRI_ACCURACY_RECOVERY_V1/historical_gpu_cache_v3_continuation', 'plan_sha256': v2.PLAN_SHA256, 'plan_units': 750,
    'retained_units_adopted': 406, 'retained_units_gpu_operations': 0, 'adopted_saved_units': 1, 'quantized_units_pending_gpu': 343,
    'previous_attempt_direct_quantize_ops': 1, 'previous_attempt_module_path_quantizations': 1, 'previous_attempt_underlying_quantizations': 2,
    'new_gpu_unit_attempts_max': 343, 'new_direct_quantize_ops_max': 343, 'new_module_path_quantizations_max': 343, 'new_underlying_quantizations_max': 686,
    'cumulative_underlying_quantizations_max': MAX_CUMULATIVE_UNDERLYING, 'quantization': dict(qc.QUANT), 'full_model_loaded': False,
    'model_forward_calls': 0, 'model_backward_calls': 0, 'generation_calls': 0, 'optimizer_updates': 0, 'device': 'gpu',
    'module_construction': 'temporary single nn.Linear/nn.Embedding lookup verification only', 'work_root_name': WORK_ROOT_NAME,
    'previous_work_root_name_read_only': v2.WORK_ROOT_NAME, 'resource_policy': POLICY, 'first_payload_mismatch': 'STOP_no_second_attempt',
    'automatic_retry': False, 'one_shot_no_rerun_no_resume_no_new_root_retry': True,
    'continuation': 'evidence-bound continuation of the closed v2 attempt; not a generic retry; unit 0 adopted by explicit record and never re-quantized',
    'final_identity': 'pinned historical cache certificate incl. both shard SHA-256'}
SCOPE_SHA256 = qc.sha_bytes(qc.canonical(SCOPE).encode())
SCOPE_SCHEMA = 'accuracy-recovery-historical-gpu-cache-v3-continuation-scope-v1'
SCOPE_KIND = 'FUTURE_EXACT_HISTORICAL_GPU_CACHE_V3_CONTINUATION_SCOPE'
ACTION = 'accuracy_recovery_historical_gpu_cache_v3_continue_343_units_with_adopted_unit0'
ADOPTION_SCHEMA = 'accuracy-recovery-gpu-cache-v3-saved-unit-adoption-v1'
REQUIRED_CLOSURE = (*(n for n in v2.REQUIRED_CLOSURE if n != 'rebuild_accuracy_recovery_historical_gpu_cache_v2.py'),      # v3 never imports the v2 script
                    'accuracy_recovery_gpu_cache_v3.py', 'rebuild_accuracy_recovery_historical_gpu_cache_v3.py')
D = 'local_data/accuracy_recovery_v1/'
PRODUCTION_PINS = {
    'scope': '1f2bd2db5a7b6161334979fdea2a4d0d225a2b5a4b8fbe51506f820b49509b7e', 'scope_digest': '3edf070143c8a7e42eca71a8a5817005d359fb0f4ea591322e397f4cc1a41dc7',
    'human': '2eeb5c9c0035b40f56b916cc357b82e8947df558060486bbfc1221751684772d', 'review': 'ab88f04b767609ba8eac75fb94dd5520bc689fa127989f8a53186b17514b0562',
    'failure_receipt': '629a60e1bb8d06848288031b7df33a4952f75272c8a1e30a82def3deec9b4394', 'launcher_record': '1ad149d4f3cecaad1868b1da033c385740d367001d7458c82c03b3787225574d',
    'receipts': 'ffee25145812b615f9f21c3ed9c4e440f01cf2ac83cbfdb5b51e5bd607667ae1', 'resource_log': 'ef0dd867c5cdcf7600b92f053c5cb3c188818ae83395056c6b1eac47ae953fa2',
    'rss_log': '476c18377f30d05f83cffecd0c2fb979984fc01ed7feb2fcf591d67096594c4c', 'stdout': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
    'stderr': '022263a225e962b5e41a9ea82198a5c3ea22028edfa18d78904a315e04e7fcd6', 'summary': 'aba97e91f850b42d3bc7516fa177f96cf79c3c806c93c68b6e23360113a81356',
    'unit0_file': '6cc6fa0558040faa6efa1b95535d58595246fc53843618b796cfc7ebb2912258'}
ScopeRefused = v2.ScopeRefused


def _req(ok, message):
    if not ok:
        raise ScopeRefused(message)


def canonical_roots(repo_root, source_dir=None):
    r = Path(repo_root)
    return {'repo_root': str(r), 'work_root': str(r / 'work' / WORK_ROOT_NAME), 'previous_work_root': str(r / 'work' / v2.WORK_ROOT_NAME),
            'global_lock': str(r / v2.GLOBAL_LOCK_REL), 'consumed_dir': str(r / CONSUMED_REL), 'previous_consumed_dir': str(r / v2.CONSUMED_REL),
            'source_dir': str(source_dir) if source_dir else None}


def previous_paths(repo_root, scope_digest, instruction_sha=None):
    """Where the closed v2 attempt's evidence lives (all read-only). Consumed-marker names derive from the old scope digest / instruction hash."""
    r = Path(repo_root); d = r / 'local_data/accuracy_recovery_v1'; w = r / 'work' / v2.WORK_ROOT_NAME; c = r / v2.CONSUMED_REL
    stem = 'gpu_cache_v2_ACTUAL_EXECUTION_20261020_'
    paths = {'scope': d / 'historical_gpu_cache_v2_execution_scope_PRIVATE.json', 'human': d / 'HISTORICAL_GPU_CACHE_V2_ACTUAL_HUMAN_AUTHORIZATION_PRIVATE.json',
             'review': d / 'HISTORICAL_GPU_CACHE_V2_INDEPENDENT_REVIEW_PRIVATE.json', 'failure_receipt': w / 'v2_failure_001.json', 'ledger': w / 'v2_ledger.json',
             'resource_log': w / 'v2_resource.jsonl', 'rss_log': w / 'v2_rss.jsonl', 'receipts': w / 'units/receipts.jsonl',
             'launcher_record': d / (stem + 'launcher_record_PRIVATE.json'), 'summary': d / (stem + 'SUMMARY_PRIVATE.json'),
             'stdout': d / (stem + 'child_stdout_PRIVATE.txt'), 'stderr': d / (stem + 'child_stderr_PRIVATE.txt'),
             'independent_detail': d / 'HISTORICAL_GPU_CACHE_V2_ACTUAL_INDEPENDENT_DETAIL_PRIVATE.json',
             'consumed_scope': c / ('scope_consumed_%s.json' % scope_digest), 'consumed_outcome': c / ('scope_outcome_%s.json' % scope_digest)}
    if instruction_sha:
        paths['consumed_instruction'] = c / ('authorization_consumed_%s.json' % instruction_sha)
    return paths


def verify_previous_attempt(repo_root, pins=PRODUCTION_PINS, expected=cc.Expected()):
    """Authenticate the closed v2 attempt's evidence by hash and content; bind its exact operation counts. Raises ValueError on any difference."""
    ledger_p = previous_paths(repo_root, pins['scope_digest'])['ledger']
    if not ledger_p.is_file():
        raise ValueError('Previous attempt evidence file missing: ledger')
    ledger = json.loads(ledger_p.read_text())
    if ledger.get('scope_sha256') != pins['scope_digest']:
        raise ValueError('Previous attempt ledger is not for the pinned old scope digest')
    paths = previous_paths(repo_root, pins['scope_digest'], ledger.get('human_instruction_sha256'))
    files = {}
    for name, p in paths.items():
        if not p.is_file():
            raise ValueError('Previous attempt evidence file missing: %s' % name)
        h = shared.file_sha256(p)
        if pins.get(name) is not None and h != pins[name]:
            raise ValueError('Previous attempt evidence differs from its pin: %s' % name)
        files[name] = {'path': str(p), 'sha256': h}
    fail = json.loads(paths['failure_receipt'].read_text())
    counts = {k: fail.get(k) for k in OLD_COUNTS}
    if counts != OLD_COUNTS:
        raise ValueError('Previous attempt operation count mismatch: %s' % counts)
    if (fail.get('error_type') != 'ResourceStop' or fail.get('rerun_allowed') is not False or fail.get('automatic_retry') is not False or fail.get('partial') is not True):
        raise ValueError('Previous attempt failure receipt is not the closed resource-stop evidence')
    if json.loads(paths['consumed_outcome'].read_text()).get('state') != 'FAILED':
        raise ValueError('Previous attempt outcome marker is not FAILED')
    rec = [json.loads(l) for l in paths['receipts'].read_text().splitlines()]
    retained = sum(r['action'] == 'retain' for r in rec); quantized = sum(r['action'] == 'quantize' for r in rec)
    if retained != expected.retained or quantized != 0:
        raise ValueError('Previous attempt receipts differ: %d retained, %d quantized' % (retained, quantized))
    return {'files': files, 'counts': counts, 'scope_digest': pins['scope_digest'], 'instruction_sha256': ledger['human_instruction_sha256'],
            'retained_receipts': retained, 'quantized_receipts': quantized}


# ---- the saved exact unit 0 --------------------------------------------------------------------------------------
def verify_saved_unit(plan, proofs, unit, path, source_dir):
    """Independent CPU authentication (no MLX): whole file + every payload vs the pinned proof, header order/dtypes/shapes vs the plan, source bytes."""
    if unit['action'] != 'quantize' or unit['transform']:
        raise ValueError('Only a plain quantized unit can be adopted')
    p = proofs[unit['index']]
    file_sha, parts = v2.hash_unit_file(path)
    if parts != p['output_tensor_payload_sha256']:
        raise ValueError('Saved unit %d payload differs from the pinned historical proof' % unit['index'])
    if file_sha != p['output_file_sha256']:
        raise ValueError('Saved unit %d file sha differs from the pinned historical proof' % unit['index'])
    header, _ = qc._read_st_header(path)
    order = list(header)
    if order != list(cc.PART_ORDER):
        raise ValueError('Saved unit %d header order %s differs from the sorted part order' % (unit['index'], order))
    want = cc._quant_parts(unit)
    dtypes = {k: v['dtype'] for k, v in header.items()}; shapes = {k: v['shape'] for k, v in header.items()}
    if dtypes != {k: want[k][0] for k in want} or shapes != {k: want[k][1] for k in want}:
        raise ValueError('Saved unit %d dtypes/shapes differ from the plan-derived layout' % unit['index'])
    h, left = hashlib.sha256(), unit['nbytes']
    with (Path(source_dir) / unit['source_file']).open('rb') as f:
        f.seek(unit['offset'])
        while left:
            chunk = f.read(min(qc.CHUNK, left))
            if not chunk:
                raise ValueError('Short read of the source bytes for unit %d' % unit['index'])
            h.update(chunk); left -= len(chunk)
    if h.hexdigest() != p['source_bytes_sha256']:
        raise ValueError('saved unit %d source bytes differ from the proof (source fragment)' % unit['index'])
    return {'unit_index': unit['index'], 'normalized_key': unit['normalized_key'], 'unit_hash': qc.unit_hash(plan, unit), 'file': 'unit_%04d.safetensors' % unit['index'],
            'file_sha256': file_sha, 'payload_sha256': parts, 'source_bytes_sha256': h.hexdigest(), 'header_order': order, 'dtypes': dtypes, 'shapes': shapes}


def adoption_record(plan, proofs, unit, path, source_dir, prev, pins):
    ev = verify_saved_unit(plan, proofs, unit, path, source_dir)
    if pins.get('unit0_file') is not None and ev['file_sha256'] != pins['unit0_file']:
        raise ValueError('Saved unit file differs from the independently recorded unit 0 file sha')
    return {
        'schema': ADOPTION_SCHEMA, **ev, 'plan_sha256': plan['plan_sha256'], 'old_attempt_opcounts': prev['counts'],
        'old_attempt_files': {k: prev['files'][k]['sha256'] for k in ('failure_receipt', 'ledger', 'launcher_record', 'scope', 'resource_log', 'rss_log')},
        'evidence_present': ['in-memory direct quantize payload hashes were compared with the pinned proof BEFORE saving (a difference raises HistoricalPayloadMismatch; none was raised)',
                             'saved file whole-sha and all payload shas equal the pinned proof (re-verified by streaming on CPU now)', 'source bytes sha equals the proof (re-streamed now)',
                             'header order, dtypes and shapes equal the plan-derived layout', 'old run executed 1 direct and 1 module-path quantization, then stopped on a resource gate'],
        'evidence_present_notes': {'reload_equality': 'derived, not measured by the old run: in-memory payload == proof (checked before save) and saved file payload == proof (checked now)',
                                   'module_path': 'the old run executed one module-path quantization but its boolean was computed and lost with the receipt; NOT claimed'},
        'evidence_missing': ['module_path_bitwise_equal_result', 'reload_bitwise_equal_result', 'mlx_peak_mib', 'mlx_active_after_release_mib', 'residual_within_budget',
                             'post_unit_resource_receipt'],
        'old_receipt_fabricated': False, 'receipt_equivalent': False,
        'adoption_policy': 'no_new_quantization_of_unit0: adopted by this explicit record only if the human authorization and the independent review of THIS scope accept the listed '
                           'missing evidence; the unit content is byte-identical to the historical accepted bytes; the final cache must still equal the pinned historical certificate.'}


# ---- local resource controller ------------------------------------------------------------------------------------
def os_pressure_level(run=subprocess.run):
    """Real macOS dispatch memory-pressure level (1 normal, 2 warn, 4 critical); None when unavailable."""
    try:
        p = run(['sysctl', '-n', 'kern.memorystatus_vm_pressure_level'], capture_output=True, text=True, timeout=2)
        return int(p.stdout.strip()) if p.returncode == 0 else None
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


class ResourceController:
    """Local boundary controller. HOLD = no operation may start; STOP = fail closed. The inherited supervisor is logged as information only."""

    def __init__(self, *, sampler=None, os_level=None, clock=time.monotonic, sleep=time.sleep, utc=None, log_path=None, policy=POLICY):
        self.sampler = sampler or resource_supervisor.sample_macos
        self.os_level = os_level or os_pressure_level
        self.clock, self.sleep, self.policy, self.log_path = clock, sleep, policy, log_path
        self.utc = utc or (lambda: datetime.now(timezone.utc).isoformat())
        self.baseline_swap = self.low_since = self.t0 = self.last_swap = self.last_swap_time = None
        self.last = None; self.peak_rss_gib = 0.0; self.hold_samples = 0
        self._inherited = resource_supervisor.ResourcePolicy()

    def start_window(self):
        self.t0 = self.clock()

    def _decide(self, s, level, required, now):
        P = self.policy; stop, hold = [], []
        avail, pr, swap, rss = s.available_gib, s.pressure_free_percent, s.swap_used_gib, s.process_rss_gib
        if avail is None: hold.append('available_unknown')
        if pr is None: hold.append('pressure_unknown')
        if rss is None: hold.append('rss_unknown')
        if swap is None: hold.append('swap_unknown')
        if level is None: hold.append('os_pressure_level_unknown')
        if s.other_heavy_processes is None: hold.append('other_heavy_unknown')
        elif s.other_heavy_processes: hold.append('other_heavy_process')
        if s.power_source == 'AC': pass
        elif s.power_source == 'BATTERY': hold.append('not_ac')
        else: hold.append('power_unknown')
        if s.thermal_state == 'THROTTLED': hold.append('thermal_throttled')
        if pr is not None:
            if pr <= P['stop_pressure_free_percent_max']: stop.append('pressure_free_at_or_below_15')
            elif pr < P['continue_pressure_free_percent_min']: hold.append('low_pressure_free')
        if level is not None and level & P['os_critical_mask']: stop.append('os_memory_pressure_critical')
        if rss is not None:
            self.peak_rss_gib = max(self.peak_rss_gib, rss)
            if rss > P['rss_envelope_gib']: stop.append('process_rss_above_8gib')
        if avail is not None:
            if avail < P['stop_available_gib_below']: stop.append('available_below_2gib')
            if avail < P['sustained_low_available_gib']:
                self.low_since = now if self.low_since is None else self.low_since
                if now - self.low_since >= P['sustained_low_seconds']: stop.append('sustained_available_below_4gib')
            else:
                self.low_since = None
            if avail < P['continue_available_gib_min']: hold.append('insufficient_available_memory')
            elif avail < required: hold.append('insufficient_forecast_headroom')
        growth = trend = None
        if swap is not None:
            if self.baseline_swap is None: self.baseline_swap = swap
            growth = swap - self.baseline_swap
            if growth > P['swap_cumulative_net_growth_cap_gib']: stop.append('swap_net_growth_over_cap')
            t = getattr(s, 'monotonic_seconds', None)
            if self.last_swap is not None and t is not None and self.last_swap_time is not None and t - self.last_swap_time > 0:
                trend = (swap - self.last_swap) / ((t - self.last_swap_time) / 60.0)
            self.last_swap, self.last_swap_time = swap, t
        return ('STOP' if stop else 'HOLD' if hold else 'CONTINUE'), stop + hold, growth, trend

    def _sample(self, label, unit, required, hold_start):
        s = self.sampler(); level = self.os_level(); now = self.clock()
        decision, reasons, growth, trend = self._decide(s, level, required, now)
        try:
            inherited = self._inherited.observe(s)
        except Exception as exc:                                   # informational only: never decides
            inherited = ('UNAVAILABLE', ['inherited_supervisor_unavailable:%s' % type(exc).__name__])
        row = {'utc': self.utc(), 'monotonic': float(now), 'boundary': label, 'unit': unit, 'decision': decision, 'reasons': reasons, 'required_available_gib': required,
               'available_gib': s.available_gib, 'pressure_free_percent': s.pressure_free_percent, 'swap_used_gib': s.swap_used_gib, 'baseline_swap_gib': self.baseline_swap,
               'swap_growth_gib': growth, 'swap_growth_gib_per_min_trend_only': trend, 'swap_growth_trend_only': True, 'power_source': s.power_source,
               'battery_percent': getattr(s, 'battery_percent', None), 'other_heavy_processes': s.other_heavy_processes, 'process_rss_gib': s.process_rss_gib,
               'thermal_state': s.thermal_state, 'os_pressure_level': level, 'low_available_since': self.low_since,
               'hold_elapsed_seconds': 0.0 if hold_start is None else float(now - hold_start),
               'window_elapsed_seconds': None if self.t0 is None else float(now - self.t0),
               'inherited_supervisor_level': inherited[0], 'inherited_supervisor_reasons': list(inherited[1])}
        self.last = row
        if self.log_path:
            with Path(self.log_path).open('a') as f:
                f.write(json.dumps(row, sort_keys=True) + '\n'); f.flush(); os.fsync(f.fileno())
        return s, decision, reasons, now

    def observe(self, label, unit=None, need_gib=0.0):
        """One sample, never waits or raises: used at a unit's end so a saved unit keeps its receipt."""
        s, decision, reasons, _ = self._sample(label, unit, max(self.policy['continue_available_gib_min'], need_gib), None)
        return s, decision, reasons

    def boundary(self, label, need_gib=0.0, unit=None, enforce_window=True):
        """Return a sample only when CONTINUE. HOLD resamples every 5 s up to 120 s (inside the operation-start window); STOP raises."""
        P = self.policy; required = max(P['continue_available_gib_min'], need_gib); hold_start = None
        while True:
            if enforce_window and self.t0 is not None and self.clock() - self.t0 > P['operation_start_window_seconds']:
                raise v2.WindowExceeded('Operation-start window of %.0f s elapsed (holds included) with units still pending' % P['operation_start_window_seconds'])
            s, decision, reasons, now = self._sample(label, unit, required, hold_start)
            if decision == 'CONTINUE':
                return s
            if decision == 'STOP':
                raise cc.ResourceStop('Resource controller STOP at %s: %s' % (label, ','.join(reasons)), reasons)
            self.hold_samples += 1
            hold_start = now if hold_start is None else hold_start
            if now - hold_start >= P['hold_seconds_per_boundary_max']:
                raise cc.ResourceStop('Resource controller hold timeout at %s: %s' % (label, ','.join(reasons)), [*reasons, 'hold_timeout'])
            self.sleep(P['sample_interval_seconds'])

    def preflight(self):
        return self.boundary('preflight', enforce_window=False)


# ---- local converter / assembly seam (differentially tested copies of the originals) -----------------------------------------------------------
def convert_v3(plan, source_dir, out_dir, backend, controller, *, adopted=frozenset()):
    """qc.convert, except: units in `adopted` are never touched; the start boundary HOLDs (with the unit forecast + reserve) instead of raising HeadroomBlock;
    the end-of-unit observation never raises before the receipt is written (a STOP is raised after the receipt)."""
    units = qc.validate_plan(plan, source_dir)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    binding = out_dir / 'plan_binding.json'
    if binding.exists():
        if json.loads(binding.read_text())['plan_sha256'] != plan['plan_sha256']:
            raise qc.BlockedUnit('Output directory is bound to a different plan')
    else:
        binding.write_text(json.dumps({'plan_sha256': plan['plan_sha256']}))
    done = qc.load_done(plan, out_dir)
    receipts_path = out_dir / 'receipts.jsonl'; converted = 0
    for u in units:
        if u['index'] in done or u['index'] in adopted:
            continue
        s = controller.boundary('unit_start', need_gib=qc.forecast_gib(u) + qc.RESERVE_GIB, unit=u['index'])
        backend.begin_unit()
        w, raw_sha = backend.load_unit(Path(source_dir) / u['source_file'], u)
        if u['transform'] == qc.VISION_CANON:
            w = backend.canonicalize_vision(u['normalized_key'], w)
        if backend.shape(w) != u['shape'] or backend.dtype(w) != qc.DTYPES[u['dtype']]:
            raise ValueError('Source tensor differs from plan')
        names = ('weight', 'scales', 'biases') if u['action'] == 'quantize' else ('tensor',)
        outs = backend.quantize(w) if u['action'] == 'quantize' else (w,)
        out_file = f"unit_{u['index']:04d}.safetensors"
        backend.save(out_dir / out_file, dict(zip(names, outs)))
        back = backend.reload(out_dir / out_file)
        reload_ok = all(backend.bitwise_equal(o, back[n]) for n, o in zip(names, outs))
        module_ok = retained_ok = None
        if u['action'] == 'quantize':
            via = backend.quantize_module(w, u['module_type'])
            module_ok = all(backend.bitwise_equal(a, b) for a, b in zip(outs, via))
            del via
        if u['action'] == 'retain':
            retained_ok = backend.bitwise_equal(w, back['tensor']) and (u['transform'] is not None or backend.bits_sha256(w) == raw_sha)
        peak = backend.peak_mib()
        shapes = {n: backend.shape(o) for n, o in zip(names, outs)}
        dtypes = {n: backend.dtype(o) for n, o in zip(names, outs)}
        del w, outs, back
        backend.clear()
        residual = backend.active_mib()
        s2, decision, reasons2 = controller.observe('unit_end', unit=u['index'])
        r = {'index': u['index'], 'plan_sha256': plan['plan_sha256'], 'unit_hash': qc.unit_hash(plan, u), 'source_key': u['source_key'],
             'normalized_key': u['normalized_key'], 'action': u['action'], 'transform': u['transform'], 'source_bytes_sha256': raw_sha, 'output_file': out_file,
             'output_file_sha256': qc.sha_file(out_dir / out_file), 'shapes': shapes, 'dtypes': dtypes, 'reload_bitwise_equal': reload_ok,
             'module_path_bitwise_equal': module_ok, 'retained_bitwise_equal': retained_ok, 'mlx_peak_mib': peak, 'mlx_active_after_release_mib': residual,
             'residual_within_budget': residual <= qc.MAX_RESIDUAL_ACTIVE_MIB, 'peak_within_budget': peak <= qc.MAX_UNIT_PEAK_MIB, 'forecast_gib': qc.forecast_gib(u),
             'swap_used_gib': [s.swap_used_gib, s2.swap_used_gib], 'available_gib': [s.available_gib, s2.available_gib],
             'pressure_free_percent': [s.pressure_free_percent, s2.pressure_free_percent]}
        r['passed'] = qc._passed(r)
        with receipts_path.open('a') as f:
            f.write(json.dumps(r, sort_keys=True) + '\n'); f.flush(); os.fsync(f.fileno())
        converted += 1
        if not r['passed']:
            raise qc.BlockedUnit('Check failed in unit %d (evidence preserved; unit blocked)' % u['index'])
        if decision == 'STOP':
            raise cc.ResourceStop('Resource controller STOP after unit %d (receipt written): %s' % (u['index'], ','.join(reasons2)), reasons2)
    finished = len(done) + len(adopted) + converted
    return {'units_total': len(units), 'units_done': finished, 'converted_this_call': converted, 'complete': finished == len(units)}


def assemble_v3(plan, out_dir, cache_dir, done, shard_limit_bytes=4 << 30):
    """qc.assemble_cache with the receipt map injected (so the adopted unit, which has an adoption record instead of a receipt, can be included)."""
    out_dir, cache_dir = Path(out_dir), Path(cache_dir)
    if len(done) != len(plan['units']):
        raise ValueError('Conversion is incomplete; refusing to assemble a partial cache')
    cache_dir.mkdir(parents=True, exist_ok=False)
    entries = []
    for u in plan['units']:
        header, base = qc._read_st_header(out_dir / done[u['index']]['output_file'])
        for part, meta in header.items():
            name = (u['module'] + '.' + part) if u['action'] == 'quantize' else u['normalized_key']
            entries.append((name, meta['dtype'], meta['shape'], out_dir / done[u['index']]['output_file'], base + meta['data_offsets'][0], base + meta['data_offsets'][1]))
    names = [e[0] for e in entries]
    if len(set(names)) != len(names):
        raise ValueError('Duplicate tensor names in the assembled cache')
    groups, cur, size = [], [], 0
    for e in entries:
        n = e[5] - e[4]
        if cur and size + n > shard_limit_bytes:
            groups.append(cur); cur, size = [], 0
        cur.append(e); size += n
    groups.append(cur)
    weight_map, total = {}, 0
    for gi, group in enumerate(groups, 1):
        fname = f'model-{gi:05d}-of-{len(groups):05d}.safetensors'
        header, pos = {}, 0
        for name, dtype, shape, _, b, e in group:
            header[name] = {'dtype': dtype, 'shape': shape, 'data_offsets': [pos, pos + (e - b)]}; pos += e - b
            weight_map[name] = fname
        raw = json.dumps(header, separators=(',', ':')).encode()
        raw += b' ' * (-len(raw) % 8)
        with (cache_dir / fname).open('wb') as out:
            out.write(len(raw).to_bytes(8, 'little')); out.write(raw)
            for name, dtype, shape, src, b, e in group:
                with src.open('rb') as f:
                    f.seek(b); left = e - b
                    while left:
                        chunk = f.read(min(qc.CHUNK, left)); out.write(chunk); left -= len(chunk)
            out.flush(); os.fsync(out.fileno())
        total += pos
    (cache_dir / 'model.safetensors.index.json').write_text(json.dumps({'metadata': {'total_size': total}, 'weight_map': weight_map}, sort_keys=True))
    manifest = {'plan_sha256': plan['plan_sha256'], 'tensor_count': len(names), 'shard_sha256': {p.name: qc.sha_file(p) for p in sorted(cache_dir.glob('*.safetensors'))},
                'note': 'Derived exact cache; config/tokenizer integration and loader identity certification are separate.'}
    (cache_dir / 'cache_manifest.json').write_text(json.dumps(manifest, sort_keys=True, indent=1))
    return {'shards': len(groups), 'tensors': len(names), 'total_bytes': total}


def certify_v3(plan, units_dir, cache_dir, source_dir, qmap, done):
    """quant_cache.certify_cache with the receipt map injected."""
    from mri_rl.quant_cache import WEIGHT_INDEX, WEIGHT_SUFFIXES, quant_config
    cache_dir, source_dir, units_dir = Path(cache_dir), Path(source_dir), Path(units_dir)
    if len(done) != len(plan['units']):
        raise ValueError('Not all units have passing receipts')
    index = json.loads((cache_dir / WEIGHT_INDEX).read_text())['weight_map']
    shard_headers, file_sizes = {}, {}
    for shard in sorted(set(index.values())):
        header, base = qc._read_st_header(cache_dir / shard)
        size = (cache_dir / shard).stat().st_size; end = 0
        for meta in sorted(header.values(), key=lambda m: m['data_offsets'][0]):
            if meta['data_offsets'][0] != end:
                raise ValueError('Shard tensor ranges are not contiguous: ' + shard)
            end = meta['data_offsets'][1]
        if base + end != size:
            raise ValueError('Shard size differs from its header: ' + shard)
        shard_headers[shard] = (header, base); file_sizes[shard] = size
    checked = 0
    for u in plan['units']:
        rec = done[u['index']]
        uh, ubase = qc._read_st_header(units_dir / rec['output_file'])
        parts = [('weight', 'weight'), ('scales', 'scales'), ('biases', 'biases')] if u['action'] == 'quantize' else [('tensor', None)]
        for upart, cpart in parts:
            name = (u['module'] + '.' + cpart) if cpart else u['normalized_key']
            ch, cbase = shard_headers[index[name]]
            um, cm = uh[upart], ch[name]
            if um['dtype'] != cm['dtype'] or um['shape'] != cm['shape']:
                raise ValueError('Cache tensor layout differs from its unit: ' + name)
            n = um['data_offsets'][1] - um['data_offsets'][0]
            if n != cm['data_offsets'][1] - cm['data_offsets'][0]:
                raise ValueError('Cache tensor size differs: ' + name)
            with (units_dir / rec['output_file']).open('rb') as a, (cache_dir / index[name]).open('rb') as b:
                a.seek(ubase + um['data_offsets'][0]); b.seek(cbase + cm['data_offsets'][0]); left = n
                while left:
                    k = min(64 << 20, left)
                    if a.read(k) != b.read(k):
                        raise ValueError('Cache payload bytes differ from the verified unit: ' + name)
                    left -= k
            checked += 1
    cfg_src = json.loads((source_dir / 'config.json').read_text())
    cfg_new = json.loads((cache_dir / 'config.json').read_text())
    delta = {k for k in set(cfg_src) | set(cfg_new) if cfg_src.get(k) != cfg_new.get(k)}
    if delta != {'quantization', 'quantization_config'} or cfg_new['quantization'] != quant_config(qmap) or cfg_new['quantization_config'] != quant_config(qmap):
        raise ValueError('Config differs from the source beyond the exact quantization keys')
    if qc.sha_file(cache_dir / 'config_source.json') != qc.sha_file(source_dir / 'config.json'):
        raise ValueError('config_source.json is not the original config')
    others = {}
    for p in sorted(source_dir.iterdir()):
        if p.is_file() and p.suffix not in WEIGHT_SUFFIXES and p.name not in (WEIGHT_INDEX, 'config.json'):
            if qc.sha_file(cache_dir / p.name) != qc.sha_file(p):
                raise ValueError('Non-weight source file differs in the cache: ' + p.name)
            others[p.name] = qc.sha_file(p)
    return {'plan_sha256': plan['plan_sha256'], 'units': len(plan['units']), 'tensors_checked': checked, 'quantized_modules': len(qmap),
            'retained_tensors': sum(u['action'] == 'retain' for u in plan['units']),
            'shards': {s: {'bytes': file_sizes[s], 'sha256': qc.sha_file(cache_dir / s)} for s in sorted(file_sizes)}, 'config_delta_keys': sorted(delta),
            'config_source_sha256': qc.sha_file(cache_dir / 'config_source.json'), 'config_cached_sha256': qc.sha_file(cache_dir / 'config.json'),
            'non_weight_files_identical': others, 'payload_bytes_equal_to_verified_units': True,
            'scope': 'bytes+lineage only; no model load, no inference; effective-weight loader parity needs the measured cached-load profile'}


def verify_historical_v3(plan, units_dir, cache_dir, source_dir, qmap, hist, expected, done):
    """cc.verify_historical with the receipt map injected: plan-derived layout, certificate, then the pinned historical identity."""
    cc.verify_assembly(plan, cache_dir, units_dir=units_dir, historical_shards=hist['shards'])
    cert = certify_v3(plan, units_dir, cache_dir, source_dir, qmap, done)
    diff = [k for k in cc.HISTORICAL_KEYS if cert.get(k) != hist.get(k)]
    if diff:
        raise ValueError('Reconstructed cache differs from the pinned historical certificate: ' + ','.join(diff))
    if (cert['units'], cert['quantized_modules'], cert['retained_tensors'], cert['tensors_checked'], len(cert['shards'])) != (
            expected.units, expected.quantized, expected.retained, expected.tensors, expected.shards):
        raise ValueError('Certified counts differ from the expected inventory')
    return cert


# ---- CPU preparation (never imports MLX) --------------------------------------------------------------------------------------------------
def _pending(plan, expected):
    quant = [u for u in plan['units'] if u['action'] == 'quantize']
    return [u for u in quant if u['index'] != ADOPTED_UNIT]


def plan_report_v3(*, plan_path, expected_plan_sha256, cert_path, cert_sha256, proof_path, proof_sha256, quant_map_path, source_dir, repo_root,
                   previous_pins=PRODUCTION_PINS, hash_sources=False, expected=cc.Expected(), disk_usage=shutil.disk_usage):
    plan, hist, proofs = v2.load_pins(plan_path, expected_plan_sha256, cert_path, cert_sha256, proof_path, proof_sha256, expected)
    roots = canonical_roots(repo_root, v2.canon_path(source_dir))
    prev = verify_previous_attempt(repo_root, previous_pins, expected)
    qmap = json.loads(Path(quant_map_path).read_text()); quant = [u for u in plan['units'] if u['action'] == 'quantize']
    if (len(plan['units']), len(quant)) != (expected.units, expected.quantized) or set(qmap) != {u['module'] for u in quant} or any(v != qc.QUANT for v in qmap.values()):
        raise ValueError('Plan/quant-map inventory differs from the expected scope')
    old_units = Path(roots['previous_work_root']) / 'units'
    retained = v2.authenticate_retained(plan, proofs, old_units)
    u0 = plan['units'][ADOPTED_UNIT]
    adoption = adoption_record(plan, proofs, u0, old_units / ('unit_%04d.safetensors' % ADOPTED_UNIT), source_dir, prev, previous_pins)
    qc.validate_plan(plan, source_dir); src = 'plan_validated'
    if hash_sources:
        qc.verify_source_shards(plan, source_dir); src = 'plan_validated_and_shards_hashed'
    pending = _pending(plan, expected)
    need = v2.disk_estimate_bytes(plan, pending)
    return {'command': 'plan', 'executed': False, 'scope_sha256': SCOPE_SHA256, 'plan_sha256': plan['plan_sha256'], 'plan_units': len(plan['units']),
            'retained_units_adopted': len(retained), 'adopted_saved_units': 1, 'quantized_pending_gpu': len(pending),
            'previous_attempt_underlying_quantizations': prev['counts']['gpu_underlying_quantizations'],
            'new_underlying_quantizations_max': 2 * len(pending), 'cumulative_underlying_quantizations_max': MAX_CUMULATIVE_UNDERLYING,
            'cumulative_within_cap': prev['counts']['gpu_underlying_quantizations'] + 2 * len(pending) <= MAX_CUMULATIVE_UNDERLYING,
            'historical_shards': {k: v['bytes'] for k, v in hist['shards'].items()}, 'disk_need_gib': round(need / 1024**3, 2),
            'disk_free_gib': round(disk_usage(v2._existing_ancestor(roots['work_root'])).free / 1024**3, 2), 'source_checks': src,
            'work_root_exists': Path(roots['work_root']).exists(), 'mlx_imported': any(m == 'mlx' or m.startswith('mlx.') for m in sys.modules),
            'module_sha256': qc.sha_file(__file__), '_prev': prev, '_adoption': adoption, '_roots': roots, '_plan': plan, '_proofs': proofs, '_hist': hist, '_qmap': qmap}


def _public(report):
    return {k: v for k, v in report.items() if not k.startswith('_')}


def prepare_scope_v3(*, plan_path, expected_plan_sha256, cert_path, cert_sha256, proof_path, proof_sha256, quant_map_path, source_dir, scope_output, repo_root,
                     previous_pins=PRODUCTION_PINS, expected=cc.Expected(), installed=v2.installed_fingerprint, entries=None, require_mlx=True,
                     disk_usage=shutil.disk_usage):
    """CPU only, never imports MLX: stream-hash every source shard, authenticate the closed v2 evidence, the 406 retained and the saved unit 0, fingerprint the
    whole local source closure and the installed MLX, and publish an IMMUTABLE private scope (never overwritten). It authorizes nothing."""
    r = Path(repo_root); out = v2.canon_path(scope_output)
    v2._req(out.parent == r / SCOPE_DIR_REL and not out.exists(), 'Scope output must be a NEW file directly under %s' % (r / SCOPE_DIR_REL))
    rep = plan_report_v3(plan_path=plan_path, expected_plan_sha256=expected_plan_sha256, cert_path=cert_path, cert_sha256=cert_sha256, proof_path=proof_path,
                         proof_sha256=proof_sha256, quant_map_path=quant_map_path, source_dir=source_dir, repo_root=repo_root, previous_pins=previous_pins,
                         hash_sources=True, expected=expected, disk_usage=disk_usage)
    plan, prev, adoption, roots = rep['_plan'], rep['_prev'], rep['_adoption'], rep['_roots']
    inst = installed()
    v2._req(not require_mlx or (inst.get('mlx_version') and inst.get('mlx_tree_sha256')), 'Installed MLX is not discoverable by this interpreter')
    inventory = [{'index': u['index'], 'action': u['action'], 'key': u['normalized_key'], 'file': u['source_file'], 'offset': u['offset'], 'nbytes': u['nbytes'],
                  'shape': u['shape'], 'module': u['module'], 'module_type': u['module_type'], 'unit_hash': qc.unit_hash(plan, u),
                  'proof_payload_sha256': rep['_proofs'][u['index']]['output_tensor_payload_sha256']} for u in plan['units']]
    files = {'plan': v2._file_binding(plan_path), 'quant_map': v2._file_binding(quant_map_path), 'historical_certificate': v2._file_binding(cert_path),
             'historical_proof': v2._file_binding(proof_path), **{'prev_' + k: dict(v) for k, v in prev['files'].items()}}
    previous_attempt = {k: prev[k] for k in ('counts', 'scope_digest', 'instruction_sha256', 'retained_receipts', 'quantized_receipts')}
    inputs = {'roots': roots, 'files': files, 'source_shards': {k: dict(v) for k, v in plan['source_shards'].items()}, 'inventory_sha256': shared.identity(inventory),
              'plan_sha256': plan['plan_sha256'], 'expected': asdict(expected), 'installed': inst, 'previous_attempt': previous_attempt, 'adoption': adoption}
    scope = {'schema': SCOPE_SCHEMA, 'scope_kind': SCOPE_KIND, 'experiment': 'VERO_TO_MRI_ACCURACY_RECOVERY_V1', 'action': ACTION, 'implemented_by': v2.IMPLEMENTED_BY,
             'static_scope_sha256': SCOPE_SHA256, 'limits': dict(SCOPE), **inputs,
             'counts': {'plan_units': rep['plan_units'], 'retained_units_adopted': rep['retained_units_adopted'], 'adopted_saved_units': 1,
                        'quantized_pending_gpu': rep['quantized_pending_gpu']},
             'source_sha256': v2.local_closure(entries or default_entries()), 'input_sha256': shared.identity(inputs)}
    shared.publish_once(out, scope)
    return {'scope_path': str(out), 'scope_file_sha256': shared.file_sha256(out), 'scope_sha256': shared.scope_digest(scope), 'input_sha256': scope['input_sha256'],
            'source_files_in_closure': len(scope['source_sha256']), 'source_shards_verified': len(scope['source_shards']), 'quantized_pending_gpu': rep['quantized_pending_gpu']}


def default_entries():
    return [Path(__file__), Path(__file__).resolve().parents[2] / 'scripts' / 'rebuild_accuracy_recovery_historical_gpu_cache_v3.py']


def adopt_continuation(*, plan, proofs, repo_root, source_dir, previous_pins=PRODUCTION_PINS, expected=cc.Expected()):
    """CPU file copy into the NEW v3 work root: the 406 authenticated retained units (+ receipt lines) and the exact saved unit 0 (no receipt) with its
    explicit adoption record. Everything is verified BEFORE anything is created; the old roots are only read."""
    roots = canonical_roots(repo_root, source_dir); work = Path(roots['work_root']); old_units = Path(roots['previous_work_root']) / 'units'
    prev = verify_previous_attempt(repo_root, previous_pins, expected)
    auth = v2.authenticate_retained(plan, proofs, old_units)
    u0 = plan['units'][ADOPTED_UNIT]; saved = old_units / ('unit_%04d.safetensors' % ADOPTED_UNIT)
    record = adoption_record(plan, proofs, u0, saved, source_dir, prev, previous_pins)
    units = work / 'units'
    units.mkdir(parents=True, exist_ok=False)
    for i, (line, src) in sorted(auth.items()):
        with src.open('rb') as a, (units / src.name).open('xb') as b:
            shutil.copyfileobj(a, b, qc.CHUNK); b.flush(); os.fsync(b.fileno())
    with saved.open('rb') as a, (units / saved.name).open('xb') as b:
        shutil.copyfileobj(a, b, qc.CHUNK); b.flush(); os.fsync(b.fileno())
    with (units / 'receipts.jsonl').open('x') as f:
        f.write(''.join(line + '\n' for _, (line, _) in sorted(auth.items()))); f.flush(); os.fsync(f.fileno())
    (units / 'plan_binding.json').write_text(json.dumps({'plan_sha256': plan['plan_sha256']}))
    v2.authenticate_retained(plan, proofs, units)
    if adoption_record(plan, proofs, u0, units / saved.name, source_dir, prev, previous_pins) != record:
        raise ValueError('Adopted copy of the saved unit differs from the verified original')
    (work / 'adoption').mkdir()
    rec_path = work / 'adoption' / 'unit0_adoption_PRIVATE.json'
    shared.publish_once(rec_path, record)
    manifest = {'plan_sha256': plan['plan_sha256'], 'scope_sha256': SCOPE_SHA256, 'adopted_retained_units': len(auth), 'adopted_saved_units': 1,
                'adopted_indices_sha256': qc.sha_bytes(qc.canonical(sorted(auth)).encode()), 'adoption_record_sha256': shared.file_sha256(rec_path),
                'quantized_cpu_outputs_adopted': 0}
    with (work / 'adopt_manifest.json').open('x') as f:
        f.write(json.dumps(manifest, sort_keys=True, indent=1))
    return manifest


# ---- admission ------------------------------------------------------------------------------------------------------------------------------------
class Admission3(v2.Admission):
    def __init__(self, scope, human, installed, consumed_dir, previous_consumed_dir):
        super().__init__(scope, human, installed, consumed_dir)
        self.previous_consumed_dir = Path(previous_consumed_dir)

    def already_consumed(self):
        sha = self.human['human_instruction_sha256']
        if (self.consumed_dir / ('scope_consumed_%s.json' % self.digest)).exists():
            return 'This exact scope was already consumed; a new scope with fresh human + independent review is required'
        if any((d / ('authorization_consumed_%s.json' % sha)).exists() for d in (self.consumed_dir, self.previous_consumed_dir)):
            return 'Human authorization instruction was already consumed; a fresh instruction is required'
        return None

    def consume(self):
        msg = self.already_consumed()
        if msg:
            raise ScopeRefused(msg)
        super().consume()


def admit_v3(*, scope_path, scope_sha256, human_path, human_sha256, review_path, review_sha256, now, repo_root, expected=cc.Expected(),
             installed=v2.installed_fingerprint, previous_pins=PRODUCTION_PINS):
    scope = v2._load_pinned(scope_path, scope_sha256, 'Prepared v3 execution scope')
    human = v2._load_pinned(human_path, human_sha256, 'Human authorization')
    review = v2._load_pinned(review_path, review_sha256, 'Independent review')
    _req(isinstance(scope, dict) and scope.get('schema') == SCOPE_SCHEMA and scope.get('scope_kind') == SCOPE_KIND and scope.get('action') == ACTION and
         'mode' not in scope and v2._flag_free(scope), 'Not a prepared historical GPU cache v3 continuation scope (or it carries an approval flag)')
    _req(scope.get('static_scope_sha256') == SCOPE_SHA256 and scope.get('limits') == SCOPE, 'Scope limits/policy differ from the reviewed v3 budget and resource policy')
    _req(scope.get('implemented_by') == v2.IMPLEMENTED_BY, 'Scope must name its implementing agent kind')
    _req(expected != cc.Expected() or scope.get('plan_sha256') == v2.PLAN_SHA256, 'Production scope is not bound to the frozen plan')
    _req(scope.get('expected') == asdict(expected), 'Scope inventory expectation differs (production: 750/344/406/1438/2 shards)')
    roots = scope.get('roots') or {}
    _req(roots == canonical_roots(repo_root, roots.get('source_dir')) and roots.get('source_dir') and str(v2.canon_path(roots['source_dir'])) == roots['source_dir'] and
         Path(roots['source_dir']).is_dir(), 'Scope roots differ from the canonical roots derived from the git common dir')
    inputs = {k: scope.get(k) for k in ('roots', 'files', 'source_shards', 'inventory_sha256', 'plan_sha256', 'expected', 'installed', 'previous_attempt', 'adoption')}
    _req(scope.get('input_sha256') == shared.identity(inputs), 'Scope input digest is inconsistent with its inputs')
    pa, ad = scope.get('previous_attempt'), scope.get('adoption')
    _req(isinstance(pa, dict) and pa.get('counts') == OLD_COUNTS and pa.get('scope_digest') == previous_pins.get('scope_digest') and pa.get('quantized_receipts') == 0,
         'Scope does not bind the closed v2 attempt (counts / old scope digest)')
    _req(isinstance(ad, dict) and ad.get('schema') == ADOPTION_SCHEMA and ad.get('unit_index') == ADOPTED_UNIT and ad.get('old_receipt_fabricated') is False and
         ad.get('receipt_equivalent') is False and ad.get('old_attempt_opcounts') == OLD_COUNTS, 'Scope lacks the explicit saved-unit adoption record')
    _req(all(('prev_' + k) in scope.get('files', {}) for k in ('scope', 'human', 'review', 'failure_receipt', 'ledger', 'consumed_scope', 'consumed_instruction', 'consumed_outcome')),
         'Scope does not hash-bind the previous attempt evidence')
    closure = scope.get('source_sha256')
    _req(isinstance(closure, dict) and all(shared.sha(v) for v in closure.values()) and all(any(p.endswith('/' + n) for p in closure) for n in REQUIRED_CLOSURE),
         'Complete local source closure required')
    adm = Admission3(scope, human, installed, roots['consumed_dir'], roots['previous_consumed_dir'])
    adm.check('admission', full=True)
    binding = adm.digest
    _req(isinstance(human, dict) and human.get('authorized') is True and human.get('action') == ACTION and human.get('scope_sha256') == binding and
         shared.sha(human.get('human_instruction_sha256')) and human.get('approved_by'), 'Fresh exact human authorization (instruction hash, action, scope digest) required')
    try:
        issued, expires = (datetime.fromisoformat(human[k].replace('Z', '+00:00')) for k in ('issued_at_utc', 'expires_at_utc'))
    except (KeyError, ValueError, AttributeError):
        raise ScopeRefused('Human authorization lacks issue/expiry times') from None
    _req(issued <= now < expires and (expires - issued).total_seconds() <= v2.MAX_AUTH_HOURS * 3600, 'Human authorization is not current or lasts too long')
    _req(isinstance(review, dict) and review.get('verdict') in ('PASS', 'PASS_WITH_LIMITATIONS') and review.get('scope_sha256') == binding and review.get('reviewer_id') and
         review.get('reviewer_kind') in v2.REVIEWER_KINDS and review.get('reviewer_kind') != scope['implemented_by'] and review.get('source_sha256') == closure and
         review.get('input_sha256') == scope['input_sha256'],
         'Independent review of this exact scope/source closure/inputs by a different agent kind (PASS or PASS_WITH_LIMITATIONS) required')
    msg = adm.already_consumed()
    _req(msg is None, msg or '')
    return adm


# ---- execute ---------------------------------------------------------------------------------------------------------------------------------------
def refuse_if_attempted(work_root):
    work_root = Path(work_root)
    for name in ('v3_ledger.json', 'v3_active.lock'):
        if (work_root / name).exists():
            raise ScopeRefused('%s exists: the single authorized v3 GPU execution was already started; no rerun' % name)
    if list(work_root.glob('v3_failure_*.json')):
        raise ScopeRefused('A v3 failure receipt exists; STOP, no second attempt')


def write_failure(work_root, plan, exc, backend, extra):
    for n in range(1, 10_000):
        path = Path(work_root) / ('v3_failure_%03d.json' % n)
        try:
            f = path.open('x')
        except FileExistsError:
            continue
        with f:
            f.write(json.dumps({'plan_sha256': plan['plan_sha256'], 'scope_sha256': SCOPE_SHA256, 'error_type': type(exc).__name__, 'error': str(exc)[:500],
                                'reasons': getattr(exc, 'reasons', None), 'automatic_retry': False, 'rerun_allowed': False,
                                'new_gpu_unit_attempts': getattr(backend, 'attempts', 0), 'new_direct_quantize_ops': getattr(backend, 'quantize_ops', 0),
                                'new_module_path_quantizations': getattr(backend, 'module_path_checks', 0),
                                'new_underlying_quantizations': getattr(backend, 'underlying_quantizations', 0), 'current_unit': getattr(backend, 'current', None),
                                **extra}, sort_keys=True))
            f.flush(); os.fsync(f.fileno())
        return path


def execute(*, scope_path, scope_sha256, human_path, human_sha256, review_path, review_sha256, repo_root=None, expected=cc.Expected(),
            backend_factory=v2.make_gpu_backend, sampler=None, os_level=None, rss_gib=cc._ps_rss_gib, clock=time.monotonic, sleep=time.sleep,
            now=lambda: datetime.now(timezone.utc), installed=v2.installed_fingerprint, disk_usage=shutil.disk_usage, previous_pins=PRODUCTION_PINS):
    from mri_rl.quant_cache import finalize_cache
    adm = admit_v3(scope_path=scope_path, scope_sha256=scope_sha256, human_path=human_path, human_sha256=human_sha256, review_path=review_path,
                   review_sha256=review_sha256, now=now(), repo_root=repo_root or v2.repo_root_from_git(), expected=expected, installed=installed, previous_pins=previous_pins)
    scope = adm.scope; roots, files = scope['roots'], scope['files']
    work = Path(roots['work_root']); units_dir, cache_dir, cert_out = work / 'units', work / 'cache', work / 'certificate.json'
    lock_path = Path(roots['global_lock'])
    _req(lock_path.exists() and stat.S_ISREG(os.lstat(lock_path).st_mode) and v2.canon_path(lock_path) == lock_path,
         'The canonical remaining20 global lock must be the existing regular, non-symlink file')
    lock_fd = os.open(lock_path, os.O_RDWR)          # existing file only: never created, truncated, written or unlinked
    try:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ScopeRefused('Canonical remaining20 global lock is held by another job; refused, not forced, deleted or modified') from None
        refuse_if_attempted(work)
        adm.check('after_global_lock', full=True)
        _, hist, proofs = v2.load_pins(files['plan']['path'], scope['plan_sha256'], files['historical_certificate']['path'], files['historical_certificate']['sha256'],
                                       files['historical_proof']['path'], files['historical_proof']['sha256'], expected)
        plan, qmap = cc.prepare(files['plan']['path'], scope['plan_sha256'], roots['source_dir'], files['quant_map']['path'],
                                {k: v['sha256'] for k, v in scope['source_shards'].items()}, expected)       # re-streams every source shard
        prev = verify_previous_attempt(roots['repo_root'], previous_pins, expected)
        _req({k: prev[k] for k in scope['previous_attempt']} == scope['previous_attempt'], 'Previous attempt evidence differs from the prepared scope')
        if cache_dir.exists() or cert_out.exists():
            raise ValueError('Cache dir and certificate must not already exist')
        if not (work / 'adopt_manifest.json').is_file() or not (work / 'adoption/unit0_adoption_PRIVATE.json').is_file():
            raise ValueError('Continuation not adopted: run adopt-continuation first')
        record = json.loads((work / 'adoption/unit0_adoption_PRIVATE.json').read_text())
        _req(record == scope['adoption'], 'Adoption record differs from the prepared scope')
        u0 = plan['units'][ADOPTED_UNIT]
        if adoption_record(plan, proofs, u0, units_dir / ('unit_%04d.safetensors' % ADOPTED_UNIT), roots['source_dir'], prev, previous_pins) != record:
            raise ValueError('Adopted unit 0 in the work root differs from its adoption record')
        retained = v2.authenticate_retained(plan, proofs, units_dir)
        pending = [u for u in plan['units'] if u['index'] not in retained and u['index'] != ADOPTED_UNIT]
        if any(u['action'] != 'quantize' for u in pending) or len(pending) != expected.quantized - 1 or \
                {p.name for p in units_dir.glob('unit_*.safetensors')} != {f.name for _, f in retained.values()} | {record['file']}:
            raise ValueError('Work root must hold exactly the authenticated retained units and the adopted unit; every other unit is pending GPU quantization')
        _req(prev['counts']['gpu_underlying_quantizations'] + 2 * len(pending) <= MAX_CUMULATIVE_UNDERLYING, 'Cumulative underlying quantizations would exceed the cap')
        ctrl = ResourceController(sampler=sampler, os_level=os_level, clock=clock, sleep=sleep, log_path=work / 'v3_resource.jsonl')
        ctrl.preflight()
        v2.disk_gate(work, v2.disk_estimate_bytes(plan, pending), usage=disk_usage)
        adm.check('before_consumption', full=True)
        adm.consume()                                   # scope + human instruction consumed; everything below is final, no retry
        lock = work / 'v3_active.lock'; ledger = work / 'v3_ledger.json'
        with ledger.open('x') as f:
            f.write(json.dumps({'scope_sha256': adm.digest, 'human_instruction_sha256': adm.human['human_instruction_sha256'], 'started_utc': now().isoformat(),
                                'pending_quantized_units': len(pending), 'previous_attempt_scope_digest': prev['scope_digest']}, sort_keys=True)); f.flush(); os.fsync(f.fileno())
        with lock.open('x') as f:
            f.write(str(os.getpid())); f.flush(); os.fsync(f.fileno())
        backend = None; t0 = clock(); ctrl.start_window()
        try:
            adm.check('before_backend', full=True)                   # last gate before the first MLX import
            backend = backend_factory(proofs, clock=clock, t0=t0, rss_gib=rss_gib, rss_log=work / 'v3_rss.jsonl', work_root=work, disk_usage=disk_usage,
                                      disk_need=lambda u: int(u['nbytes'] * qc.QUANT_OUT_FRACTION), max_attempts=len(pending), max_quantize=len(pending),
                                      max_module=len(pending), max_underlying=2 * len(pending), phase_check=adm.check)
            res = convert_v3(plan, roots['source_dir'], units_dir, backend, ctrl, adopted=frozenset({ADOPTED_UNIT}))
            if not res['complete']:
                raise qc.BlockedUnit('Conversion did not complete')
            for u in pending:                                                          # saved files re-checked against the proof
                v2.check_unit_file(plan, u, proofs[u['index']], units_dir / f"unit_{u['index']:04d}.safetensors")
            done = qc.load_done(plan, units_dir)
            done = {**done, ADOPTED_UNIT: {'output_file': record['file'], 'adopted': True}}
            ctrl.boundary('pre_assembly', enforce_window=False)
            assemble_v3(plan, units_dir, cache_dir, done)
            finalize_cache(plan, units_dir, cache_dir, roots['source_dir'], qmap)
            cert = verify_historical_v3(plan, units_dir, cache_dir, roots['source_dir'], qmap, hist, expected, done)
            ctrl.observe('post_verify')
        except Exception as exc:
            doneN = qc.load_done(plan, units_dir) if (units_dir / 'receipts.jsonl').exists() else {}
            write_failure(work, plan, exc, backend, {'stage': 'execute', 'last_resource_sample': ctrl.last, 'units_with_passing_receipts': len(doneN), 'partial': True,
                                                     'elapsed_seconds': clock() - t0, 'hold_samples': ctrl.hold_samples})
            adm.outcome('FAILED', '%s: %s' % (type(exc).__name__, exc))
            raise
        finally:
            if backend is not None:
                backend.release()
            lock.unlink(missing_ok=True)
        cert['gpu_reconstruction'] = {
            'device': 'gpu', 'scope_sha256': adm.digest, 'retained_units_adopted': len(retained), 'retained_units_gpu_operations': 0, 'adopted_saved_units': 1,
            'adopted_unit_basis': 'v3 explicit adoption record; no old receipt fabricated', 'adopted_unit_evidence_missing': record['evidence_missing'],
            'new_gpu_unit_attempts': backend.attempts, 'new_direct_quantize_ops': backend.quantize_ops, 'new_module_path_quantizations': backend.module_path_checks,
            'new_underlying_quantizations': backend.underlying_quantizations,
            'previous_attempt_underlying_quantizations': prev['counts']['gpu_underlying_quantizations'],
            'cumulative_underlying_quantizations': prev['counts']['gpu_underlying_quantizations'] + backend.underlying_quantizations,
            'cumulative_underlying_quantizations_max': MAX_CUMULATIVE_UNDERLYING, 'model_loaded': False, 'forward_or_backward': False, 'training': False,
            'peak_process_rss_gib': max(backend.rss_peak_gib, ctrl.peak_rss_gib), 'elapsed_seconds': clock() - t0, 'hold_samples': ctrl.hold_samples,
            'every_unit_payload_equals_pinned_historical_proof': True}
        cert['scope'] += '; GPU re-quantization continuation of the original base cache, not model training or inference'
        with cert_out.open('x') as f:
            f.write(json.dumps(cert, sort_keys=True, indent=1)); f.flush(); os.fsync(f.fileno())
        adm.outcome('SUCCEEDED', cert['gpu_reconstruction']['cumulative_underlying_quantizations'])
        return {'certificate': cert}
    finally:
        os.close(lock_fd)                               # releases the flock; the canonical lock file itself is preserved


def main(argv=None, expected=cc.Expected(), repo_root=None, installed=v2.installed_fingerprint, previous_pins=PRODUCTION_PINS):
    import argparse
    p = argparse.ArgumentParser(description='Cache v3 continuation (execute needs scope + human + independent review; other commands never import MLX).')
    p.add_argument('command', choices=('plan', 'prepare-scope', 'adopt-continuation', 'execute'))
    for n in ('scope', 'scope-sha256', 'human-authorization', 'human-authorization-sha256', 'independent-review', 'independent-review-sha256', 'plan',
              'expected-plan-sha256', 'expected-cache-certificate', 'expected-cache-certificate-sha256', 'historical-proof', 'historical-proof-sha256', 'quant-map',
              'source-dir', 'scope-output'):
        p.add_argument('--' + n)
    p.add_argument('--hash-sources', action='store_true')
    a = p.parse_args(argv)
    root = Path(repo_root) if repo_root else v2.repo_root_from_git()
    if a.command == 'execute':
        need = (a.scope, a.scope_sha256, a.human_authorization, a.human_authorization_sha256, a.independent_review, a.independent_review_sha256)
        if not all(need):
            p.error('execute needs --scope, --human-authorization and --independent-review, each with its exact --*-sha256')
        res = execute(scope_path=a.scope, scope_sha256=a.scope_sha256, human_path=a.human_authorization, human_sha256=a.human_authorization_sha256,
                      review_path=a.independent_review, review_sha256=a.independent_review_sha256, repo_root=root, expected=expected, previous_pins=previous_pins)
        print(json.dumps(res['certificate']['gpu_reconstruction'], sort_keys=True)); return
    for n in ('plan', 'expected_plan_sha256', 'expected_cache_certificate', 'expected_cache_certificate_sha256', 'historical_proof', 'historical_proof_sha256', 'quant_map', 'source_dir'):
        if not getattr(a, n):
            p.error('--%s is required' % n.replace('_', '-'))
    pins = dict(plan_path=a.plan, expected_plan_sha256=a.expected_plan_sha256, cert_path=a.expected_cache_certificate, cert_sha256=a.expected_cache_certificate_sha256,
                proof_path=a.historical_proof, proof_sha256=a.historical_proof_sha256)
    if a.command == 'plan':
        rep = plan_report_v3(**pins, quant_map_path=a.quant_map, source_dir=a.source_dir, repo_root=root, previous_pins=previous_pins, hash_sources=a.hash_sources, expected=expected)
        print(json.dumps(_public(rep), sort_keys=True)); return
    if a.command == 'adopt-continuation':
        plan, hist, proofs = v2.load_pins(**pins, expected=expected)
        print(json.dumps(adopt_continuation(plan=plan, proofs=proofs, repo_root=root, source_dir=a.source_dir, previous_pins=previous_pins, expected=expected), sort_keys=True)); return
    if not a.scope_output:
        p.error('prepare-scope needs --scope-output')
    print(json.dumps(prepare_scope_v3(**pins, quant_map_path=a.quant_map, source_dir=a.source_dir, scope_output=a.scope_output, repo_root=root, previous_pins=previous_pins,
                                      expected=expected, installed=installed), sort_keys=True))


if __name__ == '__main__':
    main()
