"""Finite, gated reconstruction of the ORIGINAL GPU-produced 4-bit base cache (runtime dependency only; no new model).

Why: the CPU rebuild (accuracy_recovery_cpu_cache_v1) reproduced all 406 retained tensors byte-exactly but 0/344 quantized
payloads (mx.quantize on CPU != the historical GPU result). This module re-quantizes ONLY those 344 units with the original
NativeConvertBackend/qc.convert recipe on the default GPU, one tensor at a time, and adopts a unit only if every output part's
payload SHA-256 equals the externally pinned historical 750-unit proof. First mismatch = STOP (no second attempt, no fallback).

Scope (fixed in SCOPE / SCOPE_SHA256, never read from a config flag): 750 plan units = 406 retained (copied, authenticated, from
the failed CPU root, which is only ever READ) + 344 quantized on GPU. No Qwen model is built or loaded; the module-path check builds
one temporary nn.Linear/nn.Embedding per quantized unit exactly as the original converter did. No inference, generation, backward,
optimizer, MRI or Adam. Counts are distinct: 344 GPU unit attempts, 344 quantize ops, 344 module-path checks.

Commands: `plan` (read-only, never imports MLX), `adopt-retained` (CPU file copy, never imports MLX), `execute` (the ONLY command
that can create MLX arrays; refuses without a fresh authorization artifact whose hashes bind this exact scope, module and script,
a once-only ledger, a lock, and the strict resource gate).

Compute/memory limitations (honest): the 8 GiB RSS limit and the disk/swap/pressure checks are SAMPLED at operation boundaries,
not enforced mid-call; the MLX 8 GiB memory limit / 64 MiB cache limit are MLX allocator guidelines inherited from
NativeConvertBackend and were not independently proven to be hard caps. MLX peak/active counters are informational. A thermal
sensor that reports SENSOR_UNAVAILABLE cannot refuse. The authorization artifact is only checked for hashes/expiry/nonce; the code
cannot prove who wrote it. Nothing here promises against an OS-level failure (a watchdog panic occurred on this Mac before).
"""
from __future__ import annotations
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from mri_rl import accuracy_recovery_cpu_cache_v1 as cc, quant_convert as qc
from mri_rl.quant_convert import NativeConvertBackend

PLAN_SHA256 = '71e2ca2709ae295d2b7aa3860701fd6ddc8ab3cd04e1a1cd011c07b420f2a68d'
WORK_ROOT_NAME = 'accuracy-recovery-historical-gpu-cache-v2'
CPU_ROOT_NAME = 'accuracy-recovery-exact-cache-v1'
MLX_MEMORY_LIMIT_BYTES = 8 * 1024**3     # must equal NativeConvertBackend.__init__ (test-enforced)
MLX_CACHE_LIMIT_BYTES = 64 * 1024**2
DISK_RESERVE_GIB = 20.0
WINDOW_SECONDS = 1800.0
MAX_GPU_UNIT_ATTEMPTS = MAX_GPU_QUANTIZE_OPS = MAX_GPU_MODULE_PATH_CHECKS = 344
SCOPE = {
    'name': 'VERO_TO_MRI_ACCURACY_RECOVERY_V1/historical_gpu_cache_v2', 'plan_sha256': PLAN_SHA256,
    'plan_units': 750, 'retained_units_adopted_from_cpu_root_readonly': 406, 'quantized_units_on_gpu': 344,
    'gpu_unit_attempts_max': MAX_GPU_UNIT_ATTEMPTS, 'gpu_quantize_ops_max': MAX_GPU_QUANTIZE_OPS,
    'gpu_module_path_checks_max': MAX_GPU_MODULE_PATH_CHECKS, 'quantization': dict(qc.QUANT), 'full_model_loaded': False,
    'inference_or_generation_or_backward_or_update': False, 'device': 'gpu', 'work_root_name': WORK_ROOT_NAME,
    'cpu_root_name_read_only': CPU_ROOT_NAME, 'mlx_memory_limit_bytes': MLX_MEMORY_LIMIT_BYTES, 'mlx_cache_limit_bytes': MLX_CACHE_LIMIT_BYTES,
    'rss_envelope_gib': cc.RSS_ENVELOPE_GIB, 'swap_net_growth_cap_gib': cc.SWAP_GROWTH_CAP_GIB, 'disk_reserve_gib': DISK_RESERVE_GIB,
    'operation_start_window_seconds': WINDOW_SECONDS, 'power': 'AC strictly', 'first_payload_mismatch': 'STOP_no_second_attempt',
    'automatic_retry': False, 'final_identity': 'pinned historical cache certificate incl. both shard SHA-256'}
SCOPE_SHA256 = qc.sha_bytes(qc.canonical(SCOPE).encode())
AUTH_SCHEMA = 'accuracy-recovery-historical-gpu-cache-v2-authorization'
MAX_AUTH_HOURS = 24


class GpuGuardError(RuntimeError):
    """MLX is not on a usable GPU (the CPU would reproduce the wrong bytes); no array was created."""


class HistoricalPayloadMismatch(RuntimeError):
    """A re-quantized part differs from the pinned historical payload SHA-256; STOP, no second attempt."""


class WindowExceeded(RuntimeError):
    """The 1800 s operation-start window elapsed before all quantized units started; STOP with a partial, no rerun."""


class ScopeRefused(RuntimeError):
    """Authorization/ledger/lock/pin check refused execution; no MLX array was created."""


# ---- hashing / proof --------------------------------------------------------------------------------------------
def hash_unit_file(path):
    """One streaming pass: (whole-file sha256, {part: payload sha256}); payload ranges must be contiguous with no trailing bytes."""
    fh = hashlib.sha256(); parts = {}
    with Path(path).open('rb') as f:
        head = f.read(8); n = int.from_bytes(head, 'little'); hb = f.read(n); fh.update(head); fh.update(hb)
        header = {k: v for k, v in json.loads(hb).items() if k != '__metadata__'}
        pos = 0
        for name, m in sorted(header.items(), key=lambda kv: kv[1]['data_offsets'][0]):
            a, b = m['data_offsets']
            if a != pos:
                raise ValueError('Unit payload ranges are not contiguous: ' + str(path))
            h, left = hashlib.sha256(), b - a
            while left:
                chunk = f.read(min(qc.CHUNK, left))
                if not chunk:
                    raise ValueError('Short unit file: ' + str(path))
                h.update(chunk); fh.update(chunk); left -= len(chunk)
            parts[name] = h.hexdigest(); pos = b
        if f.read(1):
            raise ValueError('Trailing bytes in unit file: ' + str(path))
    return fh.hexdigest(), parts


def load_pins(plan_path, expected_plan_sha256, cert_path, cert_sha256, proof_path, proof_sha256, expected=cc.Expected()):
    """Plan, historical cache certificate and the 750-unit proof, each bound by an external sha256 pin and to each other."""
    if expected_plan_sha256 != PLAN_SHA256 and expected == cc.Expected():
        raise ScopeRefused('Production scope is bound to plan %s' % PLAN_SHA256)
    plan = json.loads(Path(plan_path).read_text())
    if plan.get('plan_sha256') != expected_plan_sha256:
        raise ValueError('Plan hash differs from the pinned plan hash')
    hist = cc.load_historical(cert_path, cert_sha256, expected_plan_sha256, expected)
    if not proof_sha256 or qc.sha_file(proof_path) != proof_sha256:
        raise ValueError('Historical 750-unit proof hash differs from its pin')
    proof = json.loads(Path(proof_path).read_text())
    if proof.get('plan_sha256') != expected_plan_sha256 or [p['index'] for p in proof['units']] != list(range(expected.units)):
        raise ValueError('Proof is not for this plan or is not the full contiguous unit list')
    for u, p in zip(plan['units'], proof['units']):
        want = ['biases', 'scales', 'weight'] if u['action'] == 'quantize' else ['tensor']
        if p['unit_hash'] != qc.unit_hash(plan, u) or sorted(p['output_tensor_payload_sha256']) != want:
            raise ValueError('Proof entry does not match its plan unit: %d' % u['index'])
    return plan, hist, {p['index']: p for p in proof['units']}


def check_unit_file(plan, unit, proof_unit, path):
    file_sha, parts = hash_unit_file(path)
    if parts != proof_unit['output_tensor_payload_sha256']:
        raise ValueError('Unit %d payload differs from the pinned historical proof' % unit['index'])
    return file_sha


def authenticate_retained(plan, proofs, units_dir):
    """Every retained unit in `units_dir` (READ ONLY) must have a passing receipt bound to this plan/unit and match the proof exactly:
    source bytes, whole-file sha, payload sha. Returns {index: (receipt line, file path)}."""
    units_dir = Path(units_dir); lines = {}
    for line in (units_dir / 'receipts.jsonl').read_text().splitlines():
        r = json.loads(line)
        if r['index'] in lines:
            raise ValueError('Duplicate receipt for unit %d' % r['index'])
        lines[r['index']] = (line, r)
    out = {}
    for u in plan['units']:
        if u['action'] != 'retain':
            continue
        if u['index'] not in lines:
            raise ValueError('Retained unit %d has no receipt' % u['index'])
        line, r = lines[u['index']]; p = proofs[u['index']]; f = units_dir / r['output_file']
        if (r.get('plan_sha256') != plan['plan_sha256'] or r.get('unit_hash') != qc.unit_hash(plan, u) or not qc._passed(r)
                or r['output_file'] != f"unit_{u['index']:04d}.safetensors" or r['source_bytes_sha256'] != p['source_bytes_sha256']
                or r['output_file_sha256'] != p['output_file_sha256']):
            raise ValueError('Retained unit %d receipt does not match the plan/proof' % u['index'])
        if check_unit_file(plan, u, p, f) != p['output_file_sha256']:
            raise ValueError('Retained unit %d file sha differs from the proof' % u['index'])
        out[u['index']] = (line, f)
    return out


def adopt_retained(plan, proofs, cpu_units_dir, work_root):
    """Copy ONLY the authenticated retained unit files + their receipt lines into a NEW units dir; the CPU root is only read."""
    units = Path(work_root) / 'units'
    auth = authenticate_retained(plan, proofs, cpu_units_dir)
    units.mkdir(parents=True, exist_ok=False)
    for i, (line, src) in sorted(auth.items()):
        with src.open('rb') as a, (units / src.name).open('xb') as b:
            shutil.copyfileobj(a, b, qc.CHUNK); b.flush(); os.fsync(b.fileno())
    with (units / 'receipts.jsonl').open('x') as f:
        f.write(''.join(line + '\n' for _, (line, _) in sorted(auth.items()))); f.flush(); os.fsync(f.fileno())
    (units / 'plan_binding.json').write_text(json.dumps({'plan_sha256': plan['plan_sha256']}))
    authenticate_retained(plan, proofs, units)          # the copies themselves are re-verified against the proof
    manifest = {'plan_sha256': plan['plan_sha256'], 'scope_sha256': SCOPE_SHA256, 'adopted_retained_units': len(auth),
                'adopted_indices_sha256': qc.sha_bytes(qc.canonical(sorted(auth)).encode()), 'quantized_cpu_outputs_adopted': 0}
    with (Path(work_root) / 'adopt_manifest.json').open('x') as f:
        f.write(json.dumps(manifest, sort_keys=True, indent=1))
    return manifest


# ---- gates ------------------------------------------------------------------------------------------------------
def disk_estimate_bytes(plan, pending):
    out = sum(int(u['nbytes'] * qc.QUANT_OUT_FRACTION) for u in pending)
    cache = sum(int(u['nbytes'] * (qc.QUANT_OUT_FRACTION if u['action'] == 'quantize' else 1)) for u in plan['units'])
    return out + cache


def _existing_ancestor(path):
    p = Path(path)
    while not p.exists():
        p = p.parent
    return p


def disk_gate(path, need_bytes, reserve_gib=DISK_RESERVE_GIB, usage=shutil.disk_usage):
    free = usage(_existing_ancestor(path)).free
    if free < need_bytes + reserve_gib * 1024**3:
        raise cc.ResourceStop('Free disk %.1f GiB below need %.1f GiB + %.0f GiB reserve' % (free / 1024**3, need_bytes / 1024**3, reserve_gib),
                              ['disk_reserve'])
    return free


def check_authorization(path, sha256, *, now, plan_sha256=PLAN_SHA256, module_file=__file__, script_file=None):
    """Fresh, once-only human artifact: hashes bind THIS scope, module, script, plan and pins. Code cannot prove authorship."""
    if not path or not sha256 or qc.sha_file(path) != sha256:
        raise ScopeRefused('Authorization artifact missing or its sha256 differs from the supplied pin')
    a = json.loads(Path(path).read_text())
    script_file = script_file or Path(__file__).resolve().parents[2] / 'scripts' / 'rebuild_accuracy_recovery_historical_gpu_cache_v2.py'
    for k, want in (('schema', AUTH_SCHEMA), ('scope_sha256', SCOPE_SHA256), ('plan_sha256', plan_sha256),
                    ('module_sha256', qc.sha_file(module_file)), ('script_sha256', qc.sha_file(script_file))):
        if a.get(k) != want:
            raise ScopeRefused('Authorization field differs from the exact reviewed scope: ' + k)
    for k in ('historical_certificate_sha256', 'proof_sha256', 'independent_review_artifact_sha256'):
        if not isinstance(a.get(k), str) or len(a[k]) != 64:
            raise ScopeRefused('Authorization lacks a pinned hash: ' + k)
    h = a.get('human_authorization') or {}
    if not h.get('approved_by') or not isinstance(a.get('execution_nonce'), str) or len(a['execution_nonce']) < 16:
        raise ScopeRefused('Authorization lacks a human approver or a once-only nonce')
    issued, expires = (datetime.fromisoformat(h[k].replace('Z', '+00:00')) for k in ('issued_at_utc', 'expires_at_utc'))
    if not issued <= now < expires or expires - issued > timedelta(hours=MAX_AUTH_HOURS):
        raise ScopeRefused('Authorization is not current or lasts longer than %d h' % MAX_AUTH_HOURS)
    return a


def refuse_if_attempted(work_root):
    work_root = Path(work_root)
    for name in ('v2_ledger.json', 'v2_active.lock'):
        if (work_root / name).exists():
            raise ScopeRefused('%s exists: the single authorized GPU execution was already started; no rerun' % name)
    if list(work_root.glob('v2_failure_*.json')):
        raise ScopeRefused('A v2 failure receipt exists; STOP, no second attempt')


# ---- GPU backend ------------------------------------------------------------------------------------------------
class GpuConvertBackend(NativeConvertBackend):
    """Original NativeConvertBackend (same quantize / module-path / save / reload recipe) plus: GPU-only, quantized units only,
    per-part payload SHA vs the pinned proof BEFORE saving, distinct hard counts, RSS/disk/window checks at operation boundaries."""

    def __init__(self, proofs, *, mx=None, np=None, nn=None, clock=time.monotonic, t0=None, window_seconds=WINDOW_SECONDS,
                 rss_gib=cc._ps_rss_gib, rss_log=None, disk_need=lambda unit: 0, work_root=None, disk_usage=shutil.disk_usage,
                 max_attempts=MAX_GPU_UNIT_ATTEMPTS, max_quantize=MAX_GPU_QUANTIZE_OPS, max_module=MAX_GPU_MODULE_PATH_CHECKS):
        if mx is None:
            NativeConvertBackend.__init__(self)       # imports MLX, sets the 8 GiB memory / 64 MiB cache limits (original code)
        else:
            self.mx, self.np, self.nn = mx, np, nn
            mx.set_memory_limit(MLX_MEMORY_LIMIT_BYTES); mx.set_cache_limit(MLX_CACHE_LIMIT_BYTES)
        self.proofs, self.clock, self.window_seconds = proofs, clock, window_seconds
        self.t0 = clock() if t0 is None else t0
        self.rss_gib, self.rss_log, self.disk_need, self.work_root, self.disk_usage = rss_gib, rss_log, disk_need, work_root, disk_usage
        self.max_attempts, self.max_quantize, self.max_module = (min(max_attempts, MAX_GPU_UNIT_ATTEMPTS), min(max_quantize, MAX_GPU_QUANTIZE_OPS),
                                                                 min(max_module, MAX_GPU_MODULE_PATH_CHECKS))
        self.attempts = self.quantize_ops = self.module_path_checks = 0
        self.rss_peak_gib = 0.0; self.current = None; self.stopped = None
        self._check()

    def _check(self):
        if self.stopped:
            raise self.stopped
        mx = self.mx
        if mx.default_device() != mx.gpu or not mx.metal.is_available():
            raise GpuGuardError('MLX default device is not an available GPU; refusing (CPU would produce different quantized bytes)')

    def _rss(self, op):
        g = self.rss_gib()
        if g is None or g > cc.RSS_ENVELOPE_GIB:
            raise cc.ResourceStop('RSS %s GiB unknown or above %.1f GiB after %s' % (g, cc.RSS_ENVELOPE_GIB, op), ['process_rss_unknown_or_above_envelope'])
        self.rss_peak_gib = max(self.rss_peak_gib, g)
        if self.rss_log:
            with Path(self.rss_log).open('a') as f:
                f.write(json.dumps({'unit': self.current, 'op': op, 'rss_gib': g}) + '\n'); f.flush(); os.fsync(f.fileno())

    def _stop(self, exc):
        self.stopped = exc
        raise exc

    def begin_unit(self):
        self._check()
        if self.clock() - self.t0 > self.window_seconds:
            self._stop(WindowExceeded('Operation-start window of %.0f s elapsed with units still pending' % self.window_seconds))
        NativeConvertBackend.begin_unit(self)

    def load_unit(self, shard, unit):
        self._check()
        if unit['action'] != 'quantize' or unit['transform']:
            raise cc.AttemptCapExceeded('Only plain quantized units may touch the GPU; unit %d is not' % unit['index'])
        if self.attempts >= self.max_attempts:
            raise cc.AttemptCapExceeded('GPU unit attempt cap %d reached' % self.max_attempts)
        if self.work_root:
            disk_gate(self.work_root, self.disk_need(unit), usage=self.disk_usage)
        self.attempts += 1; self.current = unit['index']
        out = NativeConvertBackend.load_unit(self, shard, unit)
        self._check(); self._rss('load_unit')
        return out

    def quantize(self, w):
        self._check()
        if self.quantize_ops >= self.max_quantize:
            raise cc.AttemptCapExceeded('GPU quantize cap %d reached' % self.max_quantize)
        self.quantize_ops += 1
        outs = NativeConvertBackend.quantize(self, w)
        want = self.proofs[self.current]['output_tensor_payload_sha256']
        got = dict(zip(('weight', 'scales', 'biases'), (self.bits_sha256(o) for o in outs)))
        self._rss('quantize')
        if got != want:
            self._stop(HistoricalPayloadMismatch('Unit %d GPU payload differs from the pinned historical proof (parts: %s); nothing saved'
                                                 % (self.current, ','.join(sorted(k for k in want if got.get(k) != want[k])))))
        return outs

    def quantize_module(self, w, module_type):
        self._check()
        if self.module_path_checks >= self.max_module:
            raise cc.AttemptCapExceeded('GPU module-path check cap %d reached' % self.max_module)
        self.module_path_checks += 1
        out = NativeConvertBackend.quantize_module(self, w, module_type)
        self._check(); self._rss('quantize_module')
        return out

    def save(self, path, arrays):
        self._check(); NativeConvertBackend.save(self, path, arrays); self._rss('save')

    def reload(self, path):
        self._check(); out = NativeConvertBackend.reload(self, path); self._check(); self._rss('reload'); return out

    def clear(self):
        self._check(); NativeConvertBackend.clear(self)


def make_gpu_backend(proofs, **kw):
    """The only constructor that imports MLX; called from execute() after every gate."""
    return GpuConvertBackend(proofs, **kw)


# ---- pipeline ---------------------------------------------------------------------------------------------------
def check_roots(work_root, cpu_root=None):
    w = Path(work_root).resolve()
    if w.name != WORK_ROOT_NAME or Path(work_root).is_symlink():
        raise ValueError('Work root must be a non-symlink directory named ' + WORK_ROOT_NAME)
    if cpu_root is not None:
        c = Path(cpu_root).resolve()
        if c.name != CPU_ROOT_NAME or c == w or w.is_relative_to(c) or c.is_relative_to(w):
            raise ValueError('CPU root must be the separate failed root named ' + CPU_ROOT_NAME)
    return w


def write_failure(work_root, plan, exc, backend, extra):
    for n in range(1, 10_000):
        path = Path(work_root) / ('v2_failure_%03d.json' % n)
        try:
            f = path.open('x')
        except FileExistsError:
            continue
        with f:
            f.write(json.dumps({'plan_sha256': plan['plan_sha256'], 'scope_sha256': SCOPE_SHA256, 'error_type': type(exc).__name__,
                                'error': str(exc)[:500], 'reasons': getattr(exc, 'reasons', None), 'automatic_retry': False,
                                'rerun_allowed': False, 'gpu_unit_attempts': getattr(backend, 'attempts', 0),
                                'gpu_quantize_ops': getattr(backend, 'quantize_ops', 0),
                                'gpu_module_path_checks': getattr(backend, 'module_path_checks', 0), 'current_unit': getattr(backend, 'current', None),
                                **extra}, sort_keys=True))
            f.flush(); os.fsync(f.fileno())
        return path


def execute(*, plan_path, expected_plan_sha256, cert_path, cert_sha256, proof_path, proof_sha256, source_dir, quant_map_path,
            expected_source_sha256, work_root, authorization, authorization_sha256, expected=cc.Expected(),
            backend_factory=make_gpu_backend, sampler=None, rss_gib=cc._ps_rss_gib, clock=time.monotonic,
            now=lambda: datetime.now(timezone.utc), script_file=None, disk_usage=shutil.disk_usage):
    from mri_rl.quant_cache import finalize_cache
    from mri_rl.resource_supervisor import ResourcePolicy
    work = check_roots(work_root); units_dir, cache_dir, cert_out = work / 'units', work / 'cache', work / 'certificate.json'
    refuse_if_attempted(work)
    auth = check_authorization(authorization, authorization_sha256, now=now(), plan_sha256=expected_plan_sha256, script_file=script_file)
    if auth['historical_certificate_sha256'] != cert_sha256 or auth['proof_sha256'] != proof_sha256:
        raise ScopeRefused('Authorization pins differ from the supplied certificate/proof pins')
    _, hist, proofs = load_pins(plan_path, expected_plan_sha256, cert_path, cert_sha256, proof_path, proof_sha256, expected)
    plan, qmap = cc.prepare(plan_path, expected_plan_sha256, source_dir, quant_map_path, expected_source_sha256, expected)   # source/shard hashes
    if cache_dir.exists() or cert_out.exists():
        raise ValueError('Cache dir and certificate must not already exist')
    if not (work / 'adopt_manifest.json').is_file():
        raise ValueError('Retained units must be adopted first (adopt-retained)')
    adopted = authenticate_retained(plan, proofs, units_dir)                       # revalidate old receipts + retained file SHAs
    pending = [u for u in plan['units'] if u['index'] not in adopted]
    if any(u['action'] != 'quantize' for u in pending) or len(pending) != expected.quantized or \
            {p.name for p in units_dir.glob('unit_*.safetensors')} != {f.name for _, f in adopted.values()}:
        raise ValueError('Work root must hold exactly the authenticated retained units; every other unit is pending GPU quantization')
    s = cc.guarded_sampler(sampler, log_path=work / 'v2_resource.jsonl'); policy = ResourcePolicy()
    qc.gate(s, policy)
    disk_gate(work, disk_estimate_bytes(plan, pending), usage=disk_usage)
    lock = work / 'v2_active.lock'; ledger = work / 'v2_ledger.json'
    with ledger.open('x') as f:
        f.write(json.dumps({'execution_nonce': auth['execution_nonce'], 'authorization_sha256': authorization_sha256, 'scope_sha256': SCOPE_SHA256,
                            'started_utc': now().isoformat(), 'pending_quantized_units': len(pending)}, sort_keys=True)); f.flush(); os.fsync(f.fileno())
    with lock.open('x') as f:
        f.write(str(os.getpid())); f.flush(); os.fsync(f.fileno())
    backend = None; t0 = clock()
    try:
        backend = backend_factory(proofs, clock=clock, t0=t0, rss_gib=rss_gib, rss_log=work / 'v2_rss.jsonl', work_root=work, disk_usage=disk_usage,
                                  disk_need=lambda u: int(u['nbytes'] * qc.QUANT_OUT_FRACTION), max_attempts=len(pending),
                                  max_quantize=len(pending), max_module=len(pending))
        res = qc.convert(plan, source_dir, units_dir, backend, s, policy)
        if not res['complete']:
            raise qc.BlockedUnit('Conversion did not complete')
        for u in pending:                                                          # saved files re-checked against the proof
            check_unit_file(plan, u, proofs[u['index']], units_dir / f"unit_{u['index']:04d}.safetensors")
        qc.assemble_cache(plan, units_dir, cache_dir); s()
        finalize_cache(plan, units_dir, cache_dir, source_dir, qmap)
        cert = cc.verify_historical(plan, units_dir, cache_dir, source_dir, qmap, hist, expected); s()
    except Exception as exc:
        done = qc.load_done(plan, units_dir) if (units_dir / 'receipts.jsonl').exists() else {}
        write_failure(work, plan, exc, backend, {'stage': 'execute', 'last_resource_sample': s.last, 'units_with_passing_receipts': len(done),
                                                 'partial': True, 'elapsed_seconds': clock() - t0})
        raise
    finally:
        lock.unlink(missing_ok=True)
    cert['gpu_reconstruction'] = {
        'device': 'gpu', 'scope_sha256': SCOPE_SHA256, 'authorization_sha256': authorization_sha256, 'retained_units_adopted': len(adopted),
        'gpu_unit_attempts': backend.attempts, 'gpu_quantize_ops': backend.quantize_ops, 'gpu_module_path_checks': backend.module_path_checks,
        'max_unit_attempts': MAX_GPU_UNIT_ATTEMPTS, 'max_quantize_ops': MAX_GPU_QUANTIZE_OPS, 'max_module_path_checks': MAX_GPU_MODULE_PATH_CHECKS,
        'model_loaded': False, 'forward_or_backward': False, 'training': False, 'peak_process_rss_gib': max(backend.rss_peak_gib, s.peak_rss_gib),
        'elapsed_seconds': clock() - t0, 'every_unit_payload_equals_pinned_historical_proof': True}
    cert['scope'] += '; GPU re-quantization of the original base cache, not model training or inference'
    with cert_out.open('x') as f:
        f.write(json.dumps(cert, sort_keys=True, indent=1)); f.flush(); os.fsync(f.fileno())
    return {'certificate': cert}


def plan_report(*, plan_path, expected_plan_sha256, cert_path, cert_sha256, proof_path, proof_sha256, cpu_root, work_root, quant_map_path,
                source_dir=None, expected_source_sha256=None, hash_sources=False, expected=cc.Expected(), disk_usage=shutil.disk_usage):
    """Read-only, CPU-only (never imports MLX, writes nothing): scope, authentication of the 406 retained CPU outputs, disk estimate."""
    plan, hist, proofs = load_pins(plan_path, expected_plan_sha256, cert_path, cert_sha256, proof_path, proof_sha256, expected)
    work = check_roots(work_root, cpu_root)
    qmap = json.loads(Path(quant_map_path).read_text())
    quant = [u for u in plan['units'] if u['action'] == 'quantize']
    if (len(plan['units']), len(quant)) != (expected.units, expected.quantized) or set(qmap) != {u['module'] for u in quant} or \
            any(v != qc.QUANT for v in qmap.values()):
        raise ValueError('Plan/quant-map inventory differs from the expected 750/344 affine-4bit-g64 scope')
    cpu_units = Path(cpu_root) / 'units'
    adopted = authenticate_retained(plan, proofs, cpu_units)
    src = 'NOT_RUN'
    if source_dir:
        qc.validate_plan(plan, source_dir); src = 'plan_validated'
        if hash_sources:
            qc.verify_source_shards(plan, source_dir)
            if {k: v['sha256'] for k, v in plan['source_shards'].items()} != dict(expected_source_sha256 or {}):
                raise ValueError('Plan source shard hashes differ from the supplied expected hashes')
            src = 'plan_validated_and_shards_hashed'
    need = disk_estimate_bytes(plan, quant)
    return {'command': 'plan', 'executed': False, 'scope_sha256': SCOPE_SHA256, 'plan_sha256': plan['plan_sha256'], 'plan_units': len(plan['units']),
            'retained_authenticated_from_cpu_root': len(adopted), 'quantized_pending_gpu': len(quant), 'quantized_cpu_outputs_adopted': 0,
            'gpu_unit_attempts_max': MAX_GPU_UNIT_ATTEMPTS, 'gpu_quantize_ops_max': MAX_GPU_QUANTIZE_OPS,
            'gpu_module_path_checks_max': MAX_GPU_MODULE_PATH_CHECKS, 'historical_shards': {k: v['bytes'] for k, v in hist['shards'].items()},
            'max_unit_forecast_gib': max(qc.forecast_gib(u) for u in quant), 'disk_need_gib': round(need / 1024**3, 2),
            'disk_free_gib': round(disk_usage(_existing_ancestor(work)).free / 1024**3, 2), 'disk_reserve_gib': DISK_RESERVE_GIB,
            'work_root_exists': work.exists(), 'source_checks': src, 'mlx_imported': any(m == 'mlx' or m.startswith('mlx.') for m in sys.modules),
            'module_sha256': qc.sha_file(__file__)}


def main(argv=None, expected=cc.Expected()):
    import argparse
    p = argparse.ArgumentParser(description='Finite GPU re-quantization of the original base cache (execute needs a fresh artifact).')
    p.add_argument('command', choices=('plan', 'adopt-retained', 'execute'))
    for n in ('plan', 'expected-plan-sha256', 'expected-cache-certificate', 'expected-cache-certificate-sha256', 'historical-proof',
              'historical-proof-sha256', 'quant-map', 'work-root'):
        p.add_argument('--' + n, required=True)
    p.add_argument('--cpu-root'); p.add_argument('--source-dir'); p.add_argument('--hash-sources', action='store_true')
    p.add_argument('--expected-source-sha256', action='append', metavar='NAME=SHA256')
    p.add_argument('--authorization-artifact'); p.add_argument('--authorization-artifact-sha256')
    a = p.parse_args(argv)
    shas = dict(x.split('=', 1) for x in (a.expected_source_sha256 or []))
    pins = dict(plan_path=a.plan, expected_plan_sha256=a.expected_plan_sha256, cert_path=a.expected_cache_certificate,
                cert_sha256=a.expected_cache_certificate_sha256, proof_path=a.historical_proof, proof_sha256=a.historical_proof_sha256)
    if a.command == 'plan':
        if not a.cpu_root:
            p.error('plan needs --cpu-root')
        print(json.dumps(plan_report(**pins, cpu_root=a.cpu_root, work_root=a.work_root, quant_map_path=a.quant_map, source_dir=a.source_dir,
                                     expected_source_sha256=shas, hash_sources=a.hash_sources, expected=expected), sort_keys=True)); return
    if a.command == 'adopt-retained':
        if not a.cpu_root:
            p.error('adopt-retained needs --cpu-root')
        check_roots(a.work_root, a.cpu_root)
        plan, hist, proofs = load_pins(**pins, expected=expected)
        print(json.dumps(adopt_retained(plan, proofs, Path(a.cpu_root) / 'units', a.work_root), sort_keys=True)); return
    if not (a.source_dir and shas and a.authorization_artifact and a.authorization_artifact_sha256):
        p.error('execute needs --source-dir, --expected-source-sha256 and a fresh --authorization-artifact with its sha256')
    res = execute(**pins, source_dir=a.source_dir, quant_map_path=a.quant_map, expected_source_sha256=shas, work_root=a.work_root,
                  authorization=a.authorization_artifact, authorization_sha256=a.authorization_artifact_sha256, expected=expected)
    print(json.dumps(res['certificate']['gpu_reconstruction'], sort_keys=True))


if __name__ == '__main__':
    main()
