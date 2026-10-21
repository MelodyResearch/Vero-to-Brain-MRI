"""Thin CPU-only wrapper around the exact quant-cache pipeline (quant_convert / quant_cache); no new quantization algorithm.

It rebuilds the derived 4-bit/affine/group-64 cache one tensor at a time: NativeConvertBackend's per-unit work is reused
unchanged but every array/module/save/reload/compare call is forced and verified to run on mx.cpu. It never loads a model,
never runs forward/backward/generation/optimizer steps and never touches MRI. The module-path bitwise comparison builds ONE
nn.Linear/nn.Embedding for the unit (not a Qwen model, not training) and replaces its weight before nn.quantize.
Honest limits: on Apple-silicon MLX the CPU stream still allocates through MLX's unified-memory allocator, so MLX
peak/active counters are informational only and are left visible to the monitor (sys.modules is never touched). The 8 GiB
process-RSS envelope is only SAMPLED at operation boundaries, not a hard mid-call guarantee. Between units the strict local
gate (paced_resources.decisions + a ResourcePolicy) refuses on any HOLD/STOP: AC power only, known pressure/availability,
no throttling, no other heavy process. Swap: the first sample is the baseline for the WHOLE reconstruction (convert, assemble,
finalize, certify; never reset) and net growth above 1 GiB, or an unknown swap sensor, stops; absolute OS swap need not be 0.
No Metal memory/cache limit is set (that would initialize the GPU allocator).
Identity: the result is only published if it matches an externally pinned historical cache certificate (see verify_historical).
"""
from __future__ import annotations
import json
import os
from dataclasses import dataclass
from pathlib import Path
from mri_rl import quant_convert as qc
from mri_rl.quant_convert import NativeConvertBackend

MAX_UNIT_ATTEMPTS = 750
MAX_QUANTIZED = 344
RSS_ENVELOPE_GIB = 8.0


@dataclass(frozen=True)
class Expected:
    units: int = 750
    quantized: int = 344
    retained: int = 406
    tensors: int = 1438      # 344 * (weight, scales, biases) + 406 retained: the historical target
    shards: int = 2          # historical assembled shard count; only tests override the Expected fields


class CpuGuardError(RuntimeError):
    """The MLX default device is not the CPU; no array was created."""


class ResourceStop(RuntimeError):
    """Resource gate refusal (HOLD/STOP, non-AC, unknown sensor, swap growth, RSS); `reasons` are the logged readings' reasons."""

    def __init__(self, message, reasons=()):
        super().__init__(message)
        self.reasons = list(reasons)


class AttemptCapExceeded(RuntimeError):
    """More unit attempts or quantizations than the exact plan allows."""


SWAP_GROWTH_CAP_GIB = 1.0


def guarded_sampler(base=None, envelope_gib=RSS_ENVELOPE_GIB, swap_cap_gib=SWAP_GROWTH_CAP_GIB, log_path=None):
    """Strict CPU-conversion gate around a real sampler: refuses on anything but CONTINUE from the existing paced decisions
    (AC strictly, unknown/occupied process inventory, pressure, availability, throttling) or a CRITICAL supervisor level,
    on unknown/over-envelope RSS, an unknown swap sensor, or net swap growth over `swap_cap_gib` since the FIRST sample."""
    from mri_rl.paced_resources import decisions
    from mri_rl.resource_supervisor import ResourcePolicy
    if base is None:
        from mri_rl.resource_supervisor import sample_macos as base
    policy = ResourcePolicy()

    def sample():
        s = base()
        level, sup = policy.observe(s)
        _, reasons = decisions(s, loading=False, supervisor_reasons=sup)
        reasons = list(reasons)
        if level == 'CRITICAL':
            reasons.append('supervisor_critical')
        if s.process_rss_gib is None or s.process_rss_gib > envelope_gib:
            reasons.append('process_rss_unknown_or_above_envelope')
        growth = None
        if s.swap_used_gib is None:
            reasons.append('swap_sensor_unknown')
        else:
            if sample.baseline_swap_gib is None:
                sample.baseline_swap_gib = s.swap_used_gib
            growth = s.swap_used_gib - sample.baseline_swap_gib
            if growth > swap_cap_gib:
                reasons.append('stage_swap_net_growth_over_cap')
        event = {'reasons': reasons, 'swap_growth_gib': growth, 'baseline_swap_gib': sample.baseline_swap_gib,
                 'available_gib': s.available_gib, 'pressure_free_percent': s.pressure_free_percent, 'swap_used_gib': s.swap_used_gib,
                 'power_source': s.power_source, 'battery_percent': s.battery_percent, 'thermal_state': s.thermal_state,
                 'other_heavy_processes': s.other_heavy_processes, 'process_rss_gib': s.process_rss_gib}
        sample.last = event
        if log_path:
            with Path(log_path).open('a') as f:
                f.write(json.dumps(event, sort_keys=True) + '\n'); f.flush(); os.fsync(f.fileno())
        if reasons:
            raise ResourceStop('Resource gate refused: ' + ','.join(reasons), reasons)
        sample.peak_rss_gib = max(sample.peak_rss_gib, s.process_rss_gib)
        return s
    sample.peak_rss_gib, sample.baseline_swap_gib, sample.last = 0.0, None, None
    return sample


def _ps_rss_gib():
    from mri_rl.resource_supervisor import _run
    t = _run(['ps', '-o', 'rss=', '-p', str(os.getpid())])
    try:
        return int((t or '').strip()) / 1024**2
    except ValueError:
        return None


class CpuConvertBackend(NativeConvertBackend):
    """NativeConvertBackend with the device forced/verified to CPU around every operation and hard unit/RSS caps."""

    def __init__(self, mx, np, nn, *, rss_gib=_ps_rss_gib, rss_log=None, max_attempts=MAX_UNIT_ATTEMPTS, max_quantized=MAX_QUANTIZED):
        self.mx, self.np, self.nn = mx, np, nn       # deliberately NOT super().__init__: it sets Metal memory/cache limits
        self.rss_gib, self.rss_log = rss_gib, rss_log
        self.max_attempts, self.max_quantized = min(max_attempts, MAX_UNIT_ATTEMPTS), min(max_quantized, MAX_QUANTIZED)
        self.attempts = self.quantized = 0
        self.rss_peak_gib = 0.0
        self.current = None
        mx.set_default_device(mx.cpu)                # before any array or module exists
        self._check()

    def _check(self):
        if self.mx.default_device() != self.mx.cpu:
            raise CpuGuardError('MLX default device is not the CPU; refusing to create arrays')

    def _rss(self, op):
        g = self.rss_gib()
        if g is None or g > RSS_ENVELOPE_GIB:
            raise ResourceStop('RSS %s GiB unknown or above %.1f GiB after %s' % (g, RSS_ENVELOPE_GIB, op))
        self.rss_peak_gib = max(self.rss_peak_gib, g)
        if self.rss_log:
            with Path(self.rss_log).open('a') as f:
                f.write(json.dumps({'unit': self.current, 'op': op, 'rss_gib': g}) + '\n'); f.flush(); os.fsync(f.fileno())

    def _run(self, op, fn, *a, **k):
        self._check()
        with self.mx.stream(self.mx.cpu):
            out = fn(self, *a, **k)
        self._check()
        self._rss(op)
        return out

    def load_unit(self, shard, unit):
        if self.attempts >= self.max_attempts:
            raise AttemptCapExceeded('Unit attempt cap %d reached' % self.max_attempts)
        self.attempts += 1; self.current = unit.get('normalized_key')
        return self._run('load_unit', NativeConvertBackend.load_unit, shard, unit)

    def quantize(self, w):
        if self.quantized >= self.max_quantized:
            raise AttemptCapExceeded('Quantize cap %d reached' % self.max_quantized)
        self.quantized += 1
        return self._run('quantize', NativeConvertBackend.quantize, w)


def _wrap(name):
    base = getattr(NativeConvertBackend, name)
    setattr(CpuConvertBackend, name, lambda self, *a, **k: self._run(name, base, *a, **k))


for _name in ('begin_unit', 'canonicalize_vision', 'quantize_module', 'bitwise_equal', 'bits_sha256', 'save', 'reload',
              'peak_mib', 'active_mib', 'clear'):
    _wrap(_name)


def make_cpu_backend(**kw):
    """The only place MLX/numpy are imported; called by rebuild() only when units are actually pending."""
    import numpy as np
    import mlx.core as mx
    import mlx.nn as nn
    return CpuConvertBackend(mx, np, nn, **kw)


# ---- identity / path gates --------------------------------------------------------------------------------------
def prepare(plan_path, expected_plan_sha256, source_dir, quant_map_path, expected_source_sha256, expected=Expected(), verify_sources=True):
    plan = json.loads(Path(plan_path).read_text())
    if plan.get('plan_sha256') != expected_plan_sha256:
        raise ValueError('Plan hash differs from the expected plan hash')
    qc.validate_plan(plan, source_dir)                  # recomputes the plan hash, pins 4-bit/affine/g64, shard ranges
    units = plan['units']
    quant = [u for u in units if u['action'] == 'quantize']
    if (len(units), len(quant), len(units) - len(quant)) != (expected.units, expected.quantized, expected.retained):
        raise ValueError('Plan unit counts differ from the expected inventory')
    if expected.tensors != 3 * len(quant) + (len(units) - len(quant)):
        raise ValueError('Expected tensor count is inconsistent with the unit counts')
    qmap = json.loads(Path(quant_map_path).read_text())
    if set(qmap) != {u['module'] for u in quant} or any(v != qc.QUANT for v in qmap.values()):
        raise ValueError('Quantization map differs from the plan modules or from affine 4-bit group 64')
    if {k: v['sha256'] for k, v in plan['source_shards'].items()} != dict(expected_source_sha256):
        raise ValueError('Plan source shard hashes differ from the supplied expected hashes')
    if verify_sources:
        qc.verify_source_shards(plan, source_dir)
    return plan, qmap


def check_paths(work_root, units_dir, cache_dir, certificate_output, source_dir=None, **_ignored):
    root = Path(work_root).resolve()
    u, c, o = (Path(p).resolve() for p in (units_dir, cache_dir, certificate_output))
    for p in (u, c, o):
        if p == root or not p.is_relative_to(root):
            raise ValueError('Output path must be strictly inside the work root: %s' % p)
    for raw in (units_dir, cache_dir, certificate_output):
        if Path(raw).is_symlink():
            raise ValueError('Output path must not be a symlink')
    if u == c or u.is_relative_to(c) or c.is_relative_to(u) or o.is_relative_to(u) or o.is_relative_to(c):
        raise ValueError('Units dir, cache dir and certificate must be disjoint')
    if c.exists() or o.exists():
        raise ValueError('Cache dir and certificate must not already exist (no overwrite)')
    if source_dir is not None:
        s = Path(source_dir).resolve()
        if any(p == s or p.is_relative_to(s) or s.is_relative_to(p) for p in (u, c, o)):
            raise ValueError('Outputs must not overlap the source dir')
    return u, c, o


HISTORICAL_KEYS = ('plan_sha256', 'units', 'tensors_checked', 'quantized_modules', 'retained_tensors', 'shards', 'config_delta_keys',
                   'config_source_sha256', 'config_cached_sha256', 'non_weight_files_identical', 'payload_bytes_equal_to_verified_units')


def load_historical(path, sha256, expected_plan_sha256, expected):
    """The externally pinned historical cache certificate: file hash, plan and counts must match; nothing is derived from this run."""
    if not path or not sha256:
        raise ValueError('An expected historical cache certificate and its sha256 are required')
    raw = Path(path).read_bytes()
    if qc.sha_bytes(raw) != sha256:
        raise ValueError('Historical cache certificate hash differs from the pinned sha256')
    h = json.loads(raw)
    if any(k not in h for k in HISTORICAL_KEYS):
        raise ValueError('Historical cache certificate lacks required fields')
    if (h['plan_sha256'], h['units'], h['quantized_modules'], h['retained_tensors'], h['tensors_checked'], len(h['shards'])) != (
            expected_plan_sha256, expected.units, expected.quantized, expected.retained, expected.tensors, expected.shards):
        raise ValueError('Historical cache certificate differs from the expected plan/counts/shard count')
    if not any(n.startswith('tokenizer') for n in h['non_weight_files_identical']):
        raise ValueError('Historical certificate does not pin tokenizer identity')
    return h


def verify_assembly(plan, cache_dir, shard_limit_bytes=4 << 30):
    """Deterministic key->shard binding and byte sizes recomputed from the PLAN (not from this run's output)."""
    entries = []
    for u in plan['units']:
        n = u['nbytes']
        if u['action'] == 'quantize':   # packed 4-bit weight, bf16 scales and biases per 64-element group
            entries += [(u['module'] + '.weight', n // 4), (u['module'] + '.scales', n // 64), (u['module'] + '.biases', n // 64)]
        else:
            entries.append((u['normalized_key'], n))
    groups, cur, size = [], [], 0
    for e in entries:                   # same grouping rule as quant_convert.assemble_cache
        if cur and size + e[1] > shard_limit_bytes:
            groups.append(cur); cur, size = [], 0
        cur.append(e); size += e[1]
    groups.append(cur)
    want, sizes = {}, {}
    for gi, group in enumerate(groups, 1):
        fname = f'model-{gi:05d}-of-{len(groups):05d}.safetensors'
        for name, n in group:
            want[name] = fname; sizes[name] = n
    cache_dir = Path(cache_dir)
    index = json.loads((cache_dir / 'model.safetensors.index.json').read_text())
    if index['weight_map'] != want or index['metadata']['total_size'] != sum(sizes.values()):
        raise ValueError('Cache index differs from the plan-derived key/shard mapping')
    if {p.name for p in cache_dir.glob('*.safetensors')} != set(want.values()):
        raise ValueError('Cache shard files differ from the plan-derived set')
    for fname in set(want.values()):
        header, _ = qc._read_st_header(cache_dir / fname)
        if {k: v['data_offsets'][1] - v['data_offsets'][0] for k, v in header.items()} != {k: sizes[k] for k, f in want.items() if f == fname}:
            raise ValueError('Cache tensor byte sizes differ from the plan-derived sizes: ' + fname)


def verify_historical(plan, units_dir, cache_dir, source_dir, qmap, hist, expected):
    """certify_cache (consistency with this run's verified units) + the pinned historical identity + plan-derived assembly."""
    from mri_rl.quant_cache import certify_cache
    verify_assembly(plan, cache_dir)
    cert = certify_cache(plan, units_dir, cache_dir, source_dir, qmap)
    diff = [k for k in HISTORICAL_KEYS if cert.get(k) != hist.get(k)]
    if diff:
        raise ValueError('Reconstructed cache differs from the pinned historical certificate: ' + ','.join(diff))
    want = (expected.units, expected.quantized, expected.retained, expected.tensors, expected.shards)
    if (cert['units'], cert['quantized_modules'], cert['retained_tensors'], cert['tensors_checked'], len(cert['shards'])) != want:
        raise ValueError('Certified counts differ from the expected inventory')
    return cert


def _write_failure(units_dir, plan, exc, backend, extra):
    d = Path(units_dir); d.mkdir(parents=True, exist_ok=True)
    for n in range(1, 10_000):
        path = d / ('cpu_failure_%03d.json' % n)
        try:
            f = path.open('x')
        except FileExistsError:
            continue
        with f:
            f.write(json.dumps({'plan_sha256': plan['plan_sha256'], 'error_type': type(exc).__name__, 'error': str(exc)[:500],
                                'unit_attempts': getattr(backend, 'attempts', 0), 'quantized': getattr(backend, 'quantized', 0),
                                'automatic_retry': False, 'completed_unit_receipts_preserved': True,
                                'reasons': getattr(exc, 'reasons', None), **extra}, sort_keys=True))
            f.flush(); os.fsync(f.fileno())
        return path


def rebuild(*, plan_path, expected_plan_sha256, source_dir, quant_map_path, expected_source_sha256, work_root, units_dir, cache_dir,
            certificate_output, expected_cache_certificate=None, expected_cache_certificate_sha256=None, expected=Expected(),
            backend_factory=make_cpu_backend, sampler=None, rss_gib=_ps_rss_gib):
    from mri_rl.quant_cache import finalize_cache
    from mri_rl.resource_supervisor import ResourcePolicy
    plan, qmap = prepare(plan_path, expected_plan_sha256, source_dir, quant_map_path, expected_source_sha256, expected)
    hist = load_historical(expected_cache_certificate, expected_cache_certificate_sha256, expected_plan_sha256, expected)
    check_paths(work_root, units_dir, cache_dir, certificate_output, source_dir)
    units_dir = Path(units_dir); units_dir.mkdir(parents=True, exist_ok=True)
    s = guarded_sampler(sampler, log_path=units_dir / 'cpu_resource.jsonl'); policy = ResourcePolicy()
    done, backend = {}, None
    try:
        binding = units_dir / 'plan_binding.json'
        if binding.exists() and json.loads(binding.read_text())['plan_sha256'] != plan['plan_sha256']:
            raise qc.BlockedUnit('Output directory is bound to a different plan')
        done = qc.load_done(plan, units_dir) if (units_dir / 'receipts.jsonl').exists() else {}
        pending = [u for u in plan['units'] if u['index'] not in done]
        if pending:
            qc.gate(s, policy)
            backend = backend_factory(rss_gib=rss_gib, rss_log=units_dir / 'cpu_rss.jsonl', max_attempts=len(pending),
                                      max_quantized=sum(u['action'] == 'quantize' for u in pending))
            res = qc.convert(plan, source_dir, units_dir, backend, s, policy)
            if not res['complete']:
                raise qc.BlockedUnit('Conversion did not complete')
        cache_dir = Path(cache_dir)
        qc.assemble_cache(plan, units_dir, cache_dir)
        s()   # same baseline: swap growth is bounded over the whole reconstruction, not per phase
        finalize_cache(plan, units_dir, cache_dir, source_dir, qmap)
        cert = verify_historical(plan, units_dir, cache_dir, source_dir, qmap, hist, expected)
        s()
    except Exception as exc:
        _write_failure(units_dir, plan, exc, backend, {'stage': 'rebuild', 'last_resource_sample': s.last})
        raise
    cert['cpu_reconstruction'] = {
        'device': 'cpu', 'unit_attempts_this_run': getattr(backend, 'attempts', 0), 'quantized_this_run': getattr(backend, 'quantized', 0),
        'adopted_units': len(done), 'model_loaded': False, 'forward_or_backward': False, 'training': False,
        'peak_process_rss_gib': max(getattr(backend, 'rss_peak_gib', 0.0), s.peak_rss_gib),
        'max_unit_attempts': MAX_UNIT_ATTEMPTS, 'max_quantized': MAX_QUANTIZED}
    cert['scope'] += '; native CPU cache reconstruction, not model training or inference'
    with Path(certificate_output).open('x') as f:
        f.write(json.dumps(cert, sort_keys=True, indent=1)); f.flush(); os.fsync(f.fileno())
    return {'certificate': cert, 'adopted_units': len(done)}


def main(argv=None, expected=Expected()):
    import argparse
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('command', choices=('plan', 'rebuild'))
    for name in ('expected-plan-sha256', 'source-dir', 'quant-map', 'plan', 'units-dir', 'cache-dir', 'certificate-output', 'work-root'):
        p.add_argument('--' + name, required=True)
    for name in ('expected-cache-certificate', 'expected-cache-certificate-sha256'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--expected-source-sha256', action='append', required=True, metavar='NAME=SHA256')
    a = p.parse_args(argv)
    shas = dict(x.split('=', 1) for x in a.expected_source_sha256)
    kw = dict(plan_path=a.plan, expected_plan_sha256=a.expected_plan_sha256, source_dir=a.source_dir, quant_map_path=a.quant_map,
              expected_source_sha256=shas, work_root=a.work_root, units_dir=a.units_dir, cache_dir=a.cache_dir,
              certificate_output=a.certificate_output, expected_cache_certificate=a.expected_cache_certificate,
              expected_cache_certificate_sha256=a.expected_cache_certificate_sha256, expected=expected)
    if a.command == 'plan':
        plan, _ = prepare(a.plan, a.expected_plan_sha256, a.source_dir, a.quant_map, shas, expected)
        load_historical(a.expected_cache_certificate, a.expected_cache_certificate_sha256, a.expected_plan_sha256, expected)
        check_paths(a.work_root, a.units_dir, a.cache_dir, a.certificate_output, a.source_dir)
        done = qc.load_done(plan, a.units_dir) if (Path(a.units_dir) / 'receipts.jsonl').exists() else {}
        pend = [u for u in plan['units'] if u['index'] not in done]
        print(json.dumps({'command': 'plan', 'executed': False, 'plan_sha256': plan['plan_sha256'], 'units': len(plan['units']),
                          'adopted_units': len(done), 'pending_units': len(pend),
                          'pending_quantized': sum(u['action'] == 'quantize' for u in pend),
                          'max_forecast_gib': max((qc.forecast_gib(u) for u in pend), default=0.0), 'device': 'cpu'}))
        return
    print(json.dumps(rebuild(**kw)['certificate']['cpu_reconstruction'], sort_keys=True))


if __name__ == '__main__':
    main()
