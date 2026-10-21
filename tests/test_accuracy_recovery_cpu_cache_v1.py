"""CPU-only tests for the exact quant-cache CPU wrapper. A numpy-backed fake mx/nn replaces MLX: no MLX import, no model."""
import contextlib
import hashlib
import json
import struct
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
from mri_rl import accuracy_recovery_cpu_cache_v1 as cc, quant_convert as qc
from tests.test_quant_convert import healthy, _st

SRC_ROOT = Path(__file__).parents[1] / 'src'


class Dt:
    def __init__(self, name): self.name = name
    def __eq__(self, o): return isinstance(o, Dt) and o.name == self.name
    def __hash__(self): return hash(self.name)
    def __str__(self): return 'mlx.core.' + self.name


class Arr:
    def __init__(self, data, dtype, dev):
        self.data, self.dtype, self.dev, self.shape = data, dtype, dev, tuple(data.shape)
    def view(self, dtype): return Arr(self.data.view(dtype_np[dtype.name]) if dtype.name == 'uint16' else self.data, dtype, self.dev)
    def __array__(self, dtype=None, copy=None): return self.data


dtype_np = {'uint16': np.uint16, 'uint32': np.uint32, 'bfloat16': np.uint16}
ST = {'uint32': 'U32', 'bfloat16': 'BF16', 'uint16': 'U16'}
BACK = {v: k for k, v in ST.items()}


class FakeMx:
    cpu, gpu = 'cpu', 'gpu'
    bfloat16, uint16, uint32 = Dt('bfloat16'), Dt('uint16'), Dt('uint32')

    def __init__(self, honor_set=True, mismatch=False, alter=0):
        self._dev, self.honor_set, self.mismatch, self.alter, self.ops, self.created = 'gpu', honor_set, mismatch, alter, [], []
    def _rec(self, name): self.ops.append((name, self._dev))
    def default_device(self): return self._dev
    def set_default_device(self, d):
        if self.honor_set: self._dev = d
    @contextlib.contextmanager
    def stream(self, d):
        old, self._dev = self._dev, d
        try: yield
        finally: self._dev = old
    def _mk(self, data, dtype):
        self.created.append(self._dev); return Arr(data, dtype, self._dev)
    def array(self, x): self._rec('array'); return self._mk(np.asarray(x), self.uint16)
    def eval(self, *a): self._rec('eval')
    def quantize(self, w, bits, group_size, mode):
        self._rec('quantize'); b = w.data.astype(np.uint32); n, k = b.shape
        s = ((b.reshape(n, k // 64, 64).sum(-1) + self.alter) % 65535).astype(np.uint16)
        return (self._mk(b.reshape(n, k // 8, 8).sum(-1).astype(np.uint32), self.uint32), self._mk(s, self.bfloat16), self._mk(s + 1, self.bfloat16))
    def array_equal(self, a, b):
        self._rec('array_equal'); r = (not self.mismatch) and bool(np.array_equal(a.data, b.data))
        return type('R', (), {'item': lambda s: r})()
    def save_safetensors(self, path, arrays):
        self._rec('save'); header, pos, data = {'__metadata__': {'format': 'mlx'}}, 0, b''
        for n in sorted(arrays):            # observed real unit header order: __metadata__, biases, scales, weight
            a = arrays[n]; raw = a.data.tobytes()
            header[n] = {'dtype': ST[a.dtype.name], 'shape': list(a.shape), 'data_offsets': [pos, pos + len(raw)]}; pos += len(raw); data += raw
        h = json.dumps(header, separators=(',', ':')).encode(); h += b' ' * (-len(h) % 8)
        Path(path).write_bytes(struct.pack('<Q', len(h)) + h + data)
    def load(self, path):
        self._rec('load'); raw = Path(path).read_bytes(); n = struct.unpack('<Q', raw[:8])[0]; h = json.loads(raw[8:8 + n]); out = {}
        for k, m in ((k, m) for k, m in h.items() if k != '__metadata__'):
            dt = BACK[m['dtype']]; a, b = m['data_offsets']
            out[k] = self._mk(np.frombuffer(raw[8 + n + a:8 + n + b], dtype=dtype_np[dt]).reshape(m['shape']), Dt(dt))
        return out
    def reset_peak_memory(self): self._rec('reset')
    def clear_cache(self): self._rec('clear')
    def get_peak_memory(self): self._rec('peak'); return 0
    def get_active_memory(self): self._rec('active'); return 0


class FakeNN:
    def __init__(self, mx): self.mx, self.built = mx, []
    class Module: pass
    def _layer(self, kind, shape):
        self.built.append((kind, tuple(shape), self.mx._dev)); o = type(kind, (), {})(); o.weight = None; return o
    def Linear(self, i, o, bias=False): return self._layer('Linear', (o, i))
    def Embedding(self, n, d): return self._layer('Embedding', (n, d))
    def quantize(self, holder, group_size, bits, mode):
        m = holder.m; m.weight, m.scales, m.biases = self.mx.quantize(m.weight, bits=bits, group_size=group_size, mode=mode)


def backend(rss=0.5, **kw):
    mx = kw.pop('mx', None) or FakeMx(); nn = FakeNN(mx); kw.setdefault('rss_gib', lambda: rss)
    return cc.CpuConvertBackend(mx, np, nn, **kw), mx, nn


def rows(n=4, k=64, seed=0): return np.random.default_rng(seed).integers(0, 60000, size=(n, k), dtype=np.uint16)


def make_history(args, plan, qmap):
    """Stand-in for the pinned historical certificate: the honest pipeline output in separate dirs, minus this wrapper's additions."""
    from mri_rl import quant_cache
    w = args['work_root']; mx = FakeMx(); b, _, _ = backend(mx=mx)
    qc.convert(plan, args['source_dir'], w / 'h_units', b, sampler())
    qc.assemble_cache(plan, w / 'h_units', w / 'h_cache'); quant_cache.finalize_cache(plan, w / 'h_units', w / 'h_cache', args['source_dir'], qmap)
    cert = quant_cache.certify_cache(plan, w / 'h_units', w / 'h_cache', args['source_dir'], qmap)
    (w / 'hist.json').write_text(json.dumps(cert, sort_keys=True))
    args['expected_cache_certificate'] = w / 'hist.json'
    args['expected_cache_certificate_sha256'] = hashlib.sha256((w / 'hist.json').read_bytes()).hexdigest()


def fixture(tmp_path):
    """Tiny real-numeric plan: linear [4,64], embedding [2,64] (both quantized), retained bf16 [4]; shard sha pinned."""
    src = tmp_path / 'src'; src.mkdir()
    parts = [rows(4, 64, 1), rows(2, 64, 2), rows(1, 4, 3).reshape(4)]
    shard = b''.join(p.tobytes() for p in parts); (src / 'a.safetensors').write_bytes(shard)
    specs = [('lin', 'linear', [4, 64]), ('embed_tokens', 'embedding', [2, 64]), (None, None, [4])]
    units, off = [], 0
    for i, ((mod, mt, shape), p) in enumerate(zip(specs, parts)):
        units.append({'index': i, 'source_key': f'k{i}', 'normalized_key': (mod + '.weight') if mod else 'norm.weight', 'source_file': 'a.safetensors',
                      'offset': off, 'nbytes': p.nbytes, 'source_shape': shape, 'dtype': 'BF16', 'shape': shape,
                      'action': 'quantize' if mod else 'retain', 'module': mod, 'module_type': mt, 'transform': None}); off += p.nbytes
    sha = hashlib.sha256(shard).hexdigest()
    plan = {'schema': 'exact-quant-plan-v1', 'quantization': dict(qc.QUANT), 'units': units,
            'source_shards': {'a.safetensors': {'bytes': len(shard), 'sha256': sha}}, 'lineage': {}}
    plan['plan_sha256'] = qc.sha_bytes(qc.canonical({k: v for k, v in plan.items() if k != 'plan_sha256'}).encode())
    (src / 'config.json').write_text(json.dumps({'model_type': 'qwen3_vl', 'a': 1})); (src / 'tokenizer.json').write_text('{"t":1}')
    work = tmp_path / 'work'; work.mkdir()
    (work / 'plan.json').write_text(json.dumps(plan))
    qmap = {'lin': dict(qc.QUANT), 'embed_tokens': dict(qc.QUANT)}; (work / 'qmap.json').write_text(json.dumps(qmap))
    args = dict(plan_path=work / 'plan.json', expected_plan_sha256=plan['plan_sha256'], source_dir=src, quant_map_path=work / 'qmap.json',
                expected_source_sha256={'a.safetensors': sha}, work_root=work, units_dir=work / 'units', cache_dir=work / 'cache',
                certificate_output=work / 'cert.json', expected=cc.Expected(units=3, quantized=2, retained=1, tensors=7, shards=1))
    make_history(args, plan, qmap)
    return plan, args


def sampler(**kw): return lambda: healthy(**kw)


def rebuild(args, mx=None, **kw):
    mx = mx or FakeMx(); made = []
    def factory(**fk):
        b, _, _ = backend(mx=mx, **fk); made.append(b); return b
    res = cc.rebuild(**args, backend_factory=factory, sampler=kw.pop('sampler', sampler()), rss_gib=lambda: 0.5, **kw)
    return res, mx, made


# ---- device guard -------------------------------------------------------------------------------------------
def test_init_refuses_when_default_device_cannot_be_forced_to_cpu_before_any_array():
    mx = FakeMx(honor_set=False)
    with pytest.raises(cc.CpuGuardError):
        backend(mx=mx)
    assert mx.created == [] and mx.ops == []


def test_init_forces_cpu_default_device_and_never_sets_metal_limits():
    b, mx, _ = backend()
    assert mx.default_device() == 'cpu' and not hasattr(mx, 'set_memory_limit')


def test_every_array_op_and_module_path_runs_under_cpu_through_full_conversion(tmp_path):
    plan, args = fixture(tmp_path)
    res, mx, _ = rebuild(args)
    assert mx.ops and {d for _, d in mx.ops} == {'cpu'} and set(mx.created) == {'cpu'}
    assert {n for n, _ in mx.ops} >= {'array', 'quantize', 'save', 'load', 'array_equal'}


def test_module_path_constructs_only_single_linear_or_embedding_on_cpu(tmp_path):
    plan, args = fixture(tmp_path)
    mx = FakeMx(); nns = []
    def factory(**fk):
        b, _, nn = backend(mx=mx, **fk); nns.append(nn); return b
    cc.rebuild(**args, backend_factory=factory, sampler=sampler(), rss_gib=lambda: 0.5)
    assert nns[0].built == [('Linear', (4, 64), 'cpu'), ('Embedding', (2, 64), 'cpu')]


def test_device_flip_to_gpu_mid_run_stops_before_the_next_array():
    b, mx, _ = backend()
    mx._dev = 'gpu'; n = len(mx.created)
    with pytest.raises(cc.CpuGuardError):
        b.begin_unit()
    with pytest.raises(cc.CpuGuardError):
        b.quantize(Arr(rows(), Dt('bfloat16'), 'cpu'))
    assert len(mx.created) == n


# ---- caps and resource envelope -----------------------------------------------------------------------------
def unit(i=0, n=512, shape=(4, 64), action='quantize'):
    return {'index': i, 'nbytes': n, 'offset': 0, 'source_shape': list(shape), 'action': action}


def test_attempt_and_quantize_caps_raise_without_extra_work(tmp_path):
    f = tmp_path / 's.bin'; f.write_bytes(rows().tobytes())
    b, mx, _ = backend(max_attempts=2, max_quantized=1)
    b.load_unit(f, unit()); b.load_unit(f, unit())
    n = len(mx.created)
    with pytest.raises(cc.AttemptCapExceeded):
        b.load_unit(f, unit())
    assert len(mx.created) == n
    w = Arr(rows(), Dt('bfloat16'), 'cpu'); b.quantize(w)
    with pytest.raises(cc.AttemptCapExceeded):
        b.quantize(w)


def test_default_caps_are_750_and_344():
    assert (cc.MAX_UNIT_ATTEMPTS, cc.MAX_QUANTIZED) == (750, 344)
    assert cc.Expected() == cc.Expected(units=750, quantized=344, retained=406, tensors=1438)


@pytest.mark.parametrize('rss', [9.0, None])
def test_actual_rss_over_envelope_or_unknown_stops_backend(tmp_path, rss):
    f = tmp_path / 's.bin'; f.write_bytes(rows().tobytes())
    b, _, _ = backend(rss=rss)
    with pytest.raises(cc.ResourceStop):
        b.load_unit(f, unit())


def test_guarded_sampler_never_mutates_sys_modules_and_enforces_rss():
    seen, dummy = [], object()
    def base():
        seen.append(sys.modules.get('mlx.core') is dummy); return healthy(process_rss_gib=0.5)
    sys.modules['mlx.core'] = dummy
    try:
        before = dict(sys.modules)
        s = cc.guarded_sampler(base)()
        assert seen == [True] and dict(sys.modules) == before and s.process_rss_gib == 0.5
    finally:
        del sys.modules['mlx.core']
    for bad in (9.0, None):
        with pytest.raises(cc.ResourceStop):
            cc.guarded_sampler(lambda: healthy(process_rss_gib=bad))()


@pytest.mark.parametrize('kw,reason', [
    (dict(power_source='BATTERY', battery_percent=50), 'power_not_verified_AC'),
    (dict(power_source='SENSOR_UNAVAILABLE'), 'power_not_verified_AC'),
    (dict(other_heavy_processes=1), 'concurrent_process_inventory_unknown_or_occupied'),
    (dict(other_heavy_processes=None), 'concurrent_process_inventory_unknown_or_occupied'),
    (dict(pressure_free_percent=20), 'memory_pressure'),
    (dict(pressure_free_percent=None), 'memory_pressure'),
    (dict(available_gib=5.0), 'insufficient_memory_headroom'),
    (dict(available_gib=None), 'insufficient_memory_headroom'),
    (dict(thermal_state='THROTTLED'), 'thermal_throttling'),
    (dict(swap_used_gib=None), 'swap_sensor_unknown')])
def test_guarded_sampler_refuses_hold_and_unknown_with_logged_reasons(tmp_path, kw, reason):
    log = tmp_path / 'r.jsonl'
    with pytest.raises(cc.ResourceStop) as e:
        cc.guarded_sampler(lambda: healthy(**kw), log_path=log)()
    assert reason in e.value.reasons and reason in json.loads(log.read_text())['reasons']
    assert cc.guarded_sampler(lambda: healthy(swap_used_gib=7.5))()      # large absolute swap is fine; only net growth is capped


def test_swap_growth_cap_uses_one_baseline_for_the_whole_run():
    vals = iter([4.0, 4.6, 5.2])
    s = cc.guarded_sampler(lambda: healthy(swap_used_gib=next(vals), monotonic_seconds=0.0))
    s(); s()
    with pytest.raises(cc.ResourceStop) as e:
        s()                       # each step is +0.6 but 1.2 GiB over the first sample: no per-phase reset
    assert 'stage_swap_net_growth_over_cap' in e.value.reasons


# ---- inherited identity gates -------------------------------------------------------------------------------
def test_prepare_requires_exact_plan_hash_source_shas_quant_map_and_counts(tmp_path):
    plan, args = fixture(tmp_path)
    p = {k: args[k] for k in ('plan_path', 'expected_plan_sha256', 'source_dir', 'quant_map_path', 'expected_source_sha256', 'expected')}
    assert cc.prepare(**p)[0]['plan_sha256'] == plan['plan_sha256']
    with pytest.raises(ValueError): cc.prepare(**{**p, 'expected_plan_sha256': '0' * 64})
    with pytest.raises(ValueError): cc.prepare(**{**p, 'expected_source_sha256': {'a.safetensors': '0' * 64}})
    with pytest.raises(ValueError): cc.prepare(**{**p, 'expected_source_sha256': {}})
    with pytest.raises(ValueError): cc.prepare(**{**p, 'expected': cc.Expected()})      # real 750/344 counts reject the tiny plan
    (tmp_path / 'work/qmap.json').write_text(json.dumps({'lin': dict(qc.QUANT)}))
    with pytest.raises(ValueError): cc.prepare(**p)
    (tmp_path / 'work/qmap.json').write_text(json.dumps({'lin': dict(qc.QUANT), 'embed_tokens': {'bits': 8, 'group_size': 64, 'mode': 'affine'}}))
    with pytest.raises(ValueError): cc.prepare(**p)


def test_tampered_source_shard_is_refused_before_any_backend(tmp_path):
    plan, args = fixture(tmp_path)
    (args['source_dir'] / 'a.safetensors').write_bytes(b'z' * 100)
    made = []
    with pytest.raises(ValueError):
        cc.rebuild(**args, backend_factory=lambda **k: made.append(1), sampler=sampler(), rss_gib=lambda: 0.5)
    assert made == []


def test_path_policy_confines_outputs_and_refuses_overwrite(tmp_path):
    plan, args = fixture(tmp_path)
    for key, val in (('units_dir', tmp_path / 'elsewhere'), ('cache_dir', tmp_path / 'elsewhere'), ('certificate_output', tmp_path / 'c.json')):
        with pytest.raises(ValueError): cc.check_paths(**{**args, key: val})
    with pytest.raises(ValueError): cc.check_paths(**{**args, 'cache_dir': args['units_dir']})
    with pytest.raises(ValueError): cc.check_paths(**{**args, 'cache_dir': args['units_dir'] / 'cache'})
    args['cache_dir'].mkdir()
    with pytest.raises(ValueError): cc.check_paths(**args)
    args['cache_dir'].rmdir(); args['certificate_output'].write_text('{}')
    with pytest.raises(ValueError): cc.check_paths(**args)


# ---- end to end ---------------------------------------------------------------------------------------------
def test_rebuild_certifies_exact_cache_with_unambiguous_cpu_counts(tmp_path):
    plan, args = fixture(tmp_path)
    res, mx, made = rebuild(args)
    cert = json.loads(args['certificate_output'].read_text())
    assert cert == json.loads(json.dumps(res['certificate'])) and cert['plan_sha256'] == plan['plan_sha256']
    assert (cert['units'], cert['quantized_modules'], cert['retained_tensors'], cert['tensors_checked']) == (3, 2, 1, 7)
    assert cert['config_delta_keys'] == ['quantization', 'quantization_config']
    assert cert['cpu_reconstruction'] == {'device': 'cpu', 'unit_attempts_this_run': 3, 'quantized_this_run': 2, 'adopted_units': 0,
                                          'model_loaded': False, 'forward_or_backward': False, 'training': False,
                                          'peak_process_rss_gib': 0.5, 'max_unit_attempts': 750, 'max_quantized': 344}
    assert json.loads((args['cache_dir'] / 'config.json').read_text())['quantization']['lin'] == qc.QUANT
    with pytest.raises(ValueError):          # never overwrites the cache or certificate
        rebuild(args)


def test_numeric_failure_stops_with_durable_receipt_and_is_never_retried(tmp_path):
    plan, args = fixture(tmp_path)
    mx = FakeMx(mismatch=True)
    with pytest.raises(qc.BlockedUnit):
        rebuild(args, mx=mx)
    assert sum(n == 'save' for n, _ in mx.ops) == 1            # first unit only: no extras after the failure
    fail = json.loads((args['units_dir'] / 'cpu_failure_001.json').read_text())
    assert fail['error_type'] == 'BlockedUnit' and fail['automatic_retry'] is False and fail['plan_sha256'] == plan['plan_sha256']
    assert not args['cache_dir'].exists() and not args['certificate_output'].exists()
    mx2 = FakeMx()
    with pytest.raises(qc.BlockedUnit):                         # rerun stays blocked; nothing is loaded
        rebuild(args, mx=mx2)
    assert mx2.created == []
    assert (args['units_dir'] / 'cpu_failure_002.json').exists()


def test_resource_failure_preserves_completed_units_and_resume_adopts_only_exact_bindings(tmp_path):
    plan, args = fixture(tmp_path)
    seq = iter([healthy(), healthy(), healthy(), healthy(available_gib=1.0)])      # 2 gate checks per unit; unit 2 start is blocked
    def s():
        try: return next(seq)
        except StopIteration: return healthy(available_gib=1.0)
    with pytest.raises(RuntimeError):
        rebuild(args, sampler=s)
    assert (args['units_dir'] / 'cpu_failure_001.json').exists()
    mx = FakeMx()
    res, _, _ = rebuild(args, mx=mx)
    assert res['certificate']['cpu_reconstruction']['adopted_units'] == 1 and res['certificate']['cpu_reconstruction']['unit_attempts_this_run'] == 2
    assert sum(n == 'save' for n, _ in mx.ops) == 2


def test_complete_units_assemble_without_creating_any_backend(tmp_path):
    plan, args = fixture(tmp_path)
    rebuild(args)
    import shutil
    shutil.rmtree(args['cache_dir']); args['certificate_output'].unlink()          # test fixture reset only; code never deletes
    res, mx, made = rebuild(args)
    assert made == [] and mx.created == [] and res['certificate']['cpu_reconstruction']['adopted_units'] == 3


def test_foreign_plan_binding_in_units_dir_is_refused(tmp_path):
    plan, args = fixture(tmp_path)
    args['units_dir'].mkdir(); (args['units_dir'] / 'plan_binding.json').write_text(json.dumps({'plan_sha256': 'f' * 64}))
    with pytest.raises(qc.BlockedUnit):
        rebuild(args)


# ---- historical identity -----------------------------------------------------------------------------------
def test_uniformly_altered_quant_payload_with_equal_counts_is_refused_and_not_published(tmp_path):
    plan, args = fixture(tmp_path)
    w = args['work_root']; bad = {**args, 'units_dir': w / 'u2', 'cache_dir': w / 'c2', 'certificate_output': w / 'cert2.json'}
    with pytest.raises(ValueError, match='historical'):
        rebuild(bad, mx=FakeMx(alter=1))
    assert not bad['certificate_output'].exists() and (bad['units_dir'] / 'cpu_failure_001.json').exists()
    assert rebuild(args)[0]['certificate']['cpu_reconstruction']['unit_attempts_this_run'] == 3     # honest run is accepted


def test_missing_or_wrong_historical_pin_stops_before_any_backend(tmp_path):
    plan, args = fixture(tmp_path)
    for patch in ({'expected_cache_certificate': None}, {'expected_cache_certificate_sha256': None},
                  {'expected_cache_certificate_sha256': '0' * 64}):
        made = []
        with pytest.raises(ValueError):
            cc.rebuild(**{**args, **patch}, backend_factory=lambda **k: made.append(1), sampler=sampler())
        assert made == [] and not args['certificate_output'].exists() and not args['units_dir'].exists()
    h = json.loads(args['expected_cache_certificate'].read_text()); h['plan_sha256'] = '1' * 64
    args['expected_cache_certificate'].write_text(json.dumps(h))
    args['expected_cache_certificate_sha256'] = hashlib.sha256(args['expected_cache_certificate'].read_bytes()).hexdigest()
    with pytest.raises(ValueError):                       # pinned file hash is right but it is not this plan's certificate
        rebuild(args)


# ---- plan-to-header assembly layout (confirmed checker bug: real unit headers are sorted, not weight/scales/biases) ---------
SMALL = 140      # fixture shard limit: the boundary falls inside quantized unit 0 only for one of the two part orders


def assembled_small(tmp_path):
    from mri_rl import quant_cache
    plan, args = fixture(tmp_path); w = args['work_root']
    b, _, _ = backend(mx=FakeMx())
    qc.convert(plan, args['source_dir'], w / 's_units', b, sampler())
    qc.assemble_cache(plan, w / 's_units', w / 's_cache', shard_limit_bytes=SMALL)
    return plan, args, w / 's_units', w / 's_cache'


def test_verify_assembly_accepts_real_sorted_header_order_across_a_small_shard_boundary(tmp_path):
    plan, args, units, cache = assembled_small(tmp_path)
    assert len(list(cache.glob('model-*.safetensors'))) == 3                      # the boundary is really exercised
    assert list(qc._read_st_header(units / 'unit_0000.safetensors')[0]) == ['biases', 'scales', 'weight']
    cc.verify_assembly(plan, cache, shard_limit_bytes=SMALL, units_dir=units)


def test_plan_layout_predicts_exact_shard_header_and_file_size_without_reading_the_cache(tmp_path):
    plan, args, units, cache = assembled_small(tmp_path)
    layout = cc.plan_layout(plan, SMALL)
    assert [s['file'] for s in layout] == sorted(p.name for p in cache.glob('model-*.safetensors'))
    for sh in layout:
        header, base = qc._read_st_header(cache / sh['file'])
        assert header == sh['header'] and list(header) == list(sh['header']) and (cache / sh['file']).stat().st_size == sh['file_bytes']


def test_old_weight_scales_biases_order_would_have_shifted_the_boundary(tmp_path):
    plan, args, units, cache = assembled_small(tmp_path)
    idx = json.loads((cache / 'model.safetensors.index.json').read_text())
    assert idx['weight_map']['m_never'] if False else True
    old = {}
    cur, size, k = 1, 0, 0
    entries = []
    for u in plan['units']:
        n = u['nbytes']
        entries += ([(u['module'] + '.weight', n // 4), (u['module'] + '.scales', n // 64), (u['module'] + '.biases', n // 64)]
                    if u['action'] == 'quantize' else [(u['normalized_key'], n)])
    for name, n in entries:
        if size and size + n > SMALL: cur, size = cur + 1, 0
        old[name] = cur; size += n
    real = {k: int(v.split('-')[1]) for k, v in idx['weight_map'].items()}
    assert old != real          # the previous prediction disagrees with the real assembler: this is the confirmed bug


def test_unit_header_order_deviation_or_foreign_part_name_stops(tmp_path):
    plan, args, units, cache = assembled_small(tmp_path)
    f = units / 'unit_0000.safetensors'; raw = f.read_bytes()
    n = struct.unpack('<Q', raw[:8])[0]; h = json.loads(raw[8:8 + n])
    reordered = {'__metadata__': h['__metadata__'], 'weight': h['weight'], 'scales': h['scales'], 'biases': h['biases']}
    hb = json.dumps(reordered, separators=(',', ':')).encode(); hb += b' ' * (len(raw[8:8 + n]) - len(hb))
    assert len(hb) == n
    f.write_bytes(raw[:8] + hb + raw[8 + n:])
    with pytest.raises(ValueError, match='header order'):
        cc.verify_assembly(plan, cache, shard_limit_bytes=SMALL, units_dir=units)


def test_tampered_index_or_shard_header_or_historical_bytes_fail_verify_assembly(tmp_path):
    plan, args, units, cache = assembled_small(tmp_path)
    layout = cc.plan_layout(plan, SMALL)
    hist_ok = {s['file']: {'bytes': s['file_bytes']} for s in layout}
    cc.verify_assembly(plan, cache, shard_limit_bytes=SMALL, units_dir=units, historical_shards=hist_ok)
    with pytest.raises(ValueError, match='historical'):
        cc.verify_assembly(plan, cache, shard_limit_bytes=SMALL, units_dir=units,
                           historical_shards={**hist_ok, layout[0]['file']: {'bytes': layout[0]['file_bytes'] + 8}})
    with pytest.raises(ValueError, match='historical'):
        cc.verify_assembly(plan, cache, shard_limit_bytes=SMALL, units_dir=units, historical_shards={k: v for k, v in list(hist_ok.items())[:2]})
    ip = cache / 'model.safetensors.index.json'; idx = json.loads(ip.read_text()); good = dict(idx['weight_map'])
    idx['weight_map']['m0.weight'] = 'model-00001-of-00003.safetensors'; ip.write_text(json.dumps(idx))
    with pytest.raises(ValueError, match='index'): cc.verify_assembly(plan, cache, shard_limit_bytes=SMALL, units_dir=units)
    idx['weight_map'] = good; ip.write_text(json.dumps(idx))
    shard = cache / layout[1]['file']; raw = bytearray(shard.read_bytes()); n = struct.unpack('<Q', raw[:8])[0]
    raw[8 + raw[8:8 + n].index(b'U32')] = ord('I')       # same length, different dtype in the shard header
    shard.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match='header'): cc.verify_assembly(plan, cache, shard_limit_bytes=SMALL, units_dir=units)


def test_changed_config_index_nonweight_or_source_identity_fails_historical_verification(tmp_path):
    plan, args = fixture(tmp_path)
    rebuild(args)
    hist = json.loads(args['expected_cache_certificate'].read_text()); qmap = json.loads(args['quant_map_path'].read_text())
    verify = lambda: cc.verify_historical(plan, args['units_dir'], args['cache_dir'], args['source_dir'], qmap, hist, args['expected'])
    c, src = args['cache_dir'], args['source_dir']
    assert verify()['tensors_checked'] == 7
    orig = (c / 'config.json').read_text(); (c / 'config.json').write_text(orig.replace('"a": 1', '"a": 2'))
    with pytest.raises(ValueError): verify()
    (c / 'config.json').write_text(orig)
    idx = json.loads((c / 'model.safetensors.index.json').read_text()); idx['metadata']['total_size'] += 1
    (c / 'model.safetensors.index.json').write_text(json.dumps(idx))
    with pytest.raises(ValueError, match='index'): verify()
    idx['metadata']['total_size'] -= 1; idx['weight_map']['lin.weight'] = 'model-00002-of-00002.safetensors'
    (c / 'model.safetensors.index.json').write_text(json.dumps(idx))
    with pytest.raises(ValueError, match='index'): verify()
    idx['weight_map']['lin.weight'] = 'model-00001-of-00001.safetensors'; (c / 'model.safetensors.index.json').write_text(json.dumps(idx))
    assert verify()
    (c / 'tokenizer.json').write_text('{"t":2}')
    with pytest.raises(ValueError): verify()
    (c / 'tokenizer.json').write_text('{"t":1}')
    (src / 'tokenizer.json').write_text('{"t":2}'); (c / 'tokenizer.json').write_text('{"t":2}')   # cache consistent with a CHANGED source
    with pytest.raises(ValueError, match='historical'): verify()


def test_production_expected_requires_two_historical_shards():
    assert cc.Expected().shards == 2


def test_swap_growth_after_conversion_phase_still_stops_without_publishing(tmp_path):
    plan, args = fixture(tmp_path)
    n = []
    def late():
        n.append(1); return healthy(swap_used_gib=4.0 if len(n) <= 7 else 5.5)       # initial + 3 units x 2 gates, then assembly
    with pytest.raises(cc.ResourceStop):
        rebuild(args, sampler=late)
    fail = json.loads((args['units_dir'] / 'cpu_failure_001.json').read_text())
    assert 'stage_swap_net_growth_over_cap' in fail['reasons'] and not args['certificate_output'].exists()


def test_battery_hold_stops_rebuild_with_durable_reasons_and_no_backend(tmp_path):
    plan, args = fixture(tmp_path)
    with pytest.raises(cc.ResourceStop):
        rebuild(args, sampler=sampler(power_source='BATTERY', battery_percent=50))
    fail = json.loads((args['units_dir'] / 'cpu_failure_001.json').read_text())
    assert 'power_not_verified_AC' in fail['reasons'] and fail['last_resource_sample']['power_source'] == 'BATTERY'


# ---- imports / CLI ------------------------------------------------------------------------------------------
def test_module_and_cli_plan_never_import_mlx(tmp_path):
    plan, args = fixture(tmp_path)
    argv = ['plan', '--expected-plan-sha256', plan['plan_sha256'], '--source-dir', str(args['source_dir']), '--quant-map', str(args['quant_map_path']),
            '--plan', str(args['plan_path']), '--units-dir', str(args['units_dir']), '--cache-dir', str(args['cache_dir']),
            '--certificate-output', str(args['certificate_output']), '--work-root', str(args['work_root']),
            '--expected-cache-certificate', str(args['expected_cache_certificate']),
            '--expected-cache-certificate-sha256', args['expected_cache_certificate_sha256'],
            '--expected-source-sha256', 'a.safetensors=' + args['expected_source_sha256']['a.safetensors']]
    code = ("import sys; sys.path.insert(0, %r); from mri_rl import accuracy_recovery_cpu_cache_v1 as c; c.main(%r, expected=c.Expected(units=3, quantized=2, retained=1, tensors=7, shards=1)); "
            "assert not any(m == 'mlx' or m.startswith('mlx.') for m in sys.modules), 'mlx imported'" % (str(SRC_ROOT), argv))
    r = subprocess.run([sys.executable, '-I', '-c', code], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out['command'] == 'plan' and out['pending_units'] == 3 and out['executed'] is False
    assert not args['units_dir'].exists() and not args['cache_dir'].exists()


def test_cli_rebuild_requires_every_explicit_argument():
    with pytest.raises(SystemExit):
        cc.main(['rebuild', '--plan', 'x'])
