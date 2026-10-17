"""CPU tests for the streaming converter: real public metadata for the plan, stubs for native work."""
import json
import struct
from pathlib import Path
from types import SimpleNamespace
import pytest
from mri_rl import quant_convert as qc

META = Path(__file__).parents[2].parent / 'work/exact_qwen_quant_metadata'


@pytest.mark.skipif(not META.exists(), reason='public metadata not present')
def test_real_metadata_plan_is_exact_and_includes_vision_canonicalization():
    plan = qc.build_plan(META)
    units = plan['units']
    assert len(units) == 750 and sum(u['action'] == 'quantize' for u in units) == 344
    conv = [u for u in units if u['transform'] == qc.VISION_CANON]
    assert len(conv) == 1 and conv[0]['source_shape'] == [1152, 3, 2, 16, 16] and conv[0]['shape'] == [1152, 2, 16, 16, 3]
    assert conv[0]['action'] == 'retain' and not any('position_ids' in u['normalized_key'] for u in units)
    assert plan['plan_sha256'] == qc.build_plan(META)['plan_sha256']
    order = [(u['source_file'], u['offset']) for u in units]
    assert order == sorted(order)
    assert {u['module'] for u in units if u['module']} == set(json.loads((META / 'base_quantization.json').read_text()))


class Fake:
    def __init__(self, equal=True, peak=10.0, residual=0.0):
        self.equal, self.peak, self.residual, self.loaded, self.cleared = equal, peak, residual, [], 0
    def begin_unit(self): pass
    def load_unit(self, shard, unit): self.loaded.append(unit['normalized_key']); return ('w', unit['normalized_key']), 'rawsha-' + unit['normalized_key']
    def canonicalize_vision(self, key, w): return w
    def shape(self, a): return self.cur_shape
    def dtype(self, a): return 'bfloat16' if a[0] == 'w' else 'uint32'
    def quantize(self, w): return (('q', 1), ('s', 1), ('b', 1))
    def quantize_module(self, w, t): return self.quantize(w)
    def bitwise_equal(self, a, b): return self.equal
    def bits_sha256(self, a): return 'rawsha-' + a[1]
    def save(self, path, arrays): path.write_bytes(json.dumps(sorted(arrays)).encode())
    def reload(self, path): return {n: None for n in json.loads(path.read_bytes())}
    def peak_mib(self): return self.peak
    def active_mib(self): return self.residual
    def clear(self): self.cleared += 1


def healthy(**kw):
    base = dict(available_gib=9.0, pressure_free_percent=40, power_source='AC', other_heavy_processes=0,
                thermal_state='NORMAL', mlx_active_gib=None, swap_used_gib=4.0, process_rss_gib=0.2,
                battery_percent=100, monotonic_seconds=1.0)
    return SimpleNamespace(**{**base, **kw})


def make(tmp_path, n=3, action_last='retain'):
    src = tmp_path / 'src'; src.mkdir(exist_ok=True); (src / 'a.safetensors').write_bytes(b'x' * 1000)
    units = [{'index': i, 'source_key': f'k{i}', 'normalized_key': f'm{i}.weight', 'source_file': 'a.safetensors',
              'offset': 10 * i, 'nbytes': 8, 'source_shape': [64, 64], 'dtype': 'BF16', 'shape': [64, 64],
              'action': action_last if i == n - 1 else 'quantize', 'module': None if i == n - 1 and action_last == 'retain' else f'm{i}',
              'module_type': None if i == n - 1 and action_last == 'retain' else 'linear', 'transform': None} for i in range(n)]
    plan = {'schema': 'exact-quant-plan-v1', 'quantization': dict(qc.QUANT), 'units': units,
            'source_shards': {'a.safetensors': {'bytes': 1000, 'sha256': 'x'}}, 'lineage': {}}
    plan['plan_sha256'] = qc.sha_bytes(qc.canonical({k: v for k, v in plan.items() if k != 'plan_sha256'}).encode())
    return plan, src


def run(plan, src, out, backend, **kw):
    backend.cur_shape = [64, 64]
    return qc.convert(plan, src, out, backend, lambda: healthy(), **kw)


def test_serial_conversion_receipt_per_unit_release_and_cache_clear(tmp_path):
    plan, src = make(tmp_path); b = Fake()
    assert run(plan, src, tmp_path / 'o', b) == {'units_total': 3, 'units_done': 3, 'converted_this_call': 3, 'complete': True}
    assert b.loaded == ['m0.weight', 'm1.weight', 'm2.weight'] and b.cleared == 3
    lines = [json.loads(l) for l in (tmp_path / 'o/receipts.jsonl').read_text().splitlines()]
    assert all(l['passed'] for l in lines) and lines[2]['retained_bitwise_equal'] is True


def test_counterexample_failed_receipt_is_blocked_not_completed(tmp_path):
    plan, src = make(tmp_path)
    with pytest.raises(qc.BlockedUnit):
        run(plan, src, tmp_path / 'o', Fake(equal=False))
    with pytest.raises(qc.BlockedUnit):  # resume must not treat the failed unit as done
        run(plan, src, tmp_path / 'o', Fake())


def test_counterexample_changed_plan_source_action_or_output_is_not_accepted_as_done(tmp_path):
    plan, src = make(tmp_path)
    run(plan, src, tmp_path / 'o', Fake(), max_units=2)
    for change in ('action', 'offset', 'key'):
        plan2 = json.loads(json.dumps(plan)); u = plan2['units'][0]
        if change == 'action': u['action'], u['module'], u['module_type'] = 'retain', None, None
        if change == 'offset': u['offset'] = 500
        if change == 'key': u['normalized_key'] = 'other.weight'
        plan2['plan_sha256'] = qc.sha_bytes(qc.canonical({k: v for k, v in plan2.items() if k != 'plan_sha256'}).encode())
        with pytest.raises(qc.BlockedUnit):
            run(plan2, src, tmp_path / 'o', Fake())
    (tmp_path / 'o/unit_0000.safetensors').write_bytes(b'tampered')
    with pytest.raises(qc.BlockedUnit):
        run(plan, src, tmp_path / 'o', Fake())


def test_resume_converts_only_pending_units(tmp_path):
    plan, src = make(tmp_path); run(plan, src, tmp_path / 'o', Fake(), max_units=2)
    b = Fake(); assert run(plan, src, tmp_path / 'o', b)['complete'] and b.loaded == ['m2.weight']


def test_residual_active_memory_after_release_is_a_leak_failure(tmp_path):
    plan, src = make(tmp_path)
    with pytest.raises(qc.BlockedUnit):
        run(plan, src, tmp_path / 'o', Fake(residual=5000.0))


def test_native_backend_streams_fragments_and_never_retains_shard_arrays():
    import inspect
    text = inspect.getsource(qc.NativeConvertBackend)
    assert '_shards' not in text and 'readinto' in text and 'mx.load(str(shard' not in text
    assert 'VisionModel.sanitize' in text


def test_plan_hash_shape_and_range_validation(tmp_path):
    plan, src = make(tmp_path)
    bad = json.loads(json.dumps(plan)); bad['units'][0]['offset'] = 5
    with pytest.raises(ValueError): qc.validate_plan(bad, src)  # hash no longer matches
    big = json.loads(json.dumps(plan)); big['units'][0]['offset'] = 10_000
    big['plan_sha256'] = qc.sha_bytes(qc.canonical({k: v for k, v in big.items() if k != 'plan_sha256'}).encode())
    with pytest.raises(ValueError): qc.validate_plan(big, src)


def test_shared_policy_stops_rising_swap_trace_and_over_budget_peak(tmp_path):
    plan, src = make(tmp_path)
    seq = iter(healthy(available_gib=3.5, pressure_free_percent=14, swap_used_gib=4.0 + 0.1 * i, monotonic_seconds=float(i))
               for i in range(30))
    b = Fake(); b.cur_shape = [64, 64]
    with pytest.raises(RuntimeError):
        qc.convert(plan, src, tmp_path / 'o', b, lambda: next(seq))
    (tmp_path / 'p').mkdir(); plan2, src2 = make(tmp_path / 'p')
    with pytest.raises(RuntimeError):
        run(plan2, src2, tmp_path / 'p/o', Fake(peak=99999))


def _st(path, tensors):
    header, pos, data = {}, 0, b''
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {'dtype': dtype, 'shape': shape, 'data_offsets': [pos, pos + len(raw)]}; pos += len(raw); data += raw
    h = json.dumps(header).encode(); h += b' ' * (-len(h) % 8)
    Path(path).write_bytes(struct.pack('<Q', len(h)) + h + data)


def test_assemble_cache_streams_exact_bytes_names_and_refuses_partial(tmp_path):
    plan, src = make(tmp_path); out = tmp_path / 'o'; out.mkdir()
    (out / 'plan_binding.json').write_text('{}')
    raws = {}
    lines = []
    for u in plan['units']:
        f = f"unit_{u['index']:04d}.safetensors"
        if u['action'] == 'quantize':
            t = {'weight': ('U32', [2, 2], bytes([u['index']] * 16)), 'scales': ('BF16', [2, 1], bytes([7] * 4)), 'biases': ('BF16', [2, 1], bytes([9] * 4))}
        else:
            t = {'tensor': ('BF16', [2], bytes([5] * 4))}
        _st(out / f, t)
        lines.append(json.dumps({'index': u['index'], 'plan_sha256': plan['plan_sha256'], 'unit_hash': qc.unit_hash(plan, u),
            'output_file': f, 'output_file_sha256': qc.sha_file(out / f), 'reload_bitwise_equal': True,
            'module_path_bitwise_equal': True if u['action'] == 'quantize' else None,
            'retained_bitwise_equal': True if u['action'] == 'retain' else None, 'residual_within_budget': True}))
    (out / 'receipts.jsonl').write_text('\n'.join(lines[:2]) + '\n')
    with pytest.raises(ValueError):
        qc.assemble_cache(plan, out, tmp_path / 'c0')
    (out / 'receipts.jsonl').write_text('\n'.join(lines) + '\n')
    res = qc.assemble_cache(plan, out, tmp_path / 'c', shard_limit_bytes=40)
    idx = json.loads((tmp_path / 'c/model.safetensors.index.json').read_text())['weight_map']
    assert set(idx) == {'m0.weight', 'm0.scales', 'm0.biases', 'm1.weight', 'm1.scales', 'm1.biases', 'm2.weight'} and res['shards'] >= 2
    shard = tmp_path / 'c' / idx['m1.weight']; header, base = qc._read_st_header(shard)
    meta = header['m1.weight']; raw = shard.read_bytes()[base + meta['data_offsets'][0]: base + meta['data_offsets'][1]]
    assert raw == bytes([1] * 16)
