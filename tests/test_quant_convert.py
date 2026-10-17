"""CPU stub tests for the streaming converter; no MLX, no real weights."""
import json
from types import SimpleNamespace
import pytest
from mri_rl import quant_convert as qc


class Fake:
    def __init__(self, equal=True, peak=10.0, bad_reload=False):
        self.equal, self.peak, self.bad_reload, self.loads, self.cleared = equal, peak, bad_reload, [], 0
    def load(self, path, key): self.loads.append(key); return ('w', key)
    def shape(self, a): return a if isinstance(a, list) else [64, 64]
    def dtype(self, a): return 'bfloat16' if a[0] == 'w' else 'uint32'
    def quantize(self, w): return (('q', w[1]), ('s', w[1]), ('b', w[1]))
    def quantize_module(self, w, t): return self.quantize(w)
    def bitwise_equal(self, a, b): return self.equal and not self.bad_reload
    def save(self, path, arrays): path.write_bytes(json.dumps(sorted(arrays)).encode())
    def reload(self, path): return {n: None for n in json.loads(path.read_bytes())}
    def peak_mib(self): return self.peak
    def clear(self): self.cleared += 1


def healthy(**kw):
    base = dict(available_gib=9.0, pressure_free_percent=40, power_source='AC', other_heavy_processes=0,
                thermal_state='NORMAL', mlx_active_gib=None, swap_used_gib=4.0, process_rss_gib=0.2,
                battery_percent=100, monotonic_seconds=1.0)
    return SimpleNamespace(**{**base, **kw})


def make(tmp_path, n=3):
    src = tmp_path / 'src'; src.mkdir(); (src / 'a.safetensors').write_bytes(b'x')
    units = [{'index': i, 'source_key': f'k{i}', 'source_file': 'a.safetensors',
              'action': 'retain' if i == 2 else 'quantize', 'module_type': 'linear', 'shape': [64, 64],
              'dtype': 'BF16', 'output_key': f'o{i}'} for i in range(n)]
    return {'quantization': dict(qc.QUANT), 'units': units}, src


def test_serial_conversion_with_receipt_per_unit_and_cache_clear(tmp_path):
    plan, src = make(tmp_path); b = Fake()
    r = qc.convert(plan, src, tmp_path / 'o', b, lambda: healthy())
    assert r == {'units_total': 3, 'units_done': 3, 'converted_this_call': 3, 'complete': True}
    assert b.loads == ['k0', 'k1', 'k2'] and b.cleared == 3
    lines = [json.loads(l) for l in (tmp_path / 'o/receipts.jsonl').read_text().splitlines()]
    assert [x['action'] for x in lines] == ['quantize', 'quantize', 'retain'] and lines[2]['retained_bitwise_equal'] is True


def test_resume_never_reconverts_completed_units_and_rejects_tampered_output(tmp_path):
    plan, src = make(tmp_path); b1 = Fake()
    qc.convert(plan, src, tmp_path / 'o', b1, lambda: healthy(), max_units=2)
    b2 = Fake(); r = qc.convert(plan, src, tmp_path / 'o', b2, lambda: healthy())
    assert b2.loads == ['k2'] and r['complete']
    (tmp_path / 'o/unit_0000.safetensors').write_bytes(b'tampered')
    with pytest.raises(ValueError):
        qc.convert(plan, src, tmp_path / 'o', Fake(), lambda: healthy())


def test_mismatch_preserves_receipt_and_stops(tmp_path):
    plan, src = make(tmp_path)
    with pytest.raises(RuntimeError):
        qc.convert(plan, src, tmp_path / 'o', Fake(equal=False), lambda: healthy())
    assert len((tmp_path / 'o/receipts.jsonl').read_text().splitlines()) == 1


def test_plan_validation_rejects_bad_quantization_gaps_and_escape(tmp_path):
    plan, src = make(tmp_path)
    bad = dict(plan, quantization={'bits': 8, 'group_size': 64, 'mode': 'affine'})
    with pytest.raises(ValueError): qc.validate_plan(bad, src)
    gap = dict(plan, units=[plan['units'][0], dict(plan['units'][1], index=5)])
    with pytest.raises(ValueError): qc.validate_plan(gap, src)
    esc = json.loads(json.dumps(plan)); esc['units'][0]['source_file'] = '../x'
    with pytest.raises(ValueError): qc.validate_plan(esc, src)


def test_shared_policy_stops_on_rising_swap_trace_and_budget(tmp_path):
    plan, src = make(tmp_path)
    seq = iter(healthy(available_gib=3.5, pressure_free_percent=14, swap_used_gib=4.0 + 0.1 * i, monotonic_seconds=float(i))
               for i in range(20))
    with pytest.raises(RuntimeError):
        qc.convert(plan, src, tmp_path / 'o', Fake(), lambda: next(seq))
    (tmp_path / 'p').mkdir()
    plan2, src2 = make(tmp_path / 'p')
    with pytest.raises(RuntimeError):
        qc.convert(plan2, src2, tmp_path / 'p/o', Fake(peak=99999), lambda: healthy())


def test_source_shape_mismatch_rejected(tmp_path):
    plan, src = make(tmp_path); plan['units'][0]['shape'] = [64, 128]
    class Wrong(Fake):
        def shape(self, a): return [64, 64]
    with pytest.raises(ValueError):
        qc.convert(plan, src, tmp_path / 'o', Wrong(), lambda: healthy())
