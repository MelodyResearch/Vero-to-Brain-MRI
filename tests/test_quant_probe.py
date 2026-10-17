"""CPU stub tests for the tiny quantizer probe; no MLX, no real weights."""
import hashlib
import json
import struct
from types import SimpleNamespace
import pytest
from mri_rl import quant_probe as qp

K0, K1 = 'vision_tower.blocks.0.attn.proj.weight', 'vision_tower.blocks.1.attn.proj.weight'


def fragment(tmp_path, key, shape=(1152, 1152), dtype='BF16', name=None):
    header = json.dumps({key: {'dtype': dtype, 'shape': list(shape), 'data_offsets': [0, 4]}}).encode()
    path = tmp_path / (name or key.replace('.', '_') + '.safetensors')
    path.write_bytes(struct.pack('<Q', len(header)) + header + b'\0\0\0\0')
    return {'key': key, 'file': path.name, 'shape': list(shape), 'dtype': dtype,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def manifest(tmp_path, entries):
    m = tmp_path / 'manifest.json'; m.write_text(json.dumps({'tensors': entries})); return m


def test_valid_inputs_pass_and_are_hash_bound(tmp_path):
    m = manifest(tmp_path, [fragment(tmp_path, K0), fragment(tmp_path, K1)])
    assert [e['key'] for e in qp.validate_inputs(m, tmp_path)] == [K0, K1]


@pytest.mark.parametrize('mutate', ['key', 'shape', 'dtype', 'hash', 'three', 'duplicate', 'header'])
def test_allowlist_shape_dtype_hash_and_count_are_enforced(tmp_path, mutate):
    e = fragment(tmp_path, K0)
    if mutate == 'key': e = fragment(tmp_path, 'vision_tower.blocks.2.attn.proj.weight')
    if mutate == 'shape': e = fragment(tmp_path, K0, shape=(1152, 1153))
    if mutate == 'dtype': e = fragment(tmp_path, K0, dtype='F32')
    if mutate == 'hash': e['sha256'] = '0' * 64
    entries = [e]
    if mutate == 'three': entries = [e, fragment(tmp_path, K1), dict(e, key='x')]
    if mutate == 'duplicate': entries = [e, e]
    if mutate == 'header':  # provenance says K0 but file header says K1
        other = fragment(tmp_path, K1, name='k0.safetensors'); e = dict(e, file='k0.safetensors', sha256=other['sha256'])
        entries = [e]
    with pytest.raises(ValueError):
        qp.validate_inputs(manifest(tmp_path, entries), tmp_path)


def test_path_escape_and_oversize_rejected(tmp_path):
    inner = tmp_path / 'in'; inner.mkdir()
    e = fragment(tmp_path, K0); e['file'] = '../' + e['file']
    with pytest.raises(ValueError):
        qp.validate_inputs(manifest(inner, [e]), inner)
    big = fragment(inner, K0); (inner / big['file']).write_bytes(b'\0' * (qp.MAX_FILE_BYTES + 1))
    with pytest.raises(ValueError):
        qp.validate_inputs(manifest(inner, [big]), inner)


class FakeBackend:
    def __init__(self, equal=True, peak=10.0): self.equal, self.peak_mib_v, self.log = equal, peak, []
    def active_mib(self): return 1.0
    def peak_mib(self): return self.peak_mib_v
    def load(self, path, key): self.log.append(('load', key)); return key
    def quantize_ref(self, w): return ('w', 's', 'b')
    def quantize_module(self, w): return ('w', 's', 'b')
    def bitwise_equal(self, a, b): return self.equal
    def same_layout(self, xs, ys): return True
    def shape(self, a): return [1]
    def dtype(self, a): return 'uint32'
    def save(self, path, arrays): path.write_bytes(b'x')
    def reload(self, path): return {'weight': 1, 'scales': 2, 'biases': 3}


def healthy(**kw):
    base = dict(available_gib=6.0, pressure_free_percent=40, power_source='AC', other_heavy_processes=0,
                thermal_state='NORMAL', mlx_active_gib=None, swap_used_gib=4.0, process_rss_gib=0.2,
                battery_percent=100, monotonic_seconds=1.0)
    return SimpleNamespace(**{**base, **kw})


def entries(tmp_path):
    return qp.validate_inputs(manifest(tmp_path, [fragment(tmp_path, K0), fragment(tmp_path, K1)]), tmp_path)


def test_runs_serially_with_a_receipt_per_tensor_without_the_8b_floor(tmp_path):
    result = qp.run(entries(tmp_path), tmp_path / 'out', FakeBackend(), lambda: healthy(available_gib=5.9))
    lines = (tmp_path / 'out/probe_receipts.jsonl').read_text().splitlines()
    assert result == {'tensors': 2, 'all_bitwise_equal': True} and len(lines) == 2
    assert json.loads(lines[0])['key'] == K0 and 'weight_values' not in lines[0]


def test_mismatch_is_reported_not_hidden(tmp_path):
    assert qp.run(entries(tmp_path), tmp_path / 'o', FakeBackend(equal=False), healthy)['all_bitwise_equal'] is False


@pytest.mark.parametrize('bad', [dict(other_heavy_processes=1), dict(other_heavy_processes=None),
                                 dict(thermal_state='THROTTLED'), dict(available_gib=1.0),
                                 dict(pressure_free_percent=2)])
def test_critical_conditions_stop_before_any_tensor_work(tmp_path, bad):
    b = FakeBackend()
    with pytest.raises(RuntimeError):
        qp.run(entries(tmp_path), tmp_path / 'o', b, lambda: healthy(**bad))
    assert b.log == []


def test_over_budget_peak_stops_after_receipt(tmp_path):
    with pytest.raises(RuntimeError):
        qp.run(entries(tmp_path), tmp_path / 'o', FakeBackend(peak=999), healthy)
    assert len((tmp_path / 'o/probe_receipts.jsonl').read_text().splitlines()) == 1


def test_module_import_has_no_native_runtime():
    import sys
    assert 'mlx.core' not in sys.modules
