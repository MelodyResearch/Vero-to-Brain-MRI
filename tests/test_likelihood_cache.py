"""Focused CPU tests: certified-cache binding, loaded-parameter identity, cache-path load floor and OS pressure."""
import json
import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import pytest
from mri_rl import likelihood_cache as lc, quant_cache, quant_convert as qc
from tests.test_quant_cache import build


def certified(tmp_path, monkeypatch):
    plan, src, out, c, qmap = build(tmp_path)
    quant_cache.finalize_cache(plan, out, c, src, qmap)
    cert = quant_cache.certify_cache(plan, out, c, src, qmap)
    cp = tmp_path / 'cert.json'; cp.write_text(json.dumps(cert, sort_keys=True))
    monkeypatch.setattr(lc, 'EXPECTED_TENSORS', 7)
    arts = {str(src / n): qc.sha_file(src / n) for n in ('config.json', 'tokenizer.json', 'chat_template.json')}
    return c, cp, qc.sha_file(cp), arts, plan


def test_identity_binds_certificate_shards_config_and_original_artifacts(tmp_path, monkeypatch):
    c, cp, sha, arts, _ = certified(tmp_path, monkeypatch)
    b = lc.verify_cache_identity(c, cp, sha, arts)
    assert b['tensors'] == 7 and b['certificate_sha256'] == sha
    with pytest.raises(ValueError): lc.verify_cache_identity(c, cp, '0' * 64, arts)
    bad = dict(arts); bad[next(k for k in arts if k.endswith('config.json'))] = '1' * 64
    with pytest.raises(ValueError): lc.verify_cache_identity(c, cp, sha, bad)
    shard = sorted(c.glob('model-*.safetensors'))[0]; raw = bytearray(shard.read_bytes()); raw[-1] ^= 1; shard.write_bytes(bytes(raw))
    with pytest.raises(ValueError): lc.verify_cache_identity(c, cp, sha, arts)


def test_identity_rejects_config_drift_beyond_quantization_keys(tmp_path, monkeypatch):
    c, cp, sha, arts, _ = certified(tmp_path, monkeypatch)
    cfg = json.loads((c / 'config.json').read_text()); cfg['a'] = 99; (c / 'config.json').write_text(json.dumps(cfg))
    with pytest.raises(ValueError): lc.verify_cache_identity(c, cp, sha, arts)


def params_from_cache(c):
    idx = json.loads((c / 'model.safetensors.index.json').read_text())['weight_map']; out = []
    for name, shard in idx.items():
        h, base = lc._header(c / shard); m = h[name]
        raw = (c / shard).read_bytes()[base + m['data_offsets'][0]: base + m['data_offsets'][1]]
        out.append((name, (raw, m['dtype'], m['shape'])))
    return out


def test_every_loaded_parameter_must_equal_cache_bytes_with_full_inventory(tmp_path, monkeypatch):
    c, cp, sha, arts, _ = certified(tmp_path, monkeypatch)
    ps = params_from_cache(c); ident = lambda v: v
    assert lc.verify_loaded_parameters(ps, c, ident) == 7
    flipped = [(n, (bytes([v[0][0] ^ 1]) + v[0][1:], v[1], v[2])) if i == 3 else (n, v) for i, (n, v) in enumerate(ps)]
    with pytest.raises(ValueError): lc.verify_loaded_parameters(flipped, c, ident)
    with pytest.raises(ValueError): lc.verify_loaded_parameters(ps[:-1], c, ident)          # missing tensor
    with pytest.raises(ValueError): lc.verify_loaded_parameters(ps + [('extra.weight', ps[0][1])], c, ident)
    wrong = [(n, (v[0], 'BF16' if v[1] == 'U32' else 'U32', v[2])) if i == 0 else (n, v) for i, (n, v) in enumerate(ps)]
    with pytest.raises(ValueError): lc.verify_loaded_parameters(wrong, c, ident)


def healthy(**kw):
    from mri_rl.resource_supervisor import ResourceSample
    base = dict(monotonic_seconds=1.0, available_gib=7.0, pressure_free_percent=50, swap_used_gib=4.0, process_rss_gib=0.2,
                thermal_state='NORMAL', power_source='AC', battery_percent=100, other_heavy_processes=0)
    return ResourceSample(**{**base, **kw})


def gate(tmp_path, name, sample, **kw):
    from mri_rl.likelihood_battery import BatteryDiagnosticGate
    d = tmp_path / name; d.mkdir()
    clock = [0.0]
    def sampler():
        clock[0] += 31.0                      # advancing monotonic clock satisfies the 30 s preflight interval
        return replace(sample, monotonic_seconds=clock[0])
    return BatteryDiagnosticGate(d / 'r.jsonl', sampler=sampler, sleep=lambda s: None, **kw)


def test_cache_load_floor_replaces_only_the_generic_load_heuristic(tmp_path, monkeypatch):
    from mri_rl import likelihood_battery as lb
    from mri_rl.paced_answer_likelihood import CACHE_LOAD_FLOOR_GIB
    monkeypatch.setattr(lb, 'os_pressure_level', lambda: 1)
    assert abs(CACHE_LOAD_FLOOR_GIB - 6.8163) < 1e-3
    with pytest.raises(RuntimeError):            # generic gate: 7.0 GiB < 8 GiB load heuristic
        gate(tmp_path, 'a', healthy(available_gib=7.0)).before_unit('model_load', loading=True)
    assert gate(tmp_path, 'b', healthy(available_gib=7.0), load_floor_gib=CACHE_LOAD_FLOOR_GIB).before_unit('model_load', loading=True)
    with pytest.raises(RuntimeError):            # below footprint + 2 GiB reserve
        gate(tmp_path, 'c', healthy(available_gib=6.5), load_floor_gib=CACHE_LOAD_FLOOR_GIB).before_unit('model_load', loading=True)
    with pytest.raises(RuntimeError):            # genuine memory pressure still blocks
        gate(tmp_path, 'd', healthy(available_gib=7.0, pressure_free_percent=10), load_floor_gib=CACHE_LOAD_FLOOR_GIB).before_unit('model_load', loading=True)


def test_os_pressure_level_2_logged_not_critical_and_4_stops(tmp_path, monkeypatch):
    from mri_rl import likelihood_battery as lb
    for level, ok in ((1, True), (2, True), (4, False)):
        monkeypatch.setattr(lb, 'os_pressure_level', lambda level=level: level)
        g = gate(tmp_path, str(level), healthy(available_gib=12.0))
        if ok:
            g.before_unit('record', loading=False)
        else:
            with pytest.raises(RuntimeError): g.before_unit('record', loading=False)
        assert json.loads((tmp_path / str(level) / 'r.jsonl').read_text().splitlines()[0])['os_pressure_level'] == level


def test_execution_only_registration_keeps_numerical_loop_and_adoption_profile():
    from mri_rl import likelihood_adoption as a
    assert 'likelihood_cache' in a.EXECUTION_ONLY
    assert a.numerical_loop_sha256() == 'fc51dfd7fe5998b05e723e27f57e703f0f1912ffd6dc836f6614f5be22b0cb31'
