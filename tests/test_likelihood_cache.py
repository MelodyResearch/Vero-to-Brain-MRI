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


def test_cache_profile_is_phase_aware_and_replaces_only_the_memory_headroom_reason(tmp_path, monkeypatch):
    from mri_rl import likelihood_battery as lb
    from mri_rl.paced_answer_likelihood import CACHE_PROFILE as P
    monkeypatch.setattr(lb, 'os_pressure_level', lambda: 1)
    assert abs(lb.cache_required_gib(P, 'model_load', None) - 6.8163) < 1e-3
    assert abs(lb.cache_required_gib(P, 'materialized:640', 3.0) - (P['packed_gib'] - 3.0 + 2.0)) < 1e-9
    assert lb.cache_required_gib(P, 'materialized:64', 99.0) == 2.0           # nothing left to make resident
    assert lb.cache_required_gib(P, 'before_record:x', 4.8) == 3.2             # resident: 1.2 increment + 2 reserve
    with pytest.raises(RuntimeError):            # generic gate: 7.0 GiB < 8 GiB load heuristic
        gate(tmp_path, 'a', healthy(available_gib=7.0)).before_unit('model_load', loading=True)
    assert gate(tmp_path, 'b', healthy(available_gib=7.0), cache_profile=P).before_unit('model_load', loading=True)
    with pytest.raises(RuntimeError):            # below packed + adapter + 2 GiB reserve
        gate(tmp_path, 'c', healthy(available_gib=6.5), cache_profile=P).before_unit('model_load', loading=True)
    g = gate(tmp_path, 'r', healthy(available_gib=4.0), cache_profile=P)       # resident at 4 GiB: allowed (>= 3.2), generic 6 GiB floor would hold
    g.initial = healthy(); assert g.before_unit('before_record:x', loading=False)
    g2 = gate(tmp_path, 'r2', healthy(available_gib=3.0), cache_profile=P); g2.initial = healthy()
    with pytest.raises(RuntimeError): g2.before_unit('before_record:x', loading=False)
    bads = (dict(pressure_free_percent=10), dict(thermal_state='THROTTLED'), dict(other_heavy_processes=1),
            dict(battery_percent=15, power_source='BATTERY'), dict(available_gib=1.0))
    for i, bad in enumerate(bads):
        gb = gate(tmp_path, 'bad%d' % i, healthy(**{'available_gib': 7.0, **bad}), cache_profile=P)
        gb.initial = healthy()
        with pytest.raises(RuntimeError): gb.before_unit('before_record:x', loading=False)


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


def test_cache_identity_participates_in_segment_identity_only_when_recorded():
    from mri_rl.likelihood_adoption import segment_identity
    from mri_rl.experiment_contract import canonical_hash
    old = {'contract_sha256': 'c', 'checkpoint': 'A28', 'source_hashes': {'a': '1'}, 'backend_sources': {'b': '2'}, 'battery_exception': {'x': 1}}
    assert segment_identity(old) == canonical_hash({'contract': 'c', 'checkpoint': 'A28', 'sources': {'a': '1'}, 'backend': {'b': '2'}, 'battery_diagnostic': True})
    new = {**old, 'derived_cache': {'certificate_sha256': 'z'}}
    assert segment_identity(new) != segment_identity(old)
    assert segment_identity(new) == canonical_hash({'contract': 'c', 'checkpoint': 'A28', 'sources': {'a': '1'}, 'backend': {'b': '2'}, 'battery_diagnostic': True, 'derived_cache': {'certificate_sha256': 'z'}})


def test_finite_remaining_flag_is_explicit_bounded_and_nonresident():
    import inspect
    from mri_rl import paced_answer_likelihood as pal
    src = inspect.getsource(pal.prepare_plan)
    assert "57 if finite_remaining else 8" in src and 'finite_remaining and (resident or not battery_diagnostic)' in src
    assert 'finite_remaining_stage' in src


def test_certifier_requires_both_quantization_keys(tmp_path):
    plan, src, out, c, qmap = build(tmp_path)
    quant_cache.finalize_cache(plan, out, c, src, qmap)
    cfg = json.loads((c / 'config.json').read_text()); cfg['quantization_config'] = {'bits': 8, 'group_size': 64}
    (c / 'config.json').write_text(json.dumps(cfg))
    with pytest.raises(ValueError): quant_cache.certify_cache(plan, out, c, src, qmap)
