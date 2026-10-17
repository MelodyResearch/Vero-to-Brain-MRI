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


def test_cache_materialization_checks_forecast_even_above_generic_unit_floor(tmp_path, monkeypatch):
    from mri_rl import likelihood_battery as lb
    from mri_rl.paced_answer_likelihood import CACHE_PROFILE
    monkeypatch.setattr(lb, 'os_pressure_level', lambda: 2)
    g = gate(tmp_path, 'forecast', healthy(available_gib=6.5, mlx_active_gib=0.1), cache_profile=CACHE_PROFILE)
    g.initial = healthy()
    with pytest.raises(RuntimeError, match='insufficient_memory_headroom'):
        g.before_unit('materialized:64')


# ---- cache-path swap phase machine (load envelope -> resident baseline) ----
def phased_gate(tmp_path, name, swaps, profile=True, **sample_kw):
    """Gate fed a scripted sequence of swap values (one per sample); advancing clock; healthy otherwise."""
    from mri_rl.likelihood_battery import BatteryDiagnosticGate
    from mri_rl.paced_answer_likelihood import CACHE_PROFILE as P
    d = tmp_path / name; d.mkdir(); it = iter(swaps); clock = [0.0]
    def sampler():
        clock[0] += 31.0
        return healthy(monotonic_seconds=clock[0], swap_used_gib=next(it), available_gib=9.0, **sample_kw)
    return BatteryDiagnosticGate(d / 'r.jsonl', sampler=sampler, sleep=lambda s: None, cache_profile=P if profile else None), d


def test_healthy_cache_load_with_1p073_growth_logs_and_continues_without_hold(tmp_path, monkeypatch):
    from mri_rl import likelihood_battery as lb
    monkeypatch.setattr(lb, 'os_pressure_level', lambda: 2)
    # swaps: preflight x2 at 6.714, then rising loads incl. the real 1.073 and a slow-rising rate-only tail
    g, d = phased_gate(tmp_path, 'a', [6.714, 6.714, 6.8, 7.0, 7.4, 7.787, 7.787])
    g.before_unit('before_runtime_import', loading=True)
    for u in ('materialized:64', 'materialized:640', 'materialized:1280', 'materialized:1438'):
        assert g.before_unit(u)
    assert g.before_unit('quantized_model_ready')
    ev = [json.loads(l) for l in (d / 'r.jsonl').read_text().splitlines()]
    assert all(e['action'] in ('CONTINUE', 'WARNING') for e in ev) and not any(':warning_hold' in e['unit_id'] for e in ev)
    assert ev[-1]['cache_phase'] == 'load' and abs(ev[-2]['swap_growth_total_gib'] - 1.073) < 1e-6
    assert g.resident_start is not None and ev[-1]['swap_limits_gib']['resident_total_growth_lt'] == pytest.approx(5.8163, abs=1e-4)


def test_load_growth_at_or_above_packed_stops_and_failed_ready_cannot_capture_baseline(tmp_path, monkeypatch):
    from mri_rl import likelihood_battery as lb
    monkeypatch.setattr(lb, 'os_pressure_level', lambda: 1)
    g, _ = phased_gate(tmp_path, 'b', [6.0, 6.0, 6.0 + 4.8163 + 0.001])
    g.before_unit('before_runtime_import', loading=True)
    with pytest.raises(RuntimeError): g.before_unit('quantized_model_ready')
    assert g.resident_start is None
    g2, _ = phased_gate(tmp_path, 'b2', [6.0, 6.0, 11.0])
    g2.before_unit('before_runtime_import', loading=True)
    with pytest.raises(RuntimeError): g2.before_unit('materialized:64')


def test_resident_net_and_total_limits_and_second_ready_cannot_reset(tmp_path, monkeypatch):
    from mri_rl import likelihood_battery as lb
    monkeypatch.setattr(lb, 'os_pressure_level', lambda: 1)
    # ready captured at 8.0 (growth 2.0 from 6.0); resident net 0.99 ok; net 1.0 stops
    g, _ = phased_gate(tmp_path, 'c', [6.0, 6.0, 8.0, 8.0, 8.99, 9.0])
    g.before_unit('before_runtime_import', loading=True); g.before_unit('quantized_model_ready')
    base = g.resident_start
    assert g.before_unit('quantized_model_ready') and g.resident_start is base      # second ready does not reset
    assert g.before_unit('before_record:x')                                          # net 0.99 < 1
    with pytest.raises(RuntimeError): g.before_unit('before_record:y')               # net 1.0 -> stop
    # total limit is implied: baseline growth < packed and net < 1 keep total < packed + 1 (checked at the edge)
    g2, d2 = phased_gate(tmp_path, 'c2', [6.0, 6.0, 10.8, 10.8, 11.79])
    g2.before_unit('before_runtime_import', loading=True); g2.before_unit('quantized_model_ready')
    assert g2.before_unit('before_record:x')
    last = json.loads((d2 / 'r.jsonl').read_text().splitlines()[-1])
    assert last['swap_growth_total_gib'] < last['swap_limits_gib']['resident_total_growth_lt']


def test_generic_non_cache_path_keeps_exact_1gib_stage_cap(tmp_path, monkeypatch):
    from mri_rl import likelihood_battery as lb
    monkeypatch.setattr(lb, 'os_pressure_level', lambda: 1)
    g, _ = phased_gate(tmp_path, 'd', [6.0, 6.0, 7.0], profile=False)
    g.before_unit('before_runtime_import', loading=True)
    with pytest.raises(RuntimeError): g.before_unit('materialized:64')


@pytest.mark.parametrize('kw, level', [(dict(pressure_free_percent=10), 1), (dict(thermal_state='THROTTLED'), 1),
                                       (dict(other_heavy_processes=1), 1), (dict(battery_percent=15), 1), ({}, 4)])
def test_critical_protections_remain_in_load_phase(tmp_path, monkeypatch, kw, level):
    from mri_rl import likelihood_battery as lb
    monkeypatch.setattr(lb, 'os_pressure_level', lambda: level)
    g, _ = phased_gate(tmp_path, 'e%d' % abs(hash(str(kw))), [6.0] * 6, **kw)
    with pytest.raises(RuntimeError): g.before_unit('before_runtime_import', loading=True)


# ---- cache-path corroborated memory WARNING waits via bounded HOLD (no unload) ----
def mem_gate(tmp_path, name, pressures, *, profile=True, os_level=2, **kw):
    """Scripted pressure-free sequence for the post-preflight samples; first two are healthy preflight samples."""
    from mri_rl import likelihood_battery as lb
    from mri_rl.paced_answer_likelihood import CACHE_PROFILE as P
    d = tmp_path / name; d.mkdir(); it = iter([50, 50, *pressures]); clock = [0.0]; slept = []; n = [0]
    def sampler():
        clock[0] += 31.0; n[0] += 1
        return healthy(monotonic_seconds=clock[0], pressure_free_percent=next(it), swap_used_gib=6.0,
                       available_gib=9.0 if n[0] <= 2 else 5.4, **(kw if n[0] > 2 else {}))
    g = lb.BatteryDiagnosticGate(d / 'r.jsonl', sampler=sampler, sleep=slept.append, cache_profile=P if profile else None)
    g.before_unit('before_runtime_import', loading=True)
    g.resident_start = g.initial
    return g, d, slept


def test_real_case_27_31_32_waits_two_clean_samples_then_resumes(tmp_path, monkeypatch):
    from mri_rl import likelihood_battery as lb
    monkeypatch.setattr(lb, 'os_pressure_level', lambda: 2)
    g, d, slept = mem_gate(tmp_path, 'a', [27, 31, 32])
    assert g.before_unit('candidate:x:FLAIR')                       # returns only after the memory reasons cleared
    ev = [json.loads(l) for l in (d / 'r.jsonl').read_text().splitlines()][2:]
    assert [e['action'] for e in ev] == ['HOLD', 'HOLD_RECOVERY', 'CONTINUE'] and slept[-2:] == [15, 15]
    assert ev[0]['memory_warning_hold'] is True and ev[0]['reasons']


def test_unresolved_27_27_27_times_out_with_stop_and_progress_preserved(tmp_path, monkeypatch):
    from mri_rl import likelihood_battery as lb
    monkeypatch.setattr(lb, 'os_pressure_level', lambda: 2)
    g, d, slept = mem_gate(tmp_path, 'b', [27, 27, 27, 27])
    with pytest.raises(RuntimeError, match='warning_hold_not_recovered'): g.before_unit('candidate:x:FLAIR')
    assert slept[-3:] == [15, 15, 15] and slept.count(15) == 3         # bounded: exactly the existing 3 checks


def test_attributable_historical_policy_warning_does_not_force_unload_when_reasons_are_clear(tmp_path, monkeypatch):
    from mri_rl import likelihood_battery as lb
    monkeypatch.setattr(lb, 'os_pressure_level', lambda: 2)
    g, d, _ = mem_gate(tmp_path, 'c', [31, 31, 31])
    g.policy.level = 'WARNING'; g.policy._recovery = 0; g._mem_episode = True                 # policy still WARNING after an attributable memory episode
    assert g.before_unit('candidate:x:FLAIR')                          # actual memory reasons clear -> no unload, no new HOLD
    last = json.loads((d / 'r.jsonl').read_text().splitlines()[-1])
    assert last['action'] == 'CONTINUE' and last['reasons'] == [] and last['supervisor_level'] == 'WARNING'
    g2, _, _ = mem_gate(tmp_path, 'c2', [31, 31, 31])
    g2.policy.level = 'WARNING'; g2.policy._recovery = 0                 # WARNING NOT attributable to a memory episode: still stops
    with pytest.raises(RuntimeError): g2.before_unit('candidate:x:FLAIR')


@pytest.mark.parametrize('case', ['os4', 'below_phase_headroom', 'swap_budget', 'thermal', 'battery', 'concurrency', 'pressure_le_15'])
def test_any_other_reason_during_memory_hold_stops(tmp_path, monkeypatch, case):
    from mri_rl import likelihood_battery as lb
    level = [2]
    monkeypatch.setattr(lb, 'os_pressure_level', lambda: level[0])
    kw = {'thermal': dict(thermal_state='THROTTLED'), 'battery': dict(battery_percent=15), 'concurrency': dict(other_heavy_processes=1)}.get(case, {})
    g, d, _ = mem_gate(tmp_path, case, [12 if case == 'pressure_le_15' else 27, 27, 27], **kw)
    if case == 'below_phase_headroom':
        g.sampler = (lambda inner=g.sampler: replace(inner(), available_gib=2.5))
    if case == 'swap_budget':
        base = g.sampler; g.sampler = lambda: replace(base(), swap_used_gib=6.0 + 0.0 + 1.0)   # resident net >= 1 GiB
    level[0] = 4 if case == 'os4' else 2
    with pytest.raises(RuntimeError): g.before_unit('candidate:x:FLAIR')


def test_generic_non_cache_path_has_no_memory_hold(tmp_path, monkeypatch):
    from mri_rl import likelihood_battery as lb
    monkeypatch.setattr(lb, 'os_pressure_level', lambda: 2)
    g, d, slept = mem_gate(tmp_path, 'g', [27, 31, 32], profile=False)
    with pytest.raises(RuntimeError): g.before_unit('candidate:x:FLAIR')


def test_ready_boundary_recovery_captures_resident_baseline_once(tmp_path, monkeypatch):
    from mri_rl import likelihood_battery as lb
    monkeypatch.setattr(lb, 'os_pressure_level', lambda: 2)
    g, _, _ = mem_gate(tmp_path, 'ready-recovery', [27, 31, 32, 32])
    g.resident_start = None
    original_initial, original_history = g.initial, g.history
    sampler = g.sampler
    g.sampler = lambda: replace(sampler(), mlx_active_gib=4.82)
    admitted = g.before_unit('quantized_model_ready')
    assert g.resident_start is admitted
    assert g.initial is original_initial and g.history is original_history
    g.before_unit('quantized_model_ready')
    assert g.resident_start is admitted
