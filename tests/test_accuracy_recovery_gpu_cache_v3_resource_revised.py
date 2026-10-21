"""Codex required fix batch for cache v3 (resource-revised): NEW absolute 16 GiB swap cap, carried ORIGINAL v2 baseline, finite/range essential sensors,
optional MLX active guard, exact OS-pressure parsing/wiring, post-verify STOP, truthful adoption. CPU-only; fake mx; no MLX import."""
import inspect
import json
import subprocess
from types import SimpleNamespace
import numpy as np
import pytest
from mri_rl import accuracy_recovery_cpu_cache_v1 as cc, accuracy_recovery_gpu_cache_v2 as v2, accuracy_recovery_gpu_cache_v3 as v3, paced_resources, quant_convert as qc, resource_supervisor
from tests.test_accuracy_recovery_cpu_cache_v1 import FakeNN, sampler
from tests.test_accuracy_recovery_gpu_cache_v2 import GpuMx, installed_fn, now_utc
from tests.test_accuracy_recovery_gpu_cache_v3 import (REAL, S, Clock, adm3, adopt3, admit3, ctl, env3, log, run3, seq)

BASE0 = REAL[0]['swap_used_gib']                                      # 13.96966796875: the ORIGINAL authenticated v2 baseline


def test_new_absolute_swap_used_cap_16gib_is_stated_new_and_stops_strictly_above(tmp_path):
    P = v3.POLICY
    assert P['swap_used_absolute_cap_gib'] == 16.0 and 'NEW' in P['swap_used_absolute_cap_note'] and 'no absolute swap-used cap existed' in P['swap_used_absolute_cap_note']
    c, _ = ctl(tmp_path, seq(S(swap_used_gib=16.0), S(swap_used_gib=16.0000001), S(swap_used_gib=40.0)), baseline_swap_gib=15.5)       # the growth cap alone would allow 16.0
    assert c.observe('a')[1] == 'CONTINUE'
    _, d, reasons = c.observe('b'); assert d == 'STOP' and 'swap_used_above_absolute_cap_16gib' in reasons
    _, d, reasons = c.observe('c'); assert d == 'STOP' and 'swap_used_above_absolute_cap_16gib' in reasons


def test_stage_growth_cap_is_stricter_than_the_absolute_cap_for_the_real_baseline(tmp_path):
    assert BASE0 + 1.5 == 15.46966796875 < 16.0
    c, _ = ctl(tmp_path, seq(S(swap_used_gib=BASE0 + 1.5), S(swap_used_gib=BASE0 + 1.5 + 1e-6)), baseline_swap_gib=BASE0)
    assert c.observe('endpoint')[1] == 'CONTINUE'
    _, d, reasons = c.observe('over'); assert d == 'STOP' and reasons == ['swap_net_growth_over_cap']


def test_original_baseline_is_carried_first_new_sample_reports_real_growth_and_nothing_resets_it(tmp_path):
    c, _ = ctl(tmp_path, seq(S(swap_used_gib=14.548095703125), S(swap_used_gib=13.0), S(swap_used_gib=14.0), S(swap_used_gib=15.46966796875), S(swap_used_gib=15.4697)),
               baseline_swap_gib=BASE0)
    for label in ('preflight', 'hold', 'unit_start'): c.observe(label)
    rows = log(tmp_path)
    assert rows[0]['swap_growth_gib'] == pytest.approx(0.578427734375, abs=1e-12) and rows[0]['swap_growth_gib'] != 0
    assert [r['baseline_swap_gib'] for r in rows] == [BASE0] * 3
    assert c.observe('endpoint')[1] == 'CONTINUE' and c.observe('over')[1] == 'STOP'


def test_standalone_controller_without_baseline_uses_the_first_valid_sample_only(tmp_path):
    c, _ = ctl(tmp_path, seq(S(swap_used_gib=float('nan')), S(swap_used_gib=10.0), S(swap_used_gib=11.0)))
    c.observe('a'); assert c.baseline_swap is None                    # NaN never poisons the baseline
    c.observe('b'); c.observe('c'); assert c.baseline_swap == 10.0 and log(tmp_path)[2]['swap_growth_gib'] == 1.0
    for bad in (float('nan'), -1.0, float('inf')):
        with pytest.raises(ValueError): ctl(tmp_path, seq(S()), baseline_swap_gib=bad)


def test_production_execute_carries_the_authenticated_original_baseline_into_the_controller(tmp_path):
    e = env3(tmp_path); adopt3(e)
    prev = v3.verify_previous_attempt(e.repo, e.pins, e.expected)
    assert prev['swap_baseline_gib'] == BASE0 == v3.original_swap_baseline(e.prev['resource_log'])
    assert v3.PRODUCTION_PINS['swap_baseline_gib'] == 13.96966796875
    run3(e, sampler_fn=lambda: S(swap_used_gib=14.548095703125))
    rows = [json.loads(l) for l in (e.work3 / 'v3_resource.jsonl').read_text().splitlines()]
    assert rows[0]['boundary'] == 'preflight' and rows[0]['baseline_swap_gib'] == BASE0 and rows[0]['swap_growth_gib'] == pytest.approx(0.578427734375)
    assert {r['baseline_swap_gib'] for r in rows} == {BASE0}


def test_execute_stops_when_swap_exceeds_the_stage_cap_even_though_below_16gib(tmp_path):
    e = env3(tmp_path); adopt3(e)
    with pytest.raises(cc.ResourceStop):
        run3(e, sampler_fn=lambda: S(swap_used_gib=BASE0 + 1.6))
    assert not (e.work3 / 'certificate.json').exists()


def test_original_baseline_must_come_from_the_hash_pinned_log_and_the_pinned_value(tmp_path):
    e = env3(tmp_path)
    with pytest.raises(ValueError, match='pinned'): v3.verify_previous_attempt(e.repo, {**e.pins, 'swap_baseline_gib': 12.0}, e.expected)
    e.prev['resource_log'].write_text(json.dumps(dict(REAL[0], swap_used_gib=0.0)) + '\n')
    with pytest.raises(ValueError, match='differs from its pin'): v3.verify_previous_attempt(e.repo, e.pins, e.expected)
    for bad in ('', json.dumps(dict(REAL[0], swap_used_gib=float('nan'))) + '\n', json.dumps({'a': 1}) + '\n'):
        e.prev['resource_log'].write_text(bad)
        with pytest.raises(ValueError): v3.original_swap_baseline(e.prev['resource_log'])


def test_scope_binds_the_original_baseline_and_admission_refuses_a_different_pin(tmp_path):
    e = env3(tmp_path); a = adm3(e)
    assert admit3(e, a).check('x', full=True)
    with pytest.raises(v2.ScopeRefused):
        v3.admit_v3(**a, now=now_utc(), repo_root=e.repo, expected=e.expected, installed=installed_fn, previous_pins={**e.pins, 'swap_baseline_gib': 1.0})


@pytest.mark.parametrize('field', ['available_gib', 'pressure_free_percent', 'swap_used_gib', 'process_rss_gib'])
@pytest.mark.parametrize('bad', [None, float('nan'), float('inf'), float('-inf'), -1.0, 'x', True])
def test_malformed_essential_sensors_never_continue(tmp_path, field, bad):
    c, _ = ctl(tmp_path, seq(S(**{field: bad})))
    assert c.observe('x')[1] != 'CONTINUE'
    c, _ = ctl(tmp_path, seq(S(**{field: bad})))
    with pytest.raises(cc.ResourceStop): c.boundary('x')              # HOLD, then the per-boundary bound fails closed


def test_pressure_above_100_bad_process_counts_and_unknown_os_levels_never_continue(tmp_path):
    for kw in (dict(pressure_free_percent=100.1), dict(other_heavy_processes=-1), dict(other_heavy_processes=1.5), dict(other_heavy_processes=float('nan')),
               dict(other_heavy_processes=True), dict(other_heavy_processes='0')):
        assert ctl(tmp_path, seq(S(**kw)))[0].observe('x')[1] == 'HOLD', kw
    for level in (0, 3, 5, 8, -1, float('nan'), 1.5, '1', True):
        assert ctl(tmp_path, seq(S()), level=level)[0].observe('x')[1] == 'HOLD', level
    assert ctl(tmp_path, seq(S(pressure_free_percent=100)))[0].observe('x')[1] == 'CONTINUE'


def test_the_nan_everywhere_continue_defect_is_closed(tmp_path):
    nan = float('nan')
    c, _ = ctl(tmp_path, seq(S(available_gib=nan, pressure_free_percent=nan, swap_used_gib=nan, process_rss_gib=nan, other_heavy_processes=0)), level=3)
    _, d, reasons = c.observe('x')
    assert d == 'HOLD' and {'available_invalid', 'pressure_invalid', 'swap_invalid', 'rss_invalid', 'os_pressure_level_unknown'} <= set(reasons)


def test_optional_native_mlx_active_guard_is_kept_when_reported(tmp_path):
    assert v3.POLICY['mlx_active_gib_stop_above'] == 18.0
    assert 'sample.mlx_active_gib>18' in inspect.getsource(paced_resources).replace(' ', '')       # the guard being preserved
    for v, want in ((None, 'CONTINUE'), (0.0, 'CONTINUE'), (18.0, 'CONTINUE'), (18.01, 'STOP'), (float('nan'), 'HOLD'), (-1.0, 'HOLD')):
        _, d, reasons = ctl(tmp_path, seq(S(mlx_active_gib=v)))[0].observe('x')
        assert d == want, (v, reasons)
    assert 'active_mlx_memory_above_guideline' in ctl(tmp_path, seq(S(mlx_active_gib=19.0)))[0].observe('x')[2]
    assert v3.POLICY['mlx_memory_limit_bytes'] == 8 * 1024**3 and v3.POLICY['mlx_cache_limit_bytes'] == 64 * 1024**2
    assert qc.MAX_UNIT_PEAK_MIB == 4096 and qc.MAX_RESIDUAL_ACTIVE_MIB == 256


def fake_run(stdout='1\n', rc=0, exc=None, seen=None):
    def run(argv, **kw):
        if seen is not None: seen.append((argv, kw))
        if exc: raise exc
        return SimpleNamespace(returncode=rc, stdout=stdout, stderr='')
    return run


def test_os_pressure_accepts_only_exact_1_2_4_from_the_exact_sysctl_command():
    seen = []
    for out, want in (('1\n', 1), ('2\n', 2), ('4\n', 4), ('1', 1), (' 4 \n', 4)):
        assert v3.os_pressure_level(fake_run(out, seen=seen)) == want
    assert seen[0][0] == ['sysctl', '-n', 'kern.memorystatus_vm_pressure_level'] and seen[0][1]['timeout'] == 2 and seen[0][1]['capture_output'] is True
    for out in ('0\n', '3\n', '5\n', '8\n', '16\n', '01\n', '1 2\n', '', '\n', 'x\n', '-1\n', '1.0\n', 'nan\n', None):
        assert v3.os_pressure_level(fake_run(out)) is None, out
    assert v3.os_pressure_level(fake_run('1\n', rc=1)) is None
    for exc in (PermissionError(1, 'EPERM'), OSError('x'), subprocess.TimeoutExpired('sysctl', 2), ValueError('v')):
        assert v3.os_pressure_level(fake_run(exc=exc)) is None


def test_production_controller_uses_the_actual_sysctl_parser_and_sampler_and_a_failed_read_holds(tmp_path, monkeypatch):
    c = v3.ResourceController()
    assert c.os_level is v3.os_pressure_level and c.sampler is resource_supervisor.sample_macos
    params = inspect.signature(v3.execute).parameters
    assert params['os_level'].default is None and params['sampler'].default is None
    monkeypatch.setattr(v3.subprocess, 'run', fake_run(exc=PermissionError(1, 'EPERM')))
    c2, _ = ctl(tmp_path, seq(S()), level=v3.os_pressure_level)
    _, d, reasons = c2.observe('x')
    assert d == 'HOLD' and 'os_pressure_level_unknown' in reasons and log(tmp_path)[0]['os_pressure_level'] is None
    monkeypatch.setattr(v3.subprocess, 'run', fake_run('4\n'))
    assert ctl(tmp_path, seq(S()), level=v3.os_pressure_level)[0].observe('x')[1] == 'STOP'
    monkeypatch.setattr(v3.subprocess, 'run', fake_run('2\n'))
    assert ctl(tmp_path, seq(S()), level=v3.os_pressure_level)[0].observe('x')[1] == 'CONTINUE'


def run_level(e, level_fn):
    mx = GpuMx(); adm = adm3(e); clock = Clock()
    def factory(proofs, **fk): return v2.GpuConvertBackend(proofs, mx=mx, np=np, nn=FakeNN(mx), **fk)
    return v3.execute(**adm, repo_root=e.repo, expected=e.expected, installed=installed_fn, backend_factory=factory, sampler=sampler(), os_level=level_fn,
                      rss_gib=lambda: 0.5, clock=clock, sleep=clock.sleep, previous_pins=e.pins)


def test_a_critical_reading_at_post_verify_persists_the_stop_and_never_publishes_success(tmp_path, monkeypatch):
    e = env3(tmp_path); adopt3(e); flag = []
    real = v3.verify_historical_v3
    def verify_then_go_critical(*a, **k):
        r = real(*a, **k); flag.append(1); return r
    monkeypatch.setattr(v3, 'verify_historical_v3', verify_then_go_critical)
    with pytest.raises(cc.ResourceStop):
        run_level(e, lambda: 4 if flag else 1)
    assert not (e.work3 / 'certificate.json').exists()
    fail = json.loads((e.work3 / 'v3_failure_001.json').read_text())
    assert fail['error_type'] == 'ResourceStop' and 'os_memory_pressure_critical' in fail['reasons'] and fail['last_resource_sample']['boundary'] == 'post_verify'
    rows = [json.loads(l) for l in (e.work3 / 'v3_resource.jsonl').read_text().splitlines()]
    assert rows[-1]['boundary'] == 'post_verify' and rows[-1]['decision'] == 'STOP'
    assert [json.loads(p.read_text())['state'] for p in (e.repo / v3.CONSUMED_REL).glob('scope_outcome_*.json')] == ['FAILED']
    assert (e.work3 / 'units/receipts.jsonl').exists() and not (e.work3 / 'v3_active.lock').exists()          # unit receipts kept, lock released, no quantization retry


def test_adoption_never_claims_missing_old_evidence_and_never_fabricates_a_receipt(tmp_path):
    e = env3(tmp_path); adopt3(e)
    rec = json.loads((e.work3 / 'adoption/unit0_adoption_PRIVATE.json').read_text())
    for forbidden in ('module_path', 'module_path_bitwise_equal', 'passed', 'reload_bitwise_equal'):
        assert forbidden not in rec
    assert rec['receipt_equivalent'] is False and rec['old_receipt_fabricated'] is False and 'module_path_bitwise_equal_result' in rec['evidence_missing']
    assert 'NOT claimed' in rec['evidence_present_notes']['module_path']
    res, _, _ = run3(e)
    g = res['certificate']['gpu_reconstruction']
    assert 'module_path_bitwise_equal_result' in g['adopted_unit_evidence_missing'] and g['adopted_unit_basis'].endswith('no old receipt fabricated')
    assert 0 not in [json.loads(l)['index'] for l in (e.work3 / 'units/receipts.jsonl').read_text().splitlines()]


def test_cumulative_accounting_is_686_new_and_688_including_the_old_two():
    L = v3.SCOPE
    assert 343 + 343 == L['new_underlying_quantizations_max'] == 686 and 686 + 2 == L['cumulative_underlying_quantizations_max'] == 688
