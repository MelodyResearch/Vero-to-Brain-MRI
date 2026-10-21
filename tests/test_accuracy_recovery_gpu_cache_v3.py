"""CPU-only tests for the cache v3 local resource policy + evidence-bound continuation (saved exact unit 0 adopted, remaining units on GPU).
A numpy-backed fake mx/nn stands in for MLX: no MLX import, no model. Scope / human / review / old-attempt evidence files built here are
TEST-ONLY fabrications in tmp dirs; they authorize nothing real."""
import fcntl
import hashlib
import inspect
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from mri_rl import (accuracy_recovery_cpu_cache_v1 as cc, accuracy_recovery_gpu_cache_v2 as v2, accuracy_recovery_gpu_cache_v3 as v3,
                    paced_resources, quant_cache, quant_convert as qc, resource_supervisor)
from tests.test_accuracy_recovery_cpu_cache_v1 import FakeNN, sampler
from tests.test_accuracy_recovery_gpu_cache_v2 import GpuMx, Hold, SHA, installed_fn, now_utc, snapshot
from tests.test_accuracy_recovery_gpu_cache_v2 import adopt as adopt2, env as env2, loaded as loaded2, write as write2
from tests.test_quant_convert import healthy

SRC_ROOT = Path(__file__).parents[1] / 'src'
REPO = Path(__file__).parents[1]
SCRIPT3 = REPO / 'scripts' / 'rebuild_accuracy_recovery_historical_gpu_cache_v3.py'
GUARDED = ('src/mri_rl/accuracy_recovery_cpu_cache_v1.py', 'src/mri_rl/quant_convert.py', 'src/mri_rl/quant_cache.py', 'src/mri_rl/paced_resources.py',
           'src/mri_rl/resource_supervisor.py', 'src/mri_rl/accuracy_recovery_gpu_cache_v2.py', 'src/mri_rl/quant_probe.py')


class Clock:
    def __init__(self, t=1000.0): self.t, self.sleeps = t, []
    def __call__(self): return self.t
    def sleep(self, s): self.sleeps.append(s); self.t += s


def S(**kw):
    kw.setdefault('monotonic_seconds', 0.0); return healthy(**kw)


def seq(*samples):
    """Sampler returning the given samples in order, then repeating the last one forever."""
    it, last = iter(samples), [None]
    def f():
        try: last[0] = next(it)
        except StopIteration: pass
        return last[0]
    f.calls = lambda: None
    return f


def ctl(tmp_path, sampler_fn, level=1, **kw):
    clock = Clock()
    c = v3.ResourceController(sampler=sampler_fn, os_level=(level if callable(level) else (lambda: level)), clock=clock, sleep=clock.sleep,
                              utc=lambda: '2026-10-21T00:00:00+00:00', log_path=tmp_path / 'r.jsonl', **kw)
    return c, clock


def log(tmp_path): return [json.loads(l) for l in (tmp_path / 'r.jsonl').read_text().splitlines()]


# ---- the actual saved 3-sample replay (the real v2_resource.jsonl values; the log has no timestamps, so spacing here is SYNTHETIC) ------
REAL = [dict(available_gib=7.1511993408203125, pressure_free_percent=40.0, swap_used_gib=13.96966796875, process_rss_gib=0.05938720703125),
        dict(available_gib=7.1817169189453125, pressure_free_percent=40.0, swap_used_gib=13.96966796875, process_rss_gib=0.0924224853515625),
        dict(available_gib=5.9315185546875, pressure_free_percent=35.0, swap_used_gib=14.548095703125, process_rss_gib=0.693634033203125)]


def real_samples(times=(100.0, 101.0, 103.3)):
    return [S(**r, monotonic_seconds=t, power_source='AC', battery_percent=100, other_heavy_processes=0, thermal_state='SENSOR_UNAVAILABLE')
            for r, t in zip(REAL, times)]


def test_saved_three_sample_replay_old_policy_stopped_new_policy_continues_with_swap_trend_only(tmp_path):
    samples = real_samples(); it = iter(samples)
    old = cc.guarded_sampler(lambda: next(it))
    old(); old()
    with pytest.raises(cc.ResourceStop) as e:                                   # what actually happened in the consumed run
        old()
    assert set(e.value.reasons) == {'sustained_swap_growth', 'insufficient_memory_headroom'}
    c, clock = ctl(tmp_path, seq(*samples))
    need_unit0 = qc.forecast_gib({'action': 'quantize', 'nbytes': 1187 * 1024**2, 'transform': None}) + qc.RESERVE_GIB          # ~5.30 < 5.5
    for _ in samples:
        assert c.boundary('replay', need_gib=need_unit0) is not None
    rows = log(tmp_path)
    assert [r['decision'] for r in rows] == ['CONTINUE'] * 3 and clock.sleeps == [] and all(r['reasons'] == [] for r in rows)
    assert 'sustained_swap_growth' in rows[2]['inherited_supervisor_reasons'] and 'insufficient_memory_headroom' not in rows[2]['reasons']
    assert rows[2]['swap_growth_gib'] == pytest.approx(0.5784277343749995) and rows[2]['swap_growth_trend_only'] is True


def test_replay_matches_the_persisted_log_when_present():
    logp = Path(__file__).resolve()
    for p in logp.parents:
        cand = p / 'work/accuracy-recovery-historical-gpu-cache-v2/v2_resource.jsonl'
        if cand.exists():
            rows = [json.loads(l) for l in cand.read_text().splitlines()]
            assert [(r['available_gib'], r['pressure_free_percent'], r['swap_used_gib']) for r in rows] == [(r['available_gib'], r['pressure_free_percent'], r['swap_used_gib']) for r in REAL]
            return
    pytest.skip('prior v2 log not present')


# ---- policy thresholds -------------------------------------------------------------------------------------------
@pytest.mark.parametrize('kw,decision', [
    (dict(available_gib=5.5, pressure_free_percent=30), 'CONTINUE'), (dict(available_gib=5.4999), 'HOLD'), (dict(pressure_free_percent=29.9), 'HOLD'),
    (dict(pressure_free_percent=15.1), 'HOLD'), (dict(pressure_free_percent=15), 'STOP'), (dict(pressure_free_percent=3), 'STOP'),
    (dict(available_gib=1.99), 'STOP'), (dict(available_gib=2.0), 'HOLD'), (dict(process_rss_gib=8.01), 'STOP'), (dict(process_rss_gib=8.0), 'CONTINUE'),
    (dict(power_source='BATTERY', battery_percent=90), 'HOLD'), (dict(power_source='SENSOR_UNAVAILABLE'), 'HOLD'), (dict(other_heavy_processes=1), 'HOLD'),
    (dict(thermal_state='THROTTLED'), 'HOLD'), (dict(thermal_state='SENSOR_UNAVAILABLE'), 'CONTINUE'), (dict(thermal_state='NORMAL'), 'CONTINUE'),
    (dict(available_gib=None), 'HOLD'), (dict(pressure_free_percent=None), 'HOLD'), (dict(swap_used_gib=None), 'HOLD'), (dict(other_heavy_processes=None), 'HOLD'),
    (dict(process_rss_gib=None), 'HOLD')])
def test_single_sample_decisions(tmp_path, kw, decision):
    c, _ = ctl(tmp_path, seq(S(**kw)))
    s, d, reasons = c.observe('x')
    assert d == decision, reasons


def test_os_critical_or_unknown_os_level_decisions(tmp_path):
    for level, want in ((1, 'CONTINUE'), (2, 'CONTINUE'), (4, 'STOP'), (5, 'HOLD'), (3, 'HOLD'), (0, 'HOLD'), (None, 'HOLD')):
        c, _ = ctl(tmp_path, seq(S()), level=level)
        assert c.observe('x')[1] == want


def test_every_decision_is_a_pure_local_rule_not_the_inherited_supervisor_or_paced_decisions(tmp_path):
    src = inspect.getsource(v3.ResourceController)
    assert 'decisions(' not in src and '.observe(s' not in src.replace('self._inherited.observe', '')      # inherited policy is logged, never decides


# ---- HOLD semantics ----------------------------------------------------------------------------------------------
def test_hold_then_recover_waits_in_5s_steps_logs_utc_and_monotonic_and_returns(tmp_path):
    c, clock = ctl(tmp_path, seq(S(available_gib=5.0), S(available_gib=4.5), S(available_gib=5.2), S()))
    t0 = clock()
    assert c.boundary('unit_start', unit=7) is not None
    assert clock.sleeps == [5.0, 5.0, 5.0] and clock() - t0 == 15
    rows = log(tmp_path)
    assert [r['decision'] for r in rows] == ['HOLD', 'HOLD', 'HOLD', 'CONTINUE'] and [r['hold_elapsed_seconds'] for r in rows] == [0.0, 5.0, 10.0, 15.0]
    assert all(r['utc'] and isinstance(r['monotonic'], float) and r['unit'] == 7 for r in rows) and rows[0]['required_available_gib'] == 5.5


def test_hold_is_bounded_to_120_seconds_per_boundary_then_stops(tmp_path):
    c, clock = ctl(tmp_path, seq(S(available_gib=5.0)))
    with pytest.raises(cc.ResourceStop) as e:
        c.boundary('unit_start')
    assert 'hold_timeout' in e.value.reasons and clock() - 1000.0 == 120.0 and len(clock.sleeps) == 24 and len(log(tmp_path)) == 25


def test_each_boundary_gets_its_own_120s_hold_budget(tmp_path):
    c, clock = ctl(tmp_path, seq(*([S(available_gib=5.0)] * 20 + [S()] + [S(available_gib=5.0)] * 20 + [S()])))
    c.boundary('a'); c.boundary('b')
    assert sum(clock.sleeps) == 200.0


def test_sustained_below_4gib_for_30s_stops_without_waiting_the_full_hold(tmp_path):
    c, clock = ctl(tmp_path, seq(S(available_gib=3.5)))
    with pytest.raises(cc.ResourceStop) as e:
        c.boundary('unit_start')
    assert 'sustained_available_below_4gib' in e.value.reasons and clock() - 1000.0 == 30.0 < 120


def test_brief_dip_below_4gib_recovers_without_stopping(tmp_path):
    c, clock = ctl(tmp_path, seq(S(available_gib=3.5), S(available_gib=3.9), S(available_gib=4.2), S(available_gib=3.5), S()))
    assert c.boundary('x') is not None and clock() - 1000.0 == 20.0


@pytest.mark.parametrize('kw,level,reason', [(dict(pressure_free_percent=15), 1, 'pressure_free_at_or_below_15'), (dict(), 4, 'os_memory_pressure_critical'),
                                             (dict(available_gib=1.0), 1, 'available_below_2gib'), (dict(process_rss_gib=9.0), 1, 'process_rss_above_8gib')])
def test_critical_conditions_stop_immediately_without_hold(tmp_path, kw, level, reason):
    c, clock = ctl(tmp_path, seq(S(**kw)), level=level)
    with pytest.raises(cc.ResourceStop) as e:
        c.boundary('unit_start')
    assert reason in e.value.reasons and clock.sleeps == [] and len(log(tmp_path)) == 1


@pytest.mark.parametrize('kw,reason', [(dict(power_source='BATTERY'), 'not_ac'), (dict(power_source='SENSOR_UNAVAILABLE'), 'power_unknown'),
                                       (dict(other_heavy_processes=2), 'other_heavy_process'), (dict(other_heavy_processes=None), 'other_heavy_unknown'),
                                       (dict(thermal_state='THROTTLED'), 'thermal_throttled'), (dict(swap_used_gib=None), 'swap_unknown'),
                                       (dict(process_rss_gib=None), 'rss_unknown'), (dict(available_gib=None), 'available_unknown'), (dict(pressure_free_percent=None), 'pressure_unknown')])
def test_other_hard_guards_hold_never_proceed_and_fail_closed_after_the_bound(tmp_path, kw, reason):
    c, clock = ctl(tmp_path, seq(S(**kw)))
    with pytest.raises(cc.ResourceStop) as e:
        c.boundary('unit_start')
    assert reason in e.value.reasons and 'hold_timeout' in e.value.reasons and clock() - 1000.0 == 120.0
    c2, clock2 = ctl(tmp_path, seq(S(**kw), S(**kw), S()))
    assert c2.boundary('unit_start') is not None and clock2.sleeps == [5.0, 5.0]               # recovers


def test_unknown_os_level_holds_then_fails_closed(tmp_path):
    c, clock = ctl(tmp_path, seq(S()), level=None)
    with pytest.raises(cc.ResourceStop) as e:
        c.boundary('x')
    assert 'os_pressure_level_unknown' in e.value.reasons and clock() - 1000.0 == 120.0


def test_no_operation_runs_while_holding_and_the_window_includes_holds(tmp_path):
    ops = []
    c, clock = ctl(tmp_path, seq(S(available_gib=5.0), S(available_gib=5.0), S()))
    c.start_window(); clock.t += 1795.0
    with pytest.raises(v2.WindowExceeded):
        c.boundary('unit_start')
    assert ops == [] and clock.sleeps == [5.0, 5.0] and clock() - 1000.0 > 1800.0 and len(log(tmp_path)) == 2               # the holds themselves consumed the window
    c2, clock2 = ctl(tmp_path, seq(S()))
    c2.start_window(); clock2.t += 1799.0
    assert c2.boundary('unit_start') is not None                                                   # still inside the window


def test_end_boundary_observation_never_waits_or_raises_so_a_saved_unit_keeps_its_receipt(tmp_path):
    c, clock = ctl(tmp_path, seq(S(pressure_free_percent=10)))
    s, d, reasons = c.observe('unit_end', unit=1)
    assert d == 'STOP' and clock.sleeps == [] and 'pressure_free_at_or_below_15' in reasons
    c.start_window(); clock.t += 5000.0
    assert c.observe('unit_end')[1] == 'STOP'                                                     # window is only about starting operations


# ---- swap: cumulative net growth cap 1.5 GiB from the first sample, trend-only rate ------------------------------------------------
def test_swap_cap_is_cumulative_1p5gib_from_the_first_sample_and_never_resets(tmp_path):
    assert v3.POLICY['swap_cumulative_net_growth_cap_gib'] == 1.5 and v2.cc.SWAP_GROWTH_CAP_GIB == 1.0
    c, _ = ctl(tmp_path, seq(S(swap_used_gib=14.0), S(swap_used_gib=15.0), S(swap_used_gib=15.5), S(swap_used_gib=15.5001)))
    assert [c.observe(k)[1] for k in 'abc'] == ['CONTINUE'] * 3                                  # +0.0, +1.0, +1.5 (== cap is allowed)
    s, d, reasons = c.observe('d')
    assert d == 'STOP' and 'swap_net_growth_over_cap' in reasons
    rows = log(tmp_path)
    assert rows[-1]['baseline_swap_gib'] == 14.0 and rows[-1]['swap_growth_gib'] == pytest.approx(1.5001)


def test_decreases_do_not_reset_the_baseline(tmp_path):
    c, _ = ctl(tmp_path, seq(S(swap_used_gib=12.0), S(swap_used_gib=11.0), S(swap_used_gib=13.4), S(swap_used_gib=13.6)))
    assert [c.observe(k)[1] for k in 'abc'] == ['CONTINUE'] * 3 and c.observe('d')[1] == 'STOP'


def test_fast_swap_growth_alone_is_trend_only_when_memory_is_healthy(tmp_path):
    c, _ = ctl(tmp_path, seq(S(swap_used_gib=14.0, monotonic_seconds=0.0), S(swap_used_gib=14.9, monotonic_seconds=1.0)))
    assert c.observe('a')[1] == 'CONTINUE' and c.observe('b')[1] == 'CONTINUE'
    assert log(tmp_path)[1]['swap_growth_gib_per_min_trend_only'] > 50


def test_baseline_is_the_first_known_swap_sample(tmp_path):
    c, _ = ctl(tmp_path, seq(S(swap_used_gib=None), S(swap_used_gib=12.0), S(swap_used_gib=13.0)))
    assert c.observe('a')[1] == 'HOLD' and c.observe('b')[1] == 'CONTINUE' and c.observe('c')[1] == 'CONTINUE'
    assert log(tmp_path)[2]['baseline_swap_gib'] == 12.0


def test_unit_dependent_forecast_reserve_holds_when_transient_and_does_not_lower_the_floor(tmp_path):
    c, clock = ctl(tmp_path, seq(S(available_gib=5.8), S(available_gib=6.1)))
    assert c.boundary('unit_start', need_gib=6.0) is not None and clock.sleeps == [5.0]
    assert log(tmp_path)[0]['required_available_gib'] == 6.0 and 'insufficient_forecast_headroom' in log(tmp_path)[0]['reasons']
    c2, _ = ctl(tmp_path, seq(S(available_gib=5.49)))
    assert c2.observe('x', need_gib=1.0)[1] == 'HOLD'                                          # a small forecast never lowers the 5.5 floor


def test_real_largest_unit_forecast_plus_reserve_is_below_the_new_floor():
    big = {'action': 'quantize', 'nbytes': 1187 * 1024**2, 'transform': None}
    assert qc.forecast_gib(big) + qc.RESERVE_GIB < v3.POLICY['continue_available_gib_min']


def test_log_lines_are_durable_json_with_both_clocks(tmp_path):
    c, _ = ctl(tmp_path, seq(S()))
    c.boundary('preflight')
    row = log(tmp_path)[0]
    for k in ('utc', 'monotonic', 'boundary', 'decision', 'reasons', 'available_gib', 'pressure_free_percent', 'swap_used_gib', 'baseline_swap_gib',
              'power_source', 'other_heavy_processes', 'process_rss_gib', 'thermal_state', 'os_pressure_level', 'window_elapsed_seconds'):
        assert k in row


# ---- v3 fixture: the closed v2 attempt left unit 0 saved without a receipt ----------------------------------------------------------
OLD_DIGEST = 'd' * 64
OLD_INSTRUCTION = 'e' * 64


REAL_LOG = [dict(r, boundary='x') for r in REAL]


def env3(tmp_path):
    e = env2(tmp_path); adopt2(e)                                             # old v2 work root: 406 (here 1) retained units, authenticated
    plan, hist, proofs = loaded2(e)
    mx = GpuMx(); b = v2.GpuConvertBackend(proofs, mx=mx, np=np, nn=FakeNN(mx), rss_gib=lambda: 0.5)
    qc.convert(plan, e.args['source_dir'], e.work / 'units', b, sampler(), max_units=1)       # the old run's unit 0: GPU output, then ...
    rp = e.work / 'units/receipts.jsonl'                                    # ... the post-unit resource gate stopped BEFORE its receipt
    rp.write_text(''.join(l + '\n' for l in rp.read_text().splitlines() if json.loads(l)['index'] != 0))
    prev = v3.previous_paths(e.repo, OLD_DIGEST, OLD_INSTRUCTION)
    files = {
        'failure_receipt': {'error_type': 'ResourceStop', 'error': 'Resource gate refused: sustained_swap_growth,insufficient_memory_headroom', 'automatic_retry': False,
                            'rerun_allowed': False, 'partial': True, 'gpu_unit_attempts': 1, 'gpu_direct_quantize_ops': 1, 'gpu_module_path_quantizations': 1,
                            'gpu_underlying_quantizations': 2, 'current_unit': 0, 'units_with_passing_receipts': 1, 'reasons': ['sustained_swap_growth', 'insufficient_memory_headroom']},
        'ledger': {'scope_sha256': OLD_DIGEST, 'human_instruction_sha256': OLD_INSTRUCTION, 'pending_quantized_units': 2},
        'resource_log': REAL_LOG, 'rss_log': {'b': 1}, 'launcher_record': {'child': {'native_exit_code': 1}}, 'summary': {'s': 1}, 'stdout': None, 'stderr': {'e': 1},
        'independent_detail': {'d': 1}, 'scope': {'old': 'scope'}, 'human': {'old': 'human'}, 'review': {'old': 'review'},
        'consumed_scope': {'scope_sha256': OLD_DIGEST, 'state': 'STARTED'}, 'consumed_instruction': {'consumed': OLD_INSTRUCTION},
        'consumed_outcome': {'scope_sha256': OLD_DIGEST, 'state': 'FAILED'}}
    for name, doc in files.items():
        p = prev[name]; p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('' if doc is None else ''.join(json.dumps(r) + '\n' for r in doc) if name == 'resource_log' else json.dumps(doc))
    e.pins = {n: SHA(p) for n, p in prev.items() if n != 'receipts'}
    e.pins['receipts'] = SHA(prev['receipts']); e.pins['scope_digest'] = OLD_DIGEST; e.pins['swap_baseline_gib'] = REAL[0]['swap_used_gib']
    e.prev = prev; e.work3 = e.repo / 'work' / v3.WORK_ROOT_NAME; e.old_work = e.work
    e.proofs, e.hist, e.plan = proofs, hist, plan
    return e


def prep3(e, **over):
    e.n[0] += 1
    kw = dict(plan_path=e.args['plan_path'], expected_plan_sha256=e.plan['plan_sha256'], cert_path=e.args['expected_cache_certificate'],
              cert_sha256=e.args['expected_cache_certificate_sha256'], proof_path=e.proof_path, proof_sha256=e.proof_sha,
              quant_map_path=e.args['quant_map_path'], source_dir=e.args['source_dir'], scope_output=e.repo / v3.SCOPE_DIR_REL / ('scope3_%d.json' % e.n[0]),
              repo_root=e.repo, expected=e.expected, installed=installed_fn, previous_pins=e.pins)
    return v3.prepare_scope_v3(**{**kw, **over})


def adopt3(e):
    return v3.adopt_continuation(plan=e.plan, proofs=e.proofs, repo_root=e.repo, source_dir=e.args['source_dir'], previous_pins=e.pins, expected=e.expected)


def human3(digest, n, **over):
    t = now_utc()
    d = {'authorized': True, 'action': v3.ACTION, 'scope_sha256': digest, 'human_instruction_sha256': hashlib.sha256(('v3-instruction-%d' % n).encode()).hexdigest(),
         'approved_by': 'researcher', 'issued_at_utc': (t - timedelta(minutes=1)).isoformat(), 'expires_at_utc': (t + timedelta(hours=2)).isoformat()}
    d.update(over); return d


def adm3(e, info=None, human_over=None, review_over=None, review=True, human=True):
    info = info or prep3(e); scope = json.loads(Path(info['scope_path']).read_text()); n = e.n[0]
    out = dict(scope_path=info['scope_path'], scope_sha256=info['scope_file_sha256'], human_path=None, human_sha256=None, review_path=None, review_sha256=None)
    if human:
        out['human_path'], out['human_sha256'] = write2(e, 'h3_%d.json' % n, human3(info['scope_sha256'], n, **(human_over or {})))
    if review:
        doc = {'verdict': 'PASS_WITH_LIMITATIONS', 'scope_sha256': info['scope_sha256'], 'reviewer_id': 'codex-test', 'reviewer_kind': 'isolated_codex',
               'source_sha256': scope['source_sha256'], 'input_sha256': scope['input_sha256'], **(review_over or {})}
        out['review_path'], out['review_sha256'] = write2(e, 'r3_%d.json' % n, doc)
    return out


def run3(e, mx=None, adm=None, factory_kw=None, sampler_fn=None, level=1, clock=None, **kw):
    mx = mx or GpuMx(); adm = adm or adm3(e); clock = clock or Clock()
    def factory(proofs, **fk):
        return v2.GpuConvertBackend(proofs, mx=mx, np=np, nn=FakeNN(mx), **{**fk, **(factory_kw or {})})
    base = dict(**adm, repo_root=e.repo, expected=e.expected, installed=installed_fn, backend_factory=factory, sampler=sampler_fn or sampler(), os_level=lambda: level,
                rss_gib=lambda: 0.5, clock=clock, sleep=clock.sleep, previous_pins=e.pins)
    return v3.execute(**{**base, **kw}), mx, clock


def consumed3(e):
    d = e.repo / v3.CONSUMED_REL
    return sorted(p.name for p in d.iterdir()) if d.exists() else []


# ---- previous attempt binding ---------------------------------------------------------------------------------------
def test_previous_attempt_evidence_is_authenticated_and_counts_are_bound(tmp_path):
    e = env3(tmp_path)
    prev = v3.verify_previous_attempt(e.repo, e.pins, e.expected)
    assert prev['counts'] == {'gpu_unit_attempts': 1, 'gpu_direct_quantize_ops': 1, 'gpu_module_path_quantizations': 1, 'gpu_underlying_quantizations': 2}
    assert prev['scope_digest'] == OLD_DIGEST and prev['instruction_sha256'] == OLD_INSTRUCTION and prev['retained_receipts'] == 1 and prev['quantized_receipts'] == 0
    assert set(prev['files']) >= {'scope', 'human', 'review', 'failure_receipt', 'ledger', 'resource_log', 'rss_log', 'launcher_record', 'consumed_scope'}


@pytest.mark.parametrize('name', ['failure_receipt', 'ledger', 'resource_log', 'rss_log', 'launcher_record', 'scope', 'human', 'review', 'consumed_scope',
                                  'consumed_instruction', 'consumed_outcome', 'receipts', 'summary', 'stderr', 'independent_detail'])
def test_any_changed_old_evidence_file_is_refused(tmp_path, name):
    e = env3(tmp_path)
    p = e.prev[name]; p.write_bytes(p.read_bytes() + b' ')
    with pytest.raises(ValueError, match='pin|differs|hash'): v3.verify_previous_attempt(e.repo, e.pins, e.expected)


def test_old_failure_with_other_counts_or_a_missing_consumed_marker_is_refused(tmp_path):
    e = env3(tmp_path)
    f = json.loads(e.prev['failure_receipt'].read_text()); f['gpu_underlying_quantizations'] = 3; e.prev['failure_receipt'].write_text(json.dumps(f))
    pins = {**e.pins, 'failure_receipt': SHA(e.prev['failure_receipt'])}
    with pytest.raises(ValueError, match='count'): v3.verify_previous_attempt(e.repo, pins, e.expected)
    f['gpu_underlying_quantizations'] = 2; f['rerun_allowed'] = True; e.prev['failure_receipt'].write_text(json.dumps(f))
    with pytest.raises(ValueError): v3.verify_previous_attempt(e.repo, {**e.pins, 'failure_receipt': SHA(e.prev['failure_receipt'])}, e.expected)
    e2 = env3(tmp_path / 'b') if (tmp_path / 'b').mkdir() is None else None
    e2.prev['consumed_outcome'].unlink()
    with pytest.raises(ValueError, match='marker|consumed|missing|differs|pin'): v3.verify_previous_attempt(e2.repo, e2.pins, e2.expected)


def test_production_pins_are_the_independently_recorded_hashes():
    P = v3.PRODUCTION_PINS
    assert P['scope'] == '1f2bd2db5a7b6161334979fdea2a4d0d225a2b5a4b8fbe51506f820b49509b7e' and P['scope_digest'] == '3edf070143c8a7e42eca71a8a5817005d359fb0f4ea591322e397f4cc1a41dc7'
    assert P['human'] == '2eeb5c9c0035b40f56b916cc357b82e8947df558060486bbfc1221751684772d' and P['review'] == 'ab88f04b767609ba8eac75fb94dd5520bc689fa127989f8a53186b17514b0562'
    assert P['failure_receipt'] == '629a60e1bb8d06848288031b7df33a4952f75272c8a1e30a82def3deec9b4394' and P['unit0_file'] == '6cc6fa0558040faa6efa1b95535d58595246fc53843618b796cfc7ebb2912258'
    assert all(len(v) == 64 for k, v in P.items() if k != 'swap_baseline_gib') and P['swap_baseline_gib'] == 13.96966796875


def test_real_previous_attempt_evidence_matches_the_production_pins_when_present():
    for p in Path(__file__).resolve().parents:
        if (p / 'work/accuracy-recovery-historical-gpu-cache-v2/v2_failure_001.json').exists() and (p / 'local_data/accuracy_recovery_v1').is_dir():
            prev = v3.verify_previous_attempt(p, v3.PRODUCTION_PINS, cc.Expected())
            assert prev['counts']['gpu_underlying_quantizations'] == 2 and prev['retained_receipts'] == 406 and prev['quantized_receipts'] == 0
            return
    pytest.skip('real prior attempt evidence not present')


# ---- saved unit 0: independent CPU authentication and explicit adoption ------------------------------------------------------------------
def test_saved_unit_is_authenticated_against_the_proof_source_header_shapes_and_dtypes(tmp_path):
    e = env3(tmp_path); u0 = e.plan['units'][0]
    ev = v3.verify_saved_unit(e.plan, e.proofs, u0, e.old_work / 'units/unit_0000.safetensors', e.args['source_dir'])
    assert ev['file_sha256'] == e.proofs[0]['output_file_sha256'] and ev['payload_sha256'] == e.proofs[0]['output_tensor_payload_sha256']
    assert ev['source_bytes_sha256'] == e.proofs[0]['source_bytes_sha256'] and ev['header_order'] == ['biases', 'scales', 'weight']
    assert ev['dtypes'] == {'biases': 'BF16', 'scales': 'BF16', 'weight': 'U32'} and ev['shapes']['weight'] == [4, 8] and ev['unit_hash'] == qc.unit_hash(e.plan, u0)
    assert not any(m == 'mlx' or m.startswith('mlx.') for m in sys.modules)


def test_saved_unit_tamper_cases_are_refused(tmp_path):
    e = env3(tmp_path); u0 = e.plan['units'][0]; f = e.old_work / 'units/unit_0000.safetensors'; good = f.read_bytes()
    check = lambda: v3.verify_saved_unit(e.plan, e.proofs, u0, f, e.args['source_dir'])
    f.write_bytes(good[:-1] + bytes([good[-1] ^ 1]))
    with pytest.raises(ValueError, match='proof|payload|sha'): check()
    n = int.from_bytes(good[:8], 'little'); h = good[8:8 + n]
    for old, new in ((b'U32', b'I32'), (b'"biases"', b'"bia_es"')):
        f.write_bytes(good[:8] + h.replace(old, new, 1) + good[8 + n:])
        with pytest.raises(ValueError): check()
    f.write_bytes(good)
    assert check()
    (e.args['source_dir'] / 'a.safetensors').write_bytes(b'q' * len((e.args['source_dir'] / 'a.safetensors').read_bytes()))
    with pytest.raises(ValueError, match='source'): check()


def test_a_cpu_quantized_or_otherwise_different_unit_zero_is_not_adoptable(tmp_path):
    e = env3(tmp_path); u0 = e.plan['units'][0]
    mx = GpuMx(alter=1); b = cpu_like = v2.GpuConvertBackend({0: {'output_tensor_payload_sha256': {}}}, mx=mx, np=np, nn=FakeNN(mx), rss_gib=lambda: 0.5)
    other = e.tmp / 'other_unit0.safetensors'
    mx.save_safetensors(other, dict(zip(('weight', 'scales', 'biases'), mx.quantize(mx.array(np.zeros((4, 64), np.uint16)), bits=4, group_size=64, mode='affine'))))
    with pytest.raises(ValueError, match='proof|payload'): v3.verify_saved_unit(e.plan, e.proofs, u0, other, e.args['source_dir'])


def test_adopt_continuation_copies_retained_and_the_exact_saved_unit_with_an_explicit_record_and_no_fabricated_receipt(tmp_path):
    e = env3(tmp_path); before = snapshot(e.old_work); cpu_before = snapshot(e.cpu_root)
    man = adopt3(e)
    assert snapshot(e.old_work) == before and snapshot(e.cpu_root) == cpu_before                               # old roots only read
    names = sorted(p.name for p in (e.work3 / 'units').iterdir())
    assert names == ['plan_binding.json', 'receipts.jsonl', 'unit_0000.safetensors', 'unit_0002.safetensors']
    assert [json.loads(l)['index'] for l in (e.work3 / 'units/receipts.jsonl').read_text().splitlines()] == [2]       # NO receipt for the adopted unit
    rec = json.loads((e.work3 / 'adoption/unit0_adoption_PRIVATE.json').read_text())
    assert rec['old_receipt_fabricated'] is False and rec['receipt_equivalent'] is False and rec['unit_index'] == 0
    assert {'module_path_bitwise_equal_result', 'mlx_peak_mib', 'mlx_active_after_release_mib', 'reload_bitwise_equal_result'} <= set(rec['evidence_missing'])
    assert rec['old_attempt_opcounts'] == {'gpu_unit_attempts': 1, 'gpu_direct_quantize_ops': 1, 'gpu_module_path_quantizations': 1, 'gpu_underlying_quantizations': 2}
    assert rec['file_sha256'] == e.proofs[0]['output_file_sha256'] and man['adopted_retained_units'] == 1 and man['adopted_saved_units'] == 1
    assert 'no_new_quantization_of_unit0' in rec['adoption_policy'] and 'derived' in rec['evidence_present_notes']['reload_equality']
    assert not any(m == 'mlx' or m.startswith('mlx.') for m in sys.modules)
    with pytest.raises(FileExistsError): adopt3(e)


def test_adopt_continuation_refuses_when_unit_zero_or_a_retained_unit_is_not_exact(tmp_path):
    e = env3(tmp_path); f = e.old_work / 'units/unit_0000.safetensors'; good = f.read_bytes()
    f.write_bytes(good[:-1] + bytes([good[-1] ^ 1]))
    with pytest.raises(ValueError): adopt3(e)
    assert not e.work3.exists()
    f.write_bytes(good); r = e.old_work / 'units/unit_0002.safetensors'; g = r.read_bytes(); r.write_bytes(g[:-1] + bytes([g[-1] ^ 1]))
    with pytest.raises(ValueError): adopt3(e)
    assert not e.work3.exists()


# ---- scope ---------------------------------------------------------------------------------------------------------------------
def test_scope_documents_policy_budgets_cumulative_accounting_adoption_and_previous_attempt(tmp_path):
    L = v3.SCOPE
    assert L['previous_attempt_underlying_quantizations'] == 2 and L['new_gpu_unit_attempts_max'] == 343 and L['new_direct_quantize_ops_max'] == 343
    assert L['new_module_path_quantizations_max'] == 343 and L['new_underlying_quantizations_max'] == 686 and L['cumulative_underlying_quantizations_max'] == 688
    assert 2 + L['new_underlying_quantizations_max'] <= L['cumulative_underlying_quantizations_max'] and L['retained_units_gpu_operations'] == 0
    assert L['adopted_saved_units'] == 1 and L['retained_units_adopted'] == 406 and L['plan_units'] == 750 and L['one_shot_no_rerun_no_resume_no_new_root_retry']
    assert L['resource_policy'] == v3.POLICY and v3.POLICY['continue_available_gib_min'] == 5.5 and v3.POLICY['continue_pressure_free_percent_min'] == 30.0
    assert v3.POLICY['hold_seconds_per_boundary_max'] == 120.0 and v3.POLICY['sample_interval_seconds'] == 5.0 and v3.POLICY['operation_start_window_seconds'] == 1800.0
    assert v3.POLICY['sustained_low_available_gib'] == 4.0 and v3.POLICY['sustained_low_seconds'] == 30.0 and v3.POLICY['stop_pressure_free_percent_max'] == 15.0
    assert v3.POLICY['rss_envelope_gib'] == 8.0 and v3.POLICY['disk_reserve_gib'] == 20.0 and v3.POLICY['mlx_memory_limit_bytes'] == 8 * 1024**3
    assert v3.POLICY['mlx_cache_limit_bytes'] == 64 * 1024**2 and v3.POLICY['warning_only_inherited_supervisor_reasons_terminate'] is False
    assert 'SENSOR_UNAVAILABLE' in v3.POLICY['thermal_limitation']


def test_prepare_scope_v3_binds_roots_files_previous_attempt_adoption_policy_closure_and_installed(tmp_path):
    e = env3(tmp_path); info = prep3(e); scope = json.loads(Path(info['scope_path']).read_text())
    assert scope['limits'] == v3.SCOPE
    assert scope['static_scope_sha256'] == v3.SCOPE_SHA256
    assert scope['action'] == v3.ACTION
    assert scope['installed'] == installed_fn()
    assert scope['previous_attempt']['swap_baseline_gib'] == 13.96966796875 and scope['limits']['resource_policy']['swap_used_absolute_cap_gib'] == 16.0
    assert scope['roots'] == v3.canonical_roots(e.repo, e.args['source_dir']) and scope['roots']['work_root'].endswith(v3.WORK_ROOT_NAME)
    assert scope['roots']['previous_work_root'].endswith(v2.WORK_ROOT_NAME) and scope['roots']['consumed_dir'].endswith('historical_gpu_cache_v3_consumed')
    assert {'plan', 'quant_map', 'historical_certificate', 'historical_proof'} <= set(scope['files']) and 'prev_failure_receipt' in scope['files'] and 'prev_scope' in scope['files']
    assert scope['previous_attempt']['counts']['gpu_underlying_quantizations'] == 2 and scope['previous_attempt']['scope_digest'] == OLD_DIGEST
    assert scope['adoption']['unit_index'] == 0 and scope['adoption']['file_sha256'] == e.proofs[0]['output_file_sha256'] and scope['adoption']['old_receipt_fabricated'] is False
    assert scope['counts'] == {'plan_units': 3, 'retained_units_adopted': 1, 'adopted_saved_units': 1, 'quantized_pending_gpu': 1}
    assert {'accuracy_recovery_gpu_cache_v3.py', 'accuracy_recovery_gpu_cache_v2.py', 'accuracy_recovery_cpu_cache_v1.py', 'quant_convert.py', 'quant_cache.py',
            'resource_supervisor.py', 'paced_resources.py', 'rebuild_accuracy_recovery_historical_gpu_cache_v3.py'} <= {Path(p).name for p in scope['source_sha256']}
    assert info['scope_sha256'] == v3.shared.scope_digest(scope) and info['scope_file_sha256'] == SHA(info['scope_path']) and not any(m == 'mlx' or m.startswith('mlx.') for m in sys.modules)
    with pytest.raises(v2.ScopeRefused): prep3(e, scope_output=Path(info['scope_path']))


def test_prepare_scope_v3_refuses_noncanonical_output_changed_source_old_evidence_and_cpu_quantized_unit(tmp_path):
    e = env3(tmp_path)
    with pytest.raises(v2.ScopeRefused): prep3(e, scope_output=e.tmp / 'elsewhere.json')
    with pytest.raises(v2.ScopeRefused): prep3(e, installed=lambda: {'mlx_version': None, 'mlx_tree_sha256': None})
    e.prev['failure_receipt'].write_bytes(e.prev['failure_receipt'].read_bytes() + b' ')
    with pytest.raises(ValueError): prep3(e)
    assert list((e.repo / v3.SCOPE_DIR_REL).glob('scope3_*')) == []


# ---- admission ----------------------------------------------------------------------------------------------------------------------------------
def admit3(e, a, **kw):
    return v3.admit_v3(**a, now=kw.pop('now', now_utc()), repo_root=e.repo, expected=e.expected, installed=installed_fn, previous_pins=e.pins, **kw)


def test_full_v3_admission_set_is_accepted_and_consumes_nothing_by_itself(tmp_path):
    e = env3(tmp_path)
    assert admit3(e, adm3(e)).check('x', full=True) and consumed3(e) == []


def test_v3_admission_needs_scope_human_and_authentic_independent_review(tmp_path):
    e = env3(tmp_path); a = adm3(e)
    for drop in ('review_path', 'review_sha256', 'human_path', 'human_sha256', 'scope_path', 'scope_sha256'):
        with pytest.raises(v2.ScopeRefused): admit3(e, {**a, drop: None})
    with pytest.raises(v2.ScopeRefused): admit3(e, {**a, 'review_path': e.tmp / 'missing.json', 'review_sha256': 'a' * 64})
    for over in ({'verdict': 'FAIL'}, {'reviewer_kind': 'claude_code'}, {'scope_sha256': '0' * 64}, {'source_sha256': {}}, {'input_sha256': '0' * 64}, {'reviewer_id': ''}):
        with pytest.raises(v2.ScopeRefused): admit3(e, adm3(e, review_over=over))
    for over in ({'human_instruction_sha256': None}, {'action': v2.ACTION}, {'authorized': False}, {'scope_sha256': '0' * 64}, {'approved_by': ''},
                 {'issued_at_utc': '2020-01-01T00:00:00+00:00', 'expires_at_utc': '2020-01-02T00:00:00+00:00'}):
        with pytest.raises(v2.ScopeRefused): admit3(e, adm3(e, human_over=over))


def test_old_v2_scope_or_instruction_or_any_scope_without_the_previous_attempt_cannot_authorize_v3(tmp_path):
    e = env3(tmp_path); info = prep3(e); scope = json.loads(Path(info['scope_path']).read_text())
    for mutate in (lambda s: s.pop('previous_attempt'), lambda s: s.pop('adoption'), lambda s: s.update(schema=v2.SCOPE_SCHEMA), lambda s: s.update(action=v2.ACTION),
                   lambda s: s['limits'].update(cumulative_underlying_quantizations_max=1000), lambda s: s['limits']['resource_policy'].update(continue_available_gib_min=4.0),
                   lambda s: s['limits']['resource_policy'].update(hold_seconds_per_boundary_max=600.0), lambda s: s['limits']['resource_policy'].update(swap_cumulative_net_growth_cap_gib=5.0),
                   lambda s: s['roots'].update(work_root=str(e.tmp / 'work' / v3.WORK_ROOT_NAME)), lambda s: s['roots'].update(consumed_dir=str(e.tmp / 'c')),
                   lambda s: s['adoption'].update(old_receipt_fabricated=True), lambda s: s['adoption'].update(receipt_equivalent=True),
                   lambda s: s['previous_attempt']['counts'].update(gpu_underlying_quantizations=0), lambda s: s['previous_attempt'].update(scope_digest='1' * 64),
                   lambda s: s['files'].pop('prev_consumed_instruction')):
        s2 = json.loads(json.dumps(scope)); mutate(s2)
        s2['input_sha256'] = v3.shared.identity({k: s2.get(k) for k in ('roots', 'files', 'source_shards', 'inventory_sha256', 'plan_sha256', 'expected', 'installed', 'previous_attempt', 'adoption')})
        p, sha = write2(e, 'mut3_%d.json' % e.n[0], s2); e.n[0] += 1                      # input digest recomputed: each mutation must be caught by its OWN check
        a = adm3(e, {**info, 'scope_path': str(p), 'scope_file_sha256': sha, 'scope_sha256': v3.shared.scope_digest(s2)})
        with pytest.raises(v2.ScopeRefused): admit3(e, a)
    old_instr = adm3(e, human_over={'human_instruction_sha256': OLD_INSTRUCTION})
    with pytest.raises(v2.ScopeRefused, match='consumed'): admit3(e, old_instr).consume()


def test_changed_old_evidence_input_or_closure_after_prepare_is_refused_at_admission(tmp_path, monkeypatch):
    e = env3(tmp_path); a = adm3(e)
    real = v3.shared.file_sha256
    resource = next(p for p in json.loads(Path(a['scope_path']).read_text())['source_sha256'] if p.endswith('resource_supervisor.py'))
    monkeypatch.setattr(v3.shared, 'file_sha256', lambda p: '0' * 64 if str(p) == resource else real(p))
    with pytest.raises(v2.ScopeRefused, match='Reviewed source changed'): admit3(e, a)
    monkeypatch.setattr(v3.shared, 'file_sha256', real)
    e.prev['rss_log'].write_bytes(b'tampered')
    with pytest.raises(v2.ScopeRefused, match='Input file changed'): admit3(e, a)


# ---- local converter seam, differential against the originals ------------------------------------------------------------------------
class RecController:
    def __init__(self, inner=None): self.calls, self.inner = [], inner
    def boundary(self, label, need_gib=0.0, unit=None, enforce_window=True):
        self.calls.append((label, need_gib, unit)); return healthy(available_gib=9.0, pressure_free_percent=40, power_source='AC')
    def observe(self, label, unit=None, need_gib=0.0):
        self.calls.append((label, unit)); return healthy(available_gib=9.0, pressure_free_percent=40, power_source='AC'), 'CONTINUE', []


def fresh_units(tmp_path, name):
    e = env2(tmp_path / name) if (tmp_path / name).mkdir() is None else None
    plan, hist, proofs = loaded2(e); return e, plan, proofs


def seed_retained(e, out_dir):
    """Put the authenticated retained unit (+ receipt line) where the real flow has it before any GPU work: the GPU backend never handles retained units."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    src = e.cpu_root / 'units'
    (out_dir / 'unit_0002.safetensors').write_bytes((src / 'unit_0002.safetensors').read_bytes())
    (out_dir / 'receipts.jsonl').write_text(''.join(l + '\n' for l in (src / 'receipts.jsonl').read_text().splitlines() if json.loads(l)['index'] == 2))


def strip(r): return {k: v for k, v in r.items() if k not in ('available_gib', 'pressure_free_percent', 'swap_used_gib', 'mlx_peak_mib', 'mlx_active_after_release_mib', 'hold_seconds_before_unit')}


def test_convert_v3_produces_the_same_receipts_as_qc_convert_when_nothing_is_adopted(tmp_path):
    e1, plan, proofs = fresh_units(tmp_path, 'a'); e2, _, _ = fresh_units(tmp_path, 'b')
    mx1, mx2 = GpuMx(), GpuMx()
    b1 = v2.GpuConvertBackend(proofs, mx=mx1, np=np, nn=FakeNN(mx1), rss_gib=lambda: 0.5); b2 = v2.GpuConvertBackend(proofs, mx=mx2, np=np, nn=FakeNN(mx2), rss_gib=lambda: 0.5)
    seed_retained(e1, e1.tmp / 'o1'); seed_retained(e2, e2.tmp / 'o2')
    qc.convert(plan, e1.args['source_dir'], e1.tmp / 'o1', b1, sampler())
    (e2.tmp / 'o2/plan_binding.json').write_text(json.dumps({'plan_sha256': plan['plan_sha256']}))
    res = v3.convert_v3(plan, e2.args['source_dir'], e2.tmp / 'o2', b2, RecController(), adopted=frozenset())
    r1 = [strip(json.loads(l)) for l in (e1.tmp / 'o1/receipts.jsonl').read_text().splitlines()]
    r2 = [strip(json.loads(l)) for l in (e2.tmp / 'o2/receipts.jsonl').read_text().splitlines()]
    assert r1 == r2 and res['complete'] and [n for n, _ in mx1.ops] == [n for n, _ in mx2.ops]
    assert (e1.tmp / 'o1/unit_0000.safetensors').read_bytes() == (e2.tmp / 'o2/unit_0000.safetensors').read_bytes()


def test_convert_v3_never_touches_the_adopted_unit_and_never_repeats_completed_units(tmp_path):
    e = env3(tmp_path); adopt3(e); mx = GpuMx(); rc = RecController(); u0_before = SHA(e.work3 / 'units/unit_0000.safetensors')
    b = v2.GpuConvertBackend(e.proofs, mx=mx, np=np, nn=FakeNN(mx), rss_gib=lambda: 0.5, max_attempts=1, max_quantize=1, max_module=1, max_underlying=2)
    res = v3.convert_v3(e.plan, e.args['source_dir'], e.work3 / 'units', b, rc, adopted=frozenset({0}))
    assert res == {'units_total': 3, 'units_done': 3, 'converted_this_call': 1, 'complete': True} and (b.attempts, b.quantize_ops, b.module_path_checks) == (1, 1, 1)
    assert SHA(e.work3 / 'units/unit_0000.safetensors') == u0_before and [c for c in rc.calls if c[0] == 'unit_start'] == [('unit_start', pytest.approx(qc.forecast_gib(e.plan['units'][1]) + qc.RESERVE_GIB), 1)]
    mx2 = GpuMx(); b2 = v2.GpuConvertBackend(e.proofs, mx=mx2, np=np, nn=FakeNN(mx2), rss_gib=lambda: 0.5)
    res2 = v3.convert_v3(e.plan, e.args['source_dir'], e.work3 / 'units', b2, rc, adopted=frozenset({0}))
    assert res2['converted_this_call'] == 0 and mx2.created == [] and b2.attempts == 0                         # nothing repeated


def test_convert_v3_passes_the_unit_forecast_plus_reserve_to_the_hold_boundary(tmp_path):
    e = env3(tmp_path); adopt3(e); rc = RecController(); mx = GpuMx()
    v3.convert_v3(e.plan, e.args['source_dir'], e.work3 / 'units', v2.GpuConvertBackend(e.proofs, mx=mx, np=np, nn=FakeNN(mx), rss_gib=lambda: 0.5), rc, adopted=frozenset({0}))
    assert rc.calls[0][1] == qc.forecast_gib(e.plan['units'][1]) + qc.RESERVE_GIB


def test_receipt_is_written_before_a_stop_decided_at_the_unit_end_boundary(tmp_path):
    e = env3(tmp_path); adopt3(e); mx = GpuMx()
    class EndStop(RecController):
        def observe(self, label, unit=None, need_gib=0.0):
            return healthy(available_gib=1.0, pressure_free_percent=10), 'STOP', ['pressure_free_at_or_below_15']
    with pytest.raises(cc.ResourceStop):
        v3.convert_v3(e.plan, e.args['source_dir'], e.work3 / 'units', v2.GpuConvertBackend(e.proofs, mx=mx, np=np, nn=FakeNN(mx), rss_gib=lambda: 0.5), EndStop(), adopted=frozenset({0}))
    rec = [json.loads(l) for l in (e.work3 / 'units/receipts.jsonl').read_text().splitlines()]
    assert sorted(r['index'] for r in rec) == [1, 2] and rec[-1]['passed'] is True                          # the saved unit kept its receipt (v2 lost it)


def test_no_gpu_operation_while_the_start_boundary_holds(tmp_path):
    e = env3(tmp_path); adopt3(e); mx = GpuMx(); clock = Clock()
    ops_at_hold = []
    seqs = [S(available_gib=5.0)] * 3 + [S()]
    it = iter(seqs)
    def smp():
        s = next(it, seqs[-1]); ops_at_hold.append(len(mx.ops)); return s
    c = v3.ResourceController(sampler=smp, os_level=lambda: 1, clock=clock, sleep=clock.sleep, utc=lambda: 'u', log_path=tmp_path / 'r.jsonl')
    v3.convert_v3(e.plan, e.args['source_dir'], e.work3 / 'units', v2.GpuConvertBackend(e.proofs, mx=mx, np=np, nn=FakeNN(mx), rss_gib=lambda: 0.5), c, adopted=frozenset({0}))
    assert ops_at_hold[:4] == [0, 0, 0, 0] and ops_at_hold[4] > 0 and clock.sleeps[:3] == [5.0] * 3                # ops only after CONTINUE


def test_first_historical_payload_mismatch_still_stops_before_any_save(tmp_path):
    e = env3(tmp_path); adopt3(e); mx = GpuMx(alter=1)
    with pytest.raises(v2.HistoricalPayloadMismatch):
        v3.convert_v3(e.plan, e.args['source_dir'], e.work3 / 'units', v2.GpuConvertBackend(e.proofs, mx=mx, np=np, nn=FakeNN(mx), rss_gib=lambda: 0.5), RecController(), adopted=frozenset({0}))
    assert not (e.work3 / 'units/unit_0001.safetensors').exists() and sum(n == 'save' for n, _ in mx.ops) == 0


def test_assemble_and_certify_copies_equal_the_originals_byte_for_byte(tmp_path):
    e, plan, proofs = fresh_units(tmp_path, 'd'); mx = GpuMx(); qmap = json.loads(e.args['quant_map_path'].read_text())
    seed_retained(e, e.tmp / 'u')
    qc.convert(plan, e.args['source_dir'], e.tmp / 'u', v2.GpuConvertBackend(proofs, mx=mx, np=np, nn=FakeNN(mx), rss_gib=lambda: 0.5), sampler())
    done = qc.load_done(plan, e.tmp / 'u')
    qc.assemble_cache(plan, e.tmp / 'u', e.tmp / 'c1', shard_limit_bytes=140); v3.assemble_v3(plan, e.tmp / 'u', e.tmp / 'c2', done, shard_limit_bytes=140)
    assert snapshot(e.tmp / 'c1') == snapshot(e.tmp / 'c2') and len(list((e.tmp / 'c1').glob('model-*.safetensors'))) == 3
    for c in ('c1', 'c2'): quant_cache.finalize_cache(plan, e.tmp / 'u', e.tmp / c, e.args['source_dir'], qmap)
    assert v3.certify_v3(plan, e.tmp / 'u', e.tmp / 'c2', e.args['source_dir'], qmap, done) == quant_cache.certify_cache(plan, e.tmp / 'u', e.tmp / 'c1', e.args['source_dir'], qmap)
    with pytest.raises(ValueError, match='incomplete'): v3.assemble_v3(plan, e.tmp / 'u', e.tmp / 'c3', {k: v for k, v in done.items() if k != 0})


# ---- execute ---------------------------------------------------------------------------------------------------------------------------------
def test_execute_v3_completes_with_the_adopted_unit_cumulative_accounting_and_the_historical_cache(tmp_path):
    e = env3(tmp_path); adopt3(e); cpu_before = snapshot(e.cpu_root); old_before = snapshot(e.old_work); oldc_before = snapshot(e.repo / v2.CONSUMED_REL)
    res, mx, clock = run3(e)
    g = res['certificate']['gpu_reconstruction']
    assert (g['new_gpu_unit_attempts'], g['new_direct_quantize_ops'], g['new_module_path_quantizations'], g['new_underlying_quantizations']) == (1, 1, 1, 2)
    assert g['previous_attempt_underlying_quantizations'] == 2 and g['cumulative_underlying_quantizations'] == 4 <= g['cumulative_underlying_quantizations_max'] == 688
    assert g['adopted_saved_units'] == 1 and g['retained_units_adopted'] == 1 and g['retained_units_gpu_operations'] == 0 and g['model_loaded'] is False
    assert g['adopted_unit_basis'] == 'v3 explicit adoption record; no old receipt fabricated' and g['every_unit_payload_equals_pinned_historical_proof'] is True
    assert {d for _, d in mx.ops} == {'gpu'} and sum(n == 'quantize' for n, _ in mx.ops) == 2 and mx.ops[-1][0] == 'clear'
    assert json.loads((e.work3 / 'certificate.json').read_text())['shards'] == json.loads(e.args['expected_cache_certificate'].read_text())['shards']
    assert snapshot(e.cpu_root) == cpu_before and snapshot(e.old_work) == old_before and snapshot(e.repo / v2.CONSUMED_REL) == oldc_before
    names = consumed3(e)
    assert sum(n.startswith('scope_consumed_') for n in names) == sum(n.startswith('authorization_consumed_') for n in names) == sum(n.startswith('scope_outcome_') for n in names) == 1
    assert (e.work3 / 'v3_ledger.json').exists() and not (e.work3 / 'v3_active.lock').exists() and not list(e.work3.glob('v3_failure_*')) and (e.work3 / 'v3_resource.jsonl').exists()


def test_execute_v3_holds_through_a_transient_shortage_then_runs_with_no_ops_during_the_hold(tmp_path):
    e = env3(tmp_path); adopt3(e); n = []
    def smp():
        n.append(1); return S(available_gib=5.0 if 3 <= len(n) <= 6 else 9.0)
    res, mx, clock = run3(e, sampler_fn=smp)
    rows = [json.loads(l) for l in (e.work3 / 'v3_resource.jsonl').read_text().splitlines()]
    assert any(r['decision'] == 'HOLD' for r in rows) and res['certificate']['gpu_reconstruction']['new_gpu_unit_attempts'] == 1 and sum(clock.sleeps) >= 15


def test_execute_v3_critical_stop_preserves_failure_consumes_and_cannot_be_retried(tmp_path):
    e = env3(tmp_path); adopt3(e); adm = adm3(e); n = []
    def smp():
        n.append(1); return S(pressure_free_percent=10 if len(n) >= 3 else 40)
    with pytest.raises(cc.ResourceStop):
        run3(e, adm=adm, sampler_fn=smp)
    fail = json.loads((e.work3 / 'v3_failure_001.json').read_text())
    assert fail['error_type'] == 'ResourceStop' and 'pressure_free_at_or_below_15' in fail['reasons'] and fail['rerun_allowed'] is False and fail['automatic_retry'] is False
    assert (e.work3 / 'units/unit_0000.safetensors').exists() and not (e.work3 / 'certificate.json').exists() and not (e.work3 / 'v3_active.lock').exists()
    mx2 = GpuMx()
    with pytest.raises(v2.ScopeRefused): run3(e, mx=mx2, adm=adm)
    assert mx2.created == [] and any(n.startswith('scope_outcome_') for n in consumed3(e))
    e.work3.rename(e.tmp / 'failed_v3_preserved'); adopt3(e)                                                    # a new root does not reset the consumed scope
    with pytest.raises(v2.ScopeRefused, match='consumed'): run3(e, mx=GpuMx(), adm=adm)


def test_execute_v3_first_payload_mismatch_stops_with_counts_and_no_second_attempt(tmp_path):
    e = env3(tmp_path); adopt3(e)
    with pytest.raises(v2.HistoricalPayloadMismatch): run3(e, mx=GpuMx(alter=1))
    fail = json.loads((e.work3 / 'v3_failure_001.json').read_text())
    assert fail['new_direct_quantize_ops'] == 1 and fail['new_underlying_quantizations'] == 1 and fail['error_type'] == 'HistoricalPayloadMismatch'
    assert not (e.work3 / 'units/unit_0001.safetensors').exists() and (e.work3 / 'units/unit_0000.safetensors').exists()


def test_execute_v3_swap_over_the_cumulative_cap_stops_mid_run(tmp_path):
    e = env3(tmp_path); adopt3(e); n = []
    def smp():
        n.append(1); return S(swap_used_gib=14.0 if len(n) < 4 else 15.6)
    with pytest.raises(cc.ResourceStop):
        run3(e, sampler_fn=smp)
    assert 'swap_net_growth_over_cap' in json.loads((e.work3 / 'v3_failure_001.json').read_text())['reasons']


def test_execute_v3_window_includes_holds_and_stops_partial(tmp_path):
    e = env3(tmp_path); adopt3(e); n = []; clock = Clock()
    def smp():
        n.append(1)
        if len(n) == 2: clock.t += 1795.0                                                                      # the unit-start boundary: the window is nearly spent when the hold begins
        return S(available_gib=5.0) if len(n) >= 2 else S()
    with pytest.raises(v2.WindowExceeded):
        run3(e, sampler_fn=smp, clock=clock)
    assert json.loads((e.work3 / 'v3_failure_001.json').read_text())['partial'] is True and not (e.work3 / 'certificate.json').exists()


def test_execute_v3_held_global_flock_refuses_before_mlx_without_consumption(tmp_path):
    e = env3(tmp_path); adopt3(e); mx = GpuMx(); made = []
    with Hold(e.lock):
        with pytest.raises(v2.ScopeRefused, match='held'): run3(e, mx=mx, backend_factory=lambda *a, **k: made.append(1))
    assert made == [] and mx.created == [] and consumed3(e) == [] and not (e.work3 / 'v3_ledger.json').exists()
    src = inspect.getsource(v3.execute)
    assert 'flock' in src and 'LOCK_EX' in src and 'LOCK_NB' in src and 'O_CREAT' not in src


def test_execute_v3_refusals_consume_nothing(tmp_path):
    e = env3(tmp_path); adopt3(e); mx = GpuMx()
    for kw in (dict(review_over={'verdict': 'FAIL'}), dict(human_over={'human_instruction_sha256': None}), dict(review=False)):
        with pytest.raises(v2.ScopeRefused): run3(e, mx=mx, adm=adm3(e, **kw))
    with pytest.raises(cc.ResourceStop): run3(e, mx=mx, sampler_fn=sampler(power_source='BATTERY'))              # preflight hold times out: nothing consumed
    assert mx.created == [] and consumed3(e) == [] and not (e.work3 / 'v3_ledger.json').exists()


def test_execute_v3_refuses_a_work_root_that_is_not_exactly_the_adopted_state(tmp_path):
    e = env3(tmp_path); mx = GpuMx(); e.work3.mkdir(parents=True)
    with pytest.raises(ValueError, match='adopt'): run3(e, mx=mx)
    shutil_e = env3(tmp_path / 'z') if (tmp_path / 'z').mkdir() is None else None
    adopt3(shutil_e); (shutil_e.work3 / 'units/unit_0001.safetensors').write_bytes(b'x')
    with pytest.raises(ValueError, match='exactly'): run3(shutil_e, mx=mx)
    (shutil_e.work3 / 'units/unit_0001.safetensors').unlink(); f = shutil_e.work3 / 'units/unit_0000.safetensors'; g = f.read_bytes(); f.write_bytes(g[:-1] + bytes([g[-1] ^ 1]))
    with pytest.raises(ValueError): run3(shutil_e, mx=mx)
    assert mx.created == [] and consumed3(shutil_e) == []


def test_original_modules_are_not_edited_or_mutated(tmp_path):
    r = subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', *GUARDED], cwd=REPO)
    assert r.returncode == 0
    before = {m.__name__: dict(vars(m)) for m in (cc, qc, quant_cache, paced_resources, resource_supervisor, v2)}
    e = env3(tmp_path); adopt3(e); run3(e)
    after = {m.__name__: dict(vars(m)) for m in (cc, qc, quant_cache, paced_resources, resource_supervisor, v2)}
    assert before == after and v2.SCOPE_SHA256 == qc.sha_bytes(qc.canonical(v2.SCOPE).encode()) and cc.SWAP_GROWTH_CAP_GIB == 1.0
    assert not any(m == 'mlx' or m.startswith('mlx.') for m in sys.modules)


def test_no_config_env_or_approval_flag_and_no_mlx_in_cpu_prepare(tmp_path):
    src = inspect.getsource(v3)
    assert 'environ' not in src and 'getenv' not in src
    with pytest.raises(SystemExit): v3.main(['execute', '--scope', 'x'], repo_root='/nonexistent')
    e = env3(tmp_path); a = e.args; code = (
        "import sys, json; sys.path.insert(0, %r); from mri_rl import accuracy_recovery_gpu_cache_v3 as m, accuracy_recovery_cpu_cache_v1 as c; "
        "x = c.Expected(units=3, quantized=2, retained=1, tensors=7, shards=1); r = %r; pins = %r; inst = lambda: %r; "
        "base = ['--plan', %r, '--expected-plan-sha256', %r, '--expected-cache-certificate', %r, '--expected-cache-certificate-sha256', %r, '--historical-proof', %r, '--historical-proof-sha256', %r, '--quant-map', %r, '--source-dir', %r]; "
        "m.main(['plan'] + base, expected=x, repo_root=r, previous_pins=pins); m.main(['adopt-continuation'] + base, expected=x, repo_root=r, previous_pins=pins); "
        "m.main(['prepare-scope'] + base + ['--scope-output', %r], expected=x, repo_root=r, previous_pins=pins, installed=inst); "
        "assert not any(k == 'mlx' or k.startswith('mlx.') for k in sys.modules), 'mlx imported'")
    out_scope = str(e.repo / v3.SCOPE_DIR_REL / 'cli_scope3.json')
    r = subprocess.run([sys.executable, '-I', '-c', code % (str(SRC_ROOT), str(e.repo), e.pins, installed_fn(), str(a['plan_path']), e.plan['plan_sha256'], str(a['expected_cache_certificate']),
                                                          a['expected_cache_certificate_sha256'], str(e.proof_path), e.proof_sha, str(a['quant_map_path']), str(a['source_dir']), out_scope)],
                       capture_output=True, text=True, timeout=240)
    assert r.returncode == 0, r.stderr
    lines = r.stdout.splitlines(); plan_out = json.loads(lines[0])
    assert plan_out['executed'] is False and plan_out['quantized_pending_gpu'] == 1 and plan_out['adopted_saved_units'] == 1 and plan_out['mlx_imported'] is False
    assert json.loads(lines[1])['adopted_saved_units'] == 1 and (e.work3 / 'units/unit_0000.safetensors').exists() and json.loads(lines[2])['scope_file_sha256'] == SHA(out_scope)


def test_matching_payloads_but_different_historical_certificate_is_not_published(tmp_path):
    e = env3(tmp_path); adopt3(e)
    h = json.loads(e.args['expected_cache_certificate'].read_text()); h['config_cached_sha256'] = '0' * 64
    e.args['expected_cache_certificate'].write_text(json.dumps(h)); e.args['expected_cache_certificate_sha256'] = SHA(e.args['expected_cache_certificate'])
    with pytest.raises(ValueError, match='historical'): run3(e)
    assert not (e.work3 / 'certificate.json').exists() and json.loads((e.work3 / 'v3_failure_001.json').read_text())['error_type'] == 'ValueError'


def test_cumulative_underlying_cap_is_enforced_before_consumption(tmp_path, monkeypatch):
    e = env3(tmp_path); adopt3(e); mx = GpuMx()
    monkeypatch.setattr(v3, 'MAX_CUMULATIVE_UNDERLYING', 3)                      # previous 2 + 2 new = 4 > 3
    with pytest.raises(v2.ScopeRefused, match='Cumulative'): run3(e, mx=mx)
    assert mx.created == [] and consumed3(e) == []


def test_production_cumulative_accounting_adds_up():
    L = v3.SCOPE
    assert L['previous_attempt_underlying_quantizations'] + L['new_underlying_quantizations_max'] == L['cumulative_underlying_quantizations_max'] == 688
    assert L['new_gpu_unit_attempts_max'] == L['quantized_units_pending_gpu'] == 343 and 1 + L['quantized_units_pending_gpu'] == 344
