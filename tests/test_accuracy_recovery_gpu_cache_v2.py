"""CPU-only tests for the gated GPU historical-cache v2 module. A numpy-backed fake mx/nn stands in for MLX: no MLX import,
no model. The 'historical' proof/certificate are produced by the real converter pipeline on tiny fixtures (alter=0); the failed
'CPU root' is the same pipeline with uniformly different quantized payloads (alter=1), like the real CPU-vs-GPU mismatch."""
import hashlib
import inspect
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from mri_rl import accuracy_recovery_cpu_cache_v1 as cc, accuracy_recovery_gpu_cache_v2 as v2, quant_convert as qc
from tests.test_accuracy_recovery_cpu_cache_v1 import FakeMx, FakeNN, backend as cpu_backend, fixture, sampler
from tests.test_quant_convert import healthy

SRC_ROOT = Path(__file__).parents[1] / 'src'
SCRIPT = Path(__file__).parents[1] / 'scripts' / 'rebuild_accuracy_recovery_historical_gpu_cache_v2.py'
SHA = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


class GpuMx(FakeMx):
    def __init__(self, alter=0, metal=True, on_clear=None, **kw):
        super().__init__(alter=alter, **kw)
        self.metal, self.limits, self.streams, self.on_clear = SimpleNamespace(is_available=lambda: metal), [], 0, on_clear
    def set_memory_limit(self, n): self.limits.append(('memory', n))
    def set_cache_limit(self, n): self.limits.append(('cache', n))
    def stream(self, d): self.streams += 1; return super().stream(d)
    def clear_cache(self):
        super().clear_cache()
        if self.on_clear: self.on_clear()


def snapshot(root):
    return {str(p.relative_to(root)): SHA(p) for p in sorted(Path(root).rglob('*')) if p.is_file()}


def env(tmp_path):
    plan, args = fixture(tmp_path)
    w = args['work_root']
    proof = []
    receipts = {json.loads(l)['index']: json.loads(l) for l in (w / 'h_units/receipts.jsonl').read_text().splitlines()}
    for u in plan['units']:
        fsha, parts = v2.hash_unit_file(w / 'h_units' / f"unit_{u['index']:04d}.safetensors")
        proof.append({'index': u['index'], 'output_file_sha256': fsha, 'output_tensor_payload_sha256': parts,
                      'source_bytes_sha256': receipts[u['index']]['source_bytes_sha256'], 'unit_hash': qc.unit_hash(plan, u)})
    (w / 'proof.json').write_text(json.dumps({'plan_sha256': plan['plan_sha256'], 'units': proof}))
    cpu_root = tmp_path / 'cpu' / v2.CPU_ROOT_NAME
    b, _, _ = cpu_backend(mx=FakeMx(alter=1))
    qc.convert(plan, args['source_dir'], cpu_root / 'units', b, sampler())          # the failed CPU run: retained exact, quantized different
    return SimpleNamespace(plan=plan, args=args, proof_path=w / 'proof.json', proof_sha=SHA(w / 'proof.json'), cpu_root=cpu_root,
                           work=tmp_path / 'v2' / v2.WORK_ROOT_NAME, expected=args['expected'], tmp=tmp_path)


def pins(e, **over):
    d = dict(plan_path=e.args['plan_path'], expected_plan_sha256=e.plan['plan_sha256'], cert_path=e.args['expected_cache_certificate'],
             cert_sha256=e.args['expected_cache_certificate_sha256'], proof_path=e.proof_path, proof_sha256=e.proof_sha)
    d.update(over); return d


def loaded(e):
    return v2.load_pins(**pins(e), expected=e.expected)


def adopt(e):
    plan, hist, proofs = loaded(e)
    return v2.adopt_retained(plan, proofs, e.cpu_root / 'units', e.work)


def auth(e, **over):
    now = datetime.now(timezone.utc)
    a = {'schema': v2.AUTH_SCHEMA, 'scope_sha256': v2.SCOPE_SHA256, 'plan_sha256': e.plan['plan_sha256'], 'module_sha256': SHA(v2.__file__),
         'script_sha256': SHA(SCRIPT), 'historical_certificate_sha256': e.args['expected_cache_certificate_sha256'], 'proof_sha256': e.proof_sha,
         'independent_review_artifact_sha256': 'a' * 64, 'execution_nonce': 'n' * 24,
         'human_authorization': {'approved_by': 'researcher', 'issued_at_utc': (now - timedelta(minutes=1)).isoformat(),
                                 'expires_at_utc': (now + timedelta(hours=2)).isoformat()}}
    a.update(over); p = e.tmp / 'auth.json'; p.write_text(json.dumps(a)); return p, SHA(p)


def run(e, mx=None, auth_over=None, factory_kw=None, **kw):
    mx = mx or GpuMx()
    ap, asha = auth(e, **(auth_over or {}))
    def factory(proofs, **fk):
        return v2.GpuConvertBackend(proofs, mx=mx, np=np, nn=FakeNN(mx), **{**fk, **(factory_kw or {})})
    a = e.args
    base = dict(plan_path=a['plan_path'], expected_plan_sha256=e.plan['plan_sha256'], cert_path=a['expected_cache_certificate'],
                cert_sha256=a['expected_cache_certificate_sha256'], proof_path=e.proof_path, proof_sha256=e.proof_sha, source_dir=a['source_dir'],
                quant_map_path=a['quant_map_path'], expected_source_sha256=a['expected_source_sha256'], work_root=e.work, authorization=ap,
                authorization_sha256=asha, expected=e.expected, backend_factory=factory, sampler=kw.pop('sampler', sampler()),
                rss_gib=lambda: 0.5, script_file=SCRIPT)
    return v2.execute(**{**base, **kw}), mx


# ---- scope ------------------------------------------------------------------------------------------------------
def test_scope_spells_distinct_counts_and_binds_original_backend_limits():
    assert (v2.SCOPE['plan_units'], v2.SCOPE['retained_units_adopted_from_cpu_root_readonly'], v2.SCOPE['quantized_units_on_gpu']) == (750, 406, 344)
    assert (v2.SCOPE['gpu_unit_attempts_max'], v2.SCOPE['gpu_quantize_ops_max'], v2.SCOPE['gpu_module_path_checks_max']) == (344, 344, 344)
    assert v2.SCOPE['full_model_loaded'] is False and v2.SCOPE['automatic_retry'] is False and v2.SCOPE_SHA256 == qc.sha_bytes(qc.canonical(v2.SCOPE).encode())
    text = inspect.getsource(qc.NativeConvertBackend.__init__)
    assert '8 * 1024**3' in text and '64 * 1024**2' in text and (v2.MLX_MEMORY_LIMIT_BYTES, v2.MLX_CACHE_LIMIT_BYTES) == (8 * 1024**3, 64 * 1024**2)


def test_no_config_env_or_approval_flag_can_start_execution(tmp_path):
    src = inspect.getsource(v2)
    assert 'environ' not in src and 'getenv' not in src
    import argparse
    seen = []
    orig = argparse.ArgumentParser.add_argument
    argparse.ArgumentParser.add_argument = lambda self, *a, **k: (seen.extend(a), orig(self, *a, **k))[1]
    try:
        with pytest.raises(SystemExit): v2.main(['plan'])
    finally:
        argparse.ArgumentParser.add_argument = orig
    assert not {'--approve', '--yes', '--force', '--execute', '--i-have-permission'} & set(seen)
    with pytest.raises(SystemExit): v2.main(['execute', '--plan', 'x'])


# ---- pins / proof -----------------------------------------------------------------------------------------------
def test_pins_bind_plan_certificate_and_proof(tmp_path):
    e = env(tmp_path)
    plan, hist, proofs = loaded(e)
    assert len(proofs) == 3 and hist['units'] == 3
    for bad in ({'proof_sha256': '0' * 64}, {'proof_sha256': None}, {'cert_sha256': '0' * 64}, {'expected_plan_sha256': '1' * 64}):
        with pytest.raises(ValueError): v2.load_pins(**pins(e, **bad), expected=e.expected)
    p = json.loads(e.proof_path.read_text()); p['units'][0]['unit_hash'] = 'f' * 64; e.proof_path.write_text(json.dumps(p))
    with pytest.raises(ValueError, match='Proof entry'): v2.load_pins(**pins(e, proof_sha256=SHA(e.proof_path)), expected=e.expected)
    with pytest.raises(v2.ScopeRefused): v2.load_pins(**pins(e, expected_plan_sha256='1' * 64))          # production scope: only the real plan


# ---- adoption of retained CPU outputs ---------------------------------------------------------------------------
def test_adopt_copies_only_authenticated_retained_and_never_touches_the_cpu_root(tmp_path):
    e = env(tmp_path); before = snapshot(e.cpu_root)
    man = adopt(e)
    assert snapshot(e.cpu_root) == before                                         # read-only on the failed CPU root
    assert man['adopted_retained_units'] == 1 and man['quantized_cpu_outputs_adopted'] == 0
    files = sorted(p.name for p in (e.work / 'units').iterdir())
    assert files == ['plan_binding.json', 'receipts.jsonl', 'unit_0002.safetensors']        # no CPU-quantized unit file or receipt
    assert [json.loads(l)['index'] for l in (e.work / 'units/receipts.jsonl').read_text().splitlines()] == [2]
    assert not any(m == 'mlx' or m.startswith('mlx.') for m in sys.modules)
    with pytest.raises(FileExistsError): adopt(e)                                 # never merges into or overwrites a work root


def test_adopt_refuses_tampered_retained_file_receipt_or_missing_unit(tmp_path):
    e = env(tmp_path); f = e.cpu_root / 'units/unit_0002.safetensors'; good = f.read_bytes()
    f.write_bytes(good[:-1] + bytes([good[-1] ^ 1]))
    with pytest.raises(ValueError): adopt(e)
    assert not e.work.exists()
    f.write_bytes(good); rp = e.cpu_root / 'units/receipts.jsonl'; lines = rp.read_text().splitlines(); rp.write_text('\n'.join(lines[:2]) + '\n')
    with pytest.raises(ValueError, match='no receipt'): adopt(e)
    r = json.loads(lines[2]); r['source_bytes_sha256'] = '0' * 64; rp.write_text('\n'.join(lines[:2] + [json.dumps(r)]) + '\n')
    with pytest.raises(ValueError, match='receipt'): adopt(e)


def test_check_roots_rejects_wrong_names_overlap_and_symlinks(tmp_path):
    e = env(tmp_path)
    for w in (tmp_path / 'other', e.cpu_root, e.cpu_root / 'inside' / v2.WORK_ROOT_NAME):
        with pytest.raises(ValueError): v2.check_roots(w, e.cpu_root)
    with pytest.raises(ValueError): v2.check_roots(e.work, tmp_path / 'wrongname')


# ---- execute: success on the GPU path -----------------------------------------------------------------------------
def test_execute_reproduces_historical_cache_with_gpu_only_distinct_counts_and_publishes_certificate(tmp_path):
    e = env(tmp_path); adopt(e); cpu_before = snapshot(e.cpu_root)
    res, mx = run(e)
    g = res['certificate']['gpu_reconstruction']
    assert (g['gpu_unit_attempts'], g['gpu_quantize_ops'], g['gpu_module_path_checks'], g['retained_units_adopted']) == (2, 2, 2, 1)
    assert (g['max_unit_attempts'], g['max_quantize_ops'], g['max_module_path_checks']) == (344, 344, 344)
    assert g['model_loaded'] is False and g['every_unit_payload_equals_pinned_historical_proof'] is True
    assert {d for _, d in mx.ops} == {'gpu'} and set(mx.created) == {'gpu'} and mx.streams == 0           # never a CPU stream
    assert sorted(mx.limits) == [('cache', v2.MLX_CACHE_LIMIT_BYTES), ('memory', v2.MLX_MEMORY_LIMIT_BYTES)]
    assert sum(n == 'quantize' for n, _ in mx.ops) == 4        # 2 unit quantizations + 2 module-path quantizations
    assert json.loads((e.work / 'certificate.json').read_text())['shards'] == json.loads(e.args['expected_cache_certificate'].read_text())['shards']
    assert (e.work / 'v2_ledger.json').exists() and not (e.work / 'v2_active.lock').exists() and not list(e.work.glob('v2_failure_*'))
    assert snapshot(e.cpu_root) == cpu_before
    with pytest.raises(v2.ScopeRefused, match='ledger'): run(e)                       # single authorized execution


def test_gpu_unit_files_are_the_gpu_outputs_not_adopted_cpu_quantized_payloads(tmp_path):
    e = env(tmp_path); adopt(e); run(e)
    cpu = v2.hash_unit_file(e.cpu_root / 'units/unit_0000.safetensors')[1]
    new = v2.hash_unit_file(e.work / 'units/unit_0000.safetensors')[1]
    assert cpu != new and new == json.loads(e.proof_path.read_text())['units'][0]['output_tensor_payload_sha256']


# ---- first mismatch STOP ----------------------------------------------------------------------------------------
def test_first_payload_mismatch_stops_before_saving_with_durable_receipt_and_no_second_attempt(tmp_path):
    e = env(tmp_path); adopt(e)
    with pytest.raises(v2.HistoricalPayloadMismatch):
        run(e, mx=GpuMx(alter=1))
    mx2 = GpuMx(alter=1)
    fail = json.loads((e.work / 'v2_failure_001.json').read_text())
    assert fail['error_type'] == 'HistoricalPayloadMismatch' and fail['gpu_quantize_ops'] == 1 and fail['gpu_unit_attempts'] == 1
    assert fail['automatic_retry'] is False and fail['rerun_allowed'] is False and fail['partial'] is True and fail['current_unit'] == 0
    assert not (e.work / 'units/unit_0000.safetensors').exists() and not (e.work / 'certificate.json').exists()
    assert not (e.work / 'v2_active.lock').exists()
    with pytest.raises(v2.ScopeRefused): run(e, mx=mx2)
    assert mx2.created == [] and mx2.ops == []


def test_backend_refuses_any_operation_after_a_mismatch_in_the_same_process(tmp_path):
    e = env(tmp_path); plan, hist, proofs = loaded(e); mx = GpuMx(alter=1)
    b = v2.GpuConvertBackend(proofs, mx=mx, np=np, nn=FakeNN(mx), rss_gib=lambda: 0.5)
    b.begin_unit(); w, _ = b.load_unit(e.args['source_dir'] / 'a.safetensors', plan['units'][0])
    with pytest.raises(v2.HistoricalPayloadMismatch): b.quantize(w)
    n = len(mx.created)
    with pytest.raises(v2.HistoricalPayloadMismatch): b.quantize(w)
    with pytest.raises(v2.HistoricalPayloadMismatch): b.begin_unit()
    assert len(mx.created) == n and b.quantize_ops == 1


# ---- caps -------------------------------------------------------------------------------------------------------
def test_backend_caps_are_distinct_and_only_plain_quantized_units_reach_the_gpu(tmp_path):
    e = env(tmp_path); plan, hist, proofs = loaded(e); mx = GpuMx(); src = e.args['source_dir'] / 'a.safetensors'
    b = v2.GpuConvertBackend(proofs, mx=mx, np=np, nn=FakeNN(mx), rss_gib=lambda: 0.5, max_attempts=1, max_quantize=1, max_module=1)
    with pytest.raises(cc.AttemptCapExceeded, match='not'): b.load_unit(src, plan['units'][2])               # retained: never on the GPU
    with pytest.raises(cc.AttemptCapExceeded, match='not'): b.load_unit(src, {**plan['units'][0], 'transform': qc.VISION_CANON})
    assert mx.created == []
    w, _ = b.load_unit(src, plan['units'][0])
    with pytest.raises(cc.AttemptCapExceeded, match='attempt'): b.load_unit(src, plan['units'][1])
    outs = b.quantize(w)
    with pytest.raises(cc.AttemptCapExceeded, match='quantize'): b.quantize(w)
    b.quantize_module(w, 'linear')
    with pytest.raises(cc.AttemptCapExceeded, match='module-path'): b.quantize_module(w, 'linear')
    assert (b.attempts, b.quantize_ops, b.module_path_checks) == (1, 1, 1)
    assert v2.MAX_GPU_UNIT_ATTEMPTS == v2.MAX_GPU_QUANTIZE_OPS == v2.MAX_GPU_MODULE_PATH_CHECKS == 344
    hi = v2.GpuConvertBackend(proofs, mx=mx, np=np, nn=FakeNN(mx), rss_gib=lambda: 0.5, max_attempts=10**6, max_quantize=10**6, max_module=10**6)
    assert (hi.max_attempts, hi.max_quantize, hi.max_module) == (344, 344, 344)


# ---- GPU availability ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize('mx', [lambda: GpuMx(metal=False), lambda: _cpu_default()])
def test_backend_refuses_when_gpu_is_unavailable_or_not_default_before_any_array(tmp_path, mx):
    e = env(tmp_path); plan, hist, proofs = loaded(e); m = mx()
    with pytest.raises(v2.GpuGuardError): v2.GpuConvertBackend(proofs, mx=m, np=np, nn=FakeNN(m), rss_gib=lambda: 0.5)
    assert m.created == []


def _cpu_default():
    m = GpuMx(); m._dev = 'cpu'; return m


def test_execute_on_cpu_default_device_writes_failure_receipt_and_creates_no_array(tmp_path):
    e = env(tmp_path); adopt(e); m = _cpu_default()
    with pytest.raises(v2.GpuGuardError): run(e, mx=m)
    assert m.created == [] and (e.work / 'v2_failure_001.json').exists() and not (e.work / 'certificate.json').exists()


# ---- authorization / ledger / lock --------------------------------------------------------------------------------
@pytest.mark.parametrize('over', [{'scope_sha256': '0' * 64}, {'module_sha256': '0' * 64}, {'script_sha256': '0' * 64}, {'plan_sha256': '0' * 64},
                                  {'schema': 'x'}, {'execution_nonce': 'short'}, {'independent_review_artifact_sha256': 'abc'},
                                  {'historical_certificate_sha256': '0' * 64}, {'proof_sha256': '0' * 64},
                                  {'human_authorization': {'approved_by': '', 'issued_at_utc': '2026-01-01T00:00:00+00:00', 'expires_at_utc': '2026-01-02T00:00:00+00:00'}},
                                  {'human_authorization': {'approved_by': 'r', 'issued_at_utc': '2020-01-01T00:00:00+00:00', 'expires_at_utc': '2020-01-02T00:00:00+00:00'}},
                                  {'human_authorization': {'approved_by': 'r', 'issued_at_utc': '2020-01-01T00:00:00+00:00', 'expires_at_utc': '2099-01-01T00:00:00+00:00'}}])
def test_bad_or_stale_authorization_refuses_before_ledger_or_any_array(tmp_path, over):
    e = env(tmp_path); adopt(e); mx = GpuMx()
    with pytest.raises(v2.ScopeRefused): run(e, mx=mx, auth_over=over)
    assert mx.created == [] and not (e.work / 'v2_ledger.json').exists() and not list(e.work.glob('v2_failure_*'))


def test_missing_or_wrong_artifact_sha_or_pins_refuse(tmp_path):
    e = env(tmp_path); adopt(e); mx = GpuMx()
    with pytest.raises(v2.ScopeRefused): run(e, mx=mx, authorization=None)
    with pytest.raises(v2.ScopeRefused): run(e, mx=mx, authorization_sha256='0' * 64)
    with pytest.raises(v2.ScopeRefused): run(e, mx=mx, cert_sha256='0' * 64)             # supplied pin differs from the artifact's pin
    assert mx.created == []


def test_existing_lock_or_prior_failure_receipt_blocks_execution(tmp_path):
    e = env(tmp_path); adopt(e)
    (e.work / 'v2_active.lock').write_text('1'); mx = GpuMx()
    with pytest.raises(v2.ScopeRefused, match='lock'): run(e, mx=mx)
    (e.work / 'v2_active.lock').unlink(); (e.work / 'v2_failure_001.json').write_text('{}')
    with pytest.raises(v2.ScopeRefused, match='failure'): run(e, mx=mx)
    assert mx.created == []


def test_unadopted_or_foreign_work_root_refuses(tmp_path):
    e = env(tmp_path); mx = GpuMx(); e.work.mkdir(parents=True)
    with pytest.raises(ValueError, match='adopted'): run(e, mx=mx)
    (tmp_path / 'again').mkdir(); adopt_e = env(tmp_path / 'again'); adopt(adopt_e)
    (adopt_e.work / 'units/unit_0000.safetensors').write_bytes(b'x')                 # a stray non-retained unit file is not accepted
    with pytest.raises(ValueError, match='exactly'): run(adopt_e, mx=mx)
    assert mx.created == []


# ---- resources ------------------------------------------------------------------------------------------------
def test_battery_hold_or_unknown_swap_refuses_before_ledger_and_allows_a_later_authorized_run(tmp_path):
    e = env(tmp_path); adopt(e); mx = GpuMx()
    for kw in (dict(power_source='BATTERY', battery_percent=50), dict(swap_used_gib=None), dict(other_heavy_processes=1), dict(pressure_free_percent=20)):
        with pytest.raises(cc.ResourceStop): run(e, mx=mx, sampler=sampler(**kw))
    assert mx.created == [] and not (e.work / 'v2_ledger.json').exists()                  # nothing started, so nothing consumed
    log = [json.loads(l) for l in (e.work / 'v2_resource.jsonl').read_text().splitlines()]
    assert 'power_not_verified_AC' in log[0]['reasons']
    assert run(e)[0]['certificate']['gpu_reconstruction']['gpu_unit_attempts'] == 2


def test_disk_reserve_refuses_before_any_array(tmp_path):
    e = env(tmp_path); adopt(e); mx = GpuMx()
    with pytest.raises(cc.ResourceStop) as r:
        run(e, mx=mx, disk_usage=lambda p: SimpleNamespace(free=1 * 1024**3))
    assert 'disk_reserve' in r.value.reasons and mx.created == [] and not (e.work / 'v2_ledger.json').exists()


def test_per_unit_disk_check_and_rss_envelope_stop_with_failure_receipt(tmp_path):
    e = env(tmp_path); adopt(e)
    with pytest.raises(cc.ResourceStop):
        run(e, factory_kw=dict(disk_usage=lambda p: SimpleNamespace(free=1)))
    assert 'disk_reserve' in json.loads((e.work / 'v2_failure_001.json').read_text())['reasons']
    (tmp_path / 'b').mkdir(); e2 = env(tmp_path / 'b'); adopt(e2)
    with pytest.raises(cc.ResourceStop):
        run(e2, factory_kw=dict(rss_gib=lambda: 9.0))
    (tmp_path / 'c').mkdir(); e3 = env(tmp_path / 'c'); adopt(e3)
    with pytest.raises(cc.ResourceStop):
        run(e3, factory_kw=dict(rss_gib=lambda: None))


def test_swap_growth_during_assembly_stops_and_publishes_nothing(tmp_path):
    e = env(tmp_path); adopt(e); n = []
    def late():
        n.append(1); return healthy(swap_used_gib=4.0 if len(n) <= 5 else 5.5)
    with pytest.raises(cc.ResourceStop):
        run(e, sampler=late)
    fail = json.loads((e.work / 'v2_failure_001.json').read_text())
    assert 'stage_swap_net_growth_over_cap' in fail['reasons'] and not (e.work / 'certificate.json').exists()
    assert fail['last_resource_sample']['swap_growth_gib'] > 1.0


def test_window_expiry_stops_partial_without_rerun(tmp_path):
    e = env(tmp_path); adopt(e); state = {'t': 0.0}
    mx = GpuMx(on_clear=lambda: state.update(t=5000.0))                  # time passes during unit 0
    with pytest.raises(v2.WindowExceeded):
        run(e, mx=mx, clock=lambda: state['t'])
    fail = json.loads((e.work / 'v2_failure_001.json').read_text())
    assert fail['error_type'] == 'WindowExceeded' and fail['partial'] is True and fail['units_with_passing_receipts'] == 2    # 1 adopted + 1 done
    assert fail['gpu_unit_attempts'] == 1 and not (e.work / 'certificate.json').exists()
    mx2 = GpuMx()
    with pytest.raises(v2.ScopeRefused): run(e, mx=mx2)
    assert mx2.created == []


# ---- final identity is never relaxed ----------------------------------------------------------------------------
def test_matching_payloads_but_different_historical_cache_certificate_is_not_published(tmp_path):
    e = env(tmp_path); adopt(e)
    cert_path = e.args['expected_cache_certificate']; h = json.loads(cert_path.read_text()); h['config_cached_sha256'] = '0' * 64
    cert_path.write_text(json.dumps(h)); new_sha = SHA(cert_path); e.args['expected_cache_certificate_sha256'] = new_sha
    with pytest.raises(ValueError, match='historical'): run(e)
    assert not (e.work / 'certificate.json').exists() and json.loads((e.work / 'v2_failure_001.json').read_text())['error_type'] == 'ValueError'


def test_source_shard_tamper_refuses_before_ledger(tmp_path):
    e = env(tmp_path); adopt(e); mx = GpuMx()
    (e.args['source_dir'] / 'a.safetensors').write_bytes(b'z' * 100)
    with pytest.raises(ValueError): run(e, mx=mx)
    assert mx.created == [] and not (e.work / 'v2_ledger.json').exists()


# ---- CPU prepare never touches MLX / GPU ------------------------------------------------------------------------
def test_cli_plan_and_adopt_never_import_mlx_and_report_counts(tmp_path):
    e = env(tmp_path); a = e.args
    base = ['--plan', str(a['plan_path']), '--expected-plan-sha256', e.plan['plan_sha256'], '--expected-cache-certificate', str(a['expected_cache_certificate']),
            '--expected-cache-certificate-sha256', a['expected_cache_certificate_sha256'], '--historical-proof', str(e.proof_path),
            '--historical-proof-sha256', e.proof_sha, '--quant-map', str(a['quant_map_path']), '--work-root', str(e.work), '--cpu-root', str(e.cpu_root)]
    code = ("import sys, json; sys.path.insert(0, %r); from mri_rl import accuracy_recovery_gpu_cache_v2 as m, accuracy_recovery_cpu_cache_v1 as c; "
            "x = c.Expected(units=3, quantized=2, retained=1, tensors=7, shards=1); m.main(%r, expected=x); m.main(%r, expected=x); "
            "assert not any(k == 'mlx' or k.startswith('mlx.') for k in sys.modules), 'mlx imported'")
    r = subprocess.run([sys.executable, '-I', '-c', code % (str(SRC_ROOT), ['plan'] + base, ['adopt-retained'] + base)], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout.splitlines()[0])
    assert (out['executed'], out['retained_authenticated_from_cpu_root'], out['quantized_pending_gpu'], out['quantized_cpu_outputs_adopted'], out['mlx_imported'],
            out['source_checks']) == (False, 1, 2, 0, False, 'NOT_RUN') and out['scope_sha256'] == v2.SCOPE_SHA256
    assert json.loads(r.stdout.splitlines()[1])['adopted_retained_units'] == 1 and (e.work / 'units/unit_0002.safetensors').exists()


def test_plan_report_is_read_only(tmp_path):
    e = env(tmp_path); before = snapshot(e.tmp)
    out = v2.plan_report(**pins(e), cpu_root=e.cpu_root, work_root=e.work, quant_map_path=e.args['quant_map_path'], source_dir=e.args['source_dir'],
                         expected_source_sha256=e.args['expected_source_sha256'], hash_sources=True, expected=e.expected)
    assert snapshot(e.tmp) == before and not e.work.exists() and out['source_checks'] == 'plan_validated_and_shards_hashed'
    assert not any(m == 'mlx' or m.startswith('mlx.') for m in sys.modules)


# ---- real public/technical metadata (skipped when the sibling work tree is absent) ------------------------------
REAL = next((p / 'work/exact_quant_cache' for p in Path(__file__).resolve().parents if (p / 'work/exact_quant_cache/plan.json').exists()),
            Path('/nonexistent'))


@pytest.mark.skipif(not (REAL / 'plan.json').exists(), reason='historical cache metadata not present')
def test_real_metadata_pins_proof_and_plan_layout_match_the_historical_shards():
    plan, hist, proofs = v2.load_pins(REAL / 'plan.json', v2.PLAN_SHA256, REAL / 'cache_certificate.json', qc.sha_file(REAL / 'cache_certificate.json'),
                                      REAL / 'independent750verification_PRIVATE.json', qc.sha_file(REAL / 'independent750verification_PRIVATE.json'))
    assert len(proofs) == 750 and sum(u['action'] == 'quantize' for u in plan['units']) == 344
    assert {s['file']: s['file_bytes'] for s in cc.plan_layout(plan)} == {k: v['bytes'] for k, v in hist['shards'].items()}
    assert v2.disk_estimate_bytes(plan, [u for u in plan['units'] if u['action'] == 'quantize']) > 0
