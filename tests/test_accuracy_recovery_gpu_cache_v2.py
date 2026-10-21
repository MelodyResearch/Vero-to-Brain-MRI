"""CPU-only tests for the gated GPU historical-cache v2 module. A numpy-backed fake mx/nn stands in for MLX: no MLX import,
no model. The 'historical' proof/certificate are produced by the real converter pipeline on tiny fixtures (alter=0); the failed
'CPU root' is the same pipeline with uniformly different quantized payloads (alter=1), like the real CPU-vs-GPU mismatch.
Scope / human / review files built here are TEST-ONLY fabrications in tmp dirs; they authorize nothing real."""
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
from mri_rl import accuracy_recovery_cpu_cache_v1 as cc, accuracy_recovery_gpu_cache_v2 as v2, quant_convert as qc
from tests.test_accuracy_recovery_cpu_cache_v1 import FakeMx, FakeNN, backend as cpu_backend, fixture, sampler
from tests.test_quant_convert import healthy

SRC_ROOT = Path(__file__).parents[1] / 'src'
SCRIPT = Path(__file__).parents[1] / 'scripts' / 'rebuild_accuracy_recovery_historical_gpu_cache_v2.py'
SHA = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
FAKE_INSTALLED = {'python': 'test', 'executable': '/test/python', 'prefix': '/test', 'platform': 'test', 'machine': 'arm64', 'numpy_version': '2',
                  'mlx_version': '0.0-test', 'mlx-metal_version': '0.0-test', 'mlx_tree_sha256': 't' * 64, 'mlx_file_count': 3}
installed_fn = lambda: dict(FAKE_INSTALLED)
SOURCE_RELATED = ('accuracy_recovery_cpu_cache_v1.py', 'quant_convert.py', 'resource_supervisor.py')


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
    repo = tmp_path / 'repo'; (repo / 'work').mkdir(parents=True); (repo / v2.SCOPE_DIR_REL).mkdir(parents=True)
    lock = repo / v2.GLOBAL_LOCK_REL; lock.parent.mkdir(parents=True); lock.write_bytes(b'')
    cpu_root = repo / 'work' / v2.CPU_ROOT_NAME
    b, _, _ = cpu_backend(mx=FakeMx(alter=1))
    qc.convert(plan, args['source_dir'], cpu_root / 'units', b, sampler())          # the failed CPU run: retained exact, quantized different
    return SimpleNamespace(plan=plan, args=args, proof_path=w / 'proof.json', proof_sha=SHA(w / 'proof.json'), cpu_root=cpu_root, repo=repo, lock=lock,
                           work=repo / 'work' / v2.WORK_ROOT_NAME, expected=args['expected'], tmp=tmp_path, n=[0])


def pins(e, **over):
    d = dict(plan_path=e.args['plan_path'], expected_plan_sha256=e.plan['plan_sha256'], cert_path=e.args['expected_cache_certificate'],
             cert_sha256=e.args['expected_cache_certificate_sha256'], proof_path=e.proof_path, proof_sha256=e.proof_sha)
    d.update(over); return d


def loaded(e):
    return v2.load_pins(**pins(e), expected=e.expected)


def adopt(e):
    plan, hist, proofs = loaded(e)
    return v2.adopt_retained(plan, proofs, e.cpu_root / 'units', e.work)


def prep(e, **over):
    e.n[0] += 1
    kw = dict(**pins(e), quant_map_path=e.args['quant_map_path'], source_dir=e.args['source_dir'], cpu_root=e.cpu_root, work_root=e.work,
              scope_output=e.repo / v2.SCOPE_DIR_REL / ('scope_%d.json' % e.n[0]), repo_root=e.repo, expected=e.expected, installed=installed_fn)
    return v2.prepare_scope(**{**kw, **over})


def now_utc():
    return datetime.now(timezone.utc)


def human_doc(digest, n, **over):
    t = now_utc()
    d = {'authorized': True, 'action': v2.ACTION, 'scope_sha256': digest, 'human_instruction_sha256': hashlib.sha256(('instruction-%d' % n).encode()).hexdigest(),
         'approved_by': 'researcher', 'issued_at_utc': (t - timedelta(minutes=1)).isoformat(), 'expires_at_utc': (t + timedelta(hours=2)).isoformat()}
    d.update(over); return d


def review_doc(digest, scope, **over):
    d = {'verdict': 'PASS_WITH_LIMITATIONS', 'scope_sha256': digest, 'reviewer_id': 'codex-test-reviewer', 'reviewer_kind': 'isolated_codex',
         'source_sha256': scope['source_sha256'], 'input_sha256': scope['input_sha256']}
    d.update(over); return d


def write(e, name, doc):
    p = e.tmp / name; p.write_text(json.dumps(doc)); return p, SHA(p)


def admission(e, info=None, human_over=None, review_over=None, review=True, human=True):
    """Full test-only admission set: prepared scope + human + independent review, each a separate file with its exact pin."""
    info = info or prep(e); scope = json.loads(Path(info['scope_path']).read_text()); n = e.n[0]
    out = dict(scope_path=info['scope_path'], scope_sha256=info['scope_file_sha256'], human_path=None, human_sha256=None, review_path=None, review_sha256=None)
    if human:
        out['human_path'], out['human_sha256'] = write(e, 'human_%d.json' % n, human_doc(info['scope_sha256'], n, **(human_over or {})))
    if review:
        out['review_path'], out['review_sha256'] = write(e, 'review_%d.json' % n, review_doc(info['scope_sha256'], scope, **(review_over or {})))
    return out


def auth(e, **over):
    """HUMAN authorization file only (path, sha): without a prepared scope and an authenticated independent review it cannot authorize."""
    e.n[0] += 1
    return write(e, 'human_only_%d.json' % e.n[0], human_doc('0' * 64, e.n[0], **over))


def run(e, mx=None, adm=None, factory_kw=None, **kw):
    mx = mx or GpuMx()
    adm = adm or admission(e)
    def factory(proofs, **fk):
        return v2.GpuConvertBackend(proofs, mx=mx, np=np, nn=FakeNN(mx), **{**fk, **(factory_kw or {})})
    base = dict(**adm, repo_root=e.repo, expected=e.expected, installed=installed_fn, backend_factory=factory, sampler=kw.pop('sampler', sampler()),
                rss_gib=lambda: 0.5)
    return v2.execute(**{**base, **kw}), mx


def consumed(e):
    d = e.repo / v2.CONSUMED_REL
    return sorted(p.name for p in d.iterdir()) if d.exists() else []


class Hold:
    def __enter__(self):
        self.fd = os.open(self.path, os.O_RDWR); fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB); return self
    def __exit__(self, *a): os.close(self.fd)
    def __init__(self, path): self.path = path


# ---- scope ------------------------------------------------------------------------------------------------------
def test_scope_spells_distinct_counts_688_underlying_and_binds_original_backend_limits():
    L = v2.SCOPE
    assert (L['plan_units'], L['retained_units_adopted_from_cpu_root_readonly'], L['quantized_units_on_gpu'], L['retained_units_on_gpu']) == (750, 406, 344, 0)
    assert (L['gpu_unit_attempts_max'], L['gpu_direct_quantize_ops_max'], L['gpu_module_path_quantizations_max'], L['underlying_quantizations_max']) == (344, 344, 344, 688)
    assert (L['model_forward_calls'], L['model_backward_calls'], L['generation_calls'], L['optimizer_updates'], L['full_model_loaded']) == (0, 0, 0, 0, False)
    assert 'lookup' in L['module_construction'] and L['one_shot_no_rerun_no_resume_no_new_root_retry'] and v2.SCOPE_SHA256 == qc.sha_bytes(qc.canonical(L).encode())
    text = inspect.getsource(qc.NativeConvertBackend.__init__)
    assert '8 * 1024**3' in text and '64 * 1024**2' in text and (v2.MLX_MEMORY_LIMIT_BYTES, v2.MLX_CACHE_LIMIT_BYTES) == (8 * 1024**3, 64 * 1024**2)


def test_no_config_env_or_approval_flag_can_start_execution():
    src = inspect.getsource(v2)
    assert 'environ' not in src and 'getenv' not in src
    import argparse
    seen = []
    orig = argparse.ArgumentParser.add_argument
    argparse.ArgumentParser.add_argument = lambda self, *a, **k: (seen.extend(a), orig(self, *a, **k))[1]
    try:
        with pytest.raises(SystemExit): v2.main(['plan'], repo_root='/nonexistent')
    finally:
        argparse.ArgumentParser.add_argument = orig
    assert not {'--approve', '--yes', '--force', '--execute', '--i-have-permission', '--authorization-artifact'} & set(seen)
    with pytest.raises(SystemExit): v2.main(['execute', '--scope', 'x'], repo_root='/nonexistent')


def test_real_local_closure_covers_every_imported_execution_module_and_loads_no_mlx():
    closure = v2.local_closure(v2.default_entries())
    names = {Path(p).name for p in closure}
    assert set(v2.REQUIRED_CLOSURE) <= names and {'shared_rollout_gspo_core_v1.py', 'paired_prompt_groups_v1.py', 'mlx_gspo_smoke.py'} <= names
    assert all(shared_sha == SHA(p) for p, shared_sha in closure.items())
    r = subprocess.run([sys.executable, '-I', '-c', "import sys; sys.path.insert(0, %r); import mri_rl.accuracy_recovery_gpu_cache_v2; "
                        "assert not any(m == 'mlx' or m.startswith('mlx.') for m in sys.modules)" % str(SRC_ROOT)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


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
    assert sorted(p.name for p in (e.work / 'units').iterdir()) == ['plan_binding.json', 'receipts.jsonl', 'unit_0002.safetensors']
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


# ---- prepared immutable execution scope --------------------------------------------------------------------------
def test_prepare_scope_binds_roots_files_inventory_shards_limits_closure_and_installed(tmp_path):
    e = env(tmp_path); info = prep(e); scope = json.loads(Path(info['scope_path']).read_text())
    assert scope['limits'] == v2.SCOPE and scope['static_scope_sha256'] == v2.SCOPE_SHA256 and scope['implemented_by'] == 'claude_code'
    assert scope['roots'] == v2.canonical_roots(e.repo, e.args['source_dir']) and scope['installed'] == FAKE_INSTALLED
    assert set(scope['files']) == {'plan', 'quant_map', 'historical_certificate', 'historical_proof'} and all(len(b['sha256']) == 64 for b in scope['files'].values())
    assert scope['source_shards'] == e.plan['source_shards'] and len(scope['inventory_sha256']) == 64 and scope['plan_sha256'] == e.plan['plan_sha256']
    assert set(v2.REQUIRED_CLOSURE) <= {Path(p).name for p in scope['source_sha256']} and info['source_shards_verified'] == 1
    assert (scope['counts']['retained_authenticated_from_cpu_root'], scope['counts']['quantized_pending_gpu']) == (1, 2)
    assert info['scope_sha256'] == v2.shared.scope_digest(scope) and info['scope_file_sha256'] == SHA(info['scope_path'])
    assert not any(m == 'mlx' or m.startswith('mlx.') for m in sys.modules)
    with pytest.raises(v2.ScopeRefused): prep(e, scope_output=Path(info['scope_path']))                    # immutable: never overwritten


def test_prepare_scope_refuses_noncanonical_output_roots_tampered_shards_and_missing_mlx(tmp_path):
    e = env(tmp_path)
    with pytest.raises(v2.ScopeRefused): prep(e, scope_output=e.tmp / 'elsewhere.json')
    with pytest.raises(v2.ScopeRefused): prep(e, work_root=e.tmp / 'x' / v2.WORK_ROOT_NAME)
    with pytest.raises(v2.ScopeRefused): prep(e, installed=lambda: {**FAKE_INSTALLED, 'mlx_tree_sha256': None})
    link = e.tmp / 'link'; link.symlink_to(e.args['source_dir'])
    with pytest.raises(v2.ScopeRefused): prep(e, source_dir=link)                                          # symlink drift
    (e.args['source_dir'] / 'a.safetensors').write_bytes(b'z' * 100)
    with pytest.raises(ValueError): prep(e)
    assert list((e.repo / v2.SCOPE_DIR_REL).iterdir()) == []


# ---- independent admission: forged / missing / mismatched -----------------------------------------------------------
def test_full_admission_set_is_accepted_and_consumes_nothing_by_itself(tmp_path):
    e = env(tmp_path); a = admission(e)
    adm = v2.admit(**a, now=now_utc(), repo_root=e.repo, expected=e.expected, installed=installed_fn)
    assert adm.check('x', full=True) and consumed(e) == []


def test_absent_review_or_human_or_scope_cannot_authorize(tmp_path):
    e = env(tmp_path); p, sha = auth(e)
    with pytest.raises(v2.ScopeRefused): v2.check_authorization(p, sha, now=now_utc(), plan_sha256=e.plan['plan_sha256'], script_file=SCRIPT, repo_root=e.repo)
    a = admission(e)
    for drop in ('review_path', 'review_sha256', 'human_path', 'human_sha256', 'scope_path', 'scope_sha256'):
        with pytest.raises(v2.ScopeRefused): v2.admit(**{**a, drop: None}, now=now_utc(), repo_root=e.repo, expected=e.expected, installed=installed_fn)
    with pytest.raises(v2.ScopeRefused):                                                    # a fabricated digest for a review file that does not exist
        v2.admit(**{**a, 'review_path': e.tmp / 'nonexistent_review.json', 'review_sha256': 'a' * 64}, now=now_utc(), repo_root=e.repo, expected=e.expected, installed=installed_fn)
    with pytest.raises(v2.ScopeRefused):                                                    # a real file whose sha does not match the supplied pin
        v2.admit(**{**a, 'review_sha256': 'a' * 64}, now=now_utc(), repo_root=e.repo, expected=e.expected, installed=installed_fn)


@pytest.mark.parametrize('over', [
    {'verdict': 'FAIL'}, {'verdict': 'PASS_WITH_QUESTIONS'}, {'verdict': None}, {'reviewer_kind': 'claude_code'}, {'reviewer_kind': 'someone'}, {'reviewer_id': ''},
    {'scope_sha256': '0' * 64}, {'source_sha256': {}}, {'input_sha256': '0' * 64}])
def test_forged_or_mismatched_independent_review_is_refused(tmp_path, over):
    e = env(tmp_path); a = admission(e, review_over=over)
    with pytest.raises(v2.ScopeRefused): v2.admit(**a, now=now_utc(), repo_root=e.repo, expected=e.expected, installed=installed_fn)


def test_review_bound_to_different_source_closure_hash_is_refused(tmp_path):
    e = env(tmp_path); info = prep(e); scope = json.loads(Path(info['scope_path']).read_text())
    bad = dict(scope['source_sha256']); k = sorted(bad)[0]; bad[k] = '0' * 64
    a = admission(e, info, review_over={'source_sha256': bad})
    with pytest.raises(v2.ScopeRefused): v2.admit(**a, now=now_utc(), repo_root=e.repo, expected=e.expected, installed=installed_fn)


@pytest.mark.parametrize('over', [
    {'human_instruction_sha256': None}, {'human_instruction_sha256': ''}, {'human_instruction_sha256': 'XYZ'}, {'authorized': False}, {'action': 'other'},
    {'scope_sha256': '0' * 64}, {'approved_by': ''},
    {'issued_at_utc': '2020-01-01T00:00:00+00:00', 'expires_at_utc': '2020-01-02T00:00:00+00:00'},
    {'issued_at_utc': '2020-01-01T00:00:00+00:00', 'expires_at_utc': '2099-01-01T00:00:00+00:00'}, {'issued_at_utc': 'garbage'}])
def test_human_authorization_needs_instruction_hash_action_scope_and_freshness(tmp_path, over):
    e = env(tmp_path); a = admission(e, human_over=over)
    with pytest.raises(v2.ScopeRefused): v2.admit(**a, now=now_utc(), repo_root=e.repo, expected=e.expected, installed=installed_fn)


def test_scope_with_widened_limits_or_drifted_roots_is_refused_even_if_repinned(tmp_path):
    e = env(tmp_path)
    for mutate in (lambda s: s['limits'].update(underlying_quantizations_max=1000), lambda s: s['limits'].update(gpu_direct_quantize_ops_max=345),
                   lambda s: s['limits'].update(automatic_retry=True), lambda s: s['roots'].update(global_lock=str(e.tmp / 'my.lock')),
                   lambda s: s['roots'].update(work_root=str(e.tmp / 'work' / v2.WORK_ROOT_NAME)), lambda s: s.update(approved=True),
                   lambda s: s.update(expected={**s['expected'], 'quantized': 1}), lambda s: s.update(scope_kind='x'), lambda s: s.update(implemented_by='isolated_codex')):
        info = prep(e); scope = json.loads(Path(info['scope_path']).read_text()); mutate(scope)
        p, sha = write(e, 'mut_scope_%d.json' % e.n[0], scope)
        digest = v2.shared.scope_digest(scope)
        a = admission(e, {**info, 'scope_path': str(p), 'scope_file_sha256': sha, 'scope_sha256': digest})
        with pytest.raises(v2.ScopeRefused): v2.admit(**a, now=now_utc(), repo_root=e.repo, expected=e.expected, installed=installed_fn)
    info = prep(e); scope = json.loads(Path(info['scope_path']).read_text())
    Path(info['scope_path']).write_text(json.dumps({**scope, 'extra': 1}))                       # edited after the pin was taken
    with pytest.raises(v2.ScopeRefused): v2.admit(**admission(e, info), now=now_utc(), repo_root=e.repo, expected=e.expected, installed=installed_fn)


def test_mutated_dependency_input_or_installed_fingerprint_is_refused(tmp_path, monkeypatch):
    e = env(tmp_path); a = admission(e)
    ok = v2.admit(**a, now=now_utc(), repo_root=e.repo, expected=e.expected, installed=installed_fn)
    real = v2.shared.file_sha256
    resource = next(p for p in ok.scope['source_sha256'] if p.endswith('resource_supervisor.py'))
    monkeypatch.setattr(v2.shared, 'file_sha256', lambda p: '0' * 64 if str(p) == resource else real(p))      # a dependency changed after review
    with pytest.raises(v2.ScopeRefused, match='Reviewed source changed'): v2.admit(**a, now=now_utc(), repo_root=e.repo, expected=e.expected, installed=installed_fn)
    monkeypatch.setattr(v2.shared, 'file_sha256', real)
    with pytest.raises(v2.ScopeRefused, match='fingerprint'): v2.admit(**a, now=now_utc(), repo_root=e.repo, expected=e.expected, installed=lambda: {**FAKE_INSTALLED, 'mlx_version': '9'})
    (e.args['source_dir'] / 'config.json').write_text('{}')                                       # source dir content is covered by the shard hashes at execute
    e.args['quant_map_path'].write_text(json.dumps({'lin': dict(qc.QUANT)}))                      # an input file changed after prepare
    with pytest.raises(v2.ScopeRefused, match='Input file changed'): v2.admit(**a, now=now_utc(), repo_root=e.repo, expected=e.expected, installed=installed_fn)


def test_refused_admission_in_execute_consumes_nothing_creates_no_array_and_leaves_the_lock_alone(tmp_path):
    e = env(tmp_path); adopt(e); mx = GpuMx(); lock_before = (e.lock.stat().st_ino, e.lock.stat().st_size)
    for kw in (dict(review_over={'verdict': 'FAIL'}), dict(human_over={'human_instruction_sha256': None}), dict(review=False), dict(human=False)):
        with pytest.raises(v2.ScopeRefused): run(e, mx=mx, adm=admission(e, **kw))
    assert mx.created == [] and consumed(e) == [] and not (e.work / 'v2_ledger.json').exists() and not list(e.work.glob('v2_failure_*'))
    assert (e.lock.stat().st_ino, e.lock.stat().st_size) == lock_before


# ---- canonical global flock ------------------------------------------------------------------------------------
def test_execute_acquires_the_global_flock_nonblocking_before_any_mlx_construction_and_releases_it(tmp_path):
    e = env(tmp_path); adopt(e); mx = GpuMx(); seen = []
    def probe(proofs, **fk):
        fd = os.open(e.lock, os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB); seen.append('LOCK_NOT_HELD')
        except BlockingIOError:
            seen.append('held')
        finally:
            os.close(fd)
        return v2.GpuConvertBackend(proofs, mx=mx, np=np, nn=FakeNN(mx), **fk)
    ino = e.lock.stat().st_ino
    run(e, mx=mx, backend_factory=probe)
    assert seen == ['held'] and e.lock.exists() and e.lock.stat().st_ino == ino and e.lock.stat().st_size == 0
    with Hold(e.lock): pass                                                              # released after the run
    src = inspect.getsource(v2.execute)
    assert 'flock' in src and 'LOCK_EX' in src and 'LOCK_NB' in src and 'O_CREAT' not in src and 'unlink(lock_path' not in src


def test_held_global_flock_refuses_before_any_mlx_without_forcing_or_consuming(tmp_path):
    e = env(tmp_path); adopt(e); mx = GpuMx(); made = []
    stat_before = (e.lock.stat().st_ino, e.lock.stat().st_size, e.lock.stat().st_mtime_ns)
    with Hold(e.lock):
        with pytest.raises(v2.ScopeRefused, match='held'):
            run(e, mx=mx, backend_factory=lambda *a, **k: made.append(1))
    assert made == [] and mx.created == [] and consumed(e) == [] and not (e.work / 'v2_ledger.json').exists() and not list(e.work.glob('v2_failure_*'))
    assert (e.lock.stat().st_ino, e.lock.stat().st_size, e.lock.stat().st_mtime_ns) == stat_before
    assert run(e)[0]['certificate']['gpu_reconstruction']['gpu_unit_attempts'] == 2     # nothing was consumed, so the authorized run can still go


def test_missing_or_symlinked_global_lock_is_refused_and_never_created(tmp_path):
    e = env(tmp_path); adopt(e); mx = GpuMx(); target = e.tmp / 'real.lock'; target.write_bytes(b'')
    e.lock.unlink()
    with pytest.raises(v2.ScopeRefused): run(e, mx=mx)
    assert not e.lock.exists()
    e.lock.symlink_to(target)
    with pytest.raises(v2.ScopeRefused): run(e, mx=mx)
    assert mx.created == [] and consumed(e) == []


def test_global_flock_is_released_and_preserved_after_a_failure(tmp_path):
    e = env(tmp_path); adopt(e)
    with pytest.raises(v2.HistoricalPayloadMismatch): run(e, mx=GpuMx(alter=1))
    assert e.lock.exists()
    with Hold(e.lock): pass


# ---- execute: success on the GPU path -----------------------------------------------------------------------------
def test_execute_reproduces_historical_cache_with_gpu_only_688_style_accounting_and_publishes_certificate(tmp_path):
    e = env(tmp_path); adopt(e); cpu_before = snapshot(e.cpu_root)
    res, mx = run(e)
    g = res['certificate']['gpu_reconstruction']
    assert (g['gpu_unit_attempts'], g['gpu_direct_quantize_ops'], g['gpu_module_path_checks'], g['gpu_underlying_quantizations'], g['retained_units_adopted']) == (2, 2, 2, 4, 1)
    assert g['gpu_quantize_ops'] == 2 and g['retained_units_gpu_operations'] == 0 and (g['max_quantize_ops'], g['max_underlying_quantizations']) == (344, 688)
    assert sum(n == 'quantize' for n, _ in mx.ops) == g['gpu_underlying_quantizations'] == 2 * g['gpu_quantize_ops']
    assert g['model_loaded'] is False and g['every_unit_payload_equals_pinned_historical_proof'] is True
    assert {d for _, d in mx.ops} == {'gpu'} and set(mx.created) == {'gpu'} and mx.streams == 0           # never a CPU stream
    assert sorted(mx.limits) == [('cache', v2.MLX_CACHE_LIMIT_BYTES), ('memory', v2.MLX_MEMORY_LIMIT_BYTES)] and mx.ops[-1][0] == 'clear'
    assert json.loads((e.work / 'certificate.json').read_text())['shards'] == json.loads(e.args['expected_cache_certificate'].read_text())['shards']
    assert (e.work / 'v2_ledger.json').exists() and not (e.work / 'v2_active.lock').exists() and not list(e.work.glob('v2_failure_*'))
    names = consumed(e)
    assert sum(n.startswith('scope_consumed_') for n in names) == sum(n.startswith('authorization_consumed_') for n in names) == sum(n.startswith('scope_outcome_') for n in names) == 1
    assert snapshot(e.cpu_root) == cpu_before
    with pytest.raises(v2.ScopeRefused): run(e)                                       # single authorized execution


def test_gpu_unit_files_are_the_gpu_outputs_not_adopted_cpu_quantized_payloads(tmp_path):
    e = env(tmp_path); adopt(e); run(e)
    cpu = v2.hash_unit_file(e.cpu_root / 'units/unit_0000.safetensors')[1]
    new = v2.hash_unit_file(e.work / 'units/unit_0000.safetensors')[1]
    assert cpu != new and new == json.loads(e.proof_path.read_text())['units'][0]['output_tensor_payload_sha256']


# ---- first mismatch STOP / one-shot ---------------------------------------------------------------------------
def test_first_payload_mismatch_stops_before_saving_and_the_consumed_scope_cannot_be_retried(tmp_path):
    e = env(tmp_path); adopt(e); adm = admission(e)
    with pytest.raises(v2.HistoricalPayloadMismatch):
        run(e, mx=GpuMx(alter=1), adm=adm)
    fail = json.loads((e.work / 'v2_failure_001.json').read_text())
    assert fail['error_type'] == 'HistoricalPayloadMismatch' and fail['gpu_direct_quantize_ops'] == 1 and fail['gpu_unit_attempts'] == 1
    assert fail['gpu_underlying_quantizations'] == 1 and fail['automatic_retry'] is False and fail['rerun_allowed'] is False and fail['partial'] is True
    assert not (e.work / 'units/unit_0000.safetensors').exists() and not (e.work / 'certificate.json').exists() and not (e.work / 'v2_active.lock').exists()
    outcome = [json.loads((e.repo / v2.CONSUMED_REL / n).read_text()) for n in consumed(e) if n.startswith('scope_outcome_')]
    assert outcome[0]['state'] == 'FAILED'
    mx2 = GpuMx(alter=1)
    with pytest.raises(v2.ScopeRefused): run(e, mx=mx2, adm=adm)
    assert mx2.created == [] and mx2.ops == []


def test_new_root_cannot_retry_a_consumed_scope_or_a_consumed_human_instruction(tmp_path):
    e = env(tmp_path); adopt(e); adm = admission(e)
    with pytest.raises(v2.HistoricalPayloadMismatch): run(e, mx=GpuMx(alter=1), adm=adm)
    e.work.rename(e.tmp / 'failed_root_preserved')                                          # a human moves the evidence aside, re-adopts fresh
    adopt(e); mx = GpuMx()
    with pytest.raises(v2.ScopeRefused, match='already consumed'): run(e, mx=mx, adm=adm)             # same scope: refused although the new root is clean
    scope_info = prep(e)                                                                    # same inputs => same scope digest => still consumed
    fresh_scope_same_instruction = admission(e, scope_info, human_over={'human_instruction_sha256': hashlib.sha256(b'instruction-1').hexdigest()})
    with pytest.raises(v2.ScopeRefused): run(e, mx=mx, adm=fresh_scope_same_instruction)
    assert mx.created == [] and not (e.work / 'v2_ledger.json').exists()


def test_backend_refuses_any_operation_after_a_mismatch_in_the_same_process(tmp_path):
    e = env(tmp_path); plan, hist, proofs = loaded(e); mx = GpuMx(alter=1)
    b = v2.GpuConvertBackend(proofs, mx=mx, np=np, nn=FakeNN(mx), rss_gib=lambda: 0.5)
    b.begin_unit(); w, _ = b.load_unit(e.args['source_dir'] / 'a.safetensors', plan['units'][0])
    with pytest.raises(v2.HistoricalPayloadMismatch): b.quantize(w)
    n = len(mx.created)
    with pytest.raises(v2.HistoricalPayloadMismatch): b.quantize(w)
    with pytest.raises(v2.HistoricalPayloadMismatch): b.begin_unit()
    assert len(mx.created) == n and b.quantize_ops == 1


def test_reviewed_source_change_mid_run_stops_before_the_next_unit(tmp_path, monkeypatch):
    e = env(tmp_path); adopt(e); real = v2.shared.file_sha256; state = {'bad': False}
    target = str(Path(v2.cc.__file__).resolve())
    monkeypatch.setattr(v2.shared, 'file_sha256', lambda p: '0' * 64 if state['bad'] and str(Path(p).resolve()) == target else real(p))
    with pytest.raises(v2.ScopeRefused, match='Reviewed source changed before begin_unit'):
        run(e, mx=GpuMx(on_clear=lambda: state.update(bad=True)))                      # tampering appears after unit 0 completes
    fail = json.loads((e.work / 'v2_failure_001.json').read_text())
    assert fail['gpu_unit_attempts'] == 1 and not (e.work / 'certificate.json').exists()


# ---- caps -------------------------------------------------------------------------------------------------------
def test_backend_caps_are_distinct_and_only_plain_quantized_units_reach_the_gpu(tmp_path):
    e = env(tmp_path); plan, hist, proofs = loaded(e); mx = GpuMx(); src = e.args['source_dir'] / 'a.safetensors'
    b = v2.GpuConvertBackend(proofs, mx=mx, np=np, nn=FakeNN(mx), rss_gib=lambda: 0.5, max_attempts=1, max_quantize=1, max_module=1)
    with pytest.raises(cc.AttemptCapExceeded, match='not'): b.load_unit(src, plan['units'][2])               # retained: never on the GPU
    with pytest.raises(cc.AttemptCapExceeded, match='not'): b.load_unit(src, {**plan['units'][0], 'transform': qc.VISION_CANON})
    assert mx.created == []
    w, _ = b.load_unit(src, plan['units'][0])
    with pytest.raises(cc.AttemptCapExceeded, match='attempt'): b.load_unit(src, plan['units'][1])
    b.quantize(w)
    with pytest.raises(cc.AttemptCapExceeded, match='quantize'): b.quantize(w)
    b.quantize_module(w, 'linear')
    with pytest.raises(cc.AttemptCapExceeded, match='module-path'): b.quantize_module(w, 'linear')
    assert (b.attempts, b.quantize_ops, b.module_path_checks, b.underlying_quantizations) == (1, 1, 1, 2)
    assert v2.MAX_GPU_UNIT_ATTEMPTS == v2.MAX_GPU_QUANTIZE_OPS == v2.MAX_GPU_MODULE_PATH_CHECKS == 344 and v2.MAX_UNDERLYING_QUANTIZATIONS == 688
    hi = v2.GpuConvertBackend(proofs, mx=mx, np=np, nn=FakeNN(mx), rss_gib=lambda: 0.5, max_attempts=10**6, max_quantize=10**6, max_module=10**6, max_underlying=10**6)
    assert (hi.max_attempts, hi.max_quantize, hi.max_module, hi.max_underlying) == (344, 344, 344, 688)


def test_combined_underlying_cap_stops_before_the_extra_quantization(tmp_path):
    e = env(tmp_path); plan, hist, proofs = loaded(e); mx = GpuMx(); src = e.args['source_dir'] / 'a.safetensors'
    b = v2.GpuConvertBackend(proofs, mx=mx, np=np, nn=FakeNN(mx), rss_gib=lambda: 0.5, max_underlying=2)
    w, _ = b.load_unit(src, plan['units'][0]); b.quantize(w); b.quantize_module(w, 'linear'); n = len(mx.created)
    with pytest.raises(cc.AttemptCapExceeded, match='Underlying'): b.quantize_module(w, 'linear')
    with pytest.raises(cc.AttemptCapExceeded, match='Underlying'): b.quantize(w)
    assert len(mx.created) == n and b.underlying_quantizations == 2


# ---- GPU availability ---------------------------------------------------------------------------------------------
def _cpu_default():
    m = GpuMx(); m._dev = 'cpu'; return m


@pytest.mark.parametrize('mx', [lambda: GpuMx(metal=False), _cpu_default])
def test_backend_refuses_when_gpu_is_unavailable_or_not_default_before_any_array(tmp_path, mx):
    e = env(tmp_path); plan, hist, proofs = loaded(e); m = mx()
    with pytest.raises(v2.GpuGuardError): v2.GpuConvertBackend(proofs, mx=m, np=np, nn=FakeNN(m), rss_gib=lambda: 0.5)
    assert m.created == []


def test_execute_on_cpu_default_device_writes_failure_receipt_and_creates_no_array(tmp_path):
    e = env(tmp_path); adopt(e); m = _cpu_default()
    with pytest.raises(v2.GpuGuardError): run(e, mx=m)
    assert m.created == [] and (e.work / 'v2_failure_001.json').exists() and not (e.work / 'certificate.json').exists()


def test_existing_work_root_ledger_lock_or_failure_receipt_blocks_execution(tmp_path):
    e = env(tmp_path); adopt(e); mx = GpuMx(); adm = admission(e)
    (e.work / 'v2_active.lock').write_text('1')
    with pytest.raises(v2.ScopeRefused, match='lock'): run(e, mx=mx, adm=adm)
    (e.work / 'v2_active.lock').unlink(); (e.work / 'v2_failure_001.json').write_text('{}')
    with pytest.raises(v2.ScopeRefused, match='failure'): run(e, mx=mx, adm=adm)
    assert mx.created == [] and consumed(e) == []


def test_unadopted_or_foreign_work_root_refuses_without_consuming(tmp_path):
    e = env(tmp_path); mx = GpuMx(); e.work.mkdir(parents=True)
    with pytest.raises(ValueError, match='adopted'): run(e, mx=mx)
    (tmp_path / 'again').mkdir(); e2 = env(tmp_path / 'again'); adopt(e2)
    (e2.work / 'units/unit_0000.safetensors').write_bytes(b'x')                       # a stray non-retained unit file is not accepted
    with pytest.raises(ValueError, match='exactly'): run(e2, mx=mx)
    assert mx.created == [] and consumed(e) == [] and consumed(e2) == []


# ---- resources ------------------------------------------------------------------------------------------------
def test_battery_hold_or_unknown_swap_refuses_before_consumption_and_allows_the_same_authorization_later(tmp_path):
    e = env(tmp_path); adopt(e); mx = GpuMx(); adm = admission(e)
    for kw in (dict(power_source='BATTERY', battery_percent=50), dict(swap_used_gib=None), dict(other_heavy_processes=1), dict(pressure_free_percent=20)):
        with pytest.raises(cc.ResourceStop): run(e, mx=mx, adm=adm, sampler=sampler(**kw))
    assert mx.created == [] and consumed(e) == [] and not (e.work / 'v2_ledger.json').exists()                 # nothing started, so nothing consumed
    log = [json.loads(l) for l in (e.work / 'v2_resource.jsonl').read_text().splitlines()]
    assert 'power_not_verified_AC' in log[0]['reasons']
    assert run(e, adm=adm)[0]['certificate']['gpu_reconstruction']['gpu_unit_attempts'] == 2


def test_disk_reserve_refuses_before_any_array_or_consumption(tmp_path):
    e = env(tmp_path); adopt(e); mx = GpuMx()
    with pytest.raises(cc.ResourceStop) as r:
        run(e, mx=mx, disk_usage=lambda p: SimpleNamespace(free=1 * 1024**3))
    assert 'disk_reserve' in r.value.reasons and mx.created == [] and consumed(e) == [] and not (e.work / 'v2_ledger.json').exists()


def test_per_unit_disk_check_and_rss_envelope_stop_with_failure_receipt(tmp_path):
    e = env(tmp_path); adopt(e)
    with pytest.raises(cc.ResourceStop):
        run(e, factory_kw=dict(disk_usage=lambda p: SimpleNamespace(free=1)))
    assert 'disk_reserve' in json.loads((e.work / 'v2_failure_001.json').read_text())['reasons']
    for i, bad in enumerate((9.0, None)):
        (tmp_path / ('r%d' % i)).mkdir(); e2 = env(tmp_path / ('r%d' % i)); adopt(e2)
        with pytest.raises(cc.ResourceStop):
            run(e2, factory_kw=dict(rss_gib=lambda bad=bad: bad))


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
    e = env(tmp_path); adopt(e); state = {'t': 0.0}; adm = admission(e)
    mx = GpuMx(on_clear=lambda: state.update(t=5000.0))                  # time passes during unit 0
    with pytest.raises(v2.WindowExceeded):
        run(e, mx=mx, adm=adm, clock=lambda: state['t'])
    fail = json.loads((e.work / 'v2_failure_001.json').read_text())
    assert fail['error_type'] == 'WindowExceeded' and fail['partial'] is True and fail['units_with_passing_receipts'] == 2    # 1 adopted + 1 done
    assert fail['gpu_unit_attempts'] == 1 and not (e.work / 'certificate.json').exists()
    mx2 = GpuMx()
    with pytest.raises(v2.ScopeRefused): run(e, mx=mx2, adm=adm)
    assert mx2.created == []


# ---- final identity is never relaxed ----------------------------------------------------------------------------
def test_matching_payloads_but_different_historical_cache_certificate_is_not_published(tmp_path):
    e = env(tmp_path); adopt(e)
    cert_path = e.args['expected_cache_certificate']; h = json.loads(cert_path.read_text()); h['config_cached_sha256'] = '0' * 64
    cert_path.write_text(json.dumps(h)); e.args['expected_cache_certificate_sha256'] = SHA(cert_path)
    with pytest.raises(ValueError, match='historical'): run(e)
    assert not (e.work / 'certificate.json').exists() and json.loads((e.work / 'v2_failure_001.json').read_text())['error_type'] == 'ValueError'


def test_source_shard_tamper_between_prepare_and_execute_refuses_before_consumption(tmp_path):
    e = env(tmp_path); adopt(e); mx = GpuMx(); adm = admission(e)
    (e.args['source_dir'] / 'a.safetensors').write_bytes(b'z' * 100)
    with pytest.raises(ValueError): run(e, mx=mx, adm=adm)
    assert mx.created == [] and consumed(e) == [] and not (e.work / 'v2_ledger.json').exists()


# ---- CPU prepare never touches MLX / GPU ------------------------------------------------------------------------
def test_cli_plan_adopt_and_prepare_scope_never_import_mlx(tmp_path):
    e = env(tmp_path); a = e.args
    base = ['--plan', str(a['plan_path']), '--expected-plan-sha256', e.plan['plan_sha256'], '--expected-cache-certificate', str(a['expected_cache_certificate']),
            '--expected-cache-certificate-sha256', a['expected_cache_certificate_sha256'], '--historical-proof', str(e.proof_path),
            '--historical-proof-sha256', e.proof_sha, '--quant-map', str(a['quant_map_path']), '--work-root', str(e.work), '--cpu-root', str(e.cpu_root)]
    scope_out = str(e.repo / v2.SCOPE_DIR_REL / 'cli_scope.json')
    code = ("import sys, json; sys.path.insert(0, %r); from mri_rl import accuracy_recovery_gpu_cache_v2 as m, accuracy_recovery_cpu_cache_v1 as c; "
            "x = c.Expected(units=3, quantized=2, retained=1, tensors=7, shards=1); inst = lambda: %r; r = %r; "
            "m.main(['plan'] + %r, expected=x, repo_root=r); m.main(['adopt-retained'] + %r, expected=x, repo_root=r); "
            "m.main(['prepare-scope'] + %r + ['--source-dir', %r, '--scope-output', %r], expected=x, repo_root=r, installed=inst); "
            "assert not any(k == 'mlx' or k.startswith('mlx.') for k in sys.modules), 'mlx imported'")
    r = subprocess.run([sys.executable, '-I', '-c', code % (str(SRC_ROOT), FAKE_INSTALLED, str(e.repo), base, base, base, str(a['source_dir']), scope_out)],
                       capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stderr
    lines = r.stdout.splitlines()
    out = json.loads(lines[0])
    assert (out['executed'], out['retained_authenticated_from_cpu_root'], out['quantized_pending_gpu'], out['quantized_cpu_outputs_adopted'], out['mlx_imported'],
            out['source_checks']) == (False, 1, 2, 0, False, 'NOT_RUN') and out['scope_sha256'] == v2.SCOPE_SHA256 and out['underlying_quantizations_max'] == 688
    assert json.loads(lines[1])['adopted_retained_units'] == 1 and (e.work / 'units/unit_0002.safetensors').exists()
    prepared = json.loads(lines[2])
    assert prepared['scope_file_sha256'] == SHA(scope_out) and prepared['source_shards_verified'] == 1 and Path(scope_out).exists()


def test_cli_rejects_noncanonical_roots(tmp_path):
    e = env(tmp_path); a = e.args
    argv = ['plan', '--plan', str(a['plan_path']), '--expected-plan-sha256', e.plan['plan_sha256'], '--expected-cache-certificate', str(a['expected_cache_certificate']),
            '--expected-cache-certificate-sha256', a['expected_cache_certificate_sha256'], '--historical-proof', str(e.proof_path),
            '--historical-proof-sha256', e.proof_sha, '--quant-map', str(a['quant_map_path']), '--work-root', str(e.tmp / 'x' / v2.WORK_ROOT_NAME), '--cpu-root', str(e.cpu_root)]
    with pytest.raises(v2.ScopeRefused): v2.main(argv, expected=e.expected, repo_root=e.repo)


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
