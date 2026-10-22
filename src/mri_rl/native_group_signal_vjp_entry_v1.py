"""NATIVE_GROUP_SIGNAL_VJP_V1 exact authority, one-shot journaled read-only controller and CLI (prepare / launch / child / analyze).

Author: Claude Code (claude-opus-5-5). Importing this module imports no MLX and loads nothing.
Authority is never created by a flag or by JSON written here. Execution requires (1) the exact current CPU preparation (re-derived
and compared byte-for-byte), (2) an independent exact review by a non-Claude reviewer naming the scope/source/input hashes, supplied as a
separate file with its expected SHA-256, and (3) a fresh bounded human permission file (issued <=30 min before admission), consumed once.
All three are checked BEFORE any native import or ops construction. Synthetic test scopes are refused by the production CLI.
One planned Parent Adam52 load, one fresh child, no automatic retry/recovery/reload, operation-start window 3600 s including waits.
Resources: unchanged safe_micro_dynamic_resources_v1.POLICY (GREEN/YELLOW/RED, RSS 14 GiB, MLX 12 GiB, disk 26 GiB, battery 20%; no fixed
9/10 GiB admission, no standalone swap STOP) and the existing safe_micro_watchdog_v2 guardian on its own child only.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from mri_rl import native_group_signal_vjp_v1 as ngs
from mri_rl import shared_rollout_gspo_core_v1 as shared
from mri_rl.safe_micro_journal_v1 import Journal, durable_mkdir

require = shared.require
ACTION = ngs.ACTION
_ISSUER = object()
FRESH_SECONDS = 1800
MAX_GRANT_SECONDS = 86400
RELEASE_MAX_BYTES = 128 * 1024 ** 2
KEY_PHASES = ('admission', 'model_load', 'complete', 'start', 'native_import')


def load(path):
    require(not any(x in str(path).lower() for x in ('final100', 'holdout16', 'new16', 'dev24')), 'Forbidden development input')
    return json.loads(Path(path).read_text())


def _bound(scope):
    return {**scope['source_sha256'], **scope['file_sha256'], **scope['payload_sha256']}


class Authorization:
    def __init__(self, issuer, scope, expires, start=None):
        require(issuer is _ISSUER, 'Only exact admission may issue authorization')
        self.scope = json.loads(json.dumps(scope))
        self.sha = ngs.digest(scope)
        self.expires = expires
        self.start = time.monotonic() if start is None else start
        self.window = scope['run_limits']['operation_window_seconds_including_waits']
        self.verified = {}

    def check(self, phase):
        require(datetime.now(timezone.utc) < self.expires, 'Human permission expired')
        require(time.monotonic() - self.start < self.window, 'Operation-start window exhausted (waits count, never reset)')
        require(ngs.digest(self.scope) == self.sha, 'Exact authorization scope changed')
        for p, h in _bound(self.scope).items():
            path = Path(p)
            require(path.is_file() and not path.is_symlink(), 'Bound file missing: ' + path.name)
            s = path.stat()
            stamp = (s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_ino, s.st_dev)
            if self.verified.get(p) != stamp or any(k in str(phase) for k in KEY_PHASES):
                require(shared.file_sha256(path) == h, 'Bound source/input/payload changed: ' + path.name)
                self.verified[p] = stamp
        return True


# ================================================================ admission
def _check_fixed(scope):
    from mri_rl import safe_micro_dynamic_resources_v1 as resources
    from mri_rl.safe_micro_watchdog_v2 import POLICY as GUARDIAN_POLICY
    require(scope.get('schema') == ngs.SCHEMA and scope.get('scope_kind') == ngs.KIND and scope.get('action') == ngs.ACTION and
            scope.get('implemented_by') == ngs.IMPLEMENTED_BY, 'Fresh exact native-group-signal VJP scope required')
    require(scope.get('resource_policy') == resources.POLICY and scope.get('external_guardian_policy') == GUARDIAN_POLICY and
            scope.get('run_limits') == ngs.RUN_LIMITS and scope.get('automatic_retry') is False and scope.get('training') is False,
            'Exact conservative resource/guardian/run-limit policy required')
    ngs.require_frozen_tolerances(scope.get('tolerances'))
    budget = ngs.scope_budget(scope['response_jobs_PRIVATE'], scope['margin_jobs_PRIVATE'])
    require(scope.get('budget') == budget and budget['model_loads'] == 1 and budget['generation'] == 0 and budget['optimizer_updates'] == 0 and
            budget['parameter_writes'] == 0 and budget['checkpoint_writes'] == 0, 'Budget differs from the exact job derivation')


def _check_review(scope, review):
    require(review.get('verdict') in ('PASS', 'PASS_WITH_LIMITATIONS') and review.get('scope_sha256') == ngs.digest(scope) and
            review.get('source_sha256') == scope['source_sha256'] and review.get('input_sha256') == scope['input_sha256'] and
            review.get('reviewer_kind') == 'isolated_codex' and review.get('reviewer_id') and review['reviewer_id'] != ngs.IMPLEMENTED_BY and
            'claude' not in str(review['reviewer_id']).lower(), 'Independent exact review by a non-author model family required')


def _check_human(scope, human):
    require(human.get('authorized') is True and human.get('action') == ACTION and human.get('scope_sha256') == ngs.digest(scope) and
            shared.sha(human.get('human_instruction_sha256')) and human.get('approved_by'), 'Fresh explicit human permission for this exact scope required')


def _times(human):
    issued, expires = (datetime.fromisoformat(str(human[k]).replace('Z', '+00:00')) for k in ('issued_at_utc', 'expires_at_utc'))
    require(issued.tzinfo is not None and expires.tzinfo is not None and 0 < (expires - issued).total_seconds() <= MAX_GRANT_SECONDS, 'Bounded permission times required')
    return issued, expires


def authorize(scope, human, review, *, consumed):
    _check_fixed(scope)
    _check_human(scope, human)
    _check_review(scope, review)
    issued, expires = _times(human)
    now = datetime.now(timezone.utc)
    require(issued <= now < expires and (now - issued).total_seconds() <= FRESH_SECONDS, 'Fresh permission (issued <=30 min ago, unexpired) required')
    require(callable(consumed), 'Once-only permission ledger required')
    token = Authorization(_ISSUER, scope, expires)
    token.check('admission')
    consumed(human['human_instruction_sha256'])
    return token


def _authorize_child(scope, human, review, launch, consumed):
    _check_fixed(scope)
    _check_human(scope, human)
    _check_review(scope, review)
    issued, expires = _times(human)
    started = datetime.fromisoformat(launch['started_utc'])
    require(launch['scope_sha256'] == ngs.digest(scope) and launch['human_instruction_sha256'] == human['human_instruction_sha256'] and
            issued <= started and (started - issued).total_seconds() <= FRESH_SECONDS, 'Child lacks the exact fresh launch')
    require(issued <= datetime.now(timezone.utc) < expires, 'Child grant expired')
    token = Authorization(_ISSUER, scope, expires, start=launch['started_monotonic'])
    require(type(token.start) in (int, float) and token.start <= time.monotonic(), 'Invalid launch clock')
    token.check('child_admission')
    consumed(human['human_instruction_sha256'])
    return token


# ================================================================ the one-shot journaled read-only controller (ops injected)
class _Ledger:
    def __init__(self, limits, path):
        self.limits, self.path = dict(limits), Path(path)
        self.used = {k: 0 for k in limits}
        shared.publish_once(self.path.with_name('counter_limits_PRIVATE.json'), {'limits': self.limits})

    def charge(self, positions):
        new = {'forwards': self.used['forwards'] + 1, 'vjps': self.used['vjps'] + 1, 'teacherforced_positions': self.used['teacherforced_positions'] + positions}
        require(all(new[k] <= self.limits[k] for k in new), 'Frozen forward/VJP/position budget exceeded')
        shared.replace_atomic(self.path, {'limits': self.limits, 'used': new})
        self.used = new


def _finite_list(values, n=None, nonpositive=False):
    return (isinstance(values, list) and values and (n is None or len(values) == n) and
            all(type(v) in (int, float) and math.isfinite(v) and (not nonpositive or v <= 0) for v in values))


def clear_unwound_frames(error):
    """Same logic as native_gspo_path_probe_v1.clear_unwound_exception_frames (kept local so CPU tests import no native module):
    drop locals (graphs/contexts/arrays) of COMPLETED frames along the exception chain; executing frames are left alone."""
    import traceback
    pending, seen = [error], set()
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        traceback.clear_frames(current.__traceback__)
        pending.extend((current.__cause__, current.__context__))


def _fsync_file(path):
    with Path(path).open('rb') as handle:
        os.fsync(handle.fileno())


def run_readonly(scope, token, gate, ops):
    require(isinstance(token, Authorization) and token.sha == ngs.digest(scope), 'Exact issued authorization required')
    token.check('readonly_start')
    out = Path(scope['run_output'])
    require(not out.exists(), 'Run output already exists: one-shot, no replay or overwrite')
    durable_mkdir(out)
    vdir = out / 'vectors_PRIVATE'
    durable_mkdir(vdir)
    suffix = getattr(ops, 'vector_suffix', '.npz')
    budget = scope['budget']
    ledger = _Ledger({k: budget[k] for k in ('forwards', 'vjps', 'teacherforced_positions')}, out / 'counters_PRIVATE.json')
    journal = Journal(out / 'journal_PRIVATE', {'scope': ngs.digest(scope), 'input': scope['input_sha256']}, before=gate)
    backend, release, error, status, gate_result, before, construct_attempted = None, None, None, None, None, None, False
    records = []
    try:
        def vjp_op(key, call, job, frozen_positions, validate):
            def operation(opdir):
                token.check(key)
                ledger.charge(frozen_positions)        # spent BEFORE the native call (bounds first; never refunded if the call fails)
                r = call()
                require(r['positions'] == frozen_positions, 'Observed charged positions differ from the frozen budget/job')
                validate(r)
                path = (vdir / f'{key}_vector_PRIVATE{suffix}').resolve()
                h = ops.save_vector(path, r['vector'])
                r['vector'] = None
                _fsync_file(path)                      # durable before the journal commit (Journal.run also fsyncs committed files)
                require(shared.file_sha256(path) == h, 'Saved vector bytes differ from the reported digest')
                rec = {k: v for k, v in r.items() if k not in ('vector',)}
                rec.update(index=job['index'], vector_path=str(path), vector_sha256=h)
                p = Path(opdir) / 'record_PRIVATE.json'
                shared.publish_once(p, rec)
                return {'value': rec, 'files': {str(path): h, str(p.resolve()): shared.file_sha256(p)}}
            return journal.run(key, operation, reservation={'forwards': 1, 'vjps': 1, 'teacherforced_positions': frozen_positions})

        gate('native_model_load')
        gate.claim_model_load()
        construct_attempted = True
        backend = ops.construct(scope, gate)
        loaded = ops.load_parent(backend, scope)
        require(loaded.get('optimizer_step') == ngs.PARENT_STEP, 'Parent Adam52 (step 52) required')
        parent = scope['bindings_PRIVATE'].get('parent')
        if parent is not None:
            ps = parent['common_parent_state']
            require((loaded['policy'], loaded['adam'], loaded['rng']) == (ps['policy'], ps['adam'], ps['rng']), 'Loaded parent differs from the Common32 parent state')
        before = ops.full_identity(backend)
        require(before == loaded, 'Loaded identity differs from the full identity')
        shared.publish_once(out / 'loaded_identity_PRIVATE.json', loaded)
        for job in scope['response_jobs_PRIVATE']:
            def validate(r, job=job):
                require(_finite_list(r['selected_logps'], job['tokens'], nonpositive=True), 'Finite nonpositive selected logps for every saved token required')
            records.append(vjp_op('resp_%02d' % job['index'], lambda job=job: ops.response_vjp(backend, job), job, job['tf_positions'], validate))
        gate_result = ops.reconstruction_gate(scope, records)
        shared.publish_once(out / 'reconstruction_gate_PRIVATE.json', gate_result)
        if gate_result.get('PASS') is True:
            for mjob in scope['margin_jobs_PRIVATE']:
                expected_n = len(mjob['label_ids']) if mjob['objective'] == 'LABEL_MEAN' else 2

                def validate(r, expected_n=expected_n):
                    require(_finite_list(r['label_logps'], expected_n, nonpositive=True),
                            'Finite nonpositive canonical logps of the exact length required (label length for LABEL_MEAN, 2 for FIRST_CONTRAST)')
                vjp_op('margin_%02d' % mjob['index'], lambda mjob=mjob: ops.margin_vjp(backend, mjob), mjob, mjob['positions'], validate)
            status = 'PASS'
        else:
            status = 'PARTIAL_RECONSTRUCTION_FAILED'
        after = ops.full_identity(backend)
        shared.publish_once(out / 'after_identity_PRIVATE.json', after)
        require(after == before, 'Read-only phase changed full parameters/Adam/RNG identity')
    except BaseException as caught:
        error = caught
        import traceback
        failure_text = ''.join(traceback.format_exception(caught))[-4000:]       # durable evidence captured BEFORE clearing frames
        clear_unwound_frames(caught)                                           # tracebacks must not pin graphs/contexts during release
    finally:
        if construct_attempted:                                                # also a PARTIALLY constructed resident (ops keeps its holder)
            target, backend = backend, None
            import gc
            gc.collect()
            try:
                release = ops.release(target, out)
            except BaseException as caught:
                release = {'release_verified': False, 'error_type': type(caught).__name__, 'error': str(caught)[:200]}
                clear_unwound_frames(caught)
            target = None
        if release is not None:
            shared.replace_atomic(out / 'release_record_PRIVATE.json', release)
    try:
        if error is not None:
            shared.publish_once(out / 'failure_PRIVATE.json', {'error_type': type(error).__name__, 'error': str(error)[:400], 'traceback': failure_text,
                                                               'release': release, 'spent_including_indeterminate': dict(ledger.used),
                                                               'automatic_retry': False, 'replay': False, 'status': 'INDETERMINATE_OR_FAILED_NO_REPLAY'})
            raise error
        require(release is not None and release.get('release_verified') is True and int(release.get('active_MLX_bytes', RELEASE_MAX_BYTES)) < RELEASE_MAX_BYTES,
                'Allocator release not verified (<128 MiB required before commit)')
        if status == 'PASS':
            require(ledger.used == ledger.limits, 'Exact frozen forward/VJP/position budget not consumed')
        token.check('complete')
        result = {'status': status, 'counters': dict(ledger.used), 'model_loads': 1, 'generation': 0, 'optimizer_updates': 0, 'parameter_writes': 0,
                  'checkpoint_writes': 0, 'identity_before_equals_after': True, 'release_verified': True, 'release': release,
                  'reconstruction_gate': gate_result, 'C_science': 'EVALUATED_BY_CPU_ANALYSIS' if status == 'PASS' else 'NOT_EVALUATED',
                  'resources': gate.peaks() if hasattr(gate, 'peaks') else None}
        complete = out / 'native_complete_PRIVATE.json'

        def commit(_):
            require(not complete.exists(), 'Uncommitted completion file cannot be adopted')
            shared.publish_once(complete, result)
            return {'value': result, 'files': {str(complete.resolve()): shared.file_sha256(complete)}}
        return journal.run('native_complete', commit)
    finally:
        journal.close()


# ================================================================ production resource gate (unchanged colors; one planned load)
def _resource_gate_class():
    from mri_rl import safe_micro_dynamic_resources_v1 as resources

    class ChildResourceGate(resources.ResourceGate):
        def __init__(self, scope, token):
            attempt = Path(scope['run_output']).parent
            phase = {**scope, 'output': str(attempt / 'resources_PRIVATE'), 'compound_scope_sha256': ngs.digest(scope)}
            super().__init__('B_MICRO', {'runtime_spec': scope['runtime_PRIVATE']['runtime_spec'], 'output': str(attempt)}, phase, token)

        def claim_model_load(self):
            self.authorization.check('model_load_claim')
            path = Path(self.spec['output']) / 'model_load_ledger_PRIVATE.json'
            require(not path.exists(), 'The single planned Parent load was already claimed (no reload or recovery)')
            shared.publish_once(path, {'scope': self.scope['compound_scope_sha256'], 'claimed': 1, 'limit': 1})
    return ChildResourceGate


# ================================================================ CLI: credentials, controller, child
def _credentials(args, *, launch=None, consumed):
    from mri_rl import v2_native_readonly_v1 as nr
    for name in ('scope', 'human', 'review'):
        require(shared.file_sha256(getattr(args, name)) == getattr(args, 'expected_' + name + '_sha256'), 'Exact credential FILE hash changed: ' + name)
    scope = load(args.scope)
    require(scope.get('synthetic') is not True and not scope['bindings_PRIVATE'].get('synthetic_test_fixture'), 'Synthetic scopes are refused by the production CLI')
    with nr.cpu_only_imports():
        prepared = ngs.prepare(**scope['preparation_PRIVATE'])
    require(scope == prepared['executable_scope'], 'Exact current CPU preparation differs from the reviewed scope (source/input drift)')
    human, review = load(args.human), load(args.review)
    if launch is None:
        return authorize(scope, human, review, consumed=consumed), human, scope
    return _authorize_child(scope, human, review, launch, consumed), human, scope


def _verify_child(scope, witness):
    out = Path(scope['run_output'])
    reasons = []
    if witness.get('native_exit_code') != 0:
        reasons.append('nonzero_exit')
    if witness.get('SIGTERM_sent') or witness.get('SIGKILL_after_grace'):
        reasons.append('forced_stop')
    if witness.get('monitor_log_errors'):
        reasons.append('monitor_errors')
    if witness.get('child_reaped') is not True:
        reasons.append('child_not_reaped')
    if (witness.get('post_child_global_lock') or {}).get('no_open_holders') is not True:
        reasons.append('model_lock_not_released')
    complete = out / 'native_complete_PRIVATE.json'
    result = json.loads(complete.read_text()) if complete.is_file() else {}
    if result.get('status') not in ('PASS', 'PARTIAL_RECONSTRUCTION_FAILED') or result.get('release_verified') is not True:
        reasons.append('terminal_not_committed_with_release')
    if (out / 'failure_PRIVATE.json').exists():
        reasons.append('failure_record_present')
    return not reasons, reasons, result.get('status')


def execute_chain(args):
    import fcntl
    import shutil
    import sys
    from dataclasses import asdict
    from mri_rl.accuracy_recovery_v1 import canonical_project_root
    from mri_rl.accuracy_recovery_gpu_cache_v3 import os_pressure_level
    from mri_rl.resource_supervisor import sample_macos
    from mri_rl.safe_micro_watchdog_v2 import witness_child
    from mri_rl.safe_micro_witness_v1 import inspect_lock
    from mri_rl import safe_micro_dynamic_resources_v1 as resources
    from mri_rl import v2_native_readonly_v1 as nr
    root = canonical_project_root()
    python = root / 'work/accuracy-recovery-venv-v1/bin/python'
    require(Path(sys.executable).resolve() == python.resolve(), 'Pinned production interpreter required')
    scope = load(args.scope)
    attempt = Path(scope['run_output']).parent
    token, human, scope = _credentials(args, consumed=shared.consume_once(attempt / 'permission_PRIVATE'))
    require(not Path(scope['run_output']).exists() and not (attempt / 'launch_identity_PRIVATE.json').exists(), 'Fresh attempt required: no relaunch/resume/retry')
    shared.publish_once(attempt / 'launch_identity_PRIVATE.json', {'scope_sha256': token.sha, 'human_instruction_sha256': human['human_instruction_sha256'],
                                                                    'started_utc': datetime.now(timezone.utc).isoformat(), 'started_monotonic': token.start})
    fd = os.open(attempt / 'controller.lock', os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        runtime = load(scope['runtime_PRIVATE']['runtime_spec'])
        controller = resources.Controller()
        good = 0
        while good < 3:                                        # non-resident admission samples (waits count against the window)
            token.check('nonresident_wait')
            s = sample_macos()
            r = controller.decide(s, os_pressure_level(), shutil.disk_usage(attempt).free / 1024 ** 3, time.monotonic())
            shared.replace_atomic(attempt / 'nonresident_latest_PRIVATE.json', {'sample': asdict(s), **r})
            if r['action'] == 'STOP':
                raise resources.ResourceFatal(','.join(r['reasons']))
            good = good + 1 if r['action'] == 'CONTINUE' else 0
            if good < 3:
                time.sleep(resources.POLICY['parent_wait_seconds'])
        argv = [str(python), '-I', str(Path(__file__).resolve().parents[2] / 'scripts/run_native_group_signal_vjp_v1.py'), '--child']
        for name in ('scope', 'human', 'review'):
            argv += ['--' + name, str(getattr(args, name)), '--expected-' + name + '-sha256', getattr(args, 'expected_' + name + '_sha256')]
        witness = witness_child(argv, attempt / 'child_exit_PRIVATE', scope_sha256=args.expected_scope_sha256, sampler=lambda pid: sample_macos(pid=pid),
                                level=os_pressure_level, disk_free_gib=lambda: shutil.disk_usage(attempt).free / 1024 ** 3,
                                post_sampler=lambda: asdict(sample_macos()), lock_inspector=lambda: inspect_lock(runtime['global_lock']))
        ok, reasons, status = _verify_child(scope, witness)
        if not ok:
            shared.publish_once(attempt / 'stopped_PRIVATE.json', {'status': 'STOPPED_NO_RETRY', 'reasons': reasons, 'automatic_recovery': False})
            print(json.dumps({'status': 'STOPPED_NO_RETRY', 'reasons': reasons}))
            return 2
        with nr.cpu_only_imports():
            analysis = ngs.analyze(scope)
        print(json.dumps({'status': status, 'decision': analysis['decision']}))
        return 0
    finally:
        os.close(fd)


def execute_child(args):
    import fcntl
    import signal
    from mri_rl.safe_micro_dynamic_resources_v1 import ResourcePause
    from mri_rl.safe_micro_journal_v1 import Indeterminate
    scope = load(args.scope)
    attempt = Path(scope['run_output']).parent
    launch = load(attempt / 'launch_identity_PRIVATE.json')
    require(not Path(scope['run_output']).exists(), 'Run already attempted: no replay')
    token, _, scope = _credentials(args, launch=launch, consumed=shared.consume_once(attempt / 'child_permission_PRIVATE'))
    from mri_rl.accuracy_recovery_gpu_cache_v2 import installed_fingerprint
    require(installed_fingerprint() == load(load(scope['runtime_PRIVATE']['recovery_spec'])['cache_scope'])['installed'], 'Pinned native runtime differs')
    runtime = load(scope['runtime_PRIVATE']['runtime_spec'])
    lockpath = Path(runtime['global_lock'])
    require(lockpath.is_file() and not lockpath.is_symlink(), 'Existing global model lock required')
    fd = os.open(lockpath, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        gate = _resource_gate_class()(scope, token)
        signal.signal(signal.SIGUSR1, lambda *unused: setattr(gate, 'red_requested', True))
        gate('native_model_load_preflight')
        from types import SimpleNamespace
        from mri_rl.continuous_rl_pilot import cache_binding_for
        from mri_rl.safe_micro_recovery_entry_v1 import SHARDS
        binding = cache_binding_for(SimpleNamespace(cache_dir=Path(runtime['cache_dir']), cache_certificate=Path(runtime['cache_certificate']),
                                                    cache_certificate_sha256=shared.file_sha256(runtime['cache_certificate']), model=Path(runtime['model'])))
        require({n: v['sha256'] for n, v in binding['shards'].items()} == SHARDS, 'Final cache payload mismatch (no quantization/recovery permitted)')
        token.check('before_native_import')
        from mri_rl.native_group_signal_vjp_backend_v1 import NativeOps
        result = run_readonly(scope, token, gate, NativeOps())
        return 0 if result['release_verified'] is True else 2
    except ResourcePause as error:
        shared.publish_once(attempt / 'pause_PRIVATE.json', {'status': 'PAUSED_NO_RESTART', 'reason': str(error)})
        return 75
    except Indeterminate as error:
        shared.publish_once(attempt / 'indeterminate_PRIVATE.json', {'status': 'INDETERMINATE_NO_REPLAY', 'reason': str(error)})
        return 76
    finally:
        os.close(fd)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--launch', action='store_true')
    mode.add_argument('--child', action='store_true')
    mode.add_argument('--analyze', action='store_true')
    p.add_argument('--root', type=Path)
    p.add_argument('--worktree', type=Path)
    p.add_argument('--out', type=Path, help='NEW attempt directory under ROOT/local_data/native_group_signal_vjp_v1/ (prepare)')
    p.add_argument('--run', type=Path)
    p.add_argument('--results', type=Path)
    for name in ('scope', 'human', 'review'):
        p.add_argument('--' + name, type=Path)
        p.add_argument('--expected-' + name + '-sha256')
    args = p.parse_args(argv)
    if args.prepare:
        from mri_rl.accuracy_recovery_v1 import canonical_project_root
        worktree = (args.worktree or Path(__file__).resolve().parents[2]).resolve()
        root = (args.root or canonical_project_root()).resolve()
        require(args.out is not None, '--out NEW attempt directory required')
        out = args.out.resolve()
        require(not out.exists(), 'Attempt directory exists: every preparation is a NEW attempt (no overwrite of evidence)')
        result = ngs.prepare(root=root, worktree=worktree, out_dir=out, run_dir=args.run, results_path=args.results)
        durable_mkdir(out)
        shared.publish_once(out / 'scope_PRIVATE.json', result['executable_scope'])
        shared.publish_once(out / 'prepare_summary_SAFE.json', {'scope_sha256': result['scope_sha256'], **result['public_summary']})
        print(json.dumps({'scope_sha256': result['scope_sha256'], 'scope_file_sha256': shared.file_sha256(out / 'scope_PRIVATE.json'),
                          'summary': result['public_summary']}, sort_keys=True))
        return 0
    require(args.scope and args.expected_scope_sha256, 'Exact scope file and SHA-256 required')
    if args.analyze:
        from mri_rl import v2_native_readonly_v1 as nr
        require(shared.file_sha256(args.scope) == args.expected_scope_sha256, 'Scope file hash changed')
        with nr.cpu_only_imports():
            result = ngs.analyze(load(args.scope))
        print(json.dumps({'decision': result['decision']}))
        return 0
    require(all(getattr(args, n) and getattr(args, 'expected_' + n + '_sha256') for n in ('scope', 'human', 'review')), 'Exact credential artifacts required')
    return execute_child(args) if args.child else execute_chain(args)
