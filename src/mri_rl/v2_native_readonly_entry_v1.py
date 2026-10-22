"""V2_NATIVE_READONLY_V1 exact authority: CPU prepare / serial controller (A53 then B53) / one fresh child per checkpoint.

Author: Claude Code (claude-opus-5-5). Importing this module imports no MLX and loads nothing.

Authority is never created by a flag or a JSON written by this code: execution requires (1) the exact current CPU preparation,
(2) an independent exact review by a non-Claude reviewer naming the scope/source/input hashes, and (3) a fresh, bounded human
permission file that the ROOT records only after the researcher's reply, naming both checkpoints; each instruction is consumed
once. Repository checks are not human identity authentication. Resource colors/thresholds are the unchanged
safe_micro_dynamic_resources_v1.POLICY; the external guardian is the existing safe_micro_watchdog_v2 (own child only).
"""
from __future__ import annotations
import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from mri_rl import v2_native_readonly_v1 as nr
from mri_rl import shared_rollout_gspo_core_v1 as shared
from mri_rl import safe_micro_dynamic_resources_v1 as resources

require = shared.require
MODEL_LOADS_MAX = 2
_ISSUER = object()
KEY_PHASES = ('admission', 'model_load', 'complete', 'restore', 'start')


def load(path):
    require(not any(x in str(path).lower() for x in ('final100', 'holdout16', 'new16')), 'Forbidden development input')
    return json.loads(Path(path).read_text())


def _bound(scope):
    return {**scope['source_sha256'], **scope['file_sha256'], **scope['payload_sha256']}


class Authorization:
    def __init__(self, issuer, scope, expires, parent=None):
        require(issuer is _ISSUER, 'Only exact admission may issue authorization')
        self.scope = json.loads(json.dumps(scope)); self.sha = nr.digest(scope); self.expires = expires; self.parent = parent
        self.start = time.monotonic() if parent is None else parent.start; self.verified = {}

    def check(self, phase):
        require(datetime.now(timezone.utc) < self.expires, 'Human permission expired')
        require(time.monotonic() - self.start < nr.RUN_LIMITS['total_window_seconds'], 'Total operation window exhausted (waits count)')
        require(nr.digest(self.scope) == self.sha, 'Exact authorization scope changed')
        for p, h in _bound(self.scope).items():
            path = Path(p); require(path.is_file() and not path.is_symlink(), 'Bound file missing')
            s = path.stat(); stamp = (s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_ino, s.st_dev)
            if self.verified.get(p) != stamp or any(x in phase for x in KEY_PHASES):
                require(shared.file_sha256(path) == h, 'Bound source/input/payload changed'); self.verified[p] = stamp
        if self.parent is not None:
            clock = load(Path(self.scope['run_output']) / (self.scope['checkpoint'] + '_clock_PRIVATE.json'))
            require(time.monotonic() - clock['started_monotonic'] < nr.RUN_LIMITS['per_checkpoint_window_seconds'], 'Per-checkpoint window exhausted (waits count)')
        return True

    def for_checkpoint(self, label):
        require(self.parent is None and label in nr.CHECKPOINTS, 'Only A53/B53 derive from the compound authority')
        self.check('checkpoint_admission')
        return Authorization(_ISSUER, phase_scope(self.scope, label), self.expires, self)


def phase_scope(scope, label):
    require(label in nr.CHECKPOINTS, 'Unknown checkpoint')
    run = Path(scope['run_output'])
    return {**scope, 'checkpoint': label, 'compound_scope_sha256': nr.digest(scope), 'resource_stage': 'B_MICRO',
            'output': str(run / (label + '_resources_PRIVATE')), 'checkpoint_output': str(run / (label + '_PRIVATE'))}


def _check_fixed(scope):
    require(scope.get('schema') == nr.SCHEMA and scope.get('scope_kind') == nr.KIND and scope.get('action') == nr.ACTION and
            scope.get('implemented_by') == nr.IMPLEMENTED_BY, 'Fresh exact read-only scope required')
    require(scope.get('resource_policy') == resources.POLICY and scope.get('external_guardian_policy') == nr.GUARDIAN_POLICY and
            scope.get('run_limits') == nr.RUN_LIMITS and scope.get('totals') == nr.TOTALS and scope.get('automatic_retry') is False and
            scope.get('training') is False, 'Exact conservative policy/limits required')
    require(set(scope['checkpoints_PRIVATE']) == set(nr.CHECKPOINTS) and len(scope['dev_jobs_PRIVATE']) == nr.DEV_PER_CHECKPOINT and
            scope.get('budget') == nr.budget_for(scope['tf_jobs_PRIVATE'], scope['lengths']), 'Budget/jobs differ from the exact derivation')


def _check_review(scope, review):
    require(review.get('verdict') in ('PASS', 'PASS_WITH_LIMITATIONS') and review.get('scope_sha256') == nr.digest(scope) and
            review.get('source_sha256') == scope['source_sha256'] and review.get('input_sha256') == scope['input_sha256'] and
            review.get('reviewer_kind') == 'isolated_codex' and review.get('reviewer_id') and review['reviewer_id'] != nr.IMPLEMENTED_BY and
            'claude' not in str(review['reviewer_id']).lower(), 'Independent exact review by a non-author model family required')


def _check_human(scope, human):
    require(human.get('authorized') is True and human.get('action') == nr.ACTION and human.get('scope_sha256') == nr.digest(scope) and
            human.get('included_checkpoints') == list(nr.CHECKPOINTS) and shared.sha(human.get('human_instruction_sha256')) and human.get('approved_by'),
            'Fresh explicit human permission naming A53 and B53 required')


def _times(human):
    issued, expires = (datetime.fromisoformat(human[k].replace('Z', '+00:00')) for k in ('issued_at_utc', 'expires_at_utc'))
    require(issued.tzinfo is not None and expires.tzinfo is not None and 0 < (expires - issued).total_seconds() <= 86400, 'Bounded permission times required')
    return issued, expires


def authorize(scope, human, review, *, consumed):
    _check_fixed(scope); _check_human(scope, human); _check_review(scope, review)
    issued, expires = _times(human); now = datetime.now(timezone.utc)
    require(issued <= now < expires and (now - issued).total_seconds() <= 1800, 'Fresh permission (issued <=30 min ago, unexpired) required')
    token = Authorization(_ISSUER, scope, expires); token.check('admission'); require(callable(consumed), 'Once-only ledger required')
    consumed(human['human_instruction_sha256'])
    return token


def _authorize_child(scope, human, review, launch):
    _check_fixed(scope); _check_human(scope, human); _check_review(scope, review)
    issued, expires = _times(human); started = datetime.fromisoformat(launch['started_utc'])
    require(launch['scope_sha256'] == nr.digest(scope) and launch['human_instruction_sha256'] == human['human_instruction_sha256'] and
            issued <= started and (started - issued).total_seconds() <= 1800, 'Child lacks the exact fresh compound launch')
    require(issued <= datetime.now(timezone.utc) < expires, 'Child grant expired')
    token = Authorization(_ISSUER, scope, expires); token.start = launch['started_monotonic']
    require(type(token.start) in (int, float) and token.start <= time.monotonic(), 'Invalid compound clock'); token.check('child_admission')
    return token


def _credentials(args, *, launch_record=None, consumed=None):
    for name in ('scope', 'human', 'review'):
        require(shared.file_sha256(getattr(args, name)) == getattr(args, 'expected_' + name + '_sha256'), 'Exact credential FILE hash changed: ' + name)
    scope = load(args.scope)
    prepared = nr.prepare(**scope['preparation_PRIVATE'])
    require(scope == prepared['executable_scope'], 'Exact current CPU preparation differs from the reviewed scope')
    human, review = load(args.human), load(args.review)
    if launch_record is None:
        return authorize(scope, human, review, consumed=consumed), human
    token = _authorize_child(scope, human, review, launch_record)
    require(consumed is not None, 'Child once-only ledger required'); consumed(human['human_instruction_sha256'])
    return token, human


# ================================================================ resource gate (unchanged colors; 2-load ledger)
class ReadonlyResourceGate(resources.ResourceGate):
    def __init__(self, scope, authorization):
        runtime = scope['runtime_PRIVATE']
        super().__init__('B_MICRO', {'runtime_spec': runtime['runtime_spec'], 'output': scope['run_output']}, scope, authorization)

    def claim_model_load(self):
        self.authorization.check('readonly_model_load_claim')
        path = Path(self.scope['run_output']) / 'model_load_ledger_PRIVATE.json'
        rec = load(path) if path.exists() else {'scope': self.scope['compound_scope_sha256'], 'claimed': [], 'limit': MODEL_LOADS_MAX}
        label = self.scope['checkpoint']
        require(rec['scope'] == self.scope['compound_scope_sha256'] and rec['limit'] == MODEL_LOADS_MAX and len(rec['claimed']) < MODEL_LOADS_MAX and
                label not in rec['claimed'], 'Model-load budget exhausted/changed or this checkpoint was already loaded once')
        rec['claimed'] = rec['claimed'] + [label]; shared.replace_atomic(path, rec)


# ================================================================ serial controller (no model, no MLX)
def _verify_child(base, label, witness):
    out = Path(base) / (label + '_PRIVATE')
    reasons = []
    if witness.get('native_exit_code') != 0: reasons.append('nonzero_exit')
    if witness.get('SIGTERM_sent') or witness.get('SIGKILL_after_grace'): reasons.append('forced_stop')
    if witness.get('monitor_log_errors'): reasons.append('monitor_errors')
    if witness.get('child_reaped') is not True: reasons.append('child_not_reaped')
    if (witness.get('post_child_global_lock') or {}).get('no_open_holders') is not True: reasons.append('model_lock_not_released')
    complete = out / 'native_complete_PRIVATE.json'
    result = load(complete) if complete.is_file() else {}
    if result.get('status') != 'PASS' or result.get('release_verified') is not True: reasons.append('terminal_not_PASS_released')
    if list(out.rglob('release_failure_PRIVATE.json')): reasons.append('allocator_release_failure')
    releases = list(out.rglob('release_SAFE.json'))
    if not releases or not all(load(p).get('release_verified') is True for p in releases): reasons.append('allocator_release_not_verified')
    return not reasons, reasons


def _wait_nonresident(observe, *, sleep, clock, started, total_started):
    good = 0
    while True:
        now = clock()
        require(now - started < nr.RUN_LIMITS['per_checkpoint_window_seconds'] and now - total_started < nr.RUN_LIMITS['total_window_seconds'],
                'Non-resident wait exhausted the checkpoint/total window (waits count; no reset)')
        result = observe()
        if result['action'] == 'STOP':
            raise resources.ResourceFatal(','.join(result['reasons']))
        good = good + 1 if result['action'] == 'CONTINUE' else 0
        if good >= nr.RUN_LIMITS['wait_samples_before_launch']:
            return
        sleep(nr.RUN_LIMITS['nonresident_wait_poll_seconds'])


def run_serial(scope, *, launch, observe, analyze, sleep=time.sleep, clock=time.monotonic):
    """A53 then B53, each at most once, each a fresh child. Any non-PASS stops everything: no retry, no B, no recovery."""
    base = Path(scope['run_output']); require(base.is_dir(), 'Run output required')
    control = base / 'control_PRIVATE.json'
    require(not control.exists(), 'Controller already launched on this output: no relaunch/resume/retry')
    total_started = clock()
    shared.publish_once(control, {'launched_once': True, 'order': list(nr.CHECKPOINTS), 'automatic_retry': False})
    for n, label in enumerate(nr.CHECKPOINTS):
        if label == 'B53':
            require((base / 'A53_verified_PRIVATE.json').is_file(), 'B53 requires a verified A53 child')
        shared.publish_once(base / (label + '_clock_PRIVATE.json'), {'started_monotonic': time.monotonic()})
        _wait_nonresident(observe, sleep=sleep, clock=clock, started=clock(), total_started=total_started)
        witness = launch(label)
        ok, reasons = _verify_child(base, label, witness)
        if not ok:
            stop = {'status': 'STOPPED_NO_RETRY', 'failed': label, 'reasons': reasons, 'blocked': list(nr.CHECKPOINTS[n + 1:]), 'automatic_recovery': False}
            shared.publish_once(base / 'stopped_PRIVATE.json', stop)
            return stop
        shared.publish_once(base / (label + '_verified_PRIVATE.json'), {'checkpoint': label, 'reasons': [], 'witness_exit_code': 0})
    analysis = analyze()
    result = {'status': 'COMPLETE', 'checkpoints': list(nr.CHECKPOINTS), 'analysis_written': analysis is not None, 'automatic_training': False}
    shared.publish_once(base / 'compound_complete_PRIVATE.json', result)
    return result


def execute_chain(args):
    import fcntl
    import shutil
    import sys
    from dataclasses import asdict
    from mri_rl.accuracy_recovery_v1 import canonical_project_root
    from mri_rl.accuracy_recovery_gpu_cache_v3 import os_pressure_level
    from mri_rl.resource_supervisor import sample_macos
    from mri_rl.safe_micro_journal_v1 import durable_mkdir
    from mri_rl.safe_micro_watchdog_v2 import witness_child
    from mri_rl.safe_micro_witness_v1 import inspect_lock
    scope = load(args.scope); base = Path(scope['run_output']); root = canonical_project_root()
    python = root / 'work/accuracy-recovery-venv-v1/bin/python'
    require(Path(sys.executable).resolve() == python.resolve(), 'Pinned production interpreter required')
    with nr.cpu_only_imports():
        token, human = _credentials(args, consumed=shared.consume_once(base.with_name(base.name + '_permission_PRIVATE')))
    require(not base.exists(), 'Fresh run output required'); durable_mkdir(base)
    shared.publish_once(base / 'launch_identity_PRIVATE.json', {'scope_sha256': token.sha, 'human_instruction_sha256': human['human_instruction_sha256'],
                                                                'started_utc': datetime.now(timezone.utc).isoformat(), 'started_monotonic': token.start})
    fd = os.open(base / 'controller.lock', os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        runtime = load(scope['runtime_PRIVATE']['runtime_spec']); controller = resources.Controller()
        def observe():
            token.check('nonresident_wait')
            s = sample_macos(); r = controller.decide(s, os_pressure_level(), shutil.disk_usage(base).free / 1024 ** 3, time.monotonic())
            shared.replace_atomic(base / 'nonresident_latest_PRIVATE.json', {'sample': asdict(s), **r}); return r
        def launch(label):
            token.check('launch_' + label)
            argv = [str(python), '-I', str(Path(__file__).resolve().parents[2] / 'scripts/run_v2_native_readonly_v1.py'), '--child', label]
            for name in ('scope', 'human', 'review'):
                argv += ['--' + name, str(getattr(args, name)), '--expected-' + name + '-sha256', getattr(args, 'expected_' + name + '_sha256')]
            return witness_child(argv, base / (label + '_exit_PRIVATE'), scope_sha256=args.expected_scope_sha256, sampler=lambda pid: sample_macos(pid=pid),
                                 level=os_pressure_level, disk_free_gib=lambda: shutil.disk_usage(base).free / 1024 ** 3,
                                 post_sampler=lambda: asdict(sample_macos()), lock_inspector=lambda: inspect_lock(runtime['global_lock']))
        def analyze():
            with nr.cpu_only_imports():
                return nr.run_analysis(scope)
        result = run_serial(scope, launch=launch, observe=observe, analyze=analyze)
        print(json.dumps({k: v for k, v in result.items() if k in ('status', 'failed', 'reasons', 'blocked')}))
        return 0 if result['status'] == 'COMPLETE' else 2
    finally:
        os.close(fd)


def execute_child(args):
    import fcntl
    import signal
    from mri_rl.safe_micro_dynamic_resources_v1 import ResourcePause
    from mri_rl.safe_micro_journal_v1 import Indeterminate
    label = args.child; require(label in nr.CHECKPOINTS, 'Unknown checkpoint')
    scope = load(args.scope); base = Path(scope['run_output']); launch = load(base / 'launch_identity_PRIVATE.json')
    require(not (base / (label + '_PRIVATE')).exists() and not (base / (label + '_verified_PRIVATE.json')).exists(), 'Checkpoint already attempted: no replay')
    if label == 'B53':
        require((base / 'A53_verified_PRIVATE.json').is_file(), 'B53 requires a verified A53 child and release')
    with nr.cpu_only_imports():
        token, human = _credentials(args, launch_record=launch, consumed=shared.consume_once(base / (label + '_child_permission_PRIVATE')))
    phase = token.for_checkpoint(label)
    from mri_rl.accuracy_recovery_gpu_cache_v2 import installed_fingerprint
    require(installed_fingerprint() == load(load(scope['runtime_PRIVATE']['recovery_spec'])['cache_scope'])['installed'], 'Pinned native runtime differs')
    runtime = load(scope['runtime_PRIVATE']['runtime_spec']); lockpath = Path(runtime['global_lock'])
    require(lockpath.is_file() and not lockpath.is_symlink(), 'Existing global model lock required')
    fd = os.open(lockpath, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        gate = ReadonlyResourceGate(phase.scope, phase)
        signal.signal(signal.SIGUSR1, lambda *unused: setattr(gate, 'red_requested', True))
        gate('native_model_load_preflight')
        from types import SimpleNamespace
        from mri_rl.continuous_rl_pilot import cache_binding_for
        from mri_rl.safe_micro_recovery_entry_v1 import SHARDS
        binding = cache_binding_for(SimpleNamespace(cache_dir=Path(runtime['cache_dir']), cache_certificate=Path(runtime['cache_certificate']),
                                                    cache_certificate_sha256=shared.file_sha256(runtime['cache_certificate']), model=Path(runtime['model'])))
        require({n: v['sha256'] for n, v in binding['shards'].items()} == SHARDS, 'Final cache payload mismatch (no quantization/recovery permitted)')
        phase.check('before_native_import')
        from mri_rl.v2_native_readonly_backend_v1 import run_checkpoint, NativeOps
        result = run_checkpoint(label, scope=phase.scope, authorization=phase, resource_gate=gate, ops=NativeOps())
        return 0 if result['status'] == 'PASS' and result['release_verified'] else 2
    except ResourcePause as error:
        shared.publish_once(base / (label + '_pause_PRIVATE.json'), {'status': 'PAUSED_NO_RESTART', 'reason': str(error), 'checkpoint': label})
        return 75
    except Indeterminate as error:
        shared.publish_once(base / (label + '_indeterminate_PRIVATE.json'), {'status': 'INDETERMINATE_NO_REPLAY', 'reason': str(error)})
        return 76
    finally:
        os.close(fd)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path); p.add_argument('--worktree', type=Path); p.add_argument('--scope-output', type=Path)
    mode = p.add_mutually_exclusive_group(); mode.add_argument('--launch', action='store_true'); mode.add_argument('--child', choices=nr.CHECKPOINTS)
    for name in ('scope', 'human', 'review'):
        p.add_argument('--' + name, type=Path); p.add_argument('--expected-' + name + '-sha256')
    args = p.parse_args(argv)
    if args.launch or args.child:
        require(all(getattr(args, n) and getattr(args, 'expected_' + n + '_sha256') for n in ('scope', 'human', 'review')), 'Exact credential artifacts required')
        return execute_child(args) if args.child else execute_chain(args)
    worktree = (args.worktree or Path(__file__).resolve().parents[2]).resolve()
    from mri_rl.accuracy_recovery_v1 import canonical_project_root
    root = (args.root or canonical_project_root()).resolve()
    result = nr.prepare(root=root, worktree=worktree)
    if args.scope_output:
        target = Path(args.scope_output).resolve()
        require(target.parent == Path(result['executable_scope']['preparation_PRIVATE']['out_dir']), 'Scope must be written into the private preparation directory')
        target.parent.mkdir(parents=True, exist_ok=True)
        shared.publish_once(target, result['executable_scope'])
    print(json.dumps({k: v for k, v in result.items() if k != 'executable_scope'}, sort_keys=True))
    return 0
