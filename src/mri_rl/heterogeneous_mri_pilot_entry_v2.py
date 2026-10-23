"""Heterogeneous MRI pilot V2 exact-scope authority: CPU prepare, then train -> (parent_eval -> child_eval only after a committed
update) as serial fresh child processes, then CPU analysis. Preparation imports no MLX and makes no model call.

Inherited unchanged from the v1 entry: executed Accuracy-Pilot-V2 scope re-derivation (transitive closure, CPU binding only),
journal-authenticated V2 EP Parent DEV24 rows, OSF image/border-control binding, census oracle, symlink/path checks, historical
grant ledger, bounded non-resident waits and the pre-native installed-runtime/cache-certificate/SHARDS check. New: the V2 midline
rows/suffix/measurement (with a visible-prompt leak check), the v1 executed-run proof binding (gate record, analysis, scope,
CPU diagnosis), 3 phases / max 3 loads, fresh V2 action/schema/spec/output root and a fresh exact review + human grant (never a v1
grant). Result: PREPARED_PENDING_EXACT_REVIEW_AND_FRESH_HUMAN at most; it never contains an approval.
"""
from __future__ import annotations
import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from mri_rl import heterogeneous_mri_pilot_v1 as v1
from mri_rl import heterogeneous_mri_pilot_v2 as h2
from mri_rl import heterogeneous_mri_pilot_entry_v1 as v1e
from mri_rl import accuracy_recovery_v1 as a
from mri_rl import accuracy_pilot_v2_core as v2core
from mri_rl import multistep_gspo_core_v1 as ms
from mri_rl import safe_micro_dynamic_resources_v1 as resources
from mri_rl.safe_micro_entry_v1 import _closure
from mri_rl.safe_micro_watchdog_v2 import POLICY as GUARDIAN_POLICY
from mri_rl.shared_rollout_gspo_core_v1 import publish_once, consume_once, identity

ROOT, CANONICAL = v1e.ROOT, v1e.CANONICAL
PILOT_DIR = CANONICAL / 'local_data/heterogeneous_mri_pilot_v2'
V1_DIR = v1e.PILOT_DIR
V2_SPEC, V2_SPEC_SHA256, V2_EXECUTED_SCOPE = v1e.V2_SPEC, v1e.V2_SPEC_SHA256, v1e.V2_EXECUTED_SCOPE
SCHEMA, SPEC_SCHEMA = 'heterogeneous-mri-pilot-v2-scope', 'heterogeneous-mri-pilot-v2-spec'
MEASUREMENT_SCHEMA = 'heterogeneous-midline-prompt-length-measurement-v2'
KIND = 'FRESH_EXACT_HETEROGENEOUS_MRI_PILOT_V2_SCOPE'
ACTION = 'heterogeneous_mri_pilot_v2_controls6_formatgate_rollout16_classcontrast_one_joint_update_then_parentEVAL2_childDEV24_EVAL2_controls6_once'
IMPLEMENTED_BY = v1e.IMPLEMENTED_BY
PHASES = h2.PHASES
MODEL_POLICY = dict(resources.POLICY)
PINNED = dict(v1e.PINNED)
SPEC_FILES = ('midline_manifest', 'source_qa', 'osf_receipt', 'control_inputs', 'independent_census',
              'v1_execution_scope', 'v1_gate_record', 'v1_analysis', 'v1_format_diagnosis')
V2_CONCISE_SEQUENCE_LENGTH, DATA_SCALE = v1e.V2_CONCISE_SEQUENCE_LENGTH, v1e.DATA_SCALE
# Unchanged V2 cost model (.25 s/token, +3 s/call, 90 s load+release, 5 s/old forward, 10 s/backward, 15 s/save). NOT a measured guarantee.
_CALL = 256 * .25 + 3
WORST_CASE_SECONDS = {'train': 90 + 6 * _CALL + 16 * _CALL + 16 * 5 + 16 * 10 + 15, 'parent_eval': 90 + 2 * _CALL, 'child_eval': 90 + 32 * _CALL}
WINDOWS = {'train': 2400, 'parent_eval': 900, 'child_eval': 3000}
RUN_LIMITS = {'model_loads_max': 3, 'max_attempts_per_phase': 1, 'automatic_recoveries': 0, 'incomplete_operation': 'INDETERMINATE_NO_REPLAY',
              'single_resident_model': True, 'global_window_seconds': 7200, 'nonresident_good_samples': v1e.RUN_LIMITS['nonresident_good_samples'],
              'nonresident_sample_seconds': v1e.RUN_LIMITS['nonresident_sample_seconds'], 'waits_count_against_windows': True,
              'windows_basis': 'derived from the unchanged V2 cost model + margin; not a measured runtime guarantee'}
require, H, digest = a.require, a.file_sha256, a.scope_digest
load, _inside = v1e.load, v1e._inside
measured_rows, rows_identity = v1e.measured_rows, v1e.rows_identity
_ISSUER = object()


def next_phases(train_status):
    """Phase 3 (Parent EVAL2 then child) exists ONLY after a committed update."""
    require(train_status in ('UPDATED', 'NO_GSPO_SIGNAL', 'FORMAT_BLOCKED'), 'Unknown train status')
    return ['parent_eval', 'child_eval'] if train_status == 'UPDATED' else []


# ================================================================ CPU preparation helpers
def midline_records(manifest):
    require(manifest.get('schema') == 'vqarad_midline_source_candidate_v1' and manifest.get('training_authorized') is False, 'Exact candidate manifest required')
    rows = [h2.midline_row(r) for r in manifest['rows'] if r['split'] in ('train', 'eval')]
    by = {s: [r for r in rows if r['extra_info']['split'] == s] for s in ('train', 'eval')}
    for s in by:
        require(sorted(r['reward_model']['ground_truth'] for r in by[s]) == ['no', 'yes'], 'Existing 2 TRAIN + 2 EVAL label-balanced split required')
    require(not {r['extra_info']['case_proxy_sha256'] for r in by['train']} & {r['extra_info']['case_proxy_sha256'] for r in by['eval']}, 'Case-proxy overlap')
    return by['train'], by['eval']


def prompt_leaks(prompt, row):
    """Identifiers that must never be visible to the model: row id, image path/name/stem, image/case-proxy hashes."""
    image = Path(row['images'][0])
    forbidden = {'row_id': row['id'], 'image_path': str(image), 'file_name': image.name, 'file_stem': image.stem,
                 'image_sha256': row.get('image_sha256'), 'case_proxy_sha256': row['extra_info'].get('case_proxy_sha256')}
    return sorted(k for k, v in forbidden.items() if v and v in prompt)


def measured_lengths(path, train, gate, evaluation):
    rows = measured_rows(train, gate, evaluation)
    want = {'midline_train': {r['id']: r['id'] for r in train}, 'midline_gate': {c['name']: c['row']['id'] for c in gate},
            'midline_eval': {r['id']: r['id'] for r in evaluation}}
    if not path:
        return {'sequence': V2_CONCISE_SEQUENCE_LENGTH, 'by_row': {i: None for i in rows}, **{k: {n: None for n in v} for k, v in want.items()}}, 'PENDING_PROCESSOR_MEASUREMENT'
    m = load(path)
    require(m.get('schema') == MEASUREMENT_SCHEMA and m.get('model_calls') == 0 and m.get('mlx_imported') is False and m.get('visible_prompt_leaks') == 0,
            'Exact leak-free processor-only V2 measurement required')
    require(m['rows_identity'] == rows_identity(rows) and m['midline_suffix_sha256'] == h2.MIDLINE_SUFFIX_SHA256, 'Measurement is not bound to the V2 rows/suffix')
    require(set(m['by_row']) == set(rows) and all(type(v) is int and 0 < v <= 2048 for v in m['by_row'].values()), 'Incomplete measurement')
    require(m['sequence_concise_check'] == {'expected': V2_CONCISE_SEQUENCE_LENGTH, 'actual': [V2_CONCISE_SEQUENCE_LENGTH] * 2}, 'Sequence method check failed')
    out = {'sequence': V2_CONCISE_SEQUENCE_LENGTH, 'by_row': dict(m['by_row'])}
    out.update({k: {n: m['by_row'][rid] for n, rid in v.items()} for k, v in want.items()})
    return out, 'MEASURED'


def v1_proof(spec):
    """Authenticate the executed v1 run ONCE (CPU): its scope, gate record, analysis and the CPU format diagnosis."""
    gate, analysis, diag = load(spec['v1_gate_record']), load(spec['v1_analysis']), load(spec['v1_format_diagnosis'])
    require(analysis['outcome'] == 'NO_SIGNAL' and analysis['gate_reason'] == 'INVALID_CONTROL' and gate['gate']['reason'] == 'INVALID_CONTROL', 'v1 outcome differs')
    require(diag['schema'] == 'heterogeneous-mri-pilot-v1-format-diagnosis-v1' and diag['model_calls'] == 0 and
            diag['inputs_sha256']['gate_record'] == spec['v1_gate_record_sha256'], 'v1 CPU diagnosis is not bound to the v1 gate record')
    return {'v1_outcome': analysis['outcome'], 'v1_gate_reason': analysis['gate_reason'], 'v1_controls_format': [c['parser_format'] for c in diag['controls']],
            'v1_controls_boxed': [c['boxed_count'] for c in diag['controls']], 'v1_prompts_reproduced': all(c['prompt_sha256_matches_saved'] for c in diag['controls'])}


def prepare(specpath, *, allow_started=False):
    from mri_rl import accuracy_pilot_v2_entry as v2entry
    from mri_rl.accuracy_pilot_v2_native import stage_output, load_decision
    specpath = _inside(specpath, PILOT_DIR)
    spec = load(specpath); require(spec.get('schema') == SPEC_SCHEMA, 'Exact V2 pilot spec schema required')
    require(Path(spec['v2_spec']).resolve() == V2_SPEC.resolve() and H(V2_SPEC) == V2_SPEC_SHA256 == spec['v2_spec_sha256'], 'Must bind the executed V2 spec')
    files = {str(specpath): H(specpath), str(V2_SPEC.resolve()): V2_SPEC_SHA256}
    for key in SPEC_FILES:
        h = H(spec[key]); require(h == spec[key + '_sha256'] and (key not in PINNED or h == PINNED[key]), 'Bound input changed: ' + key)
        files[str(Path(spec[key]).resolve())] = h
    proof = v1_proof(spec)
    v2scope = v2entry.prepare(V2_SPEC, allow_started=True)['executable_scope']
    require(v2scope == load(V2_EXECUTED_SCOPE), 'Executed V2 scope no longer re-derives exactly')
    files[str(V2_EXECUTED_SCOPE.resolve())] = H(V2_EXECUTED_SCOPE)
    for p, h in v2scope['file_sha256'].items():
        require(files.get(p, h) == h, 'Conflicting inherited input'); files[p] = h
    v2 = load(V2_SPEC)
    require(v2['concise_sequence_prompt_length'] == V2_CONCISE_SEQUENCE_LENGTH and load_decision(v2)['decision'] == 'ADOPT_CONCISE', 'V2 concise decision required')
    ep = stage_output(v2, 'EP'); terminal = load(ep / 'terminal_PRIVATE.json')
    require(terminal['status'] == 'PASS' and terminal['release_verified'] and terminal['prompt_condition'] == 'concise' and
            terminal['scope_sha256'] == digest(v2entry.phase_scope(v2scope, 'EP')), 'Completed V2 EP Parent DEV24 under the executed scope required')
    files[str((ep / 'terminal_PRIVATE.json').resolve())] = H(ep / 'terminal_PRIVATE.json')
    dev = load(v2['dev24'])['records_PRIVATE']
    parent_rows, row_files = v1e.authenticate_parent_rows(ep, {'scope': digest(v2scope), 'stage': 'EP', 'input': v2scope['input_sha256']}, [r['id'] for r in dev])
    files.update(row_files)
    stage_input = load(v2['stage_input'])
    unit0 = v2core.condition_units(stage_input['units_PRIVATE'][:1], 'concise')[0]
    require(unit0['kind'] == ms.SEQ_KIND and unit0['index'] == 0, 'Existing frozen TRAIN sequence pair (unit 0) required')
    seq_images = {b['record_id_PRIVATE']: b['image_binding_PRIVATE']['sha256'] for b in stage_input['planned_image_bindings_PRIVATE']}
    train, evaluation = midline_records(load(spec['midline_manifest']))
    osf = {r['image_name']: r['osf_sha256'] for r in load(spec['osf_receipt'])}
    for r in train + evaluation:
        image = Path(r['images'][0]); h = H(image)
        require(h == r['image_sha256'] == osf.get(image.name) and _inside(image, CANONICAL / 'local_data'), 'Selected image differs from its OSF payload')
        files[str(image.resolve())] = h
    borders = v1.border_controls(load(spec['control_inputs']), train)
    for b in borders.values():
        p = _inside(b['path'], V1_DIR); require(H(p) == b['sha256'], 'Border-control image changed'); files[str(p)] = b['sha256']
    gate = v1.gate_controls(train, borders)
    census = v1.census(load(spec['source_qa']))
    agrees, ours, theirs = v1e.census_agrees(census, load(spec['independent_census']))
    lengths, length_status = measured_lengths(spec.get('midline_prompt_lengths'), train, gate, evaluation)
    if spec.get('midline_prompt_lengths'):
        p = _inside(spec['midline_prompt_lengths'], PILOT_DIR); files[str(p)] = H(p)
    blockers = [] if agrees else ['CENSUS_DISAGREES_WITH_INDEPENDENT_ORACLE: %s vs %s' % (ours, theirs)]
    require(census['decision'] == DATA_SCALE, 'Census must not be promoted beyond the existing evaluated admission')
    budget = None
    if length_status != 'MEASURED':
        blockers.append('PENDING_PROCESSOR_MEASUREMENT: V2 midline prompts not yet measured with the pinned CPU processor')
    else:
        budget = h2.budget({k: lengths[k] for k in ('sequence', 'midline_train', 'midline_gate', 'midline_eval')})
    pilot_input = {'sequence_unit_PRIVATE': unit0, 'sequence_image_sha256': {c['row']['id']: seq_images[c['row']['id']] for c in unit0['contexts']},
                   'midline_train_PRIVATE': train, 'midline_eval_PRIVATE': evaluation, 'border_controls_PRIVATE': borders, 'prompt_lengths': lengths,
                   'parent_DEV24_rows_PRIVATE': parent_rows, 'census': census, 'v1_proof': proof}
    own = ('src/mri_rl/heterogeneous_mri_pilot_v2.py', 'src/mri_rl/heterogeneous_mri_pilot_native_v2.py', 'src/mri_rl/heterogeneous_mri_pilot_entry_v2.py',
           'scripts/run_heterogeneous_mri_pilot_v2.py', 'scripts/measure_heterogeneous_midline_prompts_v2.py', 'scripts/diagnose_heterogeneous_pilot_v1_format.py',
           'src/mri_rl/heterogeneous_mri_pilot_v1.py', 'src/mri_rl/heterogeneous_mri_pilot_native_v1.py', 'src/mri_rl/heterogeneous_mri_pilot_entry_v1.py')
    sources = _closure({str(ROOT / p): H(ROOT / p) for p in own})
    for p, h in v2scope['source_sha256'].items():
        require(H(p) == h and sources.get(p, h) == h, 'Inherited V2 run source changed'); sources[p] = h
    out = _inside(spec['output'], PILOT_DIR); require(allow_started or not out.exists(), 'Fresh V2 pilot output required')
    scope = {'schema': SCHEMA, 'scope_kind': KIND, 'action': ACTION, 'implemented_by': IMPLEMENTED_BY, 'source_sha256': sources, 'file_sha256': files,
             'input_sha256': identity(files), 'pilot_input_sha256': identity(pilot_input), 'output': str(out), 'resource_policy': MODEL_POLICY,
             'external_guardian_policy': GUARDIAN_POLICY, 'run_limits': RUN_LIMITS, 'windows_seconds': WINDOWS, 'worst_case_seconds': WORST_CASE_SECONDS,
             'budget': budget, 'phases': list(PHASES), 'phase_condition': 'parent_eval and child_eval only after a committed update',
             'parent': dict(a.FROZEN_PARENT), 'v2_executed_scope_sha256': digest(v2scope), 'task_weight': h2.TASK_WEIGHT,
             'midline_suffix_sha256': h2.MIDLINE_SUFFIX_SHA256, 'hard_gates': ['FORMAT: strict-valid TRUE_0 and TRUE_1',
                                                                             'SIGNAL: both tasks >=1 admitted same-prompt group with strict-valid CORRECT and WRONG',
                                                                             'GRADIENT: finite nonzero joint'],
             'diagnostic_only': ['SWAP/BORDER answers', 'v1 image-dependence classification', 'pre/post control comparison'],
             'no_image_control': v1.NO_IMAGE_CONTROL, 'data_scale': DATA_SCALE, 'parent_DEV24': 'V2_EP_CONCISE_ROWS_JOURNAL_AUTHENTICATED',
             'block_transport': v1.TRANSPORT, 'v1_preserved': True, 'forbidden_inputs': ['final100', 'sealed new16 outputs'], 'automatic_retry': False, 'third_arm': False}
    status = 'BLOCKED' if blockers else 'PREPARED_PENDING_EXACT_REVIEW_AND_FRESH_HUMAN'
    return {'status': status, 'blockers': blockers, 'executable_scope': scope, 'scope_sha256': digest(scope), 'pilot_input_PRIVATE': pilot_input,
            'census': census, 'measurement_rows_PRIVATE': measured_rows(train, gate, evaluation), 'native_calls': 0}


# ================================================================ authorization (own issuer; v1/V2 grants are never borrowed)
historical_grants, wait_nonresident, pre_native_runtime_check = v1e.historical_grants, v1e.wait_nonresident, v1e.pre_native_runtime_check


class Authorization:
    def __init__(self, issuer, scope, expires, parent=None):
        require(issuer is _ISSUER, 'Only exact V2 admission may issue authorization')
        self.scope = json.loads(json.dumps(scope)); self.sha = digest(scope); self.expires = expires; self.parent = parent
        self.start = time.monotonic() if parent is None else parent.start

    def check(self, phase, clock=time.monotonic):
        require(datetime.now(timezone.utc) < self.expires, 'Human permission expired')
        require(clock() - self.start < RUN_LIMITS['global_window_seconds'], 'Global window exhausted (waits count)')
        require(digest(self.scope) == self.sha, 'Exact authorization scope changed')
        if self.parent is not None:
            started = load(Path(self.scope['output']).parent / (self.scope['phase'] + '_clock_PRIVATE.json'))['started_monotonic']
            require(clock() - started < self.scope['window_seconds'], 'Phase window exhausted (waits count): ' + self.scope['phase'])
        for p, h in {**self.scope['source_sha256'], **self.scope['file_sha256']}.items():
            require(Path(p).is_file() and H(p) == h, 'Bound source/input changed')
        return True

    def for_phase(self, phase):
        require(self.parent is None and phase in PHASES, 'Only V2 pilot phases derive from compound authority')
        self.check('phase_admission')
        return Authorization(_ISSUER, phase_scope(self.scope, phase), self.expires, self)


def phase_scope(scope, phase):
    require(phase in PHASES and scope['budget'] is not None, 'Measured budget required before any phase scope')
    base = Path(scope['output']); greedy = h2.GREEDY[phase]; train = phase == 'train'
    reservation = {'greedy_calls': greedy, 'sampled_rollout_responses': 16 if train else 0, 'backward_calls': scope['budget']['backward_calls_max'] if train else 0,
                   'optimizer_updates': int(train), 'generated_response_tokens': ((16 if train else 0) + greedy) * h2.MAX_TOKENS}
    return {**scope, 'phase': phase, 'compound_scope_sha256': digest(scope), 'action': ACTION + '_' + phase, 'resource_stage': 'B_MICRO',
            'output': str(base / (phase + '_PRIVATE')), 'train_phase_output': str(base / 'train_PRIVATE' / 'train_phase_PRIVATE'),
            'pilot_input_path': str(base / 'pilot_input_PRIVATE.json'), 'window_seconds': WINDOWS[phase],
            'budget': {'greedy_calls_phase': greedy, 'phase_reservation': reservation}}


def _check(scope, human, review):
    require(scope.get('schema') == SCHEMA and scope.get('scope_kind') == KIND and scope.get('action') == ACTION and scope.get('implemented_by') == IMPLEMENTED_BY and
            scope.get('resource_policy') == MODEL_POLICY and scope.get('external_guardian_policy') == GUARDIAN_POLICY and scope.get('run_limits') == RUN_LIMITS and
            scope.get('windows_seconds') == WINDOWS and scope.get('budget') is not None and scope.get('automatic_retry') is False, 'Exact conservative measured V2 scope required')
    require(review.get('verdict') in ('PASS', 'PASS_WITH_LIMITATIONS') and review.get('scope_sha256') == digest(scope) and review.get('source_sha256') == scope['source_sha256'] and
            review.get('input_sha256') == scope['input_sha256'] and review.get('reviewer_kind') == 'isolated_codex' and review.get('reviewer_id') and
            'claude' not in str(review['reviewer_id']).lower(), 'Independent exact review by a non-Claude model family required')
    require(human.get('authorized') is True and human.get('action') == ACTION and human.get('scope_sha256') == digest(scope) and
            human.get('included_phases') == list(PHASES) and a.sha(human.get('human_instruction_sha256')) and human.get('approved_by'),
            'Fresh explicit human permission naming all V2 pilot phases required')


def authorize(scope, human, review, *, consumed, history, now=None):
    _check(scope, human, review)
    require(human['human_instruction_sha256'] not in history, 'Human instruction was consumed by an earlier scope; a fresh grant is required')
    now = now or datetime.now(timezone.utc)
    try:
        issued, expires = (datetime.fromisoformat(human[k].replace('Z', '+00:00')) for k in ('issued_at_utc', 'expires_at_utc'))
        valid = (issued <= now < expires and (expires - issued).total_seconds() <= 86400 and (now - issued).total_seconds() <= 1800 and
                 (expires - now).total_seconds() >= RUN_LIMITS['global_window_seconds'])
    except (KeyError, TypeError, ValueError, AttributeError):
        valid = False
    require(valid, 'Fresh (<=30 min) bounded permission covering the whole bounded flow required')
    token = Authorization(_ISSUER, scope, expires); token.check('admission'); consumed(human['human_instruction_sha256']); return token


class PilotResourceGateV2(resources.ResourceGate):
    """Unchanged color/threshold gate; only the model-load ledger limit is the V2 pilot's 3."""
    def claim_model_load(self):
        from mri_rl.shared_rollout_gspo_core_v1 import replace_atomic
        self.authorization.check('hetero_v2_model_load_claim')
        path = Path(self.authorization.parent.scope['output']) / 'model_load_ledger_PRIVATE.json'
        limit = RUN_LIMITS['model_loads_max']
        rec = load(path) if path.exists() else {'scope': self.scope['compound_scope_sha256'], 'claimed': 0, 'limit': limit}
        require(rec['scope'] == self.scope['compound_scope_sha256'] and rec['limit'] == limit == 3 and 0 <= rec['claimed'] < limit, 'V2 pilot model-load budget exhausted')
        rec['claimed'] += 1; replace_atomic(path, rec)


# ================================================================ execution
def _credentials(args):
    for name in ('scope', 'human', 'review'):
        require(H(getattr(args, name)) == getattr(args, 'expected_' + name + '_sha256'), 'Exact credential FILE hash changed: ' + name)
    scope = load(args.scope); prepared = prepare(args.spec, allow_started=True)
    require(prepared['status'] == 'PREPARED_PENDING_EXACT_REVIEW_AND_FRESH_HUMAN' and scope == prepared['executable_scope'], 'Exact measured preparation differs')
    return scope, prepared, load(args.human), load(args.review)


def execute_child(args):
    import fcntl
    import signal
    scope, prepared, human, review = _credentials(args)
    base = Path(scope['output']); launch = load(base / 'launch_identity_PRIVATE.json')
    require(launch['scope_sha256'] == digest(scope) and launch['human_instruction_sha256'] == human['human_instruction_sha256'], 'Child lacks the exact launch')
    _check(scope, human, review)
    expires = datetime.fromisoformat(human['expires_at_utc'].replace('Z', '+00:00'))
    token = Authorization(_ISSUER, scope, expires); token.start = launch['started_monotonic']
    consume_once(base / ('%s_child_permission_PRIVATE' % args.child_phase))(human['human_instruction_sha256'])      # one attempt per phase
    phase = token.for_phase(args.child_phase); v2 = load(V2_SPEC)
    runtime = load(v2['runtime_spec']); lockpath = Path(runtime['global_lock']); require(lockpath.is_file() and not lockpath.is_symlink(), 'Existing lock required')
    fd = os.open(lockpath, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        gate = PilotResourceGateV2('B_MICRO', {**v2, 'output': scope['output']}, phase.scope, phase)
        signal.signal(signal.SIGUSR1, lambda *unused: setattr(gate, 'red_requested', True))
        gate('native_model_load_preflight')
        publish_once(Path(phase.scope['output']).parent / (args.child_phase + '_runtime_admission_PRIVATE.json'), pre_native_runtime_check(v2))
        phase.check('before_native_import')
        from mri_rl.heterogeneous_mri_pilot_native_v2 import run_child
        result = run_child(args.child_phase, spec=v2, scope=phase.scope, authorization=phase, resource_gate=gate,
                           scope_file_sha256=args.expected_scope_sha256)
        publish_once(Path(phase.scope['output']) / 'terminal_PRIVATE.json', {**result, 'scope_sha256': phase.sha})
        return 0 if result['release_verified'] else 2
    finally:
        os.close(fd)


def execute_launch(args):
    import shutil
    import sys
    from dataclasses import asdict
    from mri_rl.safe_micro_witness_v1 import inspect_lock
    from mri_rl.safe_micro_watchdog_v2 import witness_child
    from mri_rl.accuracy_recovery_gpu_cache_v3 import os_pressure_level
    from mri_rl.resource_supervisor import sample_macos
    from mri_rl.safe_micro_journal_v1 import durable_mkdir
    from mri_rl.shared_rollout_gspo_core_v1 import replace_atomic
    scope, prepared, human, review = _credentials(args)
    base = Path(scope['output']); python = CANONICAL / 'work/accuracy-recovery-venv-v1/bin/python'
    require(Path(sys.executable).resolve() == python.resolve(), 'Pinned production interpreter required')
    token = authorize(scope, human, review, consumed=consume_once(base.with_name(base.name + '_permission_PRIVATE')), history=historical_grants())
    require(not base.exists(), 'Fresh output required'); durable_mkdir(base)
    publish_once(base / 'pilot_input_PRIVATE.json', prepared['pilot_input_PRIVATE'])
    v1.authenticated_input(base / 'pilot_input_PRIVATE.json', scope['pilot_input_sha256'])
    publish_once(base / 'launch_identity_PRIVATE.json', {'scope_sha256': token.sha, 'human_instruction_sha256': human['human_instruction_sha256'],
                                                          'started_utc': datetime.now(timezone.utc).isoformat(), 'started_monotonic': token.start})
    runtime = load(load(V2_SPEC)['runtime_spec']); controller = resources.Controller()

    def observe():
        s = sample_macos(); r = controller.decide(s, os_pressure_level(), shutil.disk_usage(base).free / 1024 ** 3, time.monotonic())
        replace_atomic(base / 'nonresident_latest_PRIVATE.json', {'sample': asdict(s), **r}); return r
    results, plan = {}, ['train']
    while plan:
        phase = plan.pop(0)
        started = time.monotonic()
        publish_once(base / (phase + '_clock_PRIVATE.json'), {'started_monotonic': started})
        wait_nonresident(token, started, WINDOWS[phase], observe)
        argv = [str(python), '-I', str(ROOT / 'scripts/run_heterogeneous_mri_pilot_v2.py'), '--child-phase', phase]
        for name in ('spec', 'scope', 'human', 'review'):
            argv += ['--' + name, str(getattr(args, name))]
        for name in ('scope', 'human', 'review'):
            argv += ['--expected-' + name + '-sha256', getattr(args, 'expected_' + name + '_sha256')]
        wdir = base / (phase + '_exit_PRIVATE')
        witness = witness_child(argv, wdir, scope_sha256=args.expected_scope_sha256, sampler=lambda pid: sample_macos(pid=pid), level=os_pressure_level,
                                disk_free_gib=lambda: shutil.disk_usage(base).free / 1024 ** 3, post_sampler=lambda: asdict(sample_macos()),
                                lock_inspector=lambda: inspect_lock(runtime['global_lock']))
        require(witness['post_child_global_lock'].get('no_open_holders') is True, 'Model lock not released')
        require(witness['native_exit_code'] == 0 and witness.get('child_reaped') is True and witness.get('guardian_stop') is False
                and witness.get('scope_file_sha256') == args.expected_scope_sha256 and witness.get('post_child_inspection_complete') is True
                and not witness['SIGTERM_sent'] and not witness['SIGKILL_after_grace'] and not witness['monitor_log_errors'],
                'Phase did not complete cleanly: STOP (no automatic retry): ' + phase)
        terminal = load(base / (phase + '_PRIVATE') / 'terminal_PRIVATE.json')
        require(terminal['release_verified'] is True, 'Phase release not verified')
        results[phase] = terminal
        if phase == 'train':
            plan = next_phases(terminal['status'])
            if not plan:
                publish_once(base / 'phase3_skipped_PRIVATE.json', {'reason': 'NO_COMMITTED_UPDATE', 'train_status': terminal['status'], 'model_loads': 0})
    analysis = final_analysis(base, results)
    publish_once(base / 'analysis_PRIVATE.json', analysis)
    print(json.dumps({k: analysis.get(k) for k in ('outcome', 'result', 'data_scale')})); return 0


def final_analysis(base, results):
    """CPU only. Parent = journal-authenticated V2 EP concise DEV24 + NEW Parent midline EVAL2; child = DEV24 + EVAL2 (+ 6 diagnostic controls)."""
    pinput = load(base / 'pilot_input_PRIVATE.json')
    tdir = base / 'train_PRIVATE' / 'train_phase_PRIVATE'
    train = load(tdir / 'train_phase_complete_PRIVATE.json')
    status = train['status']
    if status != 'UPDATED':
        return {**h2.outcome(None, None, status=status), 'format_gate': train['format_gate'], 'train': {k: train.get(k) for k in
                ('status', 'on_policy_responses', 'decision', 'class_contrast_groups', 'admitted_groups')}, 'model_calls_in_analysis': 0}
    parent_items = results['parent_eval']['items_PRIVATE']
    child = results['child_eval']['items_PRIVATE']
    child_scored = [x for x in child if not x.get('control')]
    parent = v1.evaluation_summary(pinput['parent_DEV24_rows_PRIVATE'] + parent_items)
    terminal = v1.evaluation_summary(child_scored)
    pre = load(tdir / 'controls_pre_PRIVATE.json')['controls']
    return {**h2.outcome(parent, terminal, status=status), 'parent': parent, 'child': terminal, 'format_gate': train['format_gate'],
            'controls_pre_post': h2.control_comparison(pre, [x for x in child if x.get('control')]), 'train': {k: train.get(k) for k in
            ('status', 'on_policy_responses', 'decision', 'class_contrast_groups', 'admitted_groups')}, 'model_calls_in_analysis': 0}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--spec', type=Path)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--launch', action='store_true'); mode.add_argument('--child-phase', choices=PHASES)
    for name in ('scope', 'human', 'review'):
        p.add_argument('--' + name, type=Path); p.add_argument('--expected-' + name + '-sha256')
    args = p.parse_args(argv)
    require(args.spec is not None, '--spec required')
    if args.launch or args.child_phase:
        require(all(getattr(args, n, None) and getattr(args, 'expected_' + n + '_sha256', None) for n in ('scope', 'human', 'review')), 'Exact artifacts required')
        return execute_child(args) if args.child_phase else execute_launch(args)
    result = prepare(args.spec)
    print(json.dumps({'status': result['status'], 'blockers': result['blockers'], 'scope_sha256': result['scope_sha256'],
                      'census_decision': result['census']['decision'], 'budget': result['executable_scope']['budget'],
                      'files_bound': len(result['executable_scope']['file_sha256']), 'sources_bound': len(result['executable_scope']['source_sha256']),
                      'v1_proof': result['pilot_input_PRIVATE']['v1_proof'], 'native_calls': 0}, sort_keys=True))
    return 0
