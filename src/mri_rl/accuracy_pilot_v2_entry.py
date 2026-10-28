"""Accuracy Pilot V2 exact-scope authority: P -> C -> A -> B -> EP/EA/EB, then CPU analysis.

Preparation reads metadata only and imports no MLX. It re-derives the executed V1
recovery scope (FILE 8be74ec8...) so every inherited runtime/DEV24/cache input is
transitively authenticated, binds the SAME frozen 4-round TRAIN schedule that V1's
round 1 came from, the measured concise prompt length and the saved V1 Parent
DEV24 rows (hash-reused only when the original prompt is kept).

The result is PREPARED_PENDING_EXACT_REVIEW_AND_FRESH_HUMAN. It never contains an
approval flag; execution requires an independent exact review artifact AND a fresh
explicit human grant naming every included stage. No CPU result authorizes models.
Resource colors/thresholds are the unchanged safe_micro_dynamic_resources_v1.POLICY.
"""
from __future__ import annotations
import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from mri_rl import accuracy_recovery_v1 as a
from mri_rl import accuracy_pilot_v2_core as core
from mri_rl import safe_micro_dynamic_resources_v1 as resources
from mri_rl.safe_micro_entry_v1 import _closure
from mri_rl.safe_micro_watchdog_v2 import POLICY as GUARDIAN_POLICY

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 'accuracy-pilot-v2-scope-v1'
SPEC_SCHEMA = 'accuracy-pilot-v2-spec-v1'
KIND = 'FRESH_EXACT_ACCURACY_PILOT_V2_PROMPT_TRAIN_DEV24_SCOPE'
ACTION = 'accuracy_pilot_v2_prompt8_common_AB_two_updates_DEV24_once'
IMPLEMENTED_BY = 'Claude Code (claude-opus-5-5)'
STAGES = core.STAGES
EXECUTED_V1_SCOPE_FILE = '8be74ec824bc4daa9a11497dd07ba7501c98cc3f261f4169c2433f3840dda1c7'
PARENT = dict(a.FROZEN_PARENT)
MODEL_POLICY = dict(resources.POLICY)          # unchanged GREEN/YELLOW/RED, RSS14/MLX12/cache256MiB/disk26/battery20
# Windows are worst-case derived from ACTUAL V1 costs (greedy/sampled 0.236 s/token -> 0.25 used; V1 B 8BW incl load 137.37s;
# load+release <=90s; +3s/call prefill; <=10s/backward; <=5s/old forward; 15s/checkpoint) plus >=15% margin. Waits count.
COST_MODEL = {'seconds_per_generated_token': .25, 'seconds_per_call_overhead': 3., 'load_release_seconds': 90., 'seconds_per_backward': 10.,
              'seconds_per_old_forward': 5., 'seconds_per_checkpoint': 15., 'V1_observed': {'Q_s': 81.84, 'B_one_arm_8BW_incl_load_s': 137.37,
              'DEV24_three_models_s': 2384.31, 'DEV24_tokens': 8488, 'greedy_s_per_token': 692.20 / 2932, 'peak_MLX_GiB': 10.36, 'peak_RSS_GiB': 5.97,
              'V1_E_margin_s': 2400 - 2384.31}}
BUDGETS = {
    'P': {'greedy_calls': 16, 'generated_response_tokens': 4096, 'backward_calls': 0, 'optimizer_updates': 0, 'model_loads': 1, 'window_seconds': 1500,
          'worst_case_seconds': 90 + 16 * (256 * .25 + 3)},
    'C': {'sampled_responses': 32, 'generated_response_tokens': 8192, 'detached_old_forwards': 32, 'backward_calls': 0, 'optimizer_updates': 0, 'model_loads': 1,
          'window_seconds': 2900, 'worst_case_seconds': 90 + 32 * (256 * .25 + 3) + 32 * 5},
    'A': {'sampled_responses': 32, 'generated_response_tokens': 8192, 'detached_old_forwards': 32, 'backward_calls': 64, 'optimizer_updates': 2, 'model_loads': 1,
          'window_seconds': 3600, 'worst_case_seconds': 90 + 64 * 10 + 32 * (256 * .25 + 3) + 32 * 5 + 2 * 15},
    'B': {'sampled_responses': 32, 'generated_response_tokens': 8192, 'detached_old_forwards': 32, 'backward_calls': 64, 'optimizer_updates': 2, 'model_loads': 1,
          'window_seconds': 3600, 'worst_case_seconds': 90 + 64 * 10 + 32 * (256 * .25 + 3) + 32 * 5 + 2 * 15},
    'EP': {'greedy_calls': 24, 'generated_response_tokens': 6144, 'backward_calls': 0, 'optimizer_updates': 0, 'model_loads': 1, 'window_seconds': 2100,
           'worst_case_seconds': 90 + 24 * (256 * .25 + 3)},
    'EA': {'greedy_calls': 24, 'generated_response_tokens': 6144, 'backward_calls': 0, 'optimizer_updates': 0, 'model_loads': 1, 'window_seconds': 2100,
           'worst_case_seconds': 90 + 24 * (256 * .25 + 3)},
    'EB': {'greedy_calls': 24, 'generated_response_tokens': 6144, 'backward_calls': 0, 'optimizer_updates': 0, 'model_loads': 1, 'window_seconds': 2100,
           'worst_case_seconds': 90 + 24 * (256 * .25 + 3)}}
RUN_LIMITS = {'global_operation_start_window_seconds': 19800, 'planned_model_loads': 7, 'model_loads_max': 9, 'max_automatic_recoveries': 2,
              'max_controller_launches': 3, 'max_attempts_per_stage': 3, 'wait_samples_before_restart': 3, 'wait_counts_against_deadlines': True,
              'incomplete_operation': 'INDETERMINATE_NO_REPLAY', 'completed_results': 'HASH_VERIFIED_REUSE_ONLY', 'cache_recovery_operations': 0,
              'single_resident_model': True, 'note': 'POLICY.compound_seconds/model_load fields belong to the executed V1 scope; V2 uses these limits.'}
CONDITIONAL_FLOW = {'P': 'always; decision by core.prompt_decision (frozen rule) on 8 TRAIN images, Parent Adam52 greedy',
                    'C': 'always; common rollout units 0-3 under the frozen prompt condition',
                    'A': 'always; fresh disk Adam52, segment0 common, segment1 own policy', 'B': 'always; identical to A except Paired-Prompt grouping',
                    'EP': 'run iff ADOPT_CONCISE; if KEEP_ORIGINAL the hash-bound saved V1 Parent DEV24 rows are reused (same protocol), no call',
                    'EA': 'run iff arm A made >=1 update, else Parent alias', 'EB': 'run iff arm B made >=1 update, else Parent alias'}
SPEC_FILE_KEYS = ('recovery_spec', 'recovery_scope', 'stage_input', 'concise_length_measurement')
require = a.require; H = a.file_sha256; digest = a.scope_digest
_ISSUER = object()


def load(path):
    require(not any(x in str(path).lower() for x in ('final100', 'holdout16', 'new16')), 'Forbidden development input')
    return json.loads(Path(path).read_text())


def metadata_hash(path):
    require(Path(path).suffix not in ('.safetensors', '.png', '.gz', '.npy', '.npz', '.bin'), 'Metadata/source only during preparation')
    return H(path)


def _output(path):
    root = a.canonical_project_root(); p = Path(path); base = root / 'local_data/accuracy_pilot_v2'
    require(p.is_absolute() and p == p.resolve() and p.is_relative_to(base) and p != base, 'Canonical private V2 output required')
    require(not base.is_symlink() and not (root / 'local_data').is_symlink(), 'No private root symlinks')
    return p


def v1_parent_rows(recovery_output):
    return v1_rows(recovery_output, 'Parent')


def v1_rows(recovery_output, label):
    """Authenticate the 24 committed V1 DEV24 greedy rows of Parent/A/B through their journal commit chain (no model)."""
    from mri_rl.shared_rollout_gspo_core_v1 import identity
    require(label in ('Parent', 'A', 'B'), 'V1 endpoint label required')
    journal = Path(recovery_output) / 'stage4_PRIVATE' / 'recovery_journal_PRIVATE'
    rows, files = [], {}
    for i in range(24):
        op = journal / ('%s_row_%02d' % (label, i)); claim = op / 'reservation_PRIVATE.json'; commit = op / 'commit_PRIVATE.json'
        rec = load(commit)
        require(rec['reservation_sha256'] == H(claim) and rec['record_sha256'] == identity({k: v for k, v in rec.items() if k != 'record_sha256'}), 'V1 row commit chain changed')
        item = rec['value']['item']; path = [p for p in rec['files'] if p.endswith('row_%02d_PRIVATE.json' % i)]
        require(len(path) == 1 and H(path[0]) == rec['files'][path[0]] and load(path[0]) == item, 'V1 row file changed')
        files[path[0]] = rec['files'][path[0]]; files[str(commit)] = H(commit)
        rows.append(item)
    return rows, files


def validate_concise_measurement(measure, stage_rows, dev_rows):
    require(measure.get('schema') == 'accuracy-pilot-v2-concise-length-v1' and measure.get('instruction_sha256') == core.CONCISE_INSTRUCTION_SHA256, 'Exact concise instruction measurement required')
    ids = sorted(r['id'] for r in stage_rows + dev_rows)
    require(sorted(measure['row_ids']) == ids and len(set(measure['deltas'].values())) == 1, 'Uniform measured token delta over all 32 sequence rows required')
    delta = next(iter(measure['deltas'].values()))
    require(type(delta) is int and 0 < delta <= 64 and measure['concise_sequence_prompt_length'] == core.ORIGINAL_PROMPT_LENGTHS['mri_sequence'] + delta,
            'Concise length must equal 213 + measured delta')
    return measure['concise_sequence_prompt_length']


def prepare(specpath, *, allow_started=False):
    from mri_rl import safe_micro_recovery_entry_v1 as recovery
    from mri_rl import accuracy_stage_entry_v1 as stage_entry
    spec = load(specpath); require(spec.get('schema') == SPEC_SCHEMA, 'Exact V2 spec schema required')
    require(set(spec['file_sha256']) == set(SPEC_FILE_KEYS), 'Exact V2 metadata inventory required')
    files = {str(Path(specpath).resolve()): metadata_hash(specpath)}; inputs = {}
    for key in SPEC_FILE_KEYS:
        p = Path(spec[key]); h = metadata_hash(p); require(h == spec['file_sha256'][key], 'V2 input changed: ' + key); files[str(p)] = h; inputs[key] = load(p)
    require(spec['file_sha256']['recovery_scope'] == EXECUTED_V1_SCOPE_FILE, 'Must inherit the exact executed V1 recovery scope')
    prior = recovery.prepare(spec['recovery_spec'], allow_started=True)['executable_scope']
    require(prior == inputs['recovery_scope'], 'Executed V1 scope no longer re-derives exactly')
    rspec = inputs['recovery_spec']
    require(spec['runtime_spec'] == rspec['runtime_spec'] and spec['dev24'] == rspec['dev24'] and spec['recovery_output'] == prior['output'], 'Runtime/DEV24/V1 output must be inherited unchanged')
    for path, h in prior['file_sha256'].items():
        require(path not in files or files[path] == h, 'Conflicting inherited metadata'); files[path] = h
    runtime, micro, dev = load(rspec['runtime_spec']), load(rspec['micro_input']), load(rspec['dev24'])
    require(Path(micro['source_stage_input_PRIVATE']).resolve() == Path(spec['stage_input']).resolve() and spec['file_sha256']['stage_input'] == micro['source_stage_input_sha256'],
            'V2 schedule must be the same frozen 4-round input whose first round V1 used')
    stage_input = inputs['stage_input']
    _, exposure = stage_entry._stage(stage_input, runtime)            # re-validates 4 complete pairs, pinned z045 images, TRAIN-only, 8 unique Dallas
    for binding in stage_input['input_capsules_PRIVATE'].values():
        files[binding['path']] = metadata_hash(binding['path'])
    dev_patients = {r['row_PRIVATE']['extra_info']['patient_key'] for r in dev['records_PRIVATE']}
    schedule = core.validate_schedule(stage_input['units_PRIVATE'], dev_patients)
    stage_rows = [c['row'] for u in stage_input['units_PRIVATE'] if u['kind'] == 'SEQ_PAIR' for c in u['contexts']]
    data56 = load(stage_input['input_capsules_PRIVATE']['DATA56_INPUT_PRIVATE.json']['path'])          # hash-bound just above
    planned = {b['record_id_PRIVATE']: b['image_binding_PRIVATE']['sha256'] for b in stage_input['planned_image_bindings_PRIVATE']}
    schedule['z045'] = core.z045_provenance(stage_rows, data56['rows'], planned)
    concise = validate_concise_measurement(inputs['concise_length_measurement'], stage_rows, [r['row_PRIVATE'] for r in dev['records_PRIVATE']])
    require(spec['concise_sequence_prompt_length'] == concise, 'Spec concise length differs from measurement')
    try:
        pairs = core.dev_pairs(dev['records_PRIVATE']); pair_status = {'pairs': len(pairs), 'field': core.PAIR_FIELD, 'status': 'DERIVED' if len(pairs) == 12 else 'UNEXPECTED_COUNT'}
    except ValueError as error:
        pair_status = {'pairs': 0, 'field': core.PAIR_FIELD, 'status': 'NOT_AVAILABLE', 'reason': str(error)}
    reuse_rows, reuse_files = v1_parent_rows(spec['recovery_output'])
    require([r['id'] for r in reuse_rows] == [r['id'] for r in dev['records_PRIVATE']], 'Saved V1 Parent rows must be the same frozen DEV24 order')
    files.update(reuse_files)
    out = _output(spec['output']); require(allow_started or not out.exists(), 'Fresh V2 output required')
    own = ('src/mri_rl/accuracy_pilot_v2_core.py', 'src/mri_rl/accuracy_pilot_v2_native.py', 'src/mri_rl/accuracy_pilot_v2_entry.py', 'scripts/run_accuracy_pilot_v2.py',
           'src/mri_rl/safe_micro_watchdog_v2.py', 'src/mri_rl/accuracy_stage_entry_v1.py')
    sources = _closure({str(ROOT / p): metadata_hash(ROOT / p) for p in own})
    for path, h in prior['source_sha256'].items():
        require(metadata_hash(path) == h and (path not in sources or sources[path] == h), 'Inherited V1 run source changed'); sources[path] = h
    sources = recovery.bind_runtime_sources(sources, runtime)
    result = {'schema': SCHEMA, 'scope_kind': KIND, 'action': ACTION, 'implemented_by': IMPLEMENTED_BY, 'status_at_preparation': 'PREPARED_PENDING_EXACT_REVIEW_AND_FRESH_HUMAN',
              'source_sha256': sources, 'file_sha256': files, 'input_sha256': a.identity(files), 'output': str(out), 'resource_policy': MODEL_POLICY,
              'external_guardian_policy': GUARDIAN_POLICY, 'budgets': BUDGETS, 'cost_model': COST_MODEL, 'run_limits': RUN_LIMITS, 'parent': PARENT,
              'stage_order': list(STAGES), 'conditional_flow': CONDITIONAL_FLOW, 'prompt': {'instruction_sha256': core.CONCISE_INSTRUCTION_SHA256,
              'instruction_source': 'generation_recovery_v1.SHORT_INSTRUCTION (verbatim)', 'applies_to': 'mri_sequence question only',
              'concise_sequence_prompt_length': concise, 'decision_rule': core.PROMPT_DECISION_RULE, 'max_tokens': core.MAX_TOKENS, 'verifier_changed': False},
              'training': {'schedule': schedule, 'exposure_report': exposure, 'segments': [list(s) for s in core.SEGMENTS], 'unit_weight': core.UNIT_WEIGHT,
                           'arm_budget_original': core.arm_budget(stage_input['units_PRIVATE'], core.ORIGINAL_PROMPT_LENGTHS),
                           'arm_budget_concise': core.arm_budget(stage_input['units_PRIVATE'], core.prompt_lengths_for('concise', concise)),
                           'only_difference_between_arms': 'sequence group construction (A same-prompt F4/T4, B Paired-Prompt F2T2)',
                           'shared': 'ONLY the immutable common parent rollout of units 0-3', 'parent_selection': 'original P_calibrated13/Adam52/boundary114; NOT V1 Adam53 A/B'},
              'evaluation': {'cohort': 'FROZEN_INTERNAL_DEV24_PREVIOUSLY_EXPLORED', 'pairs': pair_status, 'substitution_label': core.SUBSTITUTION_LABEL,
                             'scale_rule': core.SCALE_RULE, 'v1_parent_rows_reuse_if_original': True, 'candidate_calls': 0, 'no_image_control': 'NOT_RUN_BY_DESIGN'},
              'automatic_retry': False, 'automatic_round2': False, 'forbidden_inputs': ['final100', 'sealed new16 outputs']}
    return {'status': 'PREPARED_PENDING_EXACT_REVIEW_AND_FRESH_HUMAN', 'executable_scope': result, 'scope_sha256': digest(result), 'native_calls': 0}


class Authorization:
    def __init__(self, issuer, scope, expires, parent=None):
        require(issuer is _ISSUER, 'Only exact admission may issue authorization')
        self.scope = json.loads(json.dumps(scope)); self.sha = digest(scope); self.expires = expires; self.parent = parent
        self.start = time.monotonic() if parent is None else parent.start; self.verified = {}

    def check(self, phase):
        require(datetime.now(timezone.utc) < self.expires, 'Human permission expired')
        require(time.monotonic() - self.start < RUN_LIMITS['global_operation_start_window_seconds'], 'Global operation-start window exhausted')
        require(digest(self.scope) == self.sha, 'Exact authorization scope changed')
        for p, h in {**self.scope['source_sha256'], **self.scope['file_sha256']}.items():
            path = Path(p); require(path.is_file(), 'Bound file missing'); s = path.stat(); stamp = (s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_ino, s.st_dev)
            if self.verified.get(p) != stamp or any(x in phase for x in ('admission', 'model_load', 'complete', 'restore')):
                require(metadata_hash(p) == h, 'Bound source/input changed'); self.verified[p] = stamp
        if self.parent is not None:
            stage = self.scope['stage']; clock = load(Path(self.scope['output']).parent / ('stage' + stage + '_clock_PRIVATE.json'))
            require(time.monotonic() - clock['started_monotonic'] < BUDGETS[stage]['window_seconds'], 'Cumulative stage window exhausted including waits')
        return True

    def for_phase(self, stage):
        require(self.parent is None and stage in STAGES, 'Only V2 stages derive from compound authority')
        self.check('phase_admission')
        return Authorization(_ISSUER, phase_scope(self.scope, stage), self.expires, self)


def phase_scope(scope, stage):
    require(stage in STAGES, 'Unknown V2 stage')
    return {**scope, 'stage': stage, 'compound_scope_sha256': digest(scope), 'action': ACTION + '_stage' + stage, 'execution_stage': core.EXECUTION[stage],
            'resource_stage': 'B_MICRO', 'output': str(Path(scope['output']) / ('stage' + stage + '_PRIVATE')), 'budget': scope['budgets'][stage], 'disk_required_bytes': 6 * 1024 ** 3}


def _check_fixed(scope):
    require(scope.get('schema') == SCHEMA and scope.get('scope_kind') == KIND and scope.get('action') == ACTION and scope.get('implemented_by') == IMPLEMENTED_BY, 'Fresh V2 scope required')
    require(scope.get('resource_policy') == MODEL_POLICY and scope.get('external_guardian_policy') == GUARDIAN_POLICY and scope.get('budgets') == BUDGETS and
            scope.get('run_limits') == RUN_LIMITS and scope.get('parent') == PARENT and scope.get('stage_order') == list(STAGES) and
            scope.get('automatic_retry') is False and scope.get('automatic_round2') is False, 'Exact conservative V2 policy/budgets/parent required')


def _check_review(scope, review):
    require(review.get('verdict') in ('PASS', 'PASS_WITH_LIMITATIONS') and review.get('scope_sha256') == digest(scope) and review.get('source_sha256') == scope['source_sha256'] and
            review.get('input_sha256') == scope['input_sha256'] and review.get('reviewer_kind') == 'isolated_codex' and review.get('reviewer_id') and
            review['reviewer_id'] != IMPLEMENTED_BY and 'claude' not in str(review['reviewer_id']).lower(),
            'Independent exact review by a non-author model family (isolated Codex; author is Claude Code) required')


def _check_human(scope, human):
    require(human.get('authorized') is True and human.get('action') == ACTION and human.get('scope_sha256') == digest(scope) and human.get('included_stages') == list(STAGES) and
            a.sha(human.get('human_instruction_sha256')) and human.get('approved_by'), 'Fresh explicit human permission naming all V2 stages required')


def authorize(scope, human, review, *, consumed):
    _check_fixed(scope); _check_human(scope, human); _check_review(scope, review)
    try:
        issued, expires = (datetime.fromisoformat(human[k].replace('Z', '+00:00')) for k in ('issued_at_utc', 'expires_at_utc')); now = datetime.now(timezone.utc)
        valid = issued.tzinfo is not None and expires.tzinfo is not None and issued <= now < expires and (expires - issued).total_seconds() <= 86400 and (now - issued).total_seconds() <= 1800
    except (KeyError, TypeError, ValueError, AttributeError):
        valid = False
    require(valid, 'Fresh bounded permission times required')
    token = Authorization(_ISSUER, scope, expires); token.check('admission'); require(callable(consumed), 'Once-only ledger required')
    consumed(human['human_instruction_sha256']); return token


def _authorize_later_child(scope, human, review, launch):
    _check_fixed(scope); _check_human(scope, human); _check_review(scope, review)
    issued, expires = (datetime.fromisoformat(human[k].replace('Z', '+00:00')) for k in ('issued_at_utc', 'expires_at_utc'))
    started = datetime.fromisoformat(launch['started_utc'])
    require(launch['scope_sha256'] == digest(scope) and launch['human_instruction_sha256'] == human['human_instruction_sha256'] and issued <= started and
            (started - issued).total_seconds() <= 1800, 'Child lacks the exact fresh compound launch')
    require(issued <= datetime.now(timezone.utc) < expires and 0 < (expires - issued).total_seconds() <= 86400, 'Child grant expired/unbounded')
    token = Authorization(_ISSUER, scope, expires); token.start = launch['started_monotonic']
    require(type(token.start) in (int, float) and token.start <= time.monotonic(), 'Invalid compound clock'); token.check('child_admission'); return token


def _credentials(args, *, launch_record=None, consumed=None):
    for name in ('scope', 'human', 'review'):
        require(metadata_hash(getattr(args, name)) == getattr(args, 'expected_' + name + '_sha256'), 'Exact credential FILE hash changed: ' + name)
    scope = load(args.scope); prepared = prepare(args.spec, allow_started=launch_record is not None)
    require(scope == prepared['executable_scope'], 'Exact current preparation differs')
    human, review = load(args.human), load(args.review)
    if launch_record is None:
        return authorize(scope, human, review, consumed=consumed), human
    token = _authorize_later_child(scope, human, review, launch_record)
    if consumed is not None:
        consumed(human['human_instruction_sha256'])
    return token, human


class V2ResourceGate(resources.ResourceGate):
    """Unchanged color/threshold gate; only the V2 model-load ledger limit differs (7 planned + 2 recovery)."""
    def claim_model_load(self):
        from mri_rl.shared_rollout_gspo_core_v1 import replace_atomic
        self.authorization.check('v2_model_load_claim')
        path = Path(self.spec['output']) / 'model_load_ledger_PRIVATE.json'
        limit = self.scope['run_limits']['model_loads_max']
        rec = load(path) if path.exists() else {'scope': self.scope['compound_scope_sha256'], 'claimed': 0, 'limit': limit}
        require(rec['scope'] == self.scope['compound_scope_sha256'] and rec['limit'] == limit == 9 and type(rec['claimed']) is int and 0 <= rec['claimed'] < limit,
                'V2 model-load budget exhausted or changed')
        rec['claimed'] += 1; replace_atomic(path, rec)


# ================================================================ controller decisions (CPU only)
def skip_reason(spec, stage):
    """Conditional flow from ACTUAL committed terminals; None means the stage runs."""
    from mri_rl.accuracy_pilot_v2_native import load_decision, arm_terminal
    if stage == 'EP':
        return None if load_decision(spec)['decision'] == 'ADOPT_CONCISE' else 'KEEP_ORIGINAL_REUSE_HASH_BOUND_V1_PARENT_DEV24'
    if stage in ('EA', 'EB'):
        return 'ZERO_UPDATE_ARM_IS_PARENT_ALIAS' if arm_terminal(spec, stage[1])['alias_of_parent'] else None
    return None


def _completed(scope, spec, stage, expected_scope_file_sha256):
    """Exact revalidation of ONE completed stage from its bound evidence. None = no completion marker (stage not done).
    A verified stage re-checks scope, terminal hash/PASS/release, closed exit, released lock and closed logs; a skipped stage
    re-derives its skip reason from the actual committed predecessor terminals and must have no native artifacts."""
    from mri_rl.safe_micro_witness_v1 import digest as fdigest
    base = Path(scope['output'])
    verified, skipped = base / ('stage' + stage + '_verified_PRIVATE.json'), base / ('stage' + stage + '_skipped_PRIVATE.json')
    if not verified.exists() and not skipped.exists():
        return None
    require(not (verified.exists() and skipped.exists()), 'Stage is both verified and skipped')
    terminal = base / ('stage' + stage + '_PRIVATE') / 'terminal_PRIVATE.json'
    if skipped.exists():
        reason = skip_reason(spec, stage)
        require(reason is not None and load(skipped) == {'stage': stage, 'reason': reason, 'model_loads': 0, 'native_calls': 0},
                'Skip marker differs from the reason derived from actual bound evidence')
        require(not terminal.exists() and not (base / ('stage' + stage + '_clock_PRIVATE.json')).exists(), 'Skipped stage has native artifacts')
        return 'SKIPPED'
    rec = load(verified)
    require(rec['scope_file_sha256'] == expected_scope_file_sha256, 'Completed stage scope mismatch')
    result = load(terminal)
    require(result['scope_sha256'] == digest(phase_scope(scope, stage)) and result.get('stage') == stage and fdigest(terminal) == rec['terminal_sha256'] and
            result['status'] == 'PASS' and result['release_verified'] is True, 'Completed stage terminal changed/incomplete')
    witness = Path(rec['exit_path']); exit = load(witness)
    require(fdigest(witness) == rec['exit_sha256'] and exit['native_exit_code'] == 0 and exit['child_reaped'] and exit['post_child_global_lock'].get('no_open_holders') is True,
            'Completed stage exit/lock invalid')
    for name in ('stdout', 'stderr'):
        require(fdigest(witness.parent / (name + '_PRIVATE.log')) == exit[name + '_sha256'], 'Closed log changed')
    return 'VERIFIED'


def _prior(scope, spec, stage, expected_scope_file_sha256):
    """EVERY earlier stage (not only the immediate one) is revalidated at each child admission / resume."""
    for previous in STAGES[:STAGES.index(stage)]:
        require(_completed(scope, spec, previous, expected_scope_file_sha256) is not None, 'Previous stage incomplete: ' + previous)


def execute_child(args):
    import fcntl
    import signal
    from mri_rl.safe_micro_dynamic_resources_v1 import ResourcePause
    from mri_rl.safe_micro_journal_v1 import Indeterminate
    from mri_rl.shared_rollout_gspo_core_v1 import publish_once
    stage = args.child_stage; scope = load(args.scope); base = Path(scope['output']); launch = load(base / 'launch_identity_PRIVATE.json')
    control = load(base / 'control_PRIVATE.json')
    require(type(args.attempt) is int and 0 <= args.attempt < RUN_LIMITS['max_attempts_per_stage'] and args.attempt == control['attempts'][stage] - 1 and
            not (base / ('stage' + stage + '_verified_PRIVATE.json')).exists() and not (base / ('stage' + stage + '_skipped_PRIVATE.json')).exists(),
            'Unreserved/duplicate child attempt')
    token, human = _credentials(args, launch_record=launch, consumed=a.consume_once(base / ('stage%s_attempt%d_permission_PRIVATE' % (stage, args.attempt))))
    spec = load(args.spec); _prior(scope, spec, stage, args.expected_scope_sha256); phase = token.for_phase(stage)
    from mri_rl.accuracy_recovery_gpu_cache_v2 import installed_fingerprint
    rspec = load(spec['recovery_spec'])
    require(installed_fingerprint() == load(rspec['cache_scope'])['installed'], 'Pinned native runtime differs')
    runtime = load(spec['runtime_spec']); lockpath = Path(runtime['global_lock']); require(lockpath.is_file() and not lockpath.is_symlink(), 'Existing lock required')
    fd = os.open(lockpath, os.O_RDWR); gate = None
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        gate = V2ResourceGate('B_MICRO', spec, phase.scope, phase)
        signal.signal(signal.SIGUSR1, lambda *unused: setattr(gate, 'red_requested', True))
        gate('native_model_load_preflight')
        from types import SimpleNamespace
        from mri_rl.continuous_rl_pilot import cache_binding_for
        from mri_rl.safe_micro_recovery_entry_v1 import SHARDS
        binding = cache_binding_for(SimpleNamespace(cache_dir=Path(runtime['cache_dir']), cache_certificate=Path(runtime['cache_certificate']),
                                                    cache_certificate_sha256=H(runtime['cache_certificate']), model=Path(runtime['model'])))
        require({n: v['sha256'] for n, v in binding['shards'].items()} == SHARDS, 'Cache payload mismatch')
        phase.check('before_native_import')
        from mri_rl.accuracy_pilot_v2_native import run_phase
        result = run_phase(stage, spec=spec, scope=phase.scope, authorization=phase, resource_gate=gate)
        record = {**result, 'scope_sha256': phase.sha, 'compound_scope_sha256': token.sha, 'stage': stage}
        p = Path(phase.scope['output']) / 'terminal_PRIVATE.json'
        if p.exists():
            require(load(p) == record, 'Completed stage terminal changed')
        else:
            publish_once(p, record)
        return 0 if record['status'] == 'PASS' and record['release_verified'] else 2
    except ResourcePause as error:
        phaseout = Path(phase.scope['output'])
        require(not list(phaseout.rglob('release_failure_PRIVATE.json')), 'Allocator release failure: no automatic recovery')
        for receipt in phaseout.rglob('release_SAFE.json'):
            require(load(receipt).get('release_verified') is True, 'Allocator release not verified: no automatic recovery')
        publish_once(base / ('stage%s_attempt%d_pause_PRIVATE.json' % (stage, args.attempt)), {'status': 'PAUSED_EXTERNAL_REAP_REQUIRED', 'reason': str(error), 'attempt': args.attempt, 'stage': stage})
        return 75
    except Indeterminate as error:
        publish_once(base / ('stage%s_attempt%d_indeterminate_PRIVATE.json' % (stage, args.attempt)), {'status': 'INDETERMINATE_NO_REPLAY', 'reason': str(error)})
        return 76
    finally:
        os.close(fd)


def validate_pause_exit(path, scope_file, base, stage, attempt):
    from mri_rl.safe_micro_witness_v1 import digest as fdigest
    path = Path(path); require(path.is_file(), 'Prior child exit absent/active: cannot resume')
    w = load(path)
    require(w['scope_file_sha256'] == scope_file and w['child_reaped'] and w['native_exit_code'] == 75 and w['post_child_global_lock'].get('no_open_holders') is True,
            'Prior child not an exact closed cooperative pause')
    require(not w['SIGTERM_sent'] and not w['SIGKILL_after_grace'] and not w['monitor_log_errors'], 'Fatal monitor/forced stop cannot resume')
    for name in ('stdout', 'stderr'):
        require(fdigest(path.parent / (name + '_PRIVATE.log')) == w[name + '_sha256'], 'Prior closed log changed')
    pause = load(Path(base) / ('stage%s_attempt%d_pause_PRIVATE.json' % (stage, attempt)))
    require(pause['stage'] == stage and pause['attempt'] == attempt and pause['status'] == 'PAUSED_EXTERNAL_REAP_REQUIRED', 'Missing exact native cooperative pause')
    return True


def wait_nonresident(scope, token, stage, observe, sleep=time.sleep, clock=time.monotonic):
    good = 0; start = load(Path(scope['output']) / ('stage' + stage + '_clock_PRIVATE.json'))['started_monotonic']
    while True:
        token.check('nonresident_wait')
        require(clock() - start < BUDGETS[stage]['window_seconds'], 'Stage cumulative wait deadline exhausted')
        result = observe()
        if result['action'] == 'STOP':
            raise resources.ResourceFatal(','.join(result['reasons']))
        good = good + 1 if result['action'] == 'CONTINUE' else 0
        if good >= RUN_LIMITS['wait_samples_before_restart']:
            return
        sleep(5.)


def execute_chain(args):
    import fcntl
    import shutil
    import sys
    from dataclasses import asdict
    from mri_rl.safe_micro_witness_v1 import inspect_lock
    from mri_rl.safe_micro_watchdog_v2 import witness_child
    from mri_rl.accuracy_recovery_gpu_cache_v3 import os_pressure_level
    from mri_rl.resource_supervisor import sample_macos
    from mri_rl.shared_rollout_gspo_core_v1 import publish_once, replace_atomic
    from mri_rl.safe_micro_journal_v1 import Journal, durable_mkdir
    scope = load(args.scope); base = Path(scope['output']); root = a.canonical_project_root(); python = root / 'work/accuracy-recovery-venv-v1/bin/python'
    require(Path(sys.executable).resolve() == python.resolve(), 'Pinned production interpreter required')
    if args.resume:
        launch = load(base / 'launch_identity_PRIVATE.json'); token, human = _credentials(args, launch_record=launch)
    else:
        token, human = _credentials(args, consumed=a.consume_once(base.with_name(base.name + '_permission_PRIVATE')))
        require(not base.exists(), 'Fresh output required'); durable_mkdir(base)
        publish_once(base / 'launch_identity_PRIVATE.json', {'scope_sha256': token.sha, 'human_instruction_sha256': human['human_instruction_sha256'],
                                                              'started_utc': datetime.now(timezone.utc).isoformat(), 'started_monotonic': token.start})
    fd = os.open(base / 'controller.lock', os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        controlpath = base / 'control_PRIVATE.json'
        control = load(controlpath) if controlpath.exists() else {'launches': 0, 'recoveries': 0, 'attempts': {s: 0 for s in STAGES}, 'accepted_pauses': []}
        require(control['launches'] < RUN_LIMITS['max_controller_launches'], 'Controller launch limit exhausted'); control['launches'] += 1; replace_atomic(controlpath, control)
        spec = load(args.spec); runtime = load(spec['runtime_spec']); controller = resources.Controller()
        def observe():
            s = sample_macos(); r = controller.decide(s, os_pressure_level(), shutil.disk_usage(base).free / 1024 ** 3, time.monotonic())
            replace_atomic(base / 'nonresident_latest_PRIVATE.json', {'sample': asdict(s), **r}); return r
        for stage in STAGES:
            verified = base / ('stage' + stage + '_verified_PRIVATE.json'); skipped = base / ('stage' + stage + '_skipped_PRIVATE.json')
            if _completed(scope, spec, stage, args.expected_scope_sha256) is not None:     # revalidated, never trusted by existence
                continue
            _prior(scope, spec, stage, args.expected_scope_sha256)
            reason = skip_reason(spec, stage)
            if reason is not None:
                publish_once(skipped, {'stage': stage, 'reason': reason, 'model_loads': 0, 'native_calls': 0}); continue
            stageclock = base / ('stage' + stage + '_clock_PRIVATE.json')
            if not stageclock.exists():
                publish_once(stageclock, {'started_monotonic': time.monotonic()})
            while True:
                if control['attempts'][stage]:
                    previous = control['attempts'][stage] - 1; pw = base / ('stage%s_attempt%d_exit_PRIVATE' % (stage, previous)) / 'actual_exit_PRIVATE.json'
                    validate_pause_exit(pw, args.expected_scope_sha256, base, stage, previous)
                    tag = '%s:%d' % (stage, previous)
                    if tag not in control['accepted_pauses']:
                        require(control['recoveries'] < RUN_LIMITS['max_automatic_recoveries'], 'Recovery budget exhausted')
                        control['recoveries'] += 1; control['accepted_pauses'].append(tag); replace_atomic(controlpath, control)
                for bindingpath in (base / ('stage' + stage + '_PRIVATE')).rglob('binding_PRIVATE.json'):
                    with Journal(bindingpath.parent, load(bindingpath)) as journal:
                        journal.assert_recoverable()
                wait_nonresident(scope, token, stage, observe)
                require(control['attempts'][stage] < RUN_LIMITS['max_attempts_per_stage'], 'Stage attempt limit exhausted')
                n = control['attempts'][stage]; control['attempts'][stage] += 1; replace_atomic(controlpath, control)
                argv = [str(python), '-I', str(ROOT / 'scripts/run_accuracy_pilot_v2.py'), '--child-stage', stage, '--attempt', str(n)]
                for name in ('spec', 'scope', 'human', 'review'):
                    argv += ['--' + name, str(getattr(args, name))]
                for name in ('scope', 'human', 'review'):
                    argv += ['--expected-' + name + '-sha256', getattr(args, 'expected_' + name + '_sha256')]
                wdir = base / ('stage%s_attempt%d_exit_PRIVATE' % (stage, n))
                witness = witness_child(argv, wdir, scope_sha256=args.expected_scope_sha256, sampler=lambda pid: sample_macos(pid=pid), level=os_pressure_level,
                                        disk_free_gib=lambda: shutil.disk_usage(base).free / 1024 ** 3, post_sampler=lambda: asdict(sample_macos()),
                                        lock_inspector=lambda: inspect_lock(runtime['global_lock']))
                require(witness['post_child_global_lock'].get('no_open_holders') is True, 'Model lock not released')
                terminal = base / ('stage' + stage + '_PRIVATE') / 'terminal_PRIVATE.json'
                if witness['native_exit_code'] == 0 and not witness['SIGTERM_sent'] and not witness['SIGKILL_after_grace'] and not witness['monitor_log_errors']:
                    result = load(terminal)
                    require(result['scope_sha256'] == digest(phase_scope(scope, stage)) and result['status'] == 'PASS' and result['release_verified'], 'Stage is not PASS/released')
                    publish_once(verified, {'scope_file_sha256': args.expected_scope_sha256, 'terminal_sha256': H(terminal), 'exit_path': str(wdir / 'actual_exit_PRIVATE.json'),
                                            'exit_sha256': H(wdir / 'actual_exit_PRIVATE.json')})
                    break
                validate_pause_exit(wdir / 'actual_exit_PRIVATE.json', args.expected_scope_sha256, base, stage, n)
        for stage in STAGES:
            require(_completed(scope, spec, stage, args.expected_scope_sha256) is not None, 'Stage incomplete before analysis: ' + stage)
        analysis = final_analysis(spec, scope)
        publish_once(base / 'analysis_PRIVATE.json', analysis)
        publish_once(base / 'compound_complete_PRIVATE.json', {'status': 'COMPLETE', 'automatic_round2': False, 'decision': analysis['scale_decision']['decision']})
        print(json.dumps({'status': 'COMPLETE', 'automatic_round2': False, 'decision': analysis['scale_decision']['decision']})); return 0
    finally:
        os.close(fd)


# ================================================================ CPU-only final analysis (no model)
def _rows(path_glob_dir, prefix):
    rows = sorted(Path(path_glob_dir).glob(prefix + '_row_*_PRIVATE.json'))
    require(len(rows) == 24, 'Exactly 24 committed DEV rows required: ' + prefix)
    return [load(p) for p in rows]


def model_rows(spec, label):
    from mri_rl.accuracy_pilot_v2_native import stage_output, arm_terminal, load_decision
    decision = load_decision(spec)
    if label == 'Parent':
        if decision['decision'] == 'KEEP_ORIGINAL':
            rows, _ = v1_parent_rows(spec['recovery_output']); return rows, 'V1_SAVED_PARENT_ORIGINAL_PROMPT_HASH_REUSED'
        return _rows(stage_output(spec, 'EP') / 'rows_PRIVATE', 'Parent'), 'V2_EP_CONCISE_PROMPT'
    if arm_terminal(spec, label)['alias_of_parent']:
        rows, origin = model_rows(spec, 'Parent'); return rows, 'PARENT_ALIAS_ZERO_UPDATES'
    return _rows(stage_output(spec, 'E' + label) / 'rows_PRIVATE', label), 'V2_E' + label


def _strip(rows):
    return [{'id': r['id'], 'truth': r['truth'], 'text': r['text'], 'tokens': r.get('tokens', len(r['actual_PRIVATE']['response']['token_ids'])),
             'wall_seconds': r.get('wall_seconds', 0.), 'actual_PRIVATE': r['actual_PRIVATE']} for r in rows]


def arm_terminal_segment0(spec, arm):
    from mri_rl.accuracy_pilot_v2_native import stage_output
    return core.verify_segment_receipt(stage_output(spec, arm) / 'arm_PRIVATE' / 'segment_0')


def final_analysis(spec, scope):
    from mri_rl.accuracy_pilot_v2_native import load_decision, arm_terminal
    dev = load(spec['dev24']); decision = load_decision(spec)
    loaded = {label: model_rows(spec, label) for label in ('Parent', 'A', 'B')}
    rows = {label: _strip(value[0]) for label, value in loaded.items()}
    origins = {label: value[1] for label, value in loaded.items()}
    metrics = {label: core.greedy_summary(r) for label, r in rows.items()}
    v1_parent = _strip(v1_parent_rows(spec['recovery_output'])[0])
    pairs = core.dev_pairs(dev['records_PRIVATE'])
    substitution = core.substitution_equivalence(pairs, rows['Parent'])
    pair_vs_parent = {k: core.compare_to_reference(pairs, rows['Parent'], rows[k]) for k in ('A', 'B')}
    pair_b_vs_a = core.compare_to_reference(pairs, rows['A'], rows['B'])
    valid = core.valid_both_correct(rows['Parent'], rows['B'])
    seg0 = {arm: arm_terminal_segment0(spec, arm) for arm in ('A', 'B')}
    mechanism = {arm: {k: seg0[arm][k] for k in ('effective_groups', 'admitted_responses', 'zero_variance_groups', 'joint_gradient_l2', 'updated')} for arm in seg0}
    mechanism['identical_bundles'] = seg0['A']['bundle_sha256'] == seg0['B']['bundle_sha256']
    return {'mechanism_segment0_same_32_responses': mechanism,
            'schema': 'accuracy-pilot-v2-analysis-v1', 'prompt_decision': decision, 'row_origins': origins, 'metrics': metrics,
            'parent_original_prompt_V1_metrics': core.greedy_summary(v1_parent),
            'effects': core.effect_decomposition(core.greedy_summary(v1_parent), metrics['Parent'], metrics['A'], metrics['B']),
            'pair_substitution_equivalence': substitution, 'pairs_vs_parent': pair_vs_parent, 'pairs_B_vs_A': pair_b_vs_a, 'valid_both_B_vs_Parent': valid,
            'training': {arm: {k: v for k, v in arm_terminal(spec, arm).items() if k != 'receipt'} for arm in ('A', 'B')},
            'scale_decision': core.scale_decision(metrics, pair_vs_parent['B'], valid, pair_b_vs_a), 'model_calls_in_analysis': 0}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--spec', type=Path, required=True)
    mode = p.add_mutually_exclusive_group(); mode.add_argument('--launch', action='store_true'); mode.add_argument('--resume', action='store_true')
    mode.add_argument('--child-stage', choices=STAGES); p.add_argument('--attempt', type=int, default=0); p.add_argument('--scope-output', type=Path)
    for name in ('scope', 'human', 'review'):
        p.add_argument('--' + name, type=Path); p.add_argument('--expected-' + name + '-sha256')
    args = p.parse_args(argv)
    if args.launch or args.resume or args.child_stage:
        require(all(getattr(args, n, None) and getattr(args, 'expected_' + n + '_sha256', None) for n in ('scope', 'human', 'review')), 'Exact artifacts required')
        return execute_child(args) if args.child_stage else execute_chain(args)
    result = prepare(args.spec)
    if args.scope_output:
        from mri_rl.shared_rollout_gspo_core_v1 import publish_once
        publish_once(_output(args.scope_output), result['executable_scope'])
    print(json.dumps({k: v for k, v in result.items() if k != 'executable_scope'})); return 0
