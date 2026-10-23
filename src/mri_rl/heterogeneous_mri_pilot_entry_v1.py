"""Heterogeneous MRI pilot exact-scope authority: CPU census/prepare, then parent -> terminal child phases, then CPU analysis.

Preparation reads metadata only and imports no MLX. It RE-DERIVES the executed Accuracy Pilot V2 scope (v2 entry prepare,
allow_started=True, CPU binding only - never its grant) and inherits its whole transitive source/input closure (runtime, contract,
calibration, cache certificate, Parent receipts, capsules, DEV24); it authenticates every reused V2 EP Parent DEV24 row through its
committed journal record; it binds the VQA-RAD source QA JSON, the OSF download receipt, every selected image, the frozen
border-control artifacts and the independent census oracle. The result is BLOCKED_PENDING_PROCESSOR_MEASUREMENT until the frozen
midline prompts are measured with the pinned CPU processor, then PREPARED_PENDING_EXACT_REVIEW_AND_FRESH_HUMAN. Execution needs an
isolated non-Claude review AND a fresh human grant (<=30 min old, never seen in any historical consumed ledger, valid for the whole
bounded flow), consumed once. Resource colors/thresholds are the unchanged safe_micro_dynamic_resources_v1.POLICY. No retry.
"""
from __future__ import annotations
import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from mri_rl import heterogeneous_mri_pilot_v1 as hp
from mri_rl import accuracy_recovery_v1 as a
from mri_rl import accuracy_pilot_v2_core as v2core
from mri_rl import multistep_gspo_core_v1 as ms
from mri_rl import safe_micro_dynamic_resources_v1 as resources
from mri_rl.safe_micro_entry_v1 import _closure
from mri_rl.safe_micro_journal_v1 import Journal
from mri_rl.safe_micro_watchdog_v2 import POLICY as GUARDIAN_POLICY
from mri_rl.shared_rollout_gspo_core_v1 import publish_once, consume_once, identity

ROOT = Path(__file__).resolve().parents[2]
CANONICAL = a.canonical_project_root(Path(__file__))
PILOT_DIR = CANONICAL / 'local_data/heterogeneous_mri_pilot_v1'
V2_DIR = CANONICAL / 'local_data/accuracy_pilot_v2'
V2_SPEC, V2_SPEC_SHA256 = V2_DIR / 'PILOT_V2_SPEC_PRIVATE.json', 'be933948d1e8680e15c3a393ecac0beb409c14c99185d4cb6814ced0acfffb87'
V2_EXECUTED_SCOPE = V2_DIR / 'PILOT_V2_EXECUTION_SCOPE_PRIVATE.json'
SCHEMA, SPEC_SCHEMA = 'heterogeneous-mri-pilot-v1-scope', 'heterogeneous-mri-pilot-v1-spec'
MEASUREMENT_SCHEMA = 'heterogeneous-midline-prompt-length-measurement-v1'
KIND = 'FRESH_EXACT_HETEROGENEOUS_MRI_PILOT_V1_SCOPE'
ACTION = 'heterogeneous_mri_pilot_v1_gate6_parent_midline2_rollout16_one_joint_update_terminal_DEV24_midline2_once'
IMPLEMENTED_BY = 'Claude Code (claude-opus-5-5)'
PHASES = ('parent', 'terminal')
MODEL_POLICY = dict(resources.POLICY)
PINNED = {'midline_manifest': '68fad7db7c68b18778469bc8c78d86ede61e15b2477f34ee8fc449fb0f2350af',
          'source_qa': '948b7156059b864cf72344fb10669233526b175c243e5a046d20d4201d297a95'}
SPEC_FILES = ('midline_manifest', 'source_qa', 'osf_receipt', 'control_inputs', 'independent_census')
V2_CONCISE_SEQUENCE_LENGTH = 233
DATA_SCALE = 'DATA_TOO_SMALL_FOR_LEARNING_CLAIM'
RUN_LIMITS = {'model_loads_max': 2, 'max_attempts_per_phase': 1, 'automatic_recoveries': 0, 'incomplete_operation': 'INDETERMINATE_NO_REPLAY',
              'single_resident_model': True, 'global_window_seconds': 4800, 'nonresident_good_samples': 3, 'nonresident_sample_seconds': 5.,
              'waits_count_against_windows': True}
# Worst case from the unchanged V2 cost model (.25 s/token, +3 s/call, 90 s load+release, 5 s/old forward, 10 s/backward, 15 s/save):
#   parent   90 + 8 greedy*67 + 16 sampled*67 + 16*5 + 16*10 + 15 = 1953 s;  terminal 90 + 26*67 = 1832 s.  Windows add margin; waits count.
WINDOWS = {'parent': 2400, 'terminal': 2400}
WORST_CASE_SECONDS = {'parent': 90 + 8 * (256 * .25 + 3) + 16 * (256 * .25 + 3) + 16 * 5 + 16 * 10 + 15, 'terminal': 90 + 26 * (256 * .25 + 3)}
GREEDY_PER_PHASE = {'parent': len(hp.GATE_NAMES) + 2, 'terminal': 26}
require, H = a.require, a.file_sha256
digest = a.scope_digest
_ISSUER = object()


def load(path):
    return json.loads(Path(path).read_text())


def _inside(path, base):
    path = Path(path)
    require(not any(p.is_symlink() for p in (path, *path.parents) if p.exists()), 'Symlinked path refused: ' + str(path))
    resolved = path.resolve()
    require(resolved.is_relative_to(Path(base).resolve()), 'Path must stay under ' + str(base))
    return resolved


# ================================================================ CPU preparation helpers
def midline_records(manifest):
    require(manifest.get('schema') == 'vqarad_midline_source_candidate_v1' and manifest.get('training_authorized') is False, 'Exact candidate manifest required')
    rows = [hp.midline_row(r) for r in manifest['rows'] if r['split'] in ('train', 'eval')]
    by = {s: [r for r in rows if r['extra_info']['split'] == s] for s in ('train', 'eval')}
    for s in by:
        require(sorted(r['reward_model']['ground_truth'] for r in by[s]) == ['no', 'yes'], 'Existing 2 TRAIN + 2 EVAL label-balanced split required')
    require(not {r['extra_info']['case_proxy_sha256'] for r in by['train']} & {r['extra_info']['case_proxy_sha256'] for r in by['eval']}, 'Case-proxy overlap')
    return by['train'], by['eval']


def measured_rows(train, gate, evaluation):
    """Every distinct midline model input (TRUE controls reuse the TRAIN rows): 2 TRAIN, 2 SWAP, 2 BORDER, 2 EVAL."""
    rows = {r['id']: r for r in train}
    rows.update({c['row']['id']: c['row'] for c in gate})
    rows.update({r['id']: r for r in evaluation})
    require(len(rows) == 8, 'Exactly 8 distinct midline model inputs expected')
    return rows


def rows_identity(rows):
    return identity([{'id': i, 'question': r['extra_info']['question'], 'image_sha256': r['image_sha256']} for i, r in sorted(rows.items())])


def measured_lengths(path, train, gate, evaluation):
    """Per-row expanded prompt lengths from the pinned-processor CPU measurement bound to these exact rows; None = blocker."""
    rows = measured_rows(train, gate, evaluation)
    want = {'midline_train': {r['id']: r['id'] for r in train}, 'midline_gate': {c['name']: c['row']['id'] for c in gate},
            'midline_eval': {r['id']: r['id'] for r in evaluation}}
    if not path:
        return {'sequence': V2_CONCISE_SEQUENCE_LENGTH, 'by_row': {i: None for i in rows}, **{k: {n: None for n in v} for k, v in want.items()}}, 'PENDING_PROCESSOR_MEASUREMENT'
    m = load(path)
    require(m.get('schema') == MEASUREMENT_SCHEMA and m.get('model_calls') == 0 and m.get('mlx_imported') is False, 'Exact processor-only measurement required')
    require(m['rows_identity'] == rows_identity(rows) and m['midline_suffix_sha256'] == hp.MIDLINE_SUFFIX_SHA256, 'Measurement is not bound to these frozen rows/suffix')
    require(set(m['by_row']) == set(rows) and all(type(v) is int and 0 < v <= 2048 for v in m['by_row'].values()), 'Incomplete measurement')
    require(m['sequence_concise_check'] == {'expected': V2_CONCISE_SEQUENCE_LENGTH, 'actual': [V2_CONCISE_SEQUENCE_LENGTH] * 2},
            'Method check failed: the frozen V2 concise sequence rows must re-measure to 233')
    out = {'sequence': V2_CONCISE_SEQUENCE_LENGTH, 'by_row': dict(m['by_row'])}
    out.update({k: {n: m['by_row'][rid] for n, rid in v.items()} for k, v in want.items()})
    return out, 'MEASURED'


def authenticate_parent_rows(ep_dir, expected_binding, dev_ids):
    """Read-only authentication of the copied V2 EP Parent DEV24 rows: each row file must be exactly the item of its committed,
    reservation-bound journal record (Journal.get re-hashes every committed file); order/condition/ids must match DEV24.
    Never takes the historical writer lock or writes anything in the V2 tree."""
    ep_dir = Path(ep_dir)
    binding = load(ep_dir / 'v2_journal_PRIVATE' / 'binding_PRIVATE.json')
    require(binding == expected_binding, 'EP journal binding is not the executed V2 compound scope/input')
    journal = Journal.__new__(Journal)
    journal.path, journal.binding, journal.fd = (ep_dir / 'v2_journal_PRIVATE').resolve(), binding, None
    rows, files = [], {}
    for i, rid in enumerate(dev_ids):
        key = 'Parent_row_%02d' % i
        value = journal.get(key)
        require(value is not None, 'Missing committed EP row ' + key)
        item, p = value['item'], ep_dir / 'rows_PRIVATE' / (key + '_PRIVATE.json')
        require(item['key'] == key and item['id'] == rid and item['condition'] == 'concise' and load(p) == item, 'EP row differs from its committed record: ' + key)
        for name in ('reservation_PRIVATE.json', 'commit_PRIVATE.json'):
            files[str((journal.path / key / name).resolve())] = H(journal.path / key / name)
        files[str(p.resolve())] = H(p)
        rows.append({'task': hp.SEQ_TASK, 'id': rid, 'truth': item['truth'], 'text': item['text'], 'tokens': item['tokens']})
    return rows, files


def census_agrees(census, oracle):
    agg = oracle['aggregate']
    ours = {'QA': census['midline_yesno_head_questions'], 'evaluated': census['evaluated'], 'eligible': {k: census['eligible_case_components'][k] for k in ('yes', 'no')},
            'mass_shift': census['mass_shift_ambiguous_questions']}
    theirs = {'QA': agg['explicit_midline_shift_all']['QA_rows'], 'evaluated': agg['explicit_midline_shift_all']['QA_evaluation_counts']['evaluated'],
              'eligible': agg['explicit_midline_shift_evaluated_MRI_source_QA_supported']['component_answer_counts'],
              'mass_shift': agg['ambiguous_mass_shift_separate']['QA_rows']}
    return ours == theirs, ours, theirs


def prepare(specpath, *, allow_started=False):
    from mri_rl import accuracy_pilot_v2_entry as v2entry
    from mri_rl.accuracy_pilot_v2_native import stage_output, load_decision
    specpath = _inside(specpath, PILOT_DIR)
    spec = load(specpath); require(spec.get('schema') == SPEC_SCHEMA, 'Exact pilot spec schema required')
    require(Path(spec['v2_spec']).resolve() == V2_SPEC.resolve() and H(V2_SPEC) == V2_SPEC_SHA256 == spec['v2_spec_sha256'], 'Must bind the executed V2 spec')
    files = {str(specpath): H(specpath), str(V2_SPEC.resolve()): V2_SPEC_SHA256}
    for key in SPEC_FILES:
        h = H(spec[key]); require(h == spec[key + '_sha256'] and (key not in PINNED or h == PINNED[key]), 'Bound input changed: ' + key)
        files[str(Path(spec[key]).resolve())] = h
    # ---- the executed V2 scope and its whole transitive closure (CPU binding only; its grant is never reused)
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
    parent_rows, row_files = authenticate_parent_rows(ep, {'scope': digest(v2scope), 'stage': 'EP', 'input': v2scope['input_sha256']}, [r['id'] for r in dev])
    files.update(row_files)
    stage_input = load(v2['stage_input'])
    unit0 = v2core.condition_units(stage_input['units_PRIVATE'][:1], 'concise')[0]
    require(unit0['kind'] == ms.SEQ_KIND and unit0['index'] == 0, 'Existing frozen TRAIN sequence pair (unit 0) required')
    seq_images = {b['record_id_PRIVATE']: b['image_binding_PRIVATE']['sha256'] for b in stage_input['planned_image_bindings_PRIVATE']}
    # ---- midline rows, images (OSF receipt), border controls, census oracle
    train, evaluation = midline_records(load(spec['midline_manifest']))
    osf = {r['image_name']: r['osf_sha256'] for r in load(spec['osf_receipt'])}
    for r in train + evaluation:
        image = Path(r['images'][0]); h = H(image)
        require(h == r['image_sha256'] == osf.get(image.name) and _inside(image, CANONICAL / 'local_data'), 'Selected image differs from its OSF payload')
        files[str(image.resolve())] = h
    borders = hp.border_controls(load(spec['control_inputs']), train)
    for b in borders.values():
        p = _inside(b['path'], PILOT_DIR); require(H(p) == b['sha256'], 'Border-control image changed'); files[str(p)] = b['sha256']
    gate = hp.gate_controls(train, borders)
    census = hp.census(load(spec['source_qa']))
    agrees, ours, theirs = census_agrees(census, load(spec['independent_census']))
    lengths, length_status = measured_lengths(spec.get('midline_prompt_lengths'), train, gate, evaluation)
    if spec.get('midline_prompt_lengths'):
        p = _inside(spec['midline_prompt_lengths'], PILOT_DIR); files[str(p)] = H(p)
    blockers = []
    if not agrees:
        blockers.append('CENSUS_DISAGREES_WITH_INDEPENDENT_ORACLE: %s vs %s' % (ours, theirs))
    require(census['decision'] == DATA_SCALE, 'Census must not be promoted beyond the existing evaluated admission')
    budget = None
    if length_status != 'MEASURED':
        blockers.append('PENDING_PROCESSOR_MEASUREMENT: frozen midline prompts not yet measured with the pinned CPU processor')
    else:
        budget = hp.budget({k: lengths[k] for k in ('sequence', 'midline_train', 'midline_gate', 'midline_eval')})
    pilot_input = {'sequence_unit_PRIVATE': unit0, 'sequence_image_sha256': {c['row']['id']: seq_images[c['row']['id']] for c in unit0['contexts']},
                   'midline_train_PRIVATE': train, 'midline_eval_PRIVATE': evaluation, 'border_controls_PRIVATE': borders, 'prompt_lengths': lengths,
                   'parent_DEV24_rows_PRIVATE': parent_rows, 'census': census}
    own = ('src/mri_rl/heterogeneous_mri_pilot_v1.py', 'src/mri_rl/heterogeneous_mri_pilot_native_v1.py', 'src/mri_rl/heterogeneous_mri_pilot_entry_v1.py',
           'scripts/run_heterogeneous_mri_pilot_v1.py', 'scripts/measure_heterogeneous_midline_prompts_v1.py')
    sources = _closure({str(ROOT / p): H(ROOT / p) for p in own})
    for p, h in v2scope['source_sha256'].items():
        require(H(p) == h and sources.get(p, h) == h, 'Inherited V2 run source changed'); sources[p] = h
    out = _inside(spec['output'], PILOT_DIR); require(allow_started or not out.exists(), 'Fresh pilot output required')
    scope = {'schema': SCHEMA, 'scope_kind': KIND, 'action': ACTION, 'implemented_by': IMPLEMENTED_BY, 'source_sha256': sources, 'file_sha256': files,
             'input_sha256': identity(files), 'pilot_input_sha256': identity(pilot_input), 'output': str(out), 'resource_policy': MODEL_POLICY,
             'external_guardian_policy': GUARDIAN_POLICY, 'run_limits': RUN_LIMITS, 'windows_seconds': WINDOWS, 'worst_case_seconds': WORST_CASE_SECONDS,
             'budget': budget, 'phases': list(PHASES), 'parent': dict(a.FROZEN_PARENT), 'v2_executed_scope_sha256': digest(v2scope),
             'task_weight': hp.TASK_WEIGHT, 'midline_suffix_sha256': hp.MIDLINE_SUFFIX_SHA256, 'gate_rule': hp.image_dependence_gate(dict.fromkeys(hp.GATE_NAMES))['rule'],
             'gate_controls': list(hp.GATE_NAMES), 'no_image_control': hp.NO_IMAGE_CONTROL, 'data_scale': DATA_SCALE,
             'admission': 'existing evaluated MRI rule (5 case components); 3 not-evaluated MRI components reported, NOT promoted',
             'parent_DEV24': 'V2_EP_CONCISE_ROWS_JOURNAL_AUTHENTICATED', 'block_transport': hp.TRANSPORT,
             'forbidden_inputs': ['final100', 'sealed new16 outputs'], 'automatic_retry': False, 'third_arm': False}
    status = 'BLOCKED' if blockers else 'PREPARED_PENDING_EXACT_REVIEW_AND_FRESH_HUMAN'
    return {'status': status, 'blockers': blockers, 'executable_scope': scope, 'scope_sha256': digest(scope), 'pilot_input_PRIVATE': pilot_input,
            'census': census, 'measurement_rows_PRIVATE': measured_rows(train, gate, evaluation), 'native_calls': 0}


def public_census(census):
    """Counts only (no qid, image name, question text or answer per item)."""
    keys = ('source_rows', 'midline_yesno_head_questions', 'mass_shift_ambiguous_questions', 'related_paraphrase_questions', 'matched_spellings',
            'by_modality_evidence', 'evaluated', 'not_evaluated', 'evaluation_status', 'eligible_case_components', 'eligible_without_case_component',
            'MRI_components_including_not_evaluated_NOT_ADMITTED', 'admission_rule', 'threshold', 'decision', 'modality_rule', 'mass_shift_rule', 'split_unit')
    return {k: census[k] for k in keys}


# ================================================================ authorization (pattern of the executed V2 entry; its grant is never reused)
def historical_grants(root=CANONICAL / 'local_data'):
    """Every human-instruction SHA already consumed by ANY earlier once-only ledger under local_data (file names only)."""
    found = set()
    for pattern in ('*/*/authorization_consumed_*.json', '*/*/*/authorization_consumed_*.json'):
        found.update(p.stem.split('authorization_consumed_', 1)[1] for p in Path(root).glob(pattern))
    return found


class Authorization:
    def __init__(self, issuer, scope, expires, parent=None):
        require(issuer is _ISSUER, 'Only exact admission may issue authorization')
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
        require(self.parent is None and phase in PHASES, 'Only pilot phases derive from compound authority')
        self.check('phase_admission')
        return Authorization(_ISSUER, phase_scope(self.scope, phase), self.expires, self)


def phase_scope(scope, phase):
    require(phase in PHASES and scope['budget'] is not None, 'Measured budget required before any phase scope')
    base = Path(scope['output']); greedy = GREEDY_PER_PHASE[phase]; parent = phase == 'parent'
    reservation = {'greedy_calls': greedy, 'sampled_rollout_responses': 16 if parent else 0, 'backward_calls': scope['budget']['backward_calls_max'] if parent else 0,
                   'optimizer_updates': int(parent), 'generated_response_tokens': ((16 if parent else 0) + greedy) * hp.MAX_TOKENS}
    return {**scope, 'phase': phase, 'compound_scope_sha256': digest(scope), 'action': ACTION + '_' + phase, 'resource_stage': 'B_MICRO',
            'output': str(base / (phase + '_PRIVATE')), 'parent_phase_output': str(base / 'parent_PRIVATE' / 'parent_phase_PRIVATE'),
            'pilot_input_path': str(base / 'pilot_input_PRIVATE.json'), 'window_seconds': WINDOWS[phase],
            'budget': {'greedy_calls_phase': greedy, 'phase_reservation': reservation}}


def _check(scope, human, review):
    require(scope.get('schema') == SCHEMA and scope.get('scope_kind') == KIND and scope.get('action') == ACTION and scope.get('implemented_by') == IMPLEMENTED_BY and
            scope.get('resource_policy') == MODEL_POLICY and scope.get('external_guardian_policy') == GUARDIAN_POLICY and scope.get('run_limits') == RUN_LIMITS and
            scope.get('windows_seconds') == WINDOWS and scope.get('budget') is not None and scope.get('automatic_retry') is False, 'Exact conservative measured pilot scope required')
    require(review.get('verdict') in ('PASS', 'PASS_WITH_LIMITATIONS') and review.get('scope_sha256') == digest(scope) and review.get('source_sha256') == scope['source_sha256'] and
            review.get('input_sha256') == scope['input_sha256'] and review.get('reviewer_kind') == 'isolated_codex' and review.get('reviewer_id') and
            'claude' not in str(review['reviewer_id']).lower(), 'Independent exact review by a non-Claude model family required')
    require(human.get('authorized') is True and human.get('action') == ACTION and human.get('scope_sha256') == digest(scope) and
            human.get('included_phases') == list(PHASES) and a.sha(human.get('human_instruction_sha256')) and human.get('approved_by'),
            'Fresh explicit human permission naming both pilot phases required')


def authorize(scope, human, review, *, consumed, history, now=None):
    """`history`: instruction SHAs consumed by any earlier ledger (old V2/V1/micro grants are never borrowed)."""
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


class PilotResourceGate(resources.ResourceGate):
    """Unchanged color/threshold gate; only the model-load ledger limit is the pilot's 2."""
    def claim_model_load(self):
        from mri_rl.shared_rollout_gspo_core_v1 import replace_atomic
        self.authorization.check('hetero_model_load_claim')
        path = Path(self.authorization.parent.scope['output']) / 'model_load_ledger_PRIVATE.json'
        rec = load(path) if path.exists() else {'scope': self.scope['compound_scope_sha256'], 'claimed': 0, 'limit': RUN_LIMITS['model_loads_max']}
        require(rec['scope'] == self.scope['compound_scope_sha256'] and rec['limit'] == 2 and 0 <= rec['claimed'] < 2, 'Pilot model-load budget exhausted')
        rec['claimed'] += 1; replace_atomic(path, rec)


def wait_nonresident(token, phase_window_start, window, observe, sleep=time.sleep, clock=time.monotonic):
    """Bounded non-resident wait before a model attempt is consumed; waiting counts against the phase window. STOP is final."""
    good = 0
    while True:
        token.check('nonresident_wait')
        require(clock() - phase_window_start < window, 'Phase window exhausted while waiting non-resident')
        result = observe()
        if result['action'] == 'STOP':
            raise resources.ResourceFatal(','.join(result['reasons']))
        good = good + 1 if result['action'] == 'CONTINUE' else 0
        if good >= RUN_LIMITS['nonresident_good_samples']:
            return
        sleep(RUN_LIMITS['nonresident_sample_seconds'])


# ================================================================ execution
def _credentials(args):
    for name in ('scope', 'human', 'review'):
        require(H(getattr(args, name)) == getattr(args, 'expected_' + name + '_sha256'), 'Exact credential FILE hash changed: ' + name)
    scope = load(args.scope); prepared = prepare(args.spec, allow_started=True)
    require(prepared['status'] == 'PREPARED_PENDING_EXACT_REVIEW_AND_FRESH_HUMAN' and scope == prepared['executable_scope'], 'Exact measured preparation differs')
    return scope, prepared, load(args.human), load(args.review)


def pre_native_runtime_check(v2, *, installed=None, binding=None):
    """The executed V2 child's CPU admission, reused unchanged: the pinned runtime's installed fingerprint must equal the historical
    cache scope, and the derived cache must verify against its historical certificate with the exact SHARDS. Metadata/hash checks
    only - no MLX import, no quantization, no cache reconstruction, no model load."""
    from types import SimpleNamespace
    from mri_rl.safe_micro_recovery_entry_v1 import SHARDS
    if installed is None:
        from mri_rl.accuracy_recovery_gpu_cache_v2 import installed_fingerprint as installed
    if binding is None:
        from mri_rl.continuous_rl_pilot import cache_binding_for as binding
    rspec, runtime = load(v2['recovery_spec']), load(v2['runtime_spec'])
    expected = load(rspec['cache_scope'])['installed']
    require(installed() == expected, 'Pinned native runtime differs from the historical cache scope')
    bound = binding(SimpleNamespace(cache_dir=Path(runtime['cache_dir']), cache_certificate=Path(runtime['cache_certificate']),
                                    cache_certificate_sha256=H(runtime['cache_certificate']), model=Path(runtime['model'])))
    shards = {n: v['sha256'] for n, v in bound['shards'].items()}
    require(shards == SHARDS, 'Cache payload mismatch')
    return {'installed_matches_cache_scope': True, 'shards': shards, 'cache_certificate_sha256': H(runtime['cache_certificate']),
            'quantization_or_cache_rebuild': False, 'model_loaded': False}


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
        gate = PilotResourceGate('B_MICRO', {**v2, 'output': scope['output']}, phase.scope, phase)
        signal.signal(signal.SIGUSR1, lambda *unused: setattr(gate, 'red_requested', True))
        gate('native_model_load_preflight')
        publish_once(Path(phase.scope['output']).parent / (args.child_phase + '_runtime_admission_PRIVATE.json'), pre_native_runtime_check(v2))
        phase.check('before_native_import')
        from mri_rl.heterogeneous_mri_pilot_native_v1 import run_child
        result = run_child(args.child_phase, spec=v2, scope=phase.scope, authorization=phase, resource_gate=gate)
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
    hp.authenticated_input(base / 'pilot_input_PRIVATE.json', scope['pilot_input_sha256'])
    publish_once(base / 'launch_identity_PRIVATE.json', {'scope_sha256': token.sha, 'human_instruction_sha256': human['human_instruction_sha256'],
                                                          'started_utc': datetime.now(timezone.utc).isoformat(), 'started_monotonic': token.start})
    runtime = load(load(V2_SPEC)['runtime_spec']); controller = resources.Controller()

    def observe():
        s = sample_macos(); r = controller.decide(s, os_pressure_level(), shutil.disk_usage(base).free / 1024 ** 3, time.monotonic())
        replace_atomic(base / 'nonresident_latest_PRIVATE.json', {'sample': asdict(s), **r}); return r
    results = {}
    for phase in PHASES:
        if phase == 'terminal' and results['parent']['status'] != 'UPDATED':
            publish_once(base / 'terminal_skipped_PRIVATE.json', {'reason': 'ZERO_UPDATES_TERMINAL_IS_PARENT_ALIAS', 'model_loads': 0}); break
        started = time.monotonic()
        publish_once(base / (phase + '_clock_PRIVATE.json'), {'started_monotonic': started})
        wait_nonresident(token, started, WINDOWS[phase], observe)
        argv = [str(python), '-I', str(ROOT / 'scripts/run_heterogeneous_mri_pilot_v1.py'), '--child-phase', phase]
        for name in ('spec', 'scope', 'human', 'review'):
            argv += ['--' + name, str(getattr(args, name))]
        for name in ('scope', 'human', 'review'):
            argv += ['--expected-' + name + '-sha256', getattr(args, 'expected_' + name + '_sha256')]
        wdir = base / (phase + '_exit_PRIVATE')
        witness = witness_child(argv, wdir, scope_sha256=args.expected_scope_sha256, sampler=lambda pid: sample_macos(pid=pid), level=os_pressure_level,
                                disk_free_gib=lambda: shutil.disk_usage(base).free / 1024 ** 3, post_sampler=lambda: asdict(sample_macos()),
                                lock_inspector=lambda: inspect_lock(runtime['global_lock']))
        require(witness['post_child_global_lock'].get('no_open_holders') is True, 'Model lock not released')
        require(witness['native_exit_code'] == 0 and not witness['SIGTERM_sent'] and not witness['SIGKILL_after_grace'] and not witness['monitor_log_errors'],
                'Phase did not complete cleanly: STOP (no automatic retry): ' + phase)
        terminal = load(base / (phase + '_PRIVATE') / 'terminal_PRIVATE.json')
        require(terminal['release_verified'] is True, 'Phase release not verified')
        results[phase] = terminal
    analysis = final_analysis(base, results)
    publish_once(base / 'analysis_PRIVATE.json', analysis)
    print(json.dumps({k: analysis[k] for k in ('outcome', 'gate_reason', 'data_scale')})); return 0


def final_analysis(base, results):
    """CPU only: Parent = journal-authenticated V2 EP concise DEV24 rows + new Parent midline EVAL; terminal = Parent alias if 0 updates."""
    pinput = load(base / 'pilot_input_PRIVATE.json')
    pdir = base / 'parent_PRIVATE' / 'parent_phase_PRIVATE'
    gate = load(pdir / 'gate_PRIVATE.json')['gate']
    if not (pdir / 'parent_midline_eval_PRIVATE.json').exists():
        return {'outcome': 'NO_SIGNAL', 'gate_reason': gate['reason'], 'data_scale': DATA_SCALE, 'gate': gate,
                'evaluation': 'NOT_RUN_GATE_FAILED', 'optimizer_updates': 0, 'model_calls_in_analysis': 0}
    parent = hp.evaluation_summary(pinput['parent_DEV24_rows_PRIVATE'] + load(pdir / 'parent_midline_eval_PRIVATE.json')['items'])
    updated = results['parent']['status'] == 'UPDATED'
    terminal = hp.evaluation_summary(results['terminal']['items_PRIVATE']) if updated else parent
    return {**hp.outcome(parent, terminal, updated=updated, data_scale=DATA_SCALE), 'gate_reason': gate['reason'], 'gate': gate,
            'parent': parent, 'terminal': terminal, 'parent_DEV24_origin': 'V2_EP_CONCISE_ROWS_JOURNAL_AUTHENTICATED', 'model_calls_in_analysis': 0}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--spec', type=Path)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--census', type=Path, help='CPU: count source midline QA (prints counts only)')
    mode.add_argument('--launch', action='store_true'); mode.add_argument('--child-phase', choices=PHASES)
    p.add_argument('--scope-output', type=Path)
    for name in ('scope', 'human', 'review'):
        p.add_argument('--' + name, type=Path); p.add_argument('--expected-' + name + '-sha256')
    args = p.parse_args(argv)
    if args.census:
        print(json.dumps(public_census(hp.census(load(args.census))), sort_keys=True)); return 0
    require(args.spec is not None, '--spec required')
    if args.launch or args.child_phase:
        require(all(getattr(args, n, None) and getattr(args, 'expected_' + n + '_sha256', None) for n in ('scope', 'human', 'review')), 'Exact artifacts required')
        return execute_child(args) if args.child_phase else execute_launch(args)
    result = prepare(args.spec)
    if args.scope_output and result['status'] == 'PREPARED_PENDING_EXACT_REVIEW_AND_FRESH_HUMAN':
        publish_once(_inside(args.scope_output, PILOT_DIR), result['executable_scope'])
    print(json.dumps({'status': result['status'], 'blockers': result['blockers'], 'scope_sha256': result['scope_sha256'],
                      'census': public_census(result['census']), 'budget': result['executable_scope']['budget'],
                      'files_bound': len(result['executable_scope']['file_sha256']), 'sources_bound': len(result['executable_scope']['source_sha256']),
                      'native_calls': 0}, sort_keys=True))
    return 0
