"""Accuracy Pilot V2 native stages. SOURCE_IMPLEMENTED_NOT_NATIVE_RUN.

Importing this module imports no MLX and loads nothing. Every stage is one fresh
child process holding the global lock, one resident model, the proven detached
RecoveryBackend restore path, a durable journal (claimed-but-uncommitted work is
INDETERMINATE and never replayed) and an actual allocator-release proof before
its completion commit. Exact fresh human authority is required (entry module).

Stages: P prompt diagnostic (Parent, 8 TRAIN images x 2 prompts, greedy)
        C common parent rollout of units 0-3 under the frozen prompt (once)
        A / B independent fresh-disk Adam52 arm processes, <=2 joint updates each
        EP / EA / EB DEV24 greedy under the frozen prompt (EP skipped when the
        original prompt is kept: the saved V1 Parent rows are hash-reused).
"""
from __future__ import annotations
import ast
import gc
import hashlib
import inspect
import json
import textwrap
import time
import types
from pathlib import Path
from mri_rl import accuracy_pilot_v2_core as core
from mri_rl import multistep_gspo_core_v1 as ms
from mri_rl import shared_rollout_gspo_core_v1 as shared
from mri_rl import safe_micro_pilot_native_v1 as pilot
from mri_rl import safe_micro_pilot_native_v2 as resumable
from mri_rl import safe_micro_training_v2 as durable
from mri_rl.safe_micro_recovery_native_v1 import RecoveryBackend, drop_snapshot_references, release_before_commit, full_identity
from mri_rl.safe_micro_journal_v1 import Journal, durable_mkdir

require = shared.require
STATUS = 'SOURCE_IMPLEMENTED_NOT_NATIVE_RUN'
STAGES, EXECUTION = core.STAGES, core.EXECUTION
PER_CALL = {'LM': 260, 'sampler': 257}           # same per-greedy-call ceilings as the executed V1 DEV24 counter


def call_limits(calls):
    return {'generations': calls, 'generated_tokens': calls * core.MAX_TOKENS, 'LM': calls * PER_CALL['LM'], 'vision': calls, 'sampler': calls * PER_CALL['sampler']}


# ================================================================ backend: unchanged context AST, exact frozen prompt length
def _context_node():
    from mri_rl import shared_rollout_gspo_backend_v1 as module
    tree = ast.parse(textwrap.dedent(inspect.getsource(module.SharedRolloutBackend._prepare_context)))
    return module, tree.body[0]


def exact_prompt_cost(lengths):
    frozen = dict(lengths)
    def require_prompt_cost(task, length):
        require(task in frozen and length == frozen[task], 'Actual prompt length differs from the frozen V2 prompt-condition ledger')
        return length
    return require_prompt_cost


class PilotV2Backend(RecoveryBackend):
    """RecoveryBackend whose context preparation is the UNCHANGED SharedRolloutBackend._prepare_context AST, compiled in a
    copied namespace where only `require_prompt_cost` checks the frozen length of the active prompt condition."""
    def configure_v2(self, *, condition, prompt_lengths):
        require(condition in core.CONDITIONS, 'Frozen prompt condition required')
        module, node = _context_node()
        namespace = dict(vars(module))
        namespace['require_prompt_cost'] = exact_prompt_cost(prompt_lengths)
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<unchanged-_prepare_context-v2-length>', 'exec'), namespace)
        self._v2_prepare = types.MethodType(namespace['_prepare_context'], self)
        self.v2_condition, self.v2_prompt_lengths = condition, dict(prompt_lengths)
        self.v2_context_ast_sha256 = hashlib.sha256(ast.dump(node).encode()).hexdigest()

    def _prepare_context(self, modality):
        require(getattr(self, '_v2_prepare', None) is not None, 'configure_v2 must bind the frozen prompt condition first')
        return self._v2_prepare(modality)

    def close(self):
        self._v2_prepare = None
        super().close()


# ================================================================ greedy row: V1 generate_one with the frozen-condition length
def prepared_inputs_v2(backend, row, lengths):
    from mlx_vlm.generate.dispatch import _prepare_generation_inputs
    from mri_rl.native_gspo_path_probe_v1 import strict_tensor_identity
    from mri_rl import early_native_conditioning_probe_v1 as utility
    from mri_rl.paired_z045_matched_f32_eval_v1 import greedy_settings
    prompt = backend.template(backend.processor, backend.model.config, [{'role': 'system', 'content': backend.contract.system_instruction},
        {'role': 'user', 'content': row['extra_info']['question']}], num_images=1, add_generation_prompt=True, enable_thinking=backend.contract.enable_thinking)
    ids, pixels, mask, extra = _prepare_generation_inputs(backend.model, backend.processor, prompt, row['images'], None, None, greedy_settings())
    actual = {'input_ids': ids, 'pixel_values': pixels, 'attention_mask': mask, **extra}
    scorer = backend.prepare_inputs(backend.processor, images=row['images'], prompts=prompt, image_token_index=backend.model.config.image_token_index,
                                    add_special_tokens=backend.special_tokens(backend.model.config.model_type, backend.processor))
    strict_tensor_identity(backend, actual, scorer, utility)
    exact_prompt_cost(lengths)(row['ability'], int(ids.shape[-1]))
    require(bool(backend.mx.all(mask == 1).item()), 'Unpadded exact eval context required')
    return prompt, scorer


def generate_one_v2(backend, row, image_sha, *, lengths, counter, before):
    """Greedy only. Identical to the executed V1 generate_one except the exact frozen-condition prompt length."""
    from mlx_vlm.generate.dispatch import stream_generate
    from mri_rl.paired_z045_f32_backend_v1 import F32GenerationModelView, ObservedFP32Processor, capture_stream, reset_effective_eos
    from mri_rl.paired_z045_matched_f32_eval_v1 import greedy_settings
    from mri_rl.native_gspo_path_probe_v1 import strict_tensor_identity
    from mri_rl import early_native_conditioning_probe_v1 as utility
    from mri_rl.paired_z045_continuation_entry_v1 import verify_images
    mx = backend.mx; expected = {row['id']: image_sha}; before('v2_greedy'); verify_images([row], expected)
    prompt, inputs = prepared_inputs_v2(backend, row, lengths); P = int(inputs['input_ids'].shape[-1])
    observed, evidence = [], []
    def observe(ids, pixels, mask, kwargs):
        extra = {k: kwargs[k] for k in inputs if k not in ('input_ids', 'pixel_values', 'attention_mask')}
        observed.append(strict_tensor_identity(backend, {'input_ids': ids, 'pixel_values': pixels, 'attention_mask': mask, **extra}, inputs, utility))
    def guard(name):
        before(name)
        if name == 'f32_generation_LM_call': counter.charge('LM')
        if name == 'f32_generation_vision_embeddings': counter.charge('vision'); verify_images([row], expected)
    def processor(tokens, logits):
        counter.charge('sampler'); return ObservedFP32Processor(mx)(tokens, logits)
    view = F32GenerationModelView(backend.model, mx, guard, evidence, observe); eos = reset_effective_eos(backend.processor.tokenizer, backend.model.config.eos_token_id)
    counter.charge('generations'); settings = greedy_settings(); settings['temperature'] = 0.
    response = capture_stream(stream_generate(view, backend.processor, prompt, image=row['images'], **settings, logits_processors=[processor]),
                              prompt_tokens=P, max_tokens=core.MAX_TOKENS, eos_ids=eos, before=before)
    counter.charge('generated_tokens', len(response['token_ids'])); require(len(observed) == 1 and evidence, 'Actual image preparation/F32 trace missing')
    clean = getattr(backend.processor, 'clean_output', None)
    if callable(clean): response['text'] = clean(response['text'])
    result = {'response': response, 'image_sha256': image_sha, 'input_fingerprints_PRIVATE': observed[0], 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
              'F32_evidence_PRIVATE': evidence, 'prompt_length': P}
    del inputs, view, observed, evidence; gc.collect(); mx.clear_cache()
    return result


def greedy_rows(backend, jobs, *, journal, counter, before, out, session_directory):
    """jobs: [(key, record_id, row, image_sha, condition, lengths)]. Each greedy row + saved RNG is one durable indivisible milestone (as V1)."""
    output = Path(out); durable_mkdir(output); items = []; last_rng = last_digest = None
    for key, record_id, row, image_sha, condition, lengths in jobs:
        cached = journal.get(key)
        if cached is not None:
            resumable._check_completed_row(cached, counter)
            items.append(cached['item']); last_rng, last_digest = cached['rng_dir'], cached['rng_digest']; continue
        if last_rng is not None:
            backend.load_rng(last_rng); require(backend.rng_digest() == last_digest, 'Recovered row RNG differs')
        def operation(opdir, key=key, record_id=record_id, row=row, image_sha=image_sha, condition=condition, lengths=lengths):
            start = time.monotonic()
            actual = generate_one_v2(backend, row, image_sha, lengths=lengths, counter=counter, before=before)
            item = {'key': key, 'id': record_id, 'condition': condition, 'truth': row['reward_model']['ground_truth'], 'text': actual['response']['text'],
                    'tokens': len(actual['response']['token_ids']), 'finish_reason': actual['response'].get('finish_reason'),
                    'actual_PRIVATE': actual, 'wall_seconds': time.monotonic() - start}
            p = output / (key + '_PRIVATE.json'); shared.publish_once(p, item)
            rng = Path(opdir) / 'post_row_RNG_PRIVATE'; durable_mkdir(rng); backend.save_rng(rng)
            value = {'item': item, 'rng_dir': str(rng.resolve()), 'rng_digest': backend.rng_digest(), 'counters': dict(counter.used),
                     'resident_session': str(Path(session_directory).resolve())}
            return {'value': value, 'files': {str(p.resolve()): shared.file_sha256(p), **durable.files_under(rng)}}
        value = journal.run(key, operation, reservation={'generated_responses': 1, 'generated_response_tokens': core.MAX_TOKENS})
        items.append(value['item']); last_rng, last_digest = value['rng_dir'], value['rng_digest']
    return items


# ================================================================ helpers
def _json(path):
    return json.loads(Path(path).read_text())


def stage_output(spec, stage):
    return Path(spec['output']) / ('stage' + stage + '_PRIVATE')


def load_decision(spec):
    terminal = _json(stage_output(spec, 'P') / 'terminal_PRIVATE.json')
    require(terminal['status'] == 'PASS' and terminal['release_verified'] is True, 'Actual prompt diagnostic PASS required')
    decision = terminal['prompt_decision']
    require(decision['decision'] in ('ADOPT_CONCISE', 'KEEP_ORIGINAL') and decision['condition'] in core.CONDITIONS, 'Frozen prompt decision required')
    return decision


def counters_snapshot(opdir, counters):
    """Immutable per-operation counters snapshot for the journal manifest. The mutable spent ledger keeps charging later
    operations; committing IT would let segment 1 invalidate segment 0's committed file hash."""
    p = Path(opdir) / 'counters_after_PRIVATE.json'
    shared.publish_once(p, {'limits': dict(counters.limits), 'used': counters.snapshot()})
    return {str(p.resolve()): shared.file_sha256(p)}


def resume_counters(path, limits, committed):
    """Spent ledger restored for a later operation; never below any committed snapshot (no refund)."""
    used = _json(path)['used'] if Path(path).exists() else (dict(committed[-1]) if committed else None)   # absent only if nothing was ever charged
    for snap in committed:
        require(all(used[k] >= v for k, v in snap.items()), 'Spent ledger is below a committed counters snapshot')
    return shared.Counters(limits, used, lambda u: shared.replace_atomic(path, {'limits': limits, 'used': u}))


def schedule(stage_input):
    units = stage_input['units_PRIVATE']
    images = {x['record_id_PRIVATE']: x['image_binding_PRIVATE']['sha256'] for x in stage_input['planned_image_bindings_PRIVATE']}
    return units, images


def arm_terminal(spec, arm):
    """Authenticated two-segment chain of one arm; never another arm's state."""
    base = stage_output(spec, arm) / 'arm_PRIVATE'
    r0 = core.verify_segment_receipt(base / 'segment_0'); r1 = core.verify_segment_receipt(base / 'segment_1')
    require(r0['arm'] == r1['arm'] == arm and r1['prev_receipt_sha256'] == r0['receipt_sha256'] and r0['segment'] == 0 and r1['segment'] == 1, 'Broken arm receipt chain')
    require(r1['policy_sha_before'] == r0['policy_sha_after'] and r1['adam_digest_before'] == r0['adam_digest_after'] and r1['rng_digest_before'] == r0['rng_digest_after'],
            'Segment 1 did not start from the committed segment-0 state')
    updates = int(r0['updated']) + int(r1['updated'])
    require(r1['adam_step_after'] == core.PARENT_ADAM_STEP + updates, 'Adam step differs from actual updates')
    return {'updates': updates, 'alias_of_parent': updates == 0, 'checkpoint': str(base / 'segment_1' / 'checkpoint_PRIVATE'), 'receipt': r1,
            'policy_sha256': r1['policy_sha_after'], 'receipts_sha256': [r0['receipt_sha256'], r1['receipt_sha256']]}


# ================================================================ one child stage
def run_phase(stage, *, spec, scope, authorization, resource_gate):
    from mri_rl.accuracy_pilot_v2_entry import Authorization
    require(stage in STAGES and isinstance(authorization, Authorization) and authorization.parent is not None and authorization.scope == scope and
            scope.get('execution_stage') == EXECUTION[stage] and scope.get('stage') == stage, 'Exact issued V2 child authority required')
    require(resource_gate.authorization is authorization and resource_gate.scope == scope, 'Exact V2 resource gate required')
    authorization.check('v2_native_import')
    from mri_rl import accuracy_recovery_v1 as recovery
    from mri_rl.accuracy_stage_native_v1 import restore_parent
    from mri_rl.experiment_contract import load_contract
    from mri_rl.two_task_training import make_grader
    from mri_rl.paired_z045_continuation_entry_v1 import run_with_release, verify_images
    runtime = _json(spec['runtime_spec']); stage_input = _json(spec['stage_input']); dev = _json(spec['dev24'])
    out = Path(scope['output']); durable_mkdir(out)
    journal = Journal(out / 'v2_journal_PRIVATE', {'scope': scope['compound_scope_sha256'], 'stage': stage, 'input': scope['input_sha256']}, before=resource_gate)
    try:
        journal.assert_recoverable()
        done = journal.get('native_complete')
        if done is not None:
            return done
        units, images = schedule(stage_input)
        parent = Path(runtime['parent_checkpoint']); parent_disk = recovery.parent_disk_snapshot(parent)
        anchor = recovery.adam52_parent_anchor(parent, runtime['parent_receipt_sha256'])
        calibration = _json(runtime['calibration']); reward_sha = shared.file_sha256(runtime['calibration'])
        verify_images([c['row'] for u in units for c in u['contexts']], images)
        condition = 'original' if stage == 'P' else load_decision(spec)['condition']
        lengths = core.prompt_lengths_for(condition, spec['concise_sequence_prompt_length'] if condition == 'concise' else None)

        def one(label, operation, checkpoint=None, expected=None, condition=condition, lengths=lengths):
            resource_gate('native_model_load')
            directory = out / (label + '_resident_%d_PRIVATE' % time.time_ns()); durable_mkdir(directory); resident = [None]
            def call():
                resource_gate.claim_model_load(); backend = PilotV2Backend.__new__(PilotV2Backend); resident[0] = backend
                backend.__init__((Path(checkpoint) if checkpoint else parent) / 'adapter.safetensors', pilot._configuration(runtime),
                                 model_path=Path(runtime['model']), contract=load_contract(Path(runtime['contract'])), seed=13, before_unit=resource_gate,
                                 cache={'dir': runtime['cache_dir']})
                backend.copy_observations = 0
                backend.configure_stage(grader=make_grader(calibration), expected_images=images, prompt_lengths=lengths, work=directory / 'rollout_work_PRIVATE',
                                        binding={'scope_sha256': authorization.sha, 'parent_receipt_sha256': anchor['receipt_sha256'], 'stage': stage})
                backend.configure_v2(condition=condition, prompt_lengths=lengths)
                if checkpoint is None:
                    restore_parent(backend, parent, anchor)
                else:
                    manifest = _json(Path(checkpoint) / 'optimizer/manifest.json')
                    backend.stage_binding = manifest['bindings']['run']; backend.load_checkpoint(checkpoint)
                    actual = full_identity(backend)
                    require((actual['policy'], actual['adam'], actual['rng'], actual['optimizer_step']) == expected,
                            'Loaded checkpoint full policy/Adam/RNG/step differs from its committed receipt')
                    shared.publish_once(directory / 'actual_checkpoint_full_restore_PRIVATE.json', actual)
                return operation(backend, directory)
            value = run_with_release(call, resident, directory)
            require(_json(directory / 'release_SAFE.json')['release_verified'], 'Actual model release required')
            resident[0] = None; gc.collect(); return value

        if stage == 'P':
            rows = [(u, c) for u in units if u['kind'] == ms.SEQ_KIND for c in u['contexts']]
            require(len(rows) == 8, 'Eight frozen TRAIN sequence images required')
            ledgers = {c: core.prompt_lengths_for(c, spec['concise_sequence_prompt_length'] if c == 'concise' else None) for c in core.CONDITIONS}
            jobs = []
            for i, (u, c) in enumerate(rows):
                for cond in core.CONDITIONS:          # per image: original then concise (interleaved, fixed before calls)
                    jobs.append(('P_%s_%02d' % (cond, i), c['row']['id'], core.condition_row(c['row'], cond), images[c['row']['id']], cond, ledgers[cond]))
            counter = resumable.CallCounter(call_limits(16), out / 'calls_PRIVATE.json')
            def diagnose(backend, directory):
                return pilot._readonly(backend, lambda: greedy_rows(backend, jobs, journal=journal, counter=counter, before=resource_gate,
                                                                    out=out / 'rows_PRIVATE', session_directory=directory), directory)
            pending = [j for j in jobs if journal.get(j[0]) is None]
            items = one('parent_P', diagnose) if pending else [journal.get(j[0])['item'] for j in jobs]
            by = {c: [x for x in items if x['condition'] == c] for c in core.CONDITIONS}
            decision = core.prompt_decision(core.greedy_summary(by['original']), core.greedy_summary(by['concise']))
            result = {'status': 'PASS', 'prompt_decision': decision, 'calls': counter.used, 'optimizer_updates': 0, 'cohort': 'FROZEN_TRAIN8_FOUR_TRAINING_PATIENTS'}
        elif stage == 'C':
            cunits = core.condition_units(units, condition)[:4]
            budget = core.common_budget(cunits, lengths)
            def common_op(backend, directory):
                def operation(opdir):
                    path = out / 'common_counters_PRIVATE.json'
                    counters = shared.Counters(budget, None, lambda used: shared.replace_atomic(path, {'limits': budget, 'used': used}))
                    try:
                        record = core.make_common(backend=backend, units=cunits, lengths=lengths, reward_policy_sha256=reward_sha, output=out / 'common_PRIVATE',
                                                  before=resource_gate, parent_receipt_sha256=anchor['receipt_sha256'], counters=counters)
                    finally:
                        drop_snapshot_references(backend); gc.collect(); backend.mx.clear_cache()
                    release = release_before_commit(backend, directory)
                    return {'value': record, 'files': {**durable.files_under(out / 'common_PRIVATE'), **counters_snapshot(opdir, counters),
                                                        str(release.resolve()): shared.file_sha256(release)}}
                return journal.run('common', operation, reservation=budget)
            record = journal.get('common') or one('parent_C', common_op)
            result = {'status': 'PASS', 'common': record, 'prompt_condition': condition, 'optimizer_updates': 0}
        elif stage in ('A', 'B'):
            arm = stage
            cunits = core.condition_units(units, condition)
            budget = core.arm_budget(cunits, lengths)
            cdir = stage_output(spec, 'C') / 'common_PRIVATE'
            armout = out / 'arm_PRIVATE'
            cpath = out / 'arm_counters_PRIVATE.json'
            def counters():
                return resume_counters(cpath, budget['limits'], [r['counters_after'] for r in (journal.get('segment0'), journal.get('segment1')) if r is not None])
            def train(backend, directory):
                c = counters(); r0 = journal.get('segment0')
                if r0 is None:
                    common = core.load_common(cdir, cunits[:4], reward_policy_sha256=reward_sha, parent_receipt_sha256=anchor['receipt_sha256'], backend=backend)
                    require(pilot.state_identity(backend) == common.parent_state, 'Fresh disk Adam52 differs from the common rollout parent')
                    def seg0(opdir):
                        r = core.run_segment(arm=arm, index=0, backend=backend, units=cunits[:4], lengths=lengths, reward_policy_sha256=reward_sha,
                                             counters=c, output=armout, before=resource_gate, common=common)
                        return {'value': r, 'files': {**durable.files_under(armout / 'segment_0'), **counters_snapshot(opdir, c)}}
                    durable_mkdir(armout); r0 = journal.run('segment0', seg0, reservation=budget['segment0'])
                    common = None
                def seg1(opdir):
                    try:
                        r = core.run_segment(arm=arm, index=1, backend=backend, units=cunits[4:], lengths=lengths, reward_policy_sha256=reward_sha,
                                             counters=c, output=armout, before=resource_gate, previous_receipt=r0['receipt_sha256'])
                    finally:
                        drop_snapshot_references(backend); gc.collect(); backend.mx.clear_cache()
                    release = release_before_commit(backend, directory)
                    return {'value': r, 'files': {**durable.files_under(armout / 'segment_1'), **counters_snapshot(opdir, c),
                                                   str(release.resolve()): shared.file_sha256(release)}}
                return journal.run('segment1', seg1, reservation=budget['segment1'])
            if journal.get('segment1') is None:
                r0 = journal.get('segment0')
                if r0 is None:
                    one('fresh_parent_' + arm, train)
                else:     # resume a committed segment 0 in a fresh process from its full checkpoint, never from another arm
                    one('resume_segment0_' + arm, train, checkpoint=armout / 'segment_0' / 'checkpoint_PRIVATE',
                        expected=(r0['policy_sha_after'], r0['adam_digest_after'], r0['rng_digest_after'], r0['adam_step_after']))
            terminal = arm_terminal(spec, arm)
            receipts = [core.verify_segment_receipt(armout / 'segment_0'), core.verify_segment_receipt(armout / 'segment_1')]
            summary = core.arm_summary(arm, receipts, counters())
            require(summary['patients_exposed'] == 4 and summary['optimizer_updates'] <= 2, 'Four patients must be exposed under <=2 updates')
            result = {'status': 'PASS', 'arm': summary, 'terminal': {k: v for k, v in terminal.items() if k != 'receipt'}, 'prompt_condition': condition,
                      'optimizer_updates': summary['optimizer_updates'], 'fresh_disk_parent': True, 'responses_from_other_arm': 0}
        else:
            label = {'EP': 'Parent', 'EA': 'A', 'EB': 'B'}[stage]
            checkpoint = expected = None
            if label != 'Parent':
                terminal = arm_terminal(spec, label)
                require(not terminal['alias_of_parent'], 'Zero-update arm is a Parent alias; its evaluation is not run')
                r = terminal['receipt']; checkpoint = terminal['checkpoint']
                expected = (r['policy_sha_after'], r['adam_digest_after'], r['rng_digest_after'], r['adam_step_after'])
            records = dev['records_PRIVATE']
            jobs = [('%s_row_%02d' % (label, i), rec['id'], core.condition_row(rec['row_PRIVATE'], condition), rec['image_sha256'], condition, lengths)
                    for i, rec in enumerate(records)]
            require(len(jobs) == 24, 'All 24 frozen DEV rows required')
            counter = resumable.CallCounter(call_limits(24), out / 'calls_PRIVATE.json')
            def evaluate(backend, directory):
                backend.expected_image_sha256 = {rec['row_PRIVATE']['id']: rec['image_sha256'] for rec in records}
                return pilot._readonly(backend, lambda: greedy_rows(backend, jobs, journal=journal, counter=counter, before=resource_gate,
                                                                    out=out / 'rows_PRIVATE', session_directory=directory), directory)
            pending = [j for j in jobs if journal.get(j[0]) is None]
            items = one(label, evaluate, checkpoint, expected) if pending else [journal.get(j[0])['item'] for j in jobs]
            result = {'status': 'PASS', 'label': label, 'metrics': core.greedy_summary(items), 'prompt_condition': condition, 'calls': counter.used,
                      'candidate_calls': 0, 'optimizer_updates': 0, 'cohort': 'FROZEN_INTERNAL_DEV24_PREVIOUSLY_EXPLORED'}
        recovery.require_parent_unchanged(parent_disk, parent); authorization.check('v2_stage_complete')
        result.update(release_verified=True, parent_disk_unchanged=True, stage=stage, resources=resource_gate.peaks())
        return resumable.commit_native_complete(journal, out, result)
    finally:
        journal.close()
