"""Heterogeneous MRI pilot native child phases. SOURCE_IMPLEMENTED_NOT_NATIVE_RUN.

Importing this module imports no MLX and loads nothing. Each phase is ONE fresh child process holding the global lock, one
resident model, the proven RecoveryBackend/PilotV2Backend restore path, a durable journal (claimed-but-uncommitted work is
INDETERMINATE and never replayed) and an actual allocator-release proof before its completion commit.

  parent   : fresh disk Parent Adam52 -> 6 greedy TRAIN gate controls -> 2 greedy midline EVAL -> 16 rollouts -> <=1 joint step
  terminal : fresh process from the committed Adam53 checkpoint (only if updated) -> greedy DEV24 (concise) + 2 midline EVAL

On ANY failure (incl. ResourcePause/Fatal/RED) the core records STOP and performs NO in-process rollback; the resident model is
only released (run_with_release) and the Parent on disk is unchanged. The ONLY backend change vs PilotV2Backend: the unchanged SharedRolloutBackend._prepare_context AST is compiled with a
`require_prompt_cost` that checks the exact frozen length PER ROW (the V2 hook was per task; MedPix images differ in size).
"""
from __future__ import annotations
import gc
import json
import time
import types
from pathlib import Path
from mri_rl import heterogeneous_mri_pilot_v1 as hp
from mri_rl import accuracy_pilot_v2_core as v2core
from mri_rl import shared_rollout_gspo_core_v1 as shared
from mri_rl import safe_micro_pilot_native_v1 as pilot
from mri_rl import safe_micro_pilot_native_v2 as resumable
from mri_rl import safe_micro_training_v2 as durable
from mri_rl.accuracy_pilot_v2_native import PilotV2Backend, _context_node, generate_one_v2, call_limits
from mri_rl.safe_micro_recovery_native_v1 import drop_snapshot_references, release_before_commit, full_identity
from mri_rl.safe_micro_journal_v1 import Journal, durable_mkdir

require = shared.require
STATUS = 'SOURCE_IMPLEMENTED_NOT_NATIVE_RUN'
PHASES = ('parent', 'terminal')


def per_row_prompt_cost(backend, sequence_length, midline_lengths):
    """Exact frozen expanded-prompt length: one for the sequence task (V2 concise 233), one PER midline row id."""
    def require_prompt_cost(task, length):
        if task == hp.SEQ_TASK:
            require(length == sequence_length, 'Sequence prompt length differs from the frozen concise ledger')
        else:
            row = getattr(backend, '_hp_row_id', None)
            require(task == hp.MIDLINE_ABILITY and row in midline_lengths and length == midline_lengths[row],
                    'Midline prompt length differs from its frozen per-row measurement')
        return length
    return require_prompt_cost


class HeterogeneousBackend(PilotV2Backend):
    def configure_hetero(self, *, sequence_length, midline_lengths, greedy_counter, greedy_before):
        require(type(sequence_length) is int and midline_lengths and all(type(v) is int and 0 < v <= 2048 for v in midline_lengths.values()),
                'Measured frozen prompt lengths required')
        module, node = _context_node()
        namespace = dict(vars(module))
        namespace['require_prompt_cost'] = per_row_prompt_cost(self, sequence_length, dict(midline_lengths))
        exec(compile(__import__('ast').Module(body=[node], type_ignores=[]), '<unchanged-_prepare_context-per-row-length>', 'exec'), namespace)
        self._v2_prepare = types.MethodType(namespace['_prepare_context'], self)
        self.v2_condition = 'concise'
        self.hp_sequence_length, self.hp_midline_lengths = sequence_length, dict(midline_lengths)
        self.v2_prompt_lengths = {hp.SEQ_TASK: sequence_length, hp.MIDLINE_ABILITY: max(midline_lengths.values())}
        self.greedy_counter, self.greedy_before = greedy_counter, greedy_before

    def _prepare_context(self, modality):
        self._hp_row_id = self.rows[modality]['id']
        try:
            return super()._prepare_context(modality)
        finally:
            self._hp_row_id = None

    def greedy(self, row, image_sha256):
        length = self.hp_sequence_length if row['ability'] == hp.SEQ_TASK else self.hp_midline_lengths[row['id']]
        return greedy_record(generate_one_v2(self, row, image_sha256, lengths={row['ability']: length}, counter=self.greedy_counter, before=self.greedy_before))


def greedy_record(actual):
    """The scored fields plus the WHOLE generate_one_v2 trace (F32 evidence, input fingerprints, prompt/image SHA, length). No extra call."""
    return {'text': actual['response']['text'], 'token_ids': list(actual['response']['token_ids']), 'prompt_length': actual['prompt_length'],
            'actual_PRIVATE': {k: v for k, v in actual.items() if k != 'response'}}


def _native_integrity(backend):
    from mri_rl import early_native_conditioning_probe_v1 as utility
    from mri_rl.paired_z045_f32_qualification_v1 import _optimizer
    params = utility._fingerprints(backend, dict(backend.tree_flatten(backend.model.parameters())))
    require(len(params) == 1942, 'Full 1942-parameter inventory required')
    return {'parameters': params, 'Adam': _optimizer(backend, utility)}


def guarded_readonly(backend, run, out, *, integrity=_native_integrity, before=lambda name: None):
    """Success-only full integrity/RNG checks. Production passes its resource gate before each native state operation.
    Any failure (including RED after the last response) records STOP and performs no subsequent GPU restore/fingerprint.
    The caller only releases this experiment's process; there is no retry or operation refund.
    """
    out = Path(out); durable_mkdir(out)
    after = None
    try:
        before('readonly_before_integrity')
        initial = integrity(backend)
        before('readonly_before_rng_save')
        rng = out / 'readonly_pre_rng_PRIVATE'; durable_mkdir(rng); backend.save_rng(rng); rng_before = backend.rng_digest()
        value = run()
        before('readonly_after_run_before_integrity')
        after = integrity(backend)
        shared.publish_once(out / 'readonly_integrity_PRIVATE.json', {'before': initial, 'after': after, 'unchanged': initial == after})
        require(initial == after, 'Readonly evaluation mutated full parameters or Adam')
        before('readonly_before_rng_reload')
        backend.load_rng(rng); require(backend.rng_digest() == rng_before, 'Pre-evaluation RNG reload failed')
        return value
    except BaseException as error:
        shared.publish_once(out / 'readonly_STOP_PRIVATE.json', {'error_type': type(error).__name__, 'error': str(error), 'in_process_restore': False,
                            'integrity_after': 'NOT_RUN' if after is None else 'COMPLETED_BEFORE_STOP',
                            'resident_state': 'TAINTED_RELEASE_PROCESS', 'automatic_retry': False})
        raise


def _json(path):
    return json.loads(Path(path).read_text())


def run_child(phase, *, spec, scope, authorization, resource_gate):
    """Called ONLY by the entry child holding the global flock after exact admission."""
    from mri_rl.heterogeneous_mri_pilot_entry_v1 import Authorization
    require(phase in PHASES and isinstance(authorization, Authorization) and authorization.scope == scope and scope.get('phase') == phase,
            'Exact issued heterogeneous-pilot phase authority required')
    require(resource_gate.authorization is authorization and resource_gate.scope == scope, 'Exact resource gate required')
    authorization.check('hetero_native_import')
    from mri_rl import accuracy_recovery_v1 as recovery
    from mri_rl.accuracy_stage_native_v1 import restore_parent
    from mri_rl.experiment_contract import load_contract
    from mri_rl.two_task_training import make_grader
    from mri_rl.paired_z045_continuation_entry_v1 import run_with_release, verify_images
    runtime = _json(spec['runtime_spec'])
    pinput = hp.authenticated_input(scope['pilot_input_path'], scope['pilot_input_sha256'])      # canonical identity, not the file bytes
    out = Path(scope['output']); durable_mkdir(out)
    journal = Journal(out / 'journal_PRIVATE', {'scope': authorization.sha, 'phase': phase}, before=resource_gate)
    try:
        journal.assert_recoverable()
        done = journal.get('native_complete')                     # key written by resumable.commit_native_complete
        if done is not None:
            return done
        parent = Path(runtime['parent_checkpoint']); parent_disk = recovery.parent_disk_snapshot(parent)
        anchor = recovery.adam52_parent_anchor(parent, runtime['parent_receipt_sha256'])
        calibration = _json(runtime['calibration']); reward_sha = shared.file_sha256(runtime['calibration'])
        seq_unit = pinput['sequence_unit_PRIVATE']; mtrain, meval = pinput['midline_train_PRIVATE'], pinput['midline_eval_PRIVATE']
        borders, lengths = pinput['border_controls_PRIVATE'], pinput['prompt_lengths']
        controls = hp.gate_controls(mtrain, borders)
        images = {c['row']['id']: pinput['sequence_image_sha256'][c['row']['id']] for c in seq_unit['contexts']}
        images.update({c['row']['id']: c['image_sha256'] for c in controls})
        images.update({r['id']: r['image_sha256'] for r in mtrain + meval})
        verify_images([c['row'] for c in seq_unit['contexts']] + [c['row'] for c in controls] + meval, images)
        mid_lengths = dict(lengths['by_row'])
        require(set(mid_lengths) == {c['row']['id'] for c in controls} | {r['id'] for r in mtrain + meval}, 'Per-row lengths must cover every midline input')
        # rollout reservation is the TRAIN rows only; gate/EVAL rows are exact per-row greedy checks
        reserve = {hp.SEQ_TASK: lengths['sequence'], hp.MIDLINE_ABILITY: max(mid_lengths[r['id']] for r in mtrain)}
        calls = resumable.CallCounter(call_limits(scope['budget']['greedy_calls_phase']), out / 'calls_PRIVATE.json')

        def resident(label, operation, checkpoint=None, expected=None):
            resource_gate('native_model_load')
            directory = out / (label + '_resident_%d_PRIVATE' % time.time_ns()); durable_mkdir(directory); holder = [None]
            def call():
                resource_gate.claim_model_load(); backend = HeterogeneousBackend.__new__(HeterogeneousBackend); holder[0] = backend
                backend.__init__((Path(checkpoint) if checkpoint else parent) / 'adapter.safetensors', pilot._configuration(runtime),
                                 model_path=Path(runtime['model']), contract=load_contract(Path(runtime['contract'])), seed=13, before_unit=resource_gate,
                                 cache={'dir': runtime['cache_dir']})
                backend.copy_observations = 0
                backend.configure_stage(grader=hp.routing_grader(make_grader(calibration)), expected_images=images, prompt_lengths=reserve,
                                        work=directory / 'rollout_work_PRIVATE',
                                        binding={'scope_sha256': authorization.sha, 'parent_receipt_sha256': anchor['receipt_sha256'], 'phase': phase})
                backend.configure_hetero(sequence_length=lengths['sequence'], midline_lengths=mid_lengths, greedy_counter=calls, greedy_before=resource_gate)
                if checkpoint is None:
                    restore_parent(backend, parent, anchor)
                else:
                    manifest = _json(Path(checkpoint) / 'optimizer/manifest.json')
                    backend.stage_binding = manifest['bindings']['run']; backend.load_checkpoint(checkpoint)
                    actual = full_identity(backend)
                    require((actual['policy'], actual['adam'], actual['rng'], actual['optimizer_step']) == expected,
                            'Loaded checkpoint policy/Adam/RNG/step differs from its committed receipt')
                    shared.publish_once(directory / 'actual_checkpoint_full_restore_PRIVATE.json', actual)
                return operation(backend, directory)
            value = run_with_release(call, holder, directory)
            require(_json(directory / 'release_SAFE.json')['release_verified'], 'Actual model release required')
            holder[0] = None; gc.collect(); return value

        if phase == 'parent':
            def body(backend, directory):
                def operation(opdir):
                    expected = pilot.state_identity(backend)
                    shared.publish_once(directory / 'parent_full_identity_PRIVATE.json', full_identity(backend))
                    try:
                        result = hp.run_parent_phase(backend=backend, seq_unit=seq_unit, midline_train=mtrain, midline_eval=meval, borders=borders, lengths=reserve,
                                                     reward_policy_sha256=reward_sha, output=out / 'parent_phase_PRIVATE', before=resource_gate,
                                                     expected_parent=expected, parent_receipt_sha256=anchor['receipt_sha256'],
                                                     row_lengths={**{c['row']['id']: lengths['sequence'] for c in seq_unit['contexts']}, **mid_lengths})
                    finally:
                        drop_snapshot_references(backend); gc.collect(); backend.mx.clear_cache()
                    release = release_before_commit(backend, directory)
                    return {'value': result, 'files': {**durable.files_under(out / 'parent_phase_PRIVATE'), str(release.resolve()): shared.file_sha256(release)}}
                return journal.run('parent_phase', operation, reservation=scope['budget']['phase_reservation'])
            result = journal.get('parent_phase') or resident('parent', body)
        else:
            receipt = _json(Path(scope['parent_phase_output']) / 'segment' / 'receipt_PRIVATE.json')
            require(receipt['updated'] is True and receipt['receipt_sha256'] == shared.identity({k: v for k, v in receipt.items() if k != 'receipt_sha256'}),
                    'Terminal evaluation runs only from a committed updated receipt; 0 updates is the Parent alias')
            checkpoint = Path(scope['parent_phase_output']) / 'segment' / 'checkpoint_PRIVATE'
            for name, h in receipt['files_sha256'].items():
                require(shared.file_sha256(checkpoint / name) == h, 'Committed checkpoint file changed')
            expected = (receipt['policy_sha_after'], receipt['adam_digest_after'], receipt['rng_digest_after'], receipt['adam_step_after'])
            dev = _json(spec['dev24'])['records_PRIVATE']
            jobs = [(rec['id'], v2core.condition_row(rec['row_PRIVATE'], 'concise'), rec['image_sha256'], hp.SEQ_TASK) for rec in dev]
            jobs += [(r['id'], r, r['image_sha256'], hp.MIDLINE_ABILITY) for r in meval]
            require(len(jobs) == 26, 'DEV24 + 2 midline EVAL required')
            def evaluate(backend, directory):
                backend.expected_image_sha256 = {**images, **{rec['row_PRIVATE']['id']: rec['image_sha256'] for rec in dev}}
                def rows():
                    items = []
                    for i, (rid, row, sha, task) in enumerate(jobs):
                        def operation(opdir, row=row, sha=sha, task=task, rid=rid):
                            g = backend.greedy(row, sha)
                            item = {'task': task, 'id': rid, 'truth': row['reward_model']['ground_truth'], 'text': g['text'], 'tokens': len(g['token_ids']),
                                    'actual_PRIVATE': g['actual_PRIVATE']}
                            p = Path(opdir) / 'row_PRIVATE.json'; shared.publish_once(p, item)
                            return {'value': item, 'files': {str(p.resolve()): shared.file_sha256(p)}}
                        items.append(journal.run('terminal_row_%02d' % i, operation, reservation={'generated_response_tokens': hp.MAX_TOKENS}))
                    return items
                return guarded_readonly(backend, rows, directory, before=resource_gate)
            pending = [i for i in range(len(jobs)) if journal.get('terminal_row_%02d' % i) is None]
            items = resident('terminal', evaluate, checkpoint, expected) if pending else [journal.get('terminal_row_%02d' % i) for i in range(len(jobs))]
            result = {'status': 'EVALUATED', 'items_PRIVATE': items, 'summary': hp.evaluation_summary(items), 'optimizer_updates': 0}
        recovery.require_parent_unchanged(parent_disk, parent); authorization.check('hetero_phase_complete')
        result = {**result, 'release_verified': True, 'parent_disk_unchanged': True, 'phase': phase, 'resources': resource_gate.peaks(), 'calls': calls.used}
        return resumable.commit_native_complete(journal, out, result)
    finally:
        journal.close()
