"""Heterogeneous MRI pilot V2 native child phases. SOURCE_IMPLEMENTED_NOT_NATIVE_RUN.

Importing this module imports no MLX and loads nothing. Composes the unchanged v1 native pieces (HeterogeneousBackend with the
per-row exact prompt-length hook, guarded_readonly, run_with_release/release_before_commit, Journal, restore_parent/full_identity).
The ONLY backend difference: greedy() keeps the whole generate_one_v2 response (token IDs, selected logps, finish reason, EOS
coverage) and F32/input trace. Three serial fresh-process phases, one resident model each:
  train       : fresh-disk Parent Adam52 -> 6 frozen controls -> FORMAT gate -> 16 on-policy -> class-contrast gate -> <=1 step
  parent_eval : (only after a committed update) fresh-disk Parent -> 2 midline EVAL under the V2 prompt
  child_eval  : (only after parent_eval) committed Adam53 checkpoint -> DEV24 + 2 midline EVAL + the same 6 controls
Any error (incl. ResourcePause/Fatal/RED): STOP record, no in-process rollback or fingerprint, process release only, no replay.
"""
from __future__ import annotations
import gc
import json
import time
from pathlib import Path
from mri_rl import heterogeneous_mri_pilot_v1 as v1
from mri_rl import heterogeneous_mri_pilot_v2 as h2
from mri_rl import heterogeneous_mri_pilot_native_v1 as v1n
from mri_rl import accuracy_pilot_v2_core as v2core
from mri_rl import shared_rollout_gspo_core_v1 as shared
from mri_rl import safe_micro_pilot_native_v1 as pilot
from mri_rl import safe_micro_pilot_native_v2 as resumable
from mri_rl import safe_micro_training_v2 as durable
from mri_rl.accuracy_pilot_v2_native import generate_one_v2, call_limits
from mri_rl.safe_micro_recovery_native_v1 import drop_snapshot_references, release_before_commit, full_identity
from mri_rl.safe_micro_journal_v1 import Journal, durable_mkdir

require = shared.require
STATUS = 'SOURCE_IMPLEMENTED_NOT_NATIVE_RUN'
PHASES = h2.PHASES


def greedy_record_full(actual):
    """v1 greedy_record + the full stream response (finish reason, EOS coverage, selected logps). No extra call."""
    base = v1n.greedy_record(actual)
    r = actual['response']
    return {**base, 'finish_reason': r.get('finish_reason'), 'EOS_coverage': r.get('EOS_coverage'), 'generation_tokens': r.get('generation_tokens'),
            'actual_PRIVATE': {**base['actual_PRIVATE'], 'response': r}}


class HeterogeneousBackendV2(v1n.HeterogeneousBackend):
    def generate(self, unit, prompt_lengths):
        # The pinned processor has no cleaner. Refuse an unexpected cleaner BEFORE generation,
        # so a later post-clean string is never misrepresented as an observed raw string.
        require(not callable(getattr(self.processor, 'clean_output', None)), 'Unexpected clean_output: new capture qualification required')
        raw = super().generate(unit, prompt_lengths)
        for response in raw['responses']:
            mark_uncleaned_response(self, response)
        return raw

    def greedy(self, row, image_sha256):
        require(not callable(getattr(self.processor, 'clean_output', None)), 'Unexpected clean_output: new capture qualification required')
        length = self.hp_sequence_length if row['ability'] == h2.SEQ_TASK else self.hp_midline_lengths[row['id']]
        actual = generate_one_v2(self, row, image_sha256, lengths={row['ability']: length}, counter=self.greedy_counter, before=self.greedy_before)
        mark_uncleaned_response(self, actual['response'])
        return greedy_record_full(actual)


def mark_uncleaned_response(backend, response):
    require(not callable(getattr(backend.processor, 'clean_output', None)), 'Unexpected clean_output: raw text not independently observed')
    response['preclean_text_PRIVATE'] = response['text']
    response['clean_called'] = False


def authenticate_evaluation_prerequisites(compound_scope, phase, *, scope_file_sha256):
    """CPU-only: a saved inner checkpoint alone never qualifies as a committed training endpoint."""
    from mri_rl.heterogeneous_mri_pilot_entry_v2 import phase_scope, digest
    require(phase in ('parent_eval', 'child_eval'), 'Evaluation phase required')
    base = Path(compound_scope['output'])
    needed = ('train',) if phase == 'parent_eval' else ('train', 'parent_eval')
    completed = {}
    for previous in needed:
        out = base / (previous + '_PRIVATE')
        terminal_path = out / 'terminal_PRIVATE.json'
        journal_dir = out / 'journal_PRIVATE'
        require(terminal_path.is_file() and (journal_dir / 'binding_PRIVATE.json').is_file(), 'Missing completed phase terminal/journal: ' + previous)
        expected_sha = digest(phase_scope(compound_scope, previous))
        terminal = v1n._json(terminal_path)
        require(terminal.get('scope_sha256') == expected_sha and terminal.get('phase') == previous and terminal.get('release_verified') is True,
                'Completed phase terminal scope/release mismatch')
        with Journal(journal_dir, {'scope': expected_sha, 'phase': previous}) as prior:
            prior.assert_recoverable()
            native = prior.get('native_complete')
            require(native is not None and terminal == {**native, 'scope_sha256': expected_sha}, 'Missing/different committed native completion')
            if previous == 'train':
                training = prior.get('train_phase')
                require(training is not None and training['status'] == native['status'] == 'UPDATED' and training['optimizer_updates'] == 1,
                        'Evaluation requires a committed one-update train transaction')
        witness = v1n._json(base / (previous + '_exit_PRIVATE') / 'actual_exit_PRIVATE.json')
        require(witness.get('native_exit_code') == 0 and witness.get('post_child_inspection_complete') is True
                and witness.get('child_reaped') is True and witness.get('guardian_stop') is False
                and witness.get('scope_file_sha256') == scope_file_sha256
                and not witness.get('SIGTERM_sent') and not witness.get('SIGKILL_after_grace') and not witness.get('monitor_log_errors')
                and witness.get('post_child_global_lock', {}).get('no_open_holders') is True, 'Previous phase did not exit/release cleanly')
        completed[previous] = native
    receipt = v1n._json(base / 'train_PRIVATE/train_phase_PRIVATE/segment/receipt_PRIVATE.json')
    require(receipt.get('updated') is True and receipt.get('adam_step_after') == 53
            and receipt.get('receipt_sha256') == shared.identity({k: v for k, v in receipt.items() if k != 'receipt_sha256'})
            and receipt['receipt_sha256'] == completed['train']['segment_receipt_sha256'], 'Committed training receipt binding differs')
    return receipt


def run_child(phase, *, spec, scope, authorization, resource_gate, scope_file_sha256):
    """Called ONLY by the V2 entry child holding the global flock after exact admission."""
    from mri_rl.heterogeneous_mri_pilot_entry_v2 import Authorization
    require(phase in PHASES and isinstance(authorization, Authorization) and authorization.scope == scope and scope.get('phase') == phase,
            'Exact issued heterogeneous-pilot V2 phase authority required')
    require(resource_gate.authorization is authorization and resource_gate.scope == scope, 'Exact resource gate required')
    authorization.check('hetero_v2_native_import')
    if phase != 'train':
        authenticate_evaluation_prerequisites(authorization.parent.scope, phase, scope_file_sha256=scope_file_sha256)
    from mri_rl import accuracy_recovery_v1 as recovery
    from mri_rl.accuracy_stage_native_v1 import restore_parent
    from mri_rl.experiment_contract import load_contract
    from mri_rl.two_task_training import make_grader
    from mri_rl.paired_z045_continuation_entry_v1 import run_with_release, verify_images
    runtime = v1n._json(spec['runtime_spec'])
    pinput = v1.authenticated_input(scope['pilot_input_path'], scope['pilot_input_sha256'])          # canonical identity, not FILE bytes
    out = Path(scope['output']); durable_mkdir(out)
    journal = Journal(out / 'journal_PRIVATE', {'scope': authorization.sha, 'phase': phase}, before=resource_gate)
    try:
        journal.assert_recoverable()
        done = journal.get('native_complete')
        if done is not None:
            return done
        parent = Path(runtime['parent_checkpoint']); parent_disk = recovery.parent_disk_snapshot(parent)
        anchor = recovery.adam52_parent_anchor(parent, runtime['parent_receipt_sha256'])
        calibration = v1n._json(runtime['calibration']); reward_sha = shared.file_sha256(runtime['calibration'])
        seq_unit = pinput['sequence_unit_PRIVATE']; mtrain, meval = pinput['midline_train_PRIVATE'], pinput['midline_eval_PRIVATE']
        borders, lengths = pinput['border_controls_PRIVATE'], pinput['prompt_lengths']
        controls = v1.gate_controls(mtrain, borders)
        require(all(r['extra_info']['question'].endswith(h2.MIDLINE_SUFFIX) for r in mtrain + meval), 'V2 midline prompt required')
        images = {c['row']['id']: pinput['sequence_image_sha256'][c['row']['id']] for c in seq_unit['contexts']}
        images.update({c['row']['id']: c['image_sha256'] for c in controls})
        images.update({r['id']: r['image_sha256'] for r in mtrain + meval})
        verify_images([c['row'] for c in seq_unit['contexts']] + [c['row'] for c in controls] + meval, images)
        mid_lengths = dict(lengths['by_row'])
        require(set(mid_lengths) == {c['row']['id'] for c in controls} | {r['id'] for r in mtrain + meval}, 'Per-row lengths must cover every midline input')
        reserve = {h2.SEQ_TASK: lengths['sequence'], h2.MIDLINE_ABILITY: max(mid_lengths[r['id']] for r in mtrain)}
        calls = resumable.CallCounter(call_limits(scope['budget']['greedy_calls_phase']), out / 'calls_PRIVATE.json')

        def resident(label, operation, checkpoint=None, expected=None):
            resource_gate('native_model_load')
            directory = out / (label + '_resident_%d_PRIVATE' % time.time_ns()); durable_mkdir(directory); holder = [None]
            def call():
                resource_gate.claim_model_load(); backend = HeterogeneousBackendV2.__new__(HeterogeneousBackendV2); holder[0] = backend
                backend.__init__((Path(checkpoint) if checkpoint else parent) / 'adapter.safetensors', pilot._configuration(runtime),
                                 model_path=Path(runtime['model']), contract=load_contract(Path(runtime['contract'])), seed=13, before_unit=resource_gate,
                                 cache={'dir': runtime['cache_dir']})
                backend.copy_observations = 0
                backend.configure_stage(grader=v1.routing_grader(make_grader(calibration)), expected_images=images, prompt_lengths=reserve,
                                        work=directory / 'rollout_work_PRIVATE',
                                        binding={'scope_sha256': authorization.sha, 'parent_receipt_sha256': anchor['receipt_sha256'], 'phase': phase})
                backend.configure_hetero(sequence_length=lengths['sequence'], midline_lengths=mid_lengths, greedy_counter=calls, greedy_before=resource_gate)
                if checkpoint is None:
                    restore_parent(backend, parent, anchor)
                else:
                    manifest = v1n._json(Path(checkpoint) / 'optimizer/manifest.json')
                    backend.stage_binding = manifest['bindings']['run']; backend.load_checkpoint(checkpoint)
                    actual = full_identity(backend)
                    require((actual['policy'], actual['adam'], actual['rng'], actual['optimizer_step']) == expected,
                            'Loaded checkpoint policy/Adam/RNG/step differs from its committed receipt')
                    shared.publish_once(directory / 'actual_checkpoint_full_restore_PRIVATE.json', actual)
                return operation(backend, directory)
            value = run_with_release(call, holder, directory)
            require(v1n._json(directory / 'release_SAFE.json')['release_verified'], 'Actual model release required')
            holder[0] = None; gc.collect(); return value

        def evaluate_rows(jobs, label, checkpoint=None, expected=None):
            """Each greedy row is one journaled indivisible milestone; success-only integrity via v1 guarded_readonly."""
            keys = ['%s_row_%02d' % (label, i) for i in range(len(jobs))]
            if all(journal.get(k) is not None for k in keys):
                return [journal.get(k) for k in keys]

            def body(backend, directory):
                backend.expected_image_sha256 = {**backend.expected_image_sha256, **{row['id']: sha for _, _, row, sha, _ in jobs}}
                resource_gate('hetero_v2_evaluate_state')
                state = v1._state(backend)

                def rows():
                    items = []
                    for key, (name, task, row, sha, extra) in zip(keys, jobs):
                        def operation(opdir, name=name, task=task, row=row, sha=sha, extra=extra):
                            item = h2.greedy_item(backend, task=task, name=name, row=row, image_sha256=sha, parent=state, before=resource_gate, extra=extra)
                            p = Path(opdir) / 'row_PRIVATE.json'; shared.publish_once(p, item)
                            return {'value': item, 'files': {str(p.resolve()): shared.file_sha256(p)}}
                        items.append(journal.run(key, operation, reservation={'generated_response_tokens': h2.MAX_TOKENS}))
                    return items
                return v1n.guarded_readonly(backend, rows, directory, before=resource_gate)
            return resident(label, body, checkpoint, expected)

        if phase == 'train':
            def body(backend, directory):
                def operation(opdir):
                    expected = pilot.state_identity(backend)
                    shared.publish_once(directory / 'parent_full_identity_PRIVATE.json', full_identity(backend))
                    try:
                        result = h2.run_train_phase(backend=backend, seq_unit=seq_unit, midline_train=mtrain, borders=borders, lengths=reserve,
                                                    reward_policy_sha256=reward_sha, output=out / 'train_phase_PRIVATE', before=resource_gate,
                                                    expected_parent=expected, parent_receipt_sha256=anchor['receipt_sha256'],
                                                    row_lengths={**{c['row']['id']: lengths['sequence'] for c in seq_unit['contexts']}, **mid_lengths})
                    finally:
                        drop_snapshot_references(backend); gc.collect(); backend.mx.clear_cache()
                    release = release_before_commit(backend, directory)
                    return {'value': result, 'files': {**durable.files_under(out / 'train_phase_PRIVATE'), str(release.resolve()): shared.file_sha256(release)}}
                return journal.run('train_phase', operation, reservation=scope['budget']['phase_reservation'])
            result = journal.get('train_phase') or resident('train', body)
        else:
            tdir = Path(scope['train_phase_output'])
            receipt = v1n._json(tdir / 'segment' / 'receipt_PRIVATE.json')
            require(receipt['updated'] is True and receipt['receipt_sha256'] == shared.identity({k: v for k, v in receipt.items() if k != 'receipt_sha256'}),
                    'Phase 3 runs only from a committed updated receipt')
            if phase == 'parent_eval':
                items = evaluate_rows([(r['id'], h2.MIDLINE_ABILITY, r, r['image_sha256'], {}) for r in meval], 'parent_eval')
            else:
                checkpoint = tdir / 'segment' / 'checkpoint_PRIVATE'
                for name, h in receipt['files_sha256'].items():
                    require(shared.file_sha256(checkpoint / name) == h, 'Committed checkpoint file changed')
                expected = (receipt['policy_sha_after'], receipt['adam_digest_after'], receipt['rng_digest_after'], receipt['adam_step_after'])
                dev = v1n._json(spec['dev24'])['records_PRIVATE']
                dev_rows = [{**v2core.condition_row(rec['row_PRIVATE'], 'concise'), 'image_sha256': rec['image_sha256']} for rec in dev]
                items = evaluate_rows(h2.child_jobs(dev_rows, meval, controls), 'child_eval', checkpoint, expected)
            result = {'status': 'EVALUATED', 'items_PRIVATE': items, 'optimizer_updates': 0}
        recovery.require_parent_unchanged(parent_disk, parent); authorization.check('hetero_v2_phase_complete')
        result = {**result, 'release_verified': True, 'parent_disk_unchanged': True, 'phase': phase, 'resources': resource_gate.peaks(), 'calls': calls.used}
        return resumable.commit_native_complete(journal, out, result)
    finally:
        journal.close()
