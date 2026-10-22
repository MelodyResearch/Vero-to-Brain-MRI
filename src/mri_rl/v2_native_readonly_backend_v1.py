"""V2_NATIVE_READONLY_V1 per-checkpoint child: load ONE saved Adam53 endpoint, teacher-force the 32 Common TRAIN responses, run 4 greedy DEV rows.

Author: Claude Code (claude-opus-5-5). SOURCE_IMPLEMENTED_NOT_NATIVE_RUN. Importing this module imports no MLX and loads nothing;
every native leaf import is function-local inside NativeOps, reached only from an issued per-checkpoint authority after the
resource gate. Reused, unchanged, proven paths: PilotV2Backend (capped 12 GiB / 256 MiB constructor, frozen-length
_prepare_context AST), RecoveryBackend.load_checkpoint under the ORIGINAL manifest run binding + verify_loaded_terminal
(1942 parameters / 1010 Adam arrays / 504 LoRA + policy/Adam/RNG/step), StageNativeBackend.activate (recorded rollout
fingerprints), SharedRolloutBackend._context_logps (F32 full TF, charges exactly P+len-1 positions), generate_one_v2 (F32
strict greedy, actual image identity), safe_micro_pilot_native_v1._readonly (full parameter/Adam integrity, in-RAM
restore of the loaded state incl. RNG), release_before_commit and run_with_release. This module calls no gradient,
backward, optimizer, TRAIN-generation or checkpoint-save surface.
"""
from __future__ import annotations
import gc
import json
import math
import time
from pathlib import Path
from mri_rl import v2_native_readonly_v1 as nr
from mri_rl import shared_rollout_gspo_core_v1 as shared
from mri_rl.safe_micro_journal_v1 import Journal, durable_mkdir

require = shared.require
STATUS = 'SOURCE_IMPLEMENTED_NOT_NATIVE_RUN'


class NativeOps:
    """Production leaf operations (lazy imports). Tests inject a CPU fake with the same interface; no MLX in pytest."""
    def construct(self, scope, job, resident, gate, directory):
        from mri_rl.accuracy_pilot_v2_native import PilotV2Backend
        from mri_rl import safe_micro_pilot_native_v1 as pilot
        from mri_rl.experiment_contract import load_contract
        from mri_rl.two_task_training import make_grader
        runtime = nr.load_json(scope['runtime_PRIVATE']['runtime_spec'])
        backend = PilotV2Backend.__new__(PilotV2Backend); resident[0] = backend
        backend.__init__(Path(job['checkpoint_dir']) / 'adapter.safetensors', pilot._configuration(runtime), model_path=Path(runtime['model']),
                         contract=load_contract(Path(runtime['contract'])), seed=13, before_unit=gate, cache={'dir': runtime['cache_dir']})
        backend.copy_observations = 0
        backend.configure_stage(grader=make_grader(nr.load_json(runtime['calibration'])), expected_images=scope['expected_images_PRIVATE'],
                                prompt_lengths=scope['lengths'], work=Path(directory) / 'unused_rollout_work_PRIVATE', binding=job['run_binding'])
        backend.configure_v2(condition=nr.CONDITION, prompt_lengths=scope['lengths'])
        return backend

    def load_and_verify(self, backend, checkpoint_dir, receipt, run_binding):
        from mri_rl.safe_micro_recovery_native_v1 import verify_loaded_terminal
        backend.stage_binding = run_binding                   # the checkpoint's ORIGINAL run binding, never the new scope
        backend.load_checkpoint(checkpoint_dir)
        return verify_loaded_terminal(backend, receipt)

    def full_identity(self, backend):
        from mri_rl.safe_micro_recovery_native_v1 import full_identity
        return full_identity(backend)

    def rng_digest(self, backend):
        return backend.rng_digest()

    def readonly(self, backend, run, directory):
        from mri_rl import safe_micro_pilot_native_v1 as pilot
        return pilot._readonly(backend, run, directory)

    def score(self, backend, unit, recorded, key, tokens, prompt_length):
        if getattr(backend, '_readonly_unit', None) != unit['index']:
            backend.bind_unit(unit)                            # rows = THIS unit's own contexts
            backend._recorded = json.loads(json.dumps(recorded))   # the unit's recorded Common-rollout fingerprints (not current policy)
            backend._readonly_unit = unit['index']
        context = backend.activate(key)                        # prepares + verifies fingerprints / prompt text / ids / frozen length
        require(context['prompt_length'] == prompt_length, 'Actual prompt length differs from the frozen job')
        value = backend.mx.stop_gradient(backend._context_logps(context, list(tokens), 'detached_old_forwards'))
        backend.mx.eval(value); result = value.tolist(); del value
        return result

    def generate(self, backend, row, image_sha, lengths, counter, before):
        from mri_rl.accuracy_pilot_v2_native import generate_one_v2
        backend._active = backend._active_modality = backend._readonly_unit = None; gc.collect(); backend.mx.clear_cache()
        return generate_one_v2(backend, row, image_sha, lengths=lengths, counter=counter, before=before)

    def release(self, backend, directory):
        from mri_rl.safe_micro_recovery_native_v1 import release_before_commit
        backend._readonly_unit = None
        return release_before_commit(backend, directory)

    def run_with_release(self, call, resident, directory):
        from mri_rl.paired_z045_continuation_entry_v1 import run_with_release
        return run_with_release(call, resident, directory)


def _finite_nonpositive(values, n):
    return isinstance(values, list) and len(values) == n and all(type(v) in (int, float) and math.isfinite(v) and v <= 0 for v in values)


def run_checkpoint(label, *, scope, authorization, resource_gate, ops=None):
    from mri_rl.v2_native_readonly_entry_v1 import Authorization
    require(label in nr.CHECKPOINTS and isinstance(authorization, Authorization) and authorization.parent is not None and authorization.scope == scope and
            scope.get('checkpoint') == label, 'Exact issued per-checkpoint read-only authority required')
    require(resource_gate.authorization is authorization and resource_gate.scope == scope, 'Exact matching resource gate required')
    authorization.check('readonly_checkpoint_start')
    ops = ops or NativeOps()
    job = scope['checkpoints_PRIVATE'][label]
    nr.check_checkpoint_entry(job)
    out = Path(scope['checkpoint_output'])
    require(not out.exists(), 'One-time fresh checkpoint output required: no resume or replay of any claimed operation')
    durable_mkdir(out)
    budget, tf_jobs, dev_jobs = scope['budget'], scope['tf_jobs_PRIVATE'], scope['dev_jobs_PRIVATE']
    units = {u['index']: u for u in scope['units_PRIVATE']}
    receipt = job['receipt_PRIVATE']
    expected_identity = (receipt['policy_sha_after'], receipt['adam_digest_after'], receipt['rng_digest_after'], receipt['adam_step_after'])
    journal = Journal(out / 'journal_PRIVATE', {'scope': scope['compound_scope_sha256'], 'checkpoint': label, 'input': scope['input_sha256']}, before=resource_gate)
    try:
        journal.assert_recoverable()
        tf_counters = shared.Counters(budget['tf_limits'], None,
                                      lambda used: shared.replace_atomic(out / 'tf_counters_PRIVATE.json', {'limits': budget['tf_limits'], 'used': used}))
        greedy = nr.FiniteCounter(budget['greedy_limits'], out / 'greedy_counters_PRIVATE.json')
        tfdir, devdir, directory = out / 'tf_PRIVATE', out / 'dev_PRIVATE', out / 'resident_PRIVATE'
        for d in (tfdir, devdir, directory):
            durable_mkdir(d)
        resident, observed = [None], {}

        def body(backend):
            for jt in tf_jobs:
                key = 'tf_%02d' % jt['index']
                require(shared.identity(tuple(jt['token_ids'])) == jt['token_sha256'] and len(jt['token_ids']) == jt['tokens'], 'Saved token binding differs')
                def operation(opdir, jt=jt, key=key):
                    logps = ops.score(backend, units[jt['unit']], scope['recorded_contexts_PRIVATE'][str(jt['unit'])], jt['context_key'],
                                      list(jt['token_ids']), jt['prompt_length'])
                    require(_finite_nonpositive(logps, jt['tokens']), 'Finite nonpositive selected log-probability for every saved token required')
                    record = {'key': key, 'checkpoint': label, 'index': jt['index'], 'unit': jt['unit'], 'context_key': jt['context_key'],
                              'response_id': jt['response_id'], 'token_sha256': jt['token_sha256'], 'tokens': jt['tokens'],
                              'prompt_length': jt['prompt_length'], 'tf_positions': jt['tf_positions'], 'logps': [float(v) for v in logps]}
                    p = tfdir / (key + '_PRIVATE.json'); shared.publish_once(p, record)
                    return {'value': {'index': jt['index'], 'logps_sha256': shared.identity(record['logps']), 'path': str(p.resolve())},
                            'files': {str(p.resolve()): shared.file_sha256(p)}}
                before = dict(tf_counters.used)
                journal.run(key, operation, reservation={'teacherforced_scorer_forwards': 1, 'teacherforced_input_positions': jt['tf_positions']})
                require(tf_counters.used['detached_old_forwards'] == before['detached_old_forwards'] + 1 and
                        tf_counters.used['teacherforced_input_positions'] == before['teacherforced_input_positions'] + jt['tf_positions'],
                        'Charged TF forward/positions differ from the frozen exact job')
            observed['rng_after_tf'] = ops.rng_digest(backend)
            for d in dev_jobs:
                key = 'dev_%d' % d['index']
                def operation(opdir, d=d, key=key):
                    start = time.monotonic()
                    actual = ops.generate(backend, d['row_PRIVATE'], d['image_sha256'], scope['lengths'], greedy, resource_gate)
                    r = actual['response']; n = len(r['token_ids'])
                    require(0 < n <= nr.core.MAX_TOKENS and _finite_nonpositive(list(r['raw_selected_logps']), n), 'Bounded finite greedy response required')
                    require(actual['image_sha256'] == d['image_sha256'], 'Greedy image identity differs from the frozen DEV job')
                    item = {'key': key, 'checkpoint': label, 'index': d['index'], 'truth': d['truth'], 'text': r['text'], 'tokens': n,
                            'finish_reason': r.get('finish_reason'), 'actual_PRIVATE': actual, 'wall_seconds': time.monotonic() - start}
                    p = devdir / (key + '_PRIVATE.json'); shared.publish_once(p, item)
                    return {'value': {'index': d['index'], 'tokens': n, 'path': str(p.resolve())}, 'files': {str(p.resolve()): shared.file_sha256(p)}}
                journal.run(key, operation, reservation={'generated_responses': 1, 'generated_response_tokens': nr.core.MAX_TOKENS})
            observed['rng_after_dev'] = ops.rng_digest(backend)

        def call():
            resource_gate('native_model_load'); resource_gate.claim_model_load()
            backend = ops.construct(scope, job, resident, resource_gate, directory)
            loaded = ops.load_and_verify(backend, job['checkpoint_dir'], receipt, job['run_binding'])
            require((loaded['policy'], loaded['adam'], loaded['rng'], loaded['optimizer_step']) == expected_identity,
                    'Loaded checkpoint full policy/Adam/RNG/step differs from its original receipt')
            shared.publish_once(directory / 'loaded_identity_PRIVATE.json', loaded)
            backend.counters = tf_counters
            ops.readonly(backend, lambda: body(backend), directory)
            after = ops.full_identity(backend)
            shared.publish_once(directory / 'after_readonly_identity_PRIVATE.json', after)
            require(after == loaded, 'Read-only phase changed full parameters/Adam/RNG/step (after the in-RAM restore)')
            ops.release(backend, directory)
            return {'loaded': loaded, 'after': after}
        value = ops.run_with_release(call, resident, directory)
        release = nr.load_json(directory / 'release_SAFE.json')
        require(release.get('release_verified') is True and not list(out.rglob('release_failure_PRIVATE.json')), 'Actual allocator release required before commit')
        nr.check_checkpoint_entry(job)                         # checkpoint payload on disk unchanged
        require(tf_counters.used == budget['tf_limits'], 'Exactly 32 TF forwards and the exact frozen TF positions required')
        require(greedy.used['generations'] == nr.DEV_PER_CHECKPOINT and all(greedy.used[k] <= v for k, v in budget['greedy_limits'].items()), 'Exactly 4 bounded greedy rows required')
        authorization.check('readonly_checkpoint_complete')
        result = {'status': 'PASS', 'checkpoint': label, 'tf_scored': len(tf_jobs), 'dev_generated': len(dev_jobs), 'tf_counters': dict(tf_counters.used),
                  'greedy_counters': dict(greedy.used), 'loaded_identity_PRIVATE': value['loaded'], 'rng_after_readonly': value['after']['rng'],
                  'rng_consumed_during_readonly_observed': observed.get('rng_after_dev') != value['loaded']['rng'] or observed.get('rng_after_tf') != value['loaded']['rng'],
                  'rng_semantics': 'Loaded RNG restored in RAM by the read-only wrapper; on-disk checkpoint unchanged and never re-saved',
                  'optimizer_updates': 0, 'backward_calls': 0, 'new_train_generation': 0, 'release_verified': True, 'checkpoint_disk_unchanged': True,
                  'resources': resource_gate.peaks()}
        return nr.commit_complete(journal, out, result)
    finally:
        journal.close()
