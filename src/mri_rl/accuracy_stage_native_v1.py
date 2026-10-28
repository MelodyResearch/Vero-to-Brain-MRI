"""Finite multistep A/B native bridge. SOURCE_IMPLEMENTED_NOT_NATIVE_RUN.

Reuses historical F32 sampling, task grader, GSPO AST, persistent Adam and
matched evaluator. Importing this module does not import MLX or load a model.
Execution requires a separately reviewed, once-only stage authorization.
"""
from __future__ import annotations
import ast
import gc
import hashlib
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace
from mri_rl import multistep_gspo_core_v1 as ms
from mri_rl import shared_rollout_gspo_core_v1 as shared
from mri_rl.shared_rollout_gspo_backend_v1 import SharedRolloutBackend

require = shared.require
STATUS = 'SOURCE_IMPLEMENTED_NOT_NATIVE_RUN'


def context_key(response):
    key = response.context.modality
    if key in ('F', 'T'):
        return key
    key = response.response_id.split('_')[0]
    require(key in ('D1', 'D2'), 'Dallas response has no own input context')
    return key


class StageNativeBackend(SharedRolloutBackend):
    """One persistent resident policy across units; fresh inputs per unit/context."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The historical loader installs mlx_vlm.generate as an instance field.
        # Preserve it explicitly; otherwise it shadows generate(unit,lengths).
        self._legacy_generation_api = self.__dict__.pop('generate', None)

    def configure_stage(self, *, grader, expected_images, prompt_lengths, work, binding):
        self.grader, self.expected_image_sha256 = grader, expected_images
        self._stage_prompt_lengths, self.work, self.stage_binding = prompt_lengths, Path(work), binding
        self._unit = None
        self._rollout_contexts = {}
        self._active = self._active_modality = self._recorded = None
        self._arm_updates = 0

    def bind_unit(self, unit):
        self.rows = {c['key']: c['row'] for c in unit['contexts']}
        self._unit = unit
        self._active = self._active_modality = self._recorded = None
        if getattr(self, '_rollout_contexts', None):
            self._recorded = self._rollout_contexts.get((unit['index'], self.policy_sha()))

    def activate(self, key):
        require(key in self.rows, 'Response image/prompt context absent from this unit')
        if self._active_modality == key:
            return self._active
        self._active = self._active_modality = None
        gc.collect(); self.mx.clear_cache()
        context = self._prepare_context(key)
        require(context['prompt_length'] <= self._stage_prompt_lengths[self.rows[key]['ability']],
                'Actual expanded prompt exceeds frozen task reservation')
        if self._recorded is not None:
            rec = self._recorded[key]
            require(context['fingerprints_sha256'] == rec['input_fingerprints_sha256'] and
                    context['prompt_length'] == rec['prompt_length'] and
                    hashlib.sha256(context['prompt'].encode()).hexdigest() == rec['prompt_text_sha256'] and
                    shared.identity(context['prompt_ids']) == rec['rendered_prompt_token_ids_sha256'],
                    'Scorer context differs from this policy rollout')
        self._active, self._active_modality = context, key
        return context

    def policy_sha(self):
        from mri_rl import early_native_conditioning_probe_v1 as utility
        # Frozen base authenticated separately by the historical cache certificate.
        return shared.identity(utility._fingerprints(self, dict(self.tree_flatten(self.model.trainable_parameters()))))

    def adam_digest(self):
        return self.optimizer_digest()

    def snapshot(self):
        snapshot = self.transaction_snapshot()
        self._unit_snapshot = snapshot
        return snapshot

    def restore(self, state, rng=None):
        # Parent loader calls restore(Adam,RNG); transaction rollback calls restore(snapshot).
        if rng is not None:
            return super().restore(state, rng)
        return self.transaction_restore(state)

    def generate(self, unit, prompt_lengths):
        from mri_rl.paired_z045_continuation_driver_v1 import _grades
        self.bind_unit(unit)
        self._stage_prompt_lengths = prompt_lengths
        directory = self.work / ('unit_%04d_%d' % (unit['index'], time.time_ns()))
        directory.mkdir(parents=True, exist_ok=False)
        contexts, responses, eos = {}, [], {}
        policy = self.policy_sha()
        for key, row in self.rows.items():
            context = self.activate(key)
            raw = self._generate_context(context, 4, directory)
            contexts[key] = {'prompt_text_sha256': hashlib.sha256(context['prompt'].encode()).hexdigest(),
                'image_sha256': raw['image_sha256'], 'rendered_prompt_token_ids_sha256': shared.identity(context['prompt_ids']),
                'input_fingerprints_sha256': context['fingerprints_sha256'], 'prompt_length': context['prompt_length']}
            grades = _grades(row, raw, self.config, self.grader)
            require(len(grades) == len(raw['responses']) == 4, 'Exactly four true graded responses per context')
            for i, (r, grade) in enumerate(zip(raw['responses'], grades)):
                rid = f'{key}{i}' if key in ('F', 'T') else f'{key}_{i}'
                responses.append({**r, 'response_id': rid, 'context_key': key, 'grade': grade})
            eos[key] = raw['responses'][0]['effective_eos_ids_PRIVATE']
        self._recorded = contexts
        self._rollout_contexts[(unit['index'], policy)] = contexts
        self.transaction_validate(self._unit_snapshot, False)
        return {'policy_sha256': policy, 'contexts': contexts, 'responses': responses, 'effective_eos_ids': eos}

    def score_old(self, unit, responses):
        self.bind_unit(unit)
        result = {}
        for r in responses:
            value = self.mx.stop_gradient(self._context_logps(self.activate(context_key(r)), list(r.token_ids), 'detached_old_forwards'))
            self.mx.eval(value)
            require(bool(self.mx.all(self.mx.isfinite(value)).item()), 'Nonfinite own-old logp')
            result[r.response_id] = value.tolist()
        self.transaction_validate(self._unit_snapshot, False)
        return result

    def gradient(self, planned, own_old, unit):
        from mri_rl.paired_z045_f32_qualification_v1 import frozen_loss_node
        self.bind_unit(unit); mx = self.mx; node = frozen_loss_node()
        self.model.eval()
        def one(item):
            r = item.response
            context = self.activate(context_key(r)); fresh = []
            previous = mx.array(own_old[r.response_id], dtype=mx.float32)
            def score(tokens):
                v = self._context_logps(context, tokens, 'gradient_forwards'); fresh.append(mx.stop_gradient(v)); return v
            ns = {'mx': mx, 'self': SimpleNamespace(_logps=score)}
            exec(compile(ast.Module(body=[node], type_ignores=[]), '<unchanged-native-GSPO>', 'exec'), ns)
            self.counters.charge('backward_calls'); self.before_unit('stage_GSPO_backward')
            loss, grad = self.nn.value_and_grad(self.model, ns['loss_fn'])(self.model, list(r.token_ids), previous, item.advantage)
            mx.eval(loss, grad, fresh)
            require(len(fresh) == 1, 'One own-context scorer per backward')
            error = float(mx.max(mx.abs(fresh[0] - previous)).item())
            self._checked_gradient(grad)
            return {'loss': float(loss.item()), 'gradient': grad, 'old_current_max_abs': error}
        def scale(tree, weight):
            v = self.tree_map(lambda g: g.astype(mx.float32) * weight, tree); mx.eval(v); return v
        def add(a, b):
            v = self.tree_map(lambda x, y: x + y, a, b); mx.eval(v); return v
        grad, evidence = shared.weighted_gradient_sum(planned.plan, one, scale, add)
        self.transaction_validate(self._unit_snapshot, False)
        return grad, {**evidence, 'gradient_l2': 0. if grad is None else self._checked_gradient(grad),
            'native_loss_AST_sha256': hashlib.sha256(ast.dump(node).encode()).hexdigest()}

    def apply_update(self, gradient):
        # One call per unit; the runner guards unit and total update caps. Do not
        # reset Adam or use the old one-step backend's arm-lifetime update flag.
        require(self._checked_gradient(gradient) > 0, 'Nonzero finite GSPO gradient required')
        self.before_unit('stage_optimizer_update')
        self.optimizer.update(self.model, gradient)
        self.mx.eval(self.model.trainable_parameters(), self.optimizer.state)
        self.transaction_validate(self._unit_snapshot, True)
        require(all(self.validate_state()[k] for k in ('parameters_finite', 'optimizer_finite', 'second_moments_nonnegative')),
                'Updated parameters and full Adam state must remain finite')
        self.mx.clear_cache()

    def save_rng(self, directory):
        from mri_rl.optimizer_checkpoint import pack_tree
        layout, arrays = pack_tree(self.rng_state())
        directory = Path(directory); self.save_arrays(directory / 'rng.safetensors', arrays)
        shared.publish_once(directory / 'rng.json', {'layout': layout, 'tensors_sha256': shared.file_sha256(directory / 'rng.safetensors')})

    def load_rng(self, directory):
        from mri_rl.optimizer_checkpoint import unpack_tree
        directory = Path(directory); meta = json.loads((directory / 'rng.json').read_text())
        require(shared.file_sha256(directory / 'rng.safetensors') == meta['tensors_sha256'], 'Persisted RNG payload changed')
        rng = unpack_tree(meta['layout'], self.load_arrays(directory / 'rng.safetensors'))
        super().restore(self.optimizer_state(), rng)

    def save_checkpoint(self, directory):
        from mri_rl.optimizer_checkpoint import save_optimizer_checkpoint
        directory = Path(directory); self.save_adapter(directory / 'adapter.safetensors')
        state = self.optimizer_state(); step = int(state['step'].item())
        binding = {'run': self.stage_binding, 'adapter_sha256': shared.file_sha256(directory / 'adapter.safetensors'),
                   'trainable_signature': self.trainable_signature()}
        save_optimizer_checkpoint(directory / 'optimizer', optimizer_state=state, rng_state=self.rng_state(), bindings=binding,
            completed_groups=114 + 2 * (self._unit['index'] + 1), optimizer_updates=step, save_arrays=self.save_arrays)
        return ['adapter.safetensors', 'optimizer/manifest.json', 'optimizer/state.safetensors']

    def load_checkpoint(self, directory):
        from mri_rl.optimizer_checkpoint import load_optimizer_checkpoint
        directory = Path(directory); meta = json.loads((directory / 'optimizer/manifest.json').read_text())
        expected = {'run': self.stage_binding, 'adapter_sha256': shared.file_sha256(directory / 'adapter.safetensors'),
                    'trainable_signature': self.trainable_signature()}
        require(meta['bindings'] == expected, 'Checkpoint belongs to another stage/adapter')
        self.model.load_weights(str(directory / 'adapter.safetensors'), strict=False)
        state, rng, _ = load_optimizer_checkpoint(directory / 'optimizer', expected_bindings=expected, load_arrays=self.load_arrays)
        super().restore(state, rng)

    def close(self):
        self._unit_snapshot = self._active = self._active_modality = self._recorded = None
        self._rollout_contexts = {}
        super().close()


def run_arms(*, backend, units, budget, prompt_lengths, reward_policy_sha256, common, parent_snapshot, output, before):
    output = Path(output); output.mkdir(parents=True, exist_ok=False); results = {}
    parent_policy = common.parent_policy_sha256
    for arm in ('A', 'B'):
        before('arm_' + arm + '_restore_parent')
        backend.restore(parent_snapshot)
        try:
            runner = ms.ArmRunner(arm=arm, units=units, backend=backend, output=output / ('arm_' + arm), budget=budget,
                prompt_lengths=prompt_lengths, reward_policy_sha256=reward_policy_sha256, common=common,
                parent_policy_sha256=parent_policy, guards=lambda phase, info: before(arm + '_' + phase))
            results[arm] = runner.run()
        finally:
            backend.restore(parent_snapshot)
    return results


def run_stage(spec, scope, authorization, resource_gate):
    from mri_rl.accuracy_stage_entry_v1 import Authorization
    require(isinstance(authorization, Authorization) and authorization.scope == scope, 'Exact stage authorization required')
    require(callable(resource_gate), 'Explicit reviewed resource gate required')
    authorization.check('stage_start')
    from mri_rl import accuracy_recovery_gpu_cache_v4 as cache4
    from mri_rl import accuracy_recovery_v1 as recovery
    from mri_rl import paired_z045_continuation_driver_v1 as old
    from mri_rl import paired_z045_continuation_entry_v1 as entry
    from mri_rl.optimizer_checkpoint import load_optimizer_checkpoint
    from mri_rl.experiment_contract import load_contract
    from mri_rl.two_task_training import make_grader
    import fcntl
    runtime = json.loads(Path(spec['runtime_spec']).read_text())
    stage = json.loads(Path(spec['stage_input']).read_text())
    out = Path(spec['output'])
    require(not out.exists(), 'Fresh stage output required; no automatic retry')
    require(len(stage['units_PRIVATE']) == 8, 'Exactly four frozen patient rounds')
    # A power waiver is NOT permission for cache reconstruction. The exact fresh
    # cache sub-scope must be explicitly included in the approved compound scope.
    cache_spec = json.loads(Path(spec['cache_contract']['path']).read_text())
    cache_args = spec.get('cache_execution')
    require(isinstance(cache_args, dict), 'Fresh cache-scope permission/review required before any native work')
    require(set(cache_args) == {'scope_path','scope_sha256','human_path','human_sha256','review_path','review_sha256','repo_root'},
            'Only reviewed cache artifact paths and hashes may be supplied')
    require(Path(cache_args['scope_path']).resolve() == Path(spec['cache_contract']['path']).resolve(), 'Cache scope path differs')
    require(cache_args.get('scope_sha256') == shared.file_sha256(spec['cache_contract']['path']), 'Cache sub-scope differs')
    authorization.check('cache_restore')
    cache4.execute(**cache_args)
    authorization.check('cache_complete')
    certificate = Path(cache_spec['roots']['work_root']) / 'certificate.json'
    cache_dir = Path(cache_spec['roots']['work_root']) / 'cache'
    require(certificate.is_file() and cache_dir.is_dir(), 'Complete historical GPU cache required')
    from mri_rl.continuous_rl_pilot import cache_binding_for
    binding = cache_binding_for(SimpleNamespace(cache_dir=cache_dir, cache_certificate=certificate,
        cache_certificate_sha256=shared.file_sha256(certificate), model=Path(runtime['model'])))
    require({v['sha256'] for v in binding['shards'].values()} == {
        '6a332d4977c95fd6b51ac7155e039016d90041da1bf3cdd3e500192ea00fa626',
        '2c94dc6761ab4641a4e2c612eb3ff5fd62d939c5704f8301025e741650b9e1e8'}, 'Historical double-shard SHA differs')
    anchor = recovery.adam52_parent_anchor(runtime['parent_checkpoint'], runtime['parent_receipt_sha256'])
    parent_dir = Path(runtime['parent_checkpoint'])
    parent_disk = recovery.parent_disk_snapshot(parent_dir)
    units, lengths = stage['units_PRIVATE'], stage['prompt_lengths']
    budget = ms.derive_arm_budget(units, lengths, maximum_seconds=3600)
    require(budget == stage['source_API_derived_budgets']['per_arm_work_excluding_shared_first_generation_old'], 'Frozen stage budget changed')
    expected_images = {item['record_id_PRIVATE']: item['image_binding_PRIVATE']['sha256'] for item in stage['planned_image_bindings_PRIVATE']}
    all_rows = [c['row'] for u in units for c in u['contexts']]
    entry.verify_images(all_rows, expected_images)
    configuration = {**old.PROFILE, 'group_size': 4, 'max_tokens': 256}
    calibration = json.loads(Path(runtime['calibration']).read_text())
    reward_sha = shared.file_sha256(runtime['calibration'])
    with Path(runtime['global_lock']).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        authorization.check('native_lock')
        out.mkdir(parents=True, exist_ok=False)
        resident = [None]
        def before(name):
            authorization.check(name); resource_gate(name)
        def construct():
            before('native_model_load')
            backend = StageNativeBackend(parent_dir / 'adapter.safetensors', configuration, model_path=Path(runtime['model']),
                contract=load_contract(Path(runtime['contract'])), seed=13, before_unit=before, cache={'dir': str(cache_dir)})
            resident[0] = backend
            backend.configure_stage(grader=make_grader(calibration), expected_images=expected_images, prompt_lengths=lengths,
                work=out / 'rollout_work_PRIVATE', binding={'scope_sha256': shared.scope_digest(scope), 'parent_receipt_sha256': anchor['receipt_sha256']})
            restore_parent(backend, parent_dir, anchor)
            return backend
        def training():
            backend = construct()
            parent_snapshot = backend.snapshot()
            shared.publish_once(out / 'parent_preflight_PRIVATE.json', {'policy': backend.policy_sha(), 'Adam': backend.adam_digest(),
                'RNG': backend.rng_digest(), 'adapter_sha256': anchor['adapter_sha256'], 'adam_step': int(backend.optimizer_state()['step'].item()),
                'preflight_forwards': 0, 'updates': 0})
            common = ms.make_common_rollout(units[0], backend, budget, prompt_lengths=lengths, reward_policy_sha256=reward_sha,
                output=out / 'common_PRIVATE', checkpoint_receipt_sha256=anchor['receipt_sha256'])
            backend.restore(parent_snapshot)
            result = run_arms(backend=backend, units=units, budget=budget, prompt_lengths=lengths, reward_policy_sha256=reward_sha,
                common=common, parent_snapshot=parent_snapshot, output=out / 'training_PRIVATE', before=before)
            backend.transaction_validate(parent_snapshot, False)
            shared.publish_once(out / 'training_complete_PRIVATE.json', {'arms': result, 'common_counters': common.ledger,
                'parent_restored': True, 'parent_disk_unchanged': recovery.parent_disk_snapshot(parent_dir) == parent_disk})
            require(recovery.parent_disk_snapshot(parent_dir) == parent_disk, 'Parent disk changed')
            return result
        # Existing release routine verifies model release even on error.
        release_dir = out / 'training_release_PRIVATE'; release_dir.mkdir()
        training_result = entry.run_with_release(training, resident, release_dir)
        resident[0] = None; gc.collect()
        before('training_released_before_evaluation')
        evaluation = evaluate_stage(spec=spec, stage=stage, runtime=runtime, out=out, cache_dir=cache_dir,
            anchor=anchor, configuration=configuration, before=before)
        require(recovery.parent_disk_snapshot(parent_dir) == parent_disk, 'Parent disk changed after evaluation')
        authorization.check('stage_complete')
        result = {'status': 'ACTUAL_STAGE_COMPLETE', 'training': training_result, 'evaluation': evaluation,
                  'parent_disk_unchanged': True, 'accuracy_claim': 'ENGINEERING_DEV_ONLY'}
        shared.publish_once(out / 'complete_PRIVATE.json', result)
        return result


def restore_parent(backend, parent_dir, anchor):
    from mri_rl.optimizer_checkpoint import load_optimizer_checkpoint
    parent_dir = Path(parent_dir)
    require(shared.file_sha256(parent_dir / 'adapter.safetensors') == anchor['adapter_sha256'], 'Parent adapter changed')
    expected = {'run': anchor['original_binding'], 'adapter_sha256': anchor['adapter_sha256'], 'trainable_signature': backend.trainable_signature()}
    state, rng, _ = load_optimizer_checkpoint(parent_dir / 'optimizer', expected_bindings=expected, load_arrays=backend.load_arrays)
    backend.restore(state, rng)
    require(int(backend.optimizer_state()['step'].item()) == 52 and len(backend.trainable_signature()) == 504,
            'Exact Adam52/504 parent restore required')


def evaluate_stage(*, spec, stage, runtime, out, cache_dir, anchor, configuration, before):
    from mri_rl import paired_z045_matched_f32_eval_v1 as matched
    from mri_rl import accuracy_recovery_eval_v1 as evaluator
    from mri_rl import paired_z045_continuation_entry_v1 as entry
    from mri_rl.experiment_contract import load_contract
    from mri_rl.answer_likelihood import PREFIX
    from tokenizers import Tokenizer
    evaluation = json.loads(Path(spec['evaluation']).read_text())
    plan = matched.compile_records(evaluation)
    tokenizer = Tokenizer.from_file(str(Path(runtime['model']) / 'tokenizer.json'))
    tokens = {task: matched.token_plan(task, PREFIX, lambda t: tokenizer.encode(t, add_special_tokens=False).ids) for task in matched.LABELS}
    require(tokens == evaluation['token_plans_PRIVATE'], 'Actual tokenizer differs from frozen candidate token plans')
    eval_scope = {'records_PRIVATE': plan['real'], 'aliases_PRIVATE': plan['aliases'], 'token_plans_PRIVATE': tokens,
        'expected_image_sha256': {r['id']: r['image_sha256'] for r in plan['real']},
        'numerical_contract_sha256': shared.identity(shared.POLICY)}
    parent = {'path': runtime['parent_checkpoint'], 'receipt_sha256': anchor['receipt_sha256']}
    checkpoints = {'parent': parent}; arms = {}
    for arm in ('A', 'B'):
        directory = Path(out) / 'training_PRIVATE' / ('arm_' + arm)
        summary = json.loads((directory / 'complete_SAFE.json').read_text())
        require(summary['status'] == 'ARM_COMPLETE' and summary['units_committed'] == 8, 'Full arm terminal required before evaluation')
        terminal = directory / 'unit_0007'
        receipt = json.loads((terminal / 'receipt_PRIVATE.json').read_text())
        require(receipt['receipt_sha256'] == shared.identity({k:v for k,v in receipt.items() if k != 'receipt_sha256'}), 'Terminal receipt changed')
        for name, digest in receipt['files_sha256'].items():
            require(shared.file_sha256(terminal / name) == digest, 'Terminal payload changed')
        arms[arm] = {'alias_of_parent': summary['optimizer_updates'] == 0}
        if not arms[arm]['alias_of_parent']:
            checkpoints[arm] = {'path': str(terminal), 'receipt_sha256': receipt['receipt_sha256']}
    counter = evaluator.EvalCounter(len(checkpoints))
    eval_dir = Path(out) / 'evaluation_PRIVATE'; eval_dir.mkdir()
    for label, checkpoint in checkpoints.items():
        directory = eval_dir / (label + '_PRIVATE'); directory.mkdir(); resident = [None]
        def run():
            before('native_model_load')
            backend = StageNativeBackend(Path(checkpoint['path']) / 'adapter.safetensors', configuration, model_path=Path(runtime['model']),
                contract=load_contract(Path(runtime['contract'])), seed=13, before_unit=before, cache={'dir':str(cache_dir)})
            resident[0] = backend
            backend.configure_stage(grader=None, expected_images=eval_scope['expected_image_sha256'], prompt_lengths=stage['prompt_lengths'],
                work=directory / 'unused_rollout', binding={'scope': 'read_only_evaluation'})
            if label == 'parent': restore_parent(backend, Path(runtime['parent_checkpoint']), anchor)
            else:
                # Authenticate the run binding from the already hash-bound checkpoint;
                # no optimizer update is performed by the matched read-only kernel.
                meta = json.loads((Path(checkpoint['path']) / 'optimizer/manifest.json').read_text())
                backend.stage_binding = meta['bindings']['run']; backend.load_checkpoint(checkpoint['path'])
            result = matched.run_checkpoint(backend, eval_scope, label, checkpoint, directory / 'rows_PRIVATE', counter, before)
            shared.publish_once(directory / 'complete_PRIVATE.json', result)
            return {'checkpoint': label, 'cases': len(result['rows'])}
        entry.run_with_release(run, resident, directory)
        resident[0] = None; gc.collect()
    analysis = evaluator.analyze(eval_dir, arms, plan['aliases'])
    shared.publish_once(eval_dir / 'complete_PRIVATE.json', {'analysis': analysis, 'actual_calls': counter.values, 'optimizer_updates': 0})
    return analysis


def make_resource_gate(spec, scope, authorization):
    from mri_rl.accuracy_stage_resources_v1 import make_resource_gate as factory
    return factory(spec, scope, authorization)
