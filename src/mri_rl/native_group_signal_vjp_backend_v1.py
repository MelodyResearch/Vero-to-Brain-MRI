"""NATIVE_GROUP_SIGNAL_VJP_V1 native ops (lazy). Importing this module imports no MLX and loads nothing.

Author: Claude Code (claude-opus-5-5). SOURCE_IMPLEMENTED_NOT_NATIVE_RUN. Reached only from entry.execute_child after exact admission,
the global lock, the resource gate and the cache/runtime pins. Reused, unchanged paths:
  * PilotV2Backend (RecoveryBackend; capped constructor; concise frozen-length _prepare_context AST) + configure_stage/configure_v2;
  * accuracy_recovery_v1.adam52_parent_anchor + accuracy_stage_native_v1.restore_parent (ORIGINAL parent binding), then the full
    identity (1942 parameters / 1010 Adam arrays / 504 LoRA / step 52) must equal the Common32 parent state (policy/Adam/RNG);
  * StageNativeBackend.bind_unit/activate with the RECORDED Common32 contexts (fingerprints, prompt text/ids, frozen length);
  * SharedRolloutBackend._context_logps (F32 full TF, charges P+len-1) inside value_and_grad for the response mean-logp VJP;
  * class_margin_v1._forward (F32 full image features; label rows or shared-prefix first-token contrast) for canonical margins;
  * safe_micro_recovery_native_v1.full_identity and its release discipline (drop references, close, gc, clear cache, <128 MiB).
Actual integrity path: the full identity (all 1942 parameter fingerprints + full Adam + Python/MLX RNG + step) is captured right after the
Parent restore and compared with the identity after the last VJP; there is NO in-RAM snapshot/restore wrapper (pilot._readonly is not used),
so any mutation is detected and reported, never hidden by a restore. The probe itself performs no optimizer step and no finite parameter
perturbation; the backend's apply_update / save_checkpoint / save_adapter / save_rng / load_checkpoint / generate and the restored
optimizer's update are replaced by refusing stubs. This does NOT claim every internal model.update call is unreachable (autograd/model
internals use their own paths). One image autograd graph at a time; references dropped and the MLX cache cleared after each VJP; the
resident is registered before __init__ so a failed construction is still released.
"""
from __future__ import annotations
import gc
import json
from pathlib import Path

from mri_rl import native_group_signal_vjp_v1 as ngs
from mri_rl import shared_rollout_gspo_core_v1 as shared

require = shared.require


def _forbidden(*args, **kwargs):
    raise RuntimeError('Forbidden in the read-only VJP experiment: optimizer update / checkpoint or adapter write')


class NativeOps:
    vector_suffix = '.safetensors'

    def __init__(self):
        self.scope = None
        self.mx = None
        self.resident = None          # registered BEFORE __init__ so a failed construction is still released

    # ------------------------------------------------------------ construction and parent restore
    def construct(self, scope, gate):
        from mri_rl.accuracy_pilot_v2_native import PilotV2Backend
        from mri_rl import safe_micro_pilot_native_v1 as pilot
        from mri_rl.experiment_contract import load_contract
        from mri_rl.two_task_training import make_grader
        b = scope['bindings_PRIVATE']
        runtime = json.loads(Path(scope['runtime_PRIVATE']['runtime_spec']).read_text())
        parent = Path(b['parent']['dir'])
        require(Path(runtime['parent_checkpoint']).resolve() == parent, 'Runtime parent differs from the bound Parent Adam52')
        backend = PilotV2Backend.__new__(PilotV2Backend)
        self.resident = backend                                  # partial object owned by ops from here on
        backend.__init__(parent / 'adapter.safetensors', pilot._configuration(runtime), model_path=Path(runtime['model']),
                         contract=load_contract(Path(runtime['contract'])), seed=13, before_unit=gate, cache={'dir': runtime['cache_dir']})
        backend.copy_observations = 0
        backend.configure_stage(grader=make_grader(json.loads(Path(runtime['calibration']).read_text())), expected_images=b['expected_images'],
                                prompt_lengths=b['lengths'], work=Path(scope['run_output']).parent / 'unused_rollout_work_PRIVATE',
                                binding={'scope_sha256': ngs.digest(scope), 'parent_receipt_sha256': b['parent']['receipt_sha256'], 'stage': ngs.ACTION})
        backend.configure_v2(condition='concise', prompt_lengths=b['lengths'])
        for name in ('apply_update', 'save_checkpoint', 'save_adapter', 'save_rng', 'load_checkpoint', 'generate'):
            setattr(backend, name, _forbidden)
        jobs = scope['response_jobs_PRIVATE']
        limits = {k: 0 for k in shared.CHARGEABLE}
        limits.update(gradient_forwards=len(jobs), scorer_vision_embedding_calls=len(jobs), scorer_language_model_calls=len(jobs),
                      teacherforced_input_positions=sum(j['tf_positions'] for j in jobs))
        path = Path(scope['run_output']) / 'scorer_counters_PRIVATE.json'
        backend.counters = shared.Counters(limits, None, lambda used: shared.replace_atomic(path, {'limits': limits, 'used': used}))
        self.scope, self.mx = scope, backend.mx
        return backend

    def load_parent(self, backend, scope):
        from mri_rl import accuracy_recovery_v1 as recovery
        from mri_rl.accuracy_stage_native_v1 import restore_parent
        b = scope['bindings_PRIVATE']['parent']
        anchor = recovery.adam52_parent_anchor(Path(b['dir']), b['receipt_sha256'])
        require(anchor['adapter_sha256'] == b['adapter_sha256'], 'Parent anchor adapter differs from the bound scope')
        restore_parent(backend, Path(b['dir']), anchor)
        backend.optimizer.update = _forbidden                       # no optimizer update reachable after restore
        identity = self.full_identity(backend)
        require(len(identity['parameters']) == 1942 and len(identity['optimizer']['arrays']) == 1010 and len(backend.trainable_signature()) == 504
                and identity['optimizer_step'] == ngs.PARENT_STEP, 'Exact 1942/1010/504/step52 Parent inventory required')
        self._verify_tokenizer(backend)
        return identity

    def full_identity(self, backend):
        from mri_rl.safe_micro_recovery_native_v1 import full_identity
        return full_identity(backend)

    def _verify_tokenizer(self, backend):
        """Frozen CPU token plans must equal the loaded processor tokenizer (prefix, labels, full prefix+label+suffix boundary)."""
        tok = backend.processor.tokenizer
        enc = lambda s: list(tok.encode(s, add_special_tokens=False))
        for c in self.scope['bindings_PRIVATE']['margin_contexts']:
            plan = c['plan']
            require(enc(ngs.CANONICAL_PREFIX) == plan['prefix_ids'] and enc(ngs.CANONICAL_SUFFIX) == plan['suffix_ids'], 'Native prefix/suffix tokens differ from the frozen plan')
            for role in ('correct', 'wrong'):
                ids, text = plan['labels'][role]['token_ids'], plan['labels'][role]['text']
                require(enc(text) == ids and enc(ngs.CANONICAL_PREFIX + text + ngs.CANONICAL_SUFFIX) == plan['prefix_ids'] + ids + plan['suffix_ids'],
                        'Native label tokenization/boundary differs from the frozen plan')

    # ------------------------------------------------------------ one image context at a time
    def _context(self, backend, job):
        b = self.scope['bindings_PRIVATE']
        unit = job['unit']
        if getattr(backend, '_ngs_unit', None) != unit:
            backend._active = backend._active_modality = None
            gc.collect()
            self.mx.clear_cache()
            backend.bind_unit(b['units'][str(unit)])
            backend._recorded = json.loads(json.dumps(b['recorded_contexts'][str(unit)]))   # recorded Common32 fingerprints
            backend._ngs_unit = unit
        context = backend.activate(job['context_key'])
        require(context['prompt_length'] == job['prompt_length'], 'Actual prompt length differs from the frozen job')
        return context

    def _vjp(self, backend, objective):
        mx = self.mx
        backend.model.eval()
        value = grad = flat = weights = None
        try:
            value, grad = backend.nn.value_and_grad(backend.model, objective)(backend.model)
            mx.eval(value, grad)
            flat = dict(backend.tree_flatten(grad))
            weights = dict(backend.tree_flatten(backend.model.trainable_parameters()))
            require(len(flat) == 504 and set(flat) == set(weights) and all(v.shape == weights[k].shape for k, v in flat.items()),
                    'Complete exact 504-leaf trainable gradient inventory required')
            require(all(v.dtype == mx.float32 for v in flat.values()), 'Native VJP must already be float32 (no silent BF16 cast)')
            require(all(bool(mx.all(mx.isfinite(v)).item()) for v in flat.values()), 'Finite native VJP required')
            scalar = float(value.item())
            vector = flat
            flat = None
            return scalar, vector
        finally:
            value = grad = flat = weights = None                  # no graph retained by this frame on success or failure

    def response_vjp(self, backend, job):
        mx = self.mx
        require(job['prompt_length'] + job['tokens'] - 1 == job['tf_positions'], 'Frozen response TF positions inconsistent (checked BEFORE forward)')
        context = self._context(backend, job)
        old = mx.array(list(job['old_logps_PRIVATE']), dtype=mx.float32)
        captured = []
        before = dict(backend.counters.used)

        def objective(model):
            v = backend._context_logps(context, list(job['token_ids']), 'gradient_forwards')
            log_ratio = mx.stop_gradient(mx.mean(v - old))            # the native loss_fn's own F32 sequence log-ratio (0 extra calls)
            captured.append((mx.stop_gradient(v), log_ratio, mx.stop_gradient(mx.exp(mx.minimum(log_ratio, 10.)))))
            return mx.mean(v)
        try:
            scalar, vector = self._vjp(backend, objective)
            require(len(captured) == 1, 'Exactly one own-context scorer per response VJP')
            v, log_ratio, ratio = captured[0]
            logps = [float(x) for x in v.tolist()]
            require(len(logps) == job['tokens'] and abs(scalar - sum(logps) / len(logps)) <= 1e-5, 'VJP scalar differs from the detached mean selected logp')
            positions = backend.counters.used['teacherforced_input_positions'] - before['teacherforced_input_positions']
            return {'selected_logps': logps, 'positions': positions, 'vector': vector, 'mean_logp': scalar,
                    'f32_sequence_log_ratio': float(log_ratio.item()), 'f32_sequence_ratio': float(ratio.item())}
        finally:
            captured.clear()
            context = old = None
            gc.collect()
            mx.clear_cache()

    def margin_vjp(self, backend, mjob):
        from mri_rl.class_margin_v1 import _forward
        mx = self.mx
        n_prefix = len(mjob['prefix_ids'])
        expected = mjob['prompt_length'] + n_prefix + (len(mjob['label_ids']) - 1 if mjob['objective'] == 'LABEL_MEAN' else 0)
        require(expected == mjob['positions'], 'Frozen canonical TF positions inconsistent (checked BEFORE forward)')
        context = self._context(backend, mjob)
        plan = {'prefix_ids': list(mjob['prefix_ids'])}
        evidence, captured = [], []
        if mjob['objective'] == 'LABEL_MEAN':
            def objective(model):
                v = _forward(backend, context['inputs'], plan, list(mjob['label_ids']), None, evidence)
                captured.append(mx.stop_gradient(v))
                return mx.mean(v)
        else:
            require(mjob['objective'] == 'FIRST_CONTRAST', 'Unknown margin objective')
            pair = list(mjob['first_pair'])

            def objective(model):
                v = _forward(backend, context['inputs'], plan, [pair[0]], pair, evidence)
                captured.append(mx.stop_gradient(v))
                return v[0] - v[1]
        backend.before_unit('ngs_margin_' + mjob['objective'])
        try:
            scalar, vector = self._vjp(backend, objective)
            require(len(captured) == 1 and len(evidence) == 1, 'Exactly one canonical F32 forward per margin VJP')
            values = [float(x) for x in captured[0].tolist()]
            n = len(mjob['label_ids']) if mjob['objective'] == 'LABEL_MEAN' else 2
            require(len(values) == n, 'Canonical logp count differs from the frozen label length / oracle pair')
            target = sum(values) / len(values) if mjob['objective'] == 'LABEL_MEAN' else values[0] - values[1]
            require(abs(scalar - target) <= 1e-5, 'Margin VJP scalar differs from the detached canonical values')
            return {'label_logps': values, 'positions': int(evidence[0]['full_input_length']), 'vector': vector, 'evidence': evidence[0]}
        finally:
            captured.clear()
            context = None
            gc.collect()
            mx.clear_cache()

    # ------------------------------------------------------------ CPU gate inside the child (before ANY margin call)
    def reconstruction_gate(self, scope, records):
        from mri_rl.v2_regression_diagnosis_v1 import TensorFile
        saved = {x: TensorFile(scope['bindings_PRIVATE']['saved_joint_gradients'][x]['path']) for x in ('A', 'B')}
        for x in ('A', 'B'):
            require(shared.file_sha256(scope['bindings_PRIVATE']['saved_joint_gradients'][x]['path']) == scope['bindings_PRIVATE']['saved_joint_gradients'][x]['sha256'],
                    'Saved joint gradient changed')
        names = sorted(saved['A'].keys())
        for r in records:
            require(shared.file_sha256(r['vector_path']) == r['vector_sha256'], 'Committed response vector changed before the gate')
        load_vector = lambda i: (lambda f: {n: f.get(n) for n in names})(TensorFile(records[i]['vector_path']))
        return ngs.reconstruction_gate(scope, records, load_vector=load_vector, load_saved=lambda x: {n: saved[x].get(n) for n in names})

    def save_vector(self, path, vector):
        self.mx.save_safetensors(str(path), vector)
        return shared.file_sha256(path)

    def release(self, backend, directory):
        """Release the full OR partially constructed resident (the controller has already cleared unwound exception frames).
        Mirrors run_with_release / release_before_commit: drop references, close, gc, clear the MLX cache, measure active memory."""
        from mri_rl.safe_micro_recovery_native_v1 import drop_snapshot_references
        target = backend if backend is not None else self.resident
        self.resident = None
        if target is None:
            return {'model_is_none': True, 'active_MLX_bytes': 0, 'release_verified': True, 'note': 'no resident was constructed'}
        mx = getattr(target, 'mx', None) or self.mx
        errors = []
        for step in (lambda: setattr(target, '_ngs_unit', None), lambda: drop_snapshot_references(target), lambda: target.close()):
            try:
                step()
            except BaseException as error:                       # a partial object may lack attributes; keep releasing
                errors.append(type(error).__name__ + ': ' + str(error)[:120])
        model_none = getattr(target, 'model', None) is None
        target = backend = None
        gc.collect()
        if mx is None:
            import mlx.core as mx                               # native child only; never reached from CPU tests
        mx.clear_cache()
        active = int(mx.get_active_memory())
        record = {'model_is_none': model_none, 'active_MLX_bytes': active, 'release_verified': model_none and active < 128 * 1024 ** 2,
                  'release_step_errors': errors}
        shared.publish_once(Path(directory) / 'operation_release_SAFE.json', record)
        return record
