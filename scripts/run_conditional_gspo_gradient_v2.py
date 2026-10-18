"""Prepare / exact-review-gated new F32 fixed-sample GSPO diagnostic, no update."""
from __future__ import annotations
import argparse
import fcntl
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback

W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W/'src'))
from mri_rl import conditional_gspo_gradient_v2 as p
from mri_rl.native_gspo_path_probe_v1 import canonical, sha, clear_unwound_exception_frames, strict_tensor_identity
require = p.require


def load(path): return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()
    if path.exists(): require(path.read_bytes() == data, 'Preserve prior scope/output')
    else:
        with path.open('xb') as handle: handle.write(data); handle.flush(); os.fsync(handle.fileno())


def verify(scope):
    for path, expected in {**scope['bound_artifacts'], **scope['source_sha256']}.items():
        require(sha(path) == expected, 'Reviewed source/artifact changed: '+path)


def prepare(a):
    oldpath = a.historical_scope.resolve(); old = load(oldpath)
    require(sha(oldpath) == 'c6f52c5c86ddcea5d3d0dca1b98894e9ba3af7dd1ae201917fc4efb242a0d115', 'Same original two-group binding required')
    b2path = W/'local_data/native_probability_resolution_v1/scope_v3_PRIVATE.json'; b2 = load(b2path)
    proofpath = W/'results/NATIVE_PROBABILITY_RESOLUTION_ACTUAL_INDEPENDENT_REVIEW_V1.json'
    require(sha(proofpath) == '3f6298bd3d6d353fd71832f59cdd54c4d95ae5130836629e252bcf7a17515221', 'Actual independently accepted B2 proof required')
    proof = load(proofpath); run = Path(b2['output_directory']); complete = load(run/'complete_SAFE.json')
    require(proof['verdict'] == 'PASS_WITH_LIMITATIONS' and proof['B2_outcome'] == 'NUMERIC_ACCEPTABLE' and
            proof['scope_sha256'] == canonical(b2) == complete['scope_sha256'] and complete['acceptance']['B2_numeric_acceptable'] and
            complete['complete_model_parameters_unchanged'] and complete['native_release_verified'], 'Actual B2 numeric/state proof incomplete')
    exitpath = b2path.parent/'actual_exit_v3_PRIVATE.json'; stdout = b2path.parent/'native_v3_stdout_PRIVATE.log'
    require(sha(exitpath) == proof['actual_exit_sha256'] and sha(stdout) == proof['closed_stdout_sha256'] and
            load(exitpath)['actual_exit_code'] == 0, 'Actual completed/released B2 native exit differs')
    bound = {str(path): sha(path) for path in (oldpath, b2path, proofpath, exitpath, stdout,
        run/'complete_SAFE.json', run/'release_SAFE.json', run/'all_parameter_integrity_SAFE.json')}
    prior_dir = W/'local_data/conditional_gspo_gradient_v1'
    prior_scope_path = prior_dir/'scope_v2_PRIVATE.json'; prior_scope = load(prior_scope_path)
    prior_run = Path(prior_scope['output_directory']); prior_exit = load(prior_dir/'actual_exit_v2_PRIVATE.json')
    prior_stop = load(prior_run/'STOP_PRIVATE.json'); prior_release = load(prior_run/'release_SAFE.json')
    require(prior_exit['actual_exit_code'] == 1 and prior_exit['scope_file_sha256'] == sha(prior_scope_path) and
            prior_exit['closed_stdout_sha256'] == sha(prior_dir/'native_v2_stdout_PRIVATE.log') and
            prior_stop['error_type'] == 'IndexError' and prior_stop['error'] == 'list index out of range' and
            prior_stop['completed_groups'] == 0 and prior_release['release_verified'] and prior_release['model_is_none'] and
            load(prior_run/'all_parameter_integrity_PRIVATE.json')['unchanged'], 'Expected closed pre-score decoder failure differs')
    for path in (prior_scope_path, prior_dir/'actual_exit_v2_PRIVATE.json', prior_dir/'native_v2_stdout_PRIVATE.log',
                 prior_dir/'root_launcher_pre_native_error_v2_PRIVATE.json', prior_run/'STOP_PRIVATE.json',
                 prior_run/'release_SAFE.json', prior_run/'all_parameter_integrity_PRIVATE.json'):
        bound[str(path)] = sha(path)
    bound.update(prior_scope['source_sha256'])  # Preserve the executed v1 source byte-for-byte.
    preflight_path = W/'local_data/conditional_gspo_gradient_v2/actual8_tokenizer_CPU_preflight_PRIVATE.json'
    preflight = load(preflight_path)
    require(preflight['empty_prefix_pass'] and preflight['native_model_loads'] == preflight['native_framework_imports'] == 0 and
            len(preflight['cases']) == 8 and all(r['decode_matches_saved'] for r in preflight['cases']) and
            preflight['worker_sha256'] == sha(W/'src/mri_rl/conditional_gspo_gradient_v2.py'), 'Actual eight-tokenizer CPU preflight missing/stale')
    bound[str(preflight_path)] = sha(preflight_path)
    groups = old['groups_PRIVATE']; require(len(groups) == 2, 'At most exactly the original two historical groups')
    for group in groups:
        parent = Path(group['parent_directory']); child = parent.with_name(f"boundary_{int(parent.name.split('_')[-1])+1:04d}")
        for path, expected in old['bound_artifacts'].items():
            candidate = Path(path)
            if ((candidate.is_relative_to(parent) and (candidate.name in ('receipt.json', 'adapter.safetensors', 'adapter_config.json') or 'optimizer' in candidate.parts))
                or candidate in (child/'receipt.json', child/'rollout_PRIVATE.json', Path(group['row']['images'][0]))
                or candidate.parent == Path(old['model_path']) or candidate == Path(old['contract_path'])
                or candidate == a.primary_root.resolve()/'results/EXACT_QUANT_CACHE_CERTIFICATE.json'):
                # Preserve authenticated historical digests; execute/reviewer checks
                # actual bytes. CPU preparation does not reread weight payloads.
                bound[path] = expected
        classes = [p.strict_correctness(group['row']['ability'], r['text'], group['row']['reward_model']['ground_truth']) for r in group['raw']['responses']]
        require(all(v['format'] == 1 for v in classes) and all((a > 0) == v['correct'] for a, v in zip(group['advantages'], classes)),
                'Saved strict correctness/advantage binding differs; no salvage or reselection')
    sources = dict(b2['source_sha256'])
    for path in (Path(__file__), W/'src/mri_rl/conditional_gspo_gradient_v2.py', W/'tests/test_conditional_gspo_gradient_v2.py',
                 W/'src/mri_rl/paired_native_gradient_probe_v1.py', W/'src/mri_rl/vero_mri_reward.py'):
        sources[str(path)] = sha(path)
    out = a.output_directory.resolve(); require(out.is_relative_to(W/'local_data') and not out.exists(), 'Fresh private output required')
    return {'schema': 'conditional-f32-fixed-sample-gspo-v2', 'execution_tier': 'B', 'primary_root': str(a.primary_root.resolve()),
        'worktree': str(W), 'output_directory': str(out), 'groups_PRIVATE': groups, 'bound_artifacts': bound, 'source_sha256': sources,
        'B2_actual_review_sha256': sha(proofpath), 'B2_scope_sha256': canonical(b2), 'runtime_versions': old['runtime_versions'],
        'model_path': old['model_path'], 'contract_path': old['contract_path'], 'derived_cache': old['derived_cache'], 'global_lock': old['global_lock'],
        'budget': p.budget(2), 'scorer': 'f32_activation_response_logps: fresh native image features -> copied F32 activations/deepstack -> native full LM -> actual centered nn.log_softmax on response rows',
        'prior_attempt': {'status': 'PRESERVED_INDEXERROR_BEFORE_FIRST_SCORE', 'full_scoring_forwards_spent': 0, 'backward_calls_spent': 0,
            'native_wall_seconds': prior_exit['actual_wall_seconds'], 'corrected_attempt_maximum_seconds': 1700,
            'combined_operation_start_deadline_seconds': prior_exit['actual_wall_seconds']+1700,
            'deadline_limitation': 'Checks bound new operation starts; an atomic final native call and release may overrun. No hard combined wall-time guarantee; report actual closed-tool elapsed time.',
            'additional_model_loads_maximum': 1,
            'correction': 'Guard empty decoder prefix; move label/correctness/roundtrip preamble inside existing integrity try/finally. No scorer/loss/numerical-tolerance change.'},
        'counter_definition': 'Full scoring forwards count scorer API calls; autograd gradient-checkpoint internal layer recomputation is not separately counted. Wall/resource limits cover all actual work.',
        'summation_closure': 'Componentwise IEEE F32 gamma3*sumabs(scaled_response_components)+3minsubnormal for three additions; label remainder defined as full-minus-label, with measured reconstruction roundoff.',
        'sample_reference': 'FIXED_SAMPLE_SURROGATE_REFERENCE; old BF16-generated responses under new F32 condition-own detached old reference, ratio1 at fixed weights; no F32 on-policy rollout.',
        'conditions': {'REAL': ['FULL', 'STRICT_LABEL_DERIVATIVE'], 'ZERO_NORMALIZED_PIXELS': ['FULL']},
        'old_current_tolerance': 1e-5, 'label_projection': 'Strict-class overlap tokens only; seq escaped nonclass has zero mask, full response-length denominator retained.',
        'directional_test': 'Save g_i=-(a_i/K)*gradient(mean logp_i); derivative under -groupg is K/a_i*dot(groupg,g_i), and projected-label analogue. Check against strict saved correctness, no finite update.',
        'state': 'Exact each-group historical pre-rollout adapter504/Adam0/RNG; all1942model tensors before/after; compare Adam/RNG BEFORE next-parent restoration; no weights save.',
        'resource_contract': 'Existing remaining20.lock/PacedGate/AC/no-other-heavy; serial single-image graphs; corrected-attempt1700-second operation-start deadline; preserve failures and clear unwound frames before release.',
        'limitations': ['NEW F32 policy image-gradient diagnostic, not historical BF16 gradient reconstruction or corrected-training gain.',
            'Original BF16 gradient impact NOT_MEASURED; no extra BF16 gradient calls.', 'Dallas F32 KV parity NOT_MEASURED; same F32 scorer code and per-call context/dtype/own-old checks only.',
            'Only the first sequence17-token instance has actual B2 KV/full qualification. Broader response-length/context generalization NOT_VERIFIED.',
            'Gradient image sensitivity and a local descent derivative do not establish MRI accuracy improvement or task scientific eligibility.'],
        'mission_alignment': {'tag': 'ENGINEERING_UNBLOCK', 'question': 'Measure whether signed native GSPO credit retains image dependence and locally agrees with strict rewards.',
            'falsifier': 'Nonfinite/integrity/context/dtype/own-old gate failure stops; cancellation or wrong-direction scores challenge effective credit.',
            'isolation': 'Fixed original2TRAIN groups/responses/advantages/checkpoints; one declared F32 numerical policy; no generation/update.',
            'provenance': 'Reviewed actual B2, authenticated historical boundaries and exact source.', 'baseline': 'ZERO normalized pixels, condition-own old; REAL strict label derivative.',
            'resources': 'One additional model load after reviewed decoder code correction, 40full scoring/vision/LM calls and24serialbackward max; no automatic retry.'},
        'new_training': 0, 'optimizer_updates': 0, 'generation_calls': 0, 'status': 'CPU_PREPARED_REQUIRES_EXACT_B3_REVIEW'}


def execute(a, scope):
    require(a.review and a.expected_review_sha256 and sha(a.review) == a.expected_review_sha256, 'Exact independent B3 review required')
    review = load(a.review)
    require(review.get('verdict') in ('PASS', 'PASS_WITH_LIMITATIONS') and review.get('scope_sha256') == canonical(scope) and
            review.get('reviewer_kind') in ('isolated_codex', 'claude_code') and review.get('reviewer_id') and review.get('reviewer_findings'), 'B3 source/scope not independently approved')
    verify(scope); out = Path(scope['output_directory']); out.mkdir(parents=True, exist_ok=False)
    from mri_rl import early_native_conditioning_probe_v1 as utility
    from mri_rl.paced_resources import PacedGate
    from mri_rl.mlx_continuous_gspo import NativeGSPOBackend, is_loading_unit
    from mri_rl.experiment_contract import load_contract
    from mri_rl.likelihood_cache import verify_cache_identity
    from mri_rl.optimizer_checkpoint import load_optimizer_checkpoint
    started = time.monotonic(); cache = {'dir': scope['derived_cache']['cache_dir']}; gate = PacedGate(out/'resources_PRIVATE.jsonl')
    def before(name):
        require(time.monotonic()-started <= 1700, 'B3 operation-start deadline exhausted')
        sample = gate.before_unit(name, loading=is_loading_unit(name, cache))
        require(time.monotonic()-started <= 1700, 'B3 deadline exhausted during resource hold')
        return sample
    backend = None; state = rng = disk = actual = prepared = None; pending_error = None; results = []; full_before = None
    with Path(scope['global_lock']).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            before('native_model_load'); require({k: importlib.metadata.version(k) for k in scope['runtime_versions']} == scope['runtime_versions'], 'Runtime version changed')
            certpath = Path(scope['primary_root'])/'results/EXACT_QUANT_CACHE_CERTIFICATE.json'; cert = load(certpath); modelpath = Path(scope['model_path'])
            artifacts = {str(modelpath/name): sha(modelpath/name) for name in set(cert['non_weight_files_identical']) | {'config.json'}}
            require(verify_cache_identity(cache['dir'], certpath, scope['derived_cache']['certificate_sha256'], artifacts) == scope['derived_cache'], 'Base cache changed')
            first = scope['groups_PRIVATE'][0]
            backend = NativeGSPOBackend(first['adapter_path'], {'group_size': 4, 'max_tokens': 256}, model_path=modelpath,
                contract=load_contract(Path(scope['contract_path'])), seed=7, before_unit=before, cache=cache)
            full_before = utility._fingerprints(backend, dict(backend.tree_flatten(backend.model.parameters())))
            require(len(full_before) == 1942, 'All model parameter inventory required')
            for index, group in enumerate(scope['groups_PRIVATE']):
                verify(scope); before('B3_restore_parent_'+str(index))
                disk = backend.load_arrays(group['adapter_path']); actual = dict(backend.tree_flatten(backend.model.trainable_parameters()))
                require(len(actual) == 504 and set(disk) == set(actual) and all(bool(backend.mx.array_equal(v, actual[k]).item()) for k, v in disk.items()), 'Corresponding pre-rollout adapter differs')
                disk = actual = None
                state, rng, _ = load_optimizer_checkpoint(Path(group['parent_directory'])/'optimizer',
                    expected_bindings={'run': group['binding'], 'adapter_sha256': group['adapter_sha256'], 'trainable_signature': backend.trainable_signature()}, load_arrays=backend.load_arrays)
                backend.restore(state, rng); require(int(backend.optimizer_state()['step'].item()) == group['parent_optimizer_updates'] == 0, 'Actual pre-rollout Adam0 required')
                row = group['row']; require(sha(row['images'][0]) == group['raw']['image_sha256'], 'Bound TRAIN image changed')
                prompt = backend.template(backend.processor, backend.model.config,
                    [{'role': 'system', 'content': backend.contract.system_instruction}, {'role': 'user', 'content': row['extra_info']['question']}],
                    num_images=1, add_generation_prompt=True, enable_thinking=backend.contract.enable_thinking)
                require(prompt == group['raw']['rendered_prompt'], 'Historical prompt differs')
                from mlx_vlm.generate.dispatch import _prepare_generation_inputs
                before('B3_prepare_native_inputs_'+str(index))
                ids, pixels, mask, extra = _prepare_generation_inputs(backend.model, backend.processor, prompt, row['images'], None, None,
                    {'max_tokens': 256, 'temperature': 1., 'top_p': 1., 'top_k': 0, 'min_p': 0., 'repetition_penalty': None})
                prepared = {'input_ids': ids, 'pixel_values': pixels, 'attention_mask': mask, **extra}
                scoring = backend.prepare_inputs(backend.processor, images=row['images'], prompts=prompt, image_token_index=backend.model.config.image_token_index,
                    add_special_tokens=backend.special_tokens(backend.model.config.model_type, backend.processor))
                require(ids.tolist() == group['raw']['prompt_token_ids'] and bool(backend.mx.all(mask == 1).item()), 'Historical prompt IDs/mask differ')
                strict_tensor_identity(backend, prepared, scoring, utility)
                results.append(p.run_resident_group(backend, group, prepared, out/f'group_{index}_PRIVATE', utility, before,
                    scorer=p.f32_activation_response_logps, accepted_B2_binding=scope['B2_actual_review_sha256'], exact_B3_review_binding=a.expected_review_sha256))
                state = rng = prepared = scoring = ids = pixels = mask = extra = None
                gc.collect(); backend.mx.clear_cache()
            verify(scope)
        except BaseException as error:
            pending_error = error; utility.durable(out/'STOP_PRIVATE.json', {'error_type': type(error).__name__, 'error': str(error),
                'traceback': ''.join(traceback.format_exception(error)), 'completed_groups': len(results), 'optimizer_updates': 0})
            clear_unwound_exception_frames(error); raise
        finally:
            if backend is not None:
                try:
                    after = utility._fingerprints(backend, dict(backend.tree_flatten(backend.model.parameters())))
                    utility.durable(out/'all_parameter_integrity_PRIVATE.json', {'before': full_before, 'after': after, 'unchanged': full_before == after})
                    require(full_before == after, 'Full parameter integrity failure'); verify(scope)
                except BaseException as error:
                    utility.durable(out/'exit_integrity_failure_PRIVATE.json', {'error': str(error), 'original_failure_preserved': pending_error is not None})
                    clear_unwound_exception_frames(error)
                    if pending_error is None: raise
                finally:
                    original = pending_error or sys.exception()
                    if original is not None: clear_unwound_exception_frames(original)
                    state = rng = disk = actual = prepared = scoring = ids = pixels = mask = extra = None
                    try:
                        backend.close(); gc.collect(); backend.mx.clear_cache(); active = int(backend.mx.get_active_memory())
                        released = backend.model is None and active < 128*1024**2
                        utility.durable(out/'release_SAFE.json', {'model_is_none': backend.model is None, 'active_MLX_bytes': active, 'release_verified': released})
                        if original is None: require(released, 'Native release failure')
                    except BaseException as error:
                        utility.durable(out/'release_failure_PRIVATE.json', {'error': str(error), 'original_failure_preserved': original is not None})
                        clear_unwound_exception_frames(error)
                        if original is None: raise
    result = {'schema': scope['schema'], 'status': 'ACTUAL_TWO_NEW_F32_FIXED_SAMPLE_GSPO_GROUPS_COMPLETE_RELEASED', 'groups': results,
        'scope_sha256': canonical(scope), 'all1942_model_parameters_unchanged': True, 'optimizer_updates': 0, 'generation_calls': 0,
        'budget': scope['budget'], 'sample_reference': scope['sample_reference'], 'limitations': scope['limitations']}
    save(out/'complete_SAFE.json', result); print(json.dumps({'status': result['status'], 'receipt_sha256': sha(out/'complete_SAFE.json')}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('primary-root', 'historical-scope', 'output-directory', 'scope-output'): parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--review', type=Path); parser.add_argument('--expected-review-sha256'); parser.add_argument('--execute', action='store_true')
    a = parser.parse_args(); scope = prepare(a)
    require(a.scope_output.resolve().is_relative_to(W/'local_data'), 'Ignored private scope required'); save(a.scope_output, scope)
    if not a.execute:
        print(json.dumps({'status': 'CPU_PREPARED_NO_NATIVE', 'scope_sha256': canonical(scope), 'scope_file_sha256': sha(a.scope_output), 'budget': scope['budget']})); return
    execute(a, scope)


if __name__ == '__main__': main()
