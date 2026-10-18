"""Prepare/review-gated native probability resolution; no B3 or training entry.

Default is CPU-only. Root executes the exact independently reviewed scope.
"""
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
from mri_rl import native_probability_resolution_v1 as p
from mri_rl.native_gspo_path_probe_v1 import canonical, clear_unwound_exception_frames
require, sha = p.require, p.sha


def load(path): return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    content = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()
    if path.exists(): require(path.read_bytes() == content, 'Preserve immutable scope/evidence; choose a new output version')
    else:
        with path.open('xb') as handle: handle.write(content); handle.flush(); os.fsync(handle.fileno())


def prepare(a):
    R = a.primary_root.resolve(); old_scope_path = a.prior_scope.resolve(); prior = load(old_scope_path)
    require(sha(old_scope_path) == 'c6f52c5c86ddcea5d3d0dca1b98894e9ba3af7dd1ae201917fc4efb242a0d115', 'Exact preserved v3 scope required')
    old_record_path = a.prior_B2_record.resolve()
    require(sha(old_record_path) == 'c27e74c3c5001de4d5059911e2700507114900cc1ef70ef56652ba71eb0d9a1b', 'Exact first completed 17-token v3 record required')
    previous = load(old_record_path); original = prior['groups_PRIVATE'][0]
    require(original['selection_PRIVATE']['arm'] == 'S_sequence' and original['selection_PRIVATE']['seed'] == 7 and
            original['selection_PRIVATE']['group_index'] == 1 and original['row']['extra_info']['internal_split'] == 'train', 'Frozen first TRAIN selection changed')
    response = original['raw']['responses'][0]
    require(response['token_ids'] == previous['token_ids'] and len(response['token_ids']) == 17, 'Saved selected response IDs differ')
    parent = Path(original['parent_directory']); child = parent.with_name('boundary_0002')
    group = {**original, 'raw': {**original['raw'], 'responses': [response]}, 'source_response_index': 0,
             'selection': 'Already bound first TRAIN sequence group, first saved response; no new outcome selection.'}
    bound = {str(old_scope_path): sha(old_scope_path), str(old_record_path): sha(old_record_path)}
    source_image = Path(original['row']['images'][0])
    for path, expected in prior['bound_artifacts'].items():
        candidate = Path(path)
        if (candidate.is_relative_to(parent) or candidate == child/'receipt.json' or candidate == child/'rollout_PRIVATE.json'
            or candidate == source_image or candidate.parent == Path(prior['model_path']) or candidate == Path(prior['contract_path'])
            or candidate == R/'results/EXACT_QUANT_CACHE_CERTIFICATE.json'):
            require(sha(candidate) == expected, 'Prior bound checkpoint/data/contract changed')
            bound[path] = expected
    require(sha(parent/'adapter.safetensors') == original['adapter_sha256'], 'Exact pre-rollout weights changed')
    oldrun = old_record_path.parent.parent
    for name in ('STOP_PRIVATE.json', 'release_SAFE.json', 'all_parameter_integrity_SAFE.json'):
        bound[str(oldrun/name)] = sha(oldrun/name)
    require(load(oldrun/'release_SAFE.json')['release_verified'] and load(oldrun/'all_parameter_integrity_SAFE.json')['all_parameters_unchanged'], 'Previous configuration not closed')
    exit_path = oldrun.parent/'actual_exit_v3_PRIVATE.json'; stdout_path = oldrun.parent/'native_v3_stdout_PRIVATE.log'
    exit_receipt = load(exit_path)
    require(exit_receipt['actual_exit_code'] == 1 and exit_receipt['no_automatic_retry'] and
            exit_receipt['scope_file_sha256'] == sha(old_scope_path) and exit_receipt['native_stdout_sha256'] == sha(stdout_path),
            'Actual prior native exit/stdout witness differs')
    bound[str(exit_path)] = sha(exit_path); bound[str(stdout_path)] = sha(stdout_path)
    installed = (R/'../../work/mlx-venv/lib/python3.13/site-packages').resolve()
    sources = [Path(__file__), W/'src/mri_rl/native_probability_resolution_v1.py', W/'tests/test_native_probability_resolution_v1.py']
    sources += [W/'src/mri_rl'/name for name in ('native_gspo_path_probe_v1.py', 'early_native_conditioning_probe_v1.py',
        'mlx_continuous_gspo.py', 'optimizer_checkpoint.py', 'paced_resources.py', 'resource_supervisor.py',
        'likelihood_cache.py', 'likelihood_loading.py', 'experiment_contract.py', 'continuous_rl.py')]
    sources += [installed/'mlx_vlm'/name for name in ('generate/ar.py', 'generate/common.py', 'generate/dispatch.py', 'sample_utils.py',
        'utils.py', 'prompt_utils.py', 'tokenizer_utils.py', 'models/base.py', 'models/cache.py', 'models/rope_utils.py', 'models/mlp.py',
        'models/qwen3_vl/qwen3_vl.py', 'models/qwen3_vl/language.py', 'models/qwen3_vl/vision.py',
        'models/qwen3_vl/config.py', 'models/qwen3_vl/processing_qwen3_vl.py',
        'trainer/common/model.py', 'trainer/common/utils.py', 'trainer/peft/lora.py', 'trainer/peft/lora_layers.py')]
    sources += [installed/'mlx'/name for name in ('nn/layers/quantized.py', 'nn/layers/normalization.py', 'nn/layers/activations.py', 'nn/layers/base.py',
        'nn/utils.py', 'utils.py', 'core/__init__.pyi')]
    out = a.output_directory.resolve()
    require(out.is_relative_to(W/'local_data') and not out.exists(), 'Fresh private output required; no automatic retry')
    scope = {'schema': 'native-probability-resolution-v1', 'execution_tier': 'B', 'status': 'CPU_PREPARED_REQUIRES_EXACT_REVIEW',
        'worktree': str(W), 'primary_root': str(R), 'output_directory': str(out), 'group_PRIVATE': group,
        'prior_B2_record_path': str(old_record_path), 'bound_artifacts': bound,
        'source_sha256': {str(path): sha(path) for path in sources}, 'model_path': prior['model_path'], 'contract_path': prior['contract_path'],
        'derived_cache': prior['derived_cache'], 'global_lock': prior['global_lock'], 'runtime_versions': prior['runtime_versions'],
        'budget': p.budget(17), 'routes': list(p.ROUTES)+['BASELINE_FULL_TF', 'F32_SHARED_FULL_TF'],
        'prefill_schedule': {'legacy': [213], 'installed_ordinary': [212, 1], 'F32_shared_explicit_control': [212, 1],
            'source_default_prefill_step_size': 2048, 'saved_decode_forwards_per_route': 16,
            'historical_runtime_schedule': 'SOURCE_DERIVED_NOT_HISTORICALLY_RECORDED'},
        'acceptance': {'chosen_comparison': 'F32_SHARED_KV_vs_F32_SHARED_FULL', 'full_vocab_logp_max_abs': .002,
            'selected_logp_max_abs': .002, 'exact_prompt_full_feature_prefix': True,
            'direct_AR_F32_vs_actual_frozen_centered_scorer_full_vocab_and_selected_max_abs': .002,
            'legacy_native_and_stable_KV_and_actual_frozen_TF_selected_reproduction_max_abs': 1e-5,
            'decode_position_matrix_equals_cache_offset_plus_rope_delta': True, 'observed_cache_and_logits_float32': True,
            'B3_in_this_execution': False, 'future_B3': 'Requires separate root decision and independent exact review of actual B2 outputs.'},
        'stopping': 'Nonfinite/context/state/resource/budget failure stops immediately; anticipated baseline numerical failures are measured through the single predeclared control, not treated as authorization to retry.',
        'precision_intervention': 'Upcast only copied full-input embeddings/deepstack activations; explicit authoritative positions on saved decode steps; same installed chunk schedule. Do not cast weights or modify modules.',
        'dtype_limit': 'Actual input/logit/cache endpoints recorded; Q and all intermediate arithmetic dtypes not directly observed. No blanket F32-model claim.',
        'private_outputs': 'Full-vocabulary raw/native/stable score matrices, positions/tokens and per-position metrics stay ignored/private; public outputs only aggregate statistics and digests.',
        'prior_evidence_aggregate': {'tokens': 17, 'native_KV_vs_TF_max': previous['comparison']['max_abs_delta'],
            'same_KV_native_vs_F32_max': previous['native_KV_vs_stable_KV']['max_abs_delta'],
            'F32_KV_vs_TF_max': previous['stable_KV_vs_full_teacherforce']['max_abs_delta'],
            'common_offset_or_full_categorical_equality': 'NOT_IDENTIFIABLE_FROM_SAVED_SELECTED_LOGPS_ONLY'},
        'mission_alignment': {'tag': 'ENGINEERING_UNBLOCK', 'question': 'Resolve numerical policy/scorer mismatches before interpreting real MRI GSPO gradient credit.',
            'falsifier': 'The fixed F32/shared/explicit chunk control fails original .002 or exact context/dtype/state gates.',
            'isolation': 'One already bound TRAIN response/checkpoint; exact same image/prompt/IDs, 0 generation/backward/update.',
            'baseline': 'Legacy whole-prompt KV vs source-derived installed chunked KV isolates schedule; full-vocab normalization reveals offsets versus categorical changes.',
            'provenance': 'Frozen original receipts/Adam/RNG/weights, old selected evidence, source hashes and real prepared tensor fingerprints.',
            'resources': 'One resident model, existing global lock/PacedGate/AC/no-other-heavy, 55 LM calls, 2 visual embedding calls, no new heavy operation after 1800s.'},
        'scientific_claim': 'No MRI accuracy, clinical validity, generalization or training result claimed.', 'evidence_tier': 'PROPOSED_NATIVE_NOT_RUN'}
    return scope


def verify_files(scope):
    for path, expected in {**scope['bound_artifacts'], **scope['source_sha256']}.items():
        require(sha(path) == expected, 'Exact reviewed source/artifact changed: '+path)


def execute(a, scope):
    require(a.review and a.expected_review_sha256 and sha(a.review) == a.expected_review_sha256, 'Independently supplied exact review digest required')
    review = load(a.review)
    require(review.get('verdict') in ('PASS', 'PASS_WITH_LIMITATIONS') and review.get('scope_sha256') == canonical(scope) and
            review.get('reviewer_id') and review.get('reviewer_findings') and review.get('reviewer_kind') in ('isolated_codex', 'claude_code'), 'Exact scope not independently approved')
    require(review['verdict'] != 'PASS_WITH_LIMITATIONS' or review.get('limitations'), 'Review limitations missing')
    verify_files(scope); out = Path(scope['output_directory']); out.mkdir(parents=True, exist_ok=False)
    from mri_rl import early_native_conditioning_probe_v1 as utility
    from mri_rl.paced_resources import PacedGate
    from mri_rl.mlx_continuous_gspo import NativeGSPOBackend, is_loading_unit
    from mri_rl.experiment_contract import load_contract
    from mri_rl.likelihood_cache import verify_cache_identity
    from mri_rl.optimizer_checkpoint import load_optimizer_checkpoint
    started = time.monotonic(); cache = {'dir': scope['derived_cache']['cache_dir']}; gate = PacedGate(out/'resources_PRIVATE.jsonl')
    def before(name):
        require(time.monotonic()-started <= p.MAX_SECONDS, 'Reviewed operation start deadline exceeded')
        result = gate.before_unit(name, loading=is_loading_unit(name, cache))
        require(time.monotonic()-started <= p.MAX_SECONDS, 'Reviewed deadline exceeded during resource hold')
        return result
    backend = None; state = rng = disk = actual = None; full_before = None; pending_error = None; result = None
    with Path(scope['global_lock']).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            before('native_model_load')
            require({k: importlib.metadata.version(k) for k in scope['runtime_versions']} == scope['runtime_versions'], 'Native runtime version changed')
            certificate = Path(scope['primary_root'])/'results/EXACT_QUANT_CACHE_CERTIFICATE.json'; cert = load(certificate)
            model_path = Path(scope['model_path']); artifacts = {str(model_path/name): sha(model_path/name) for name in set(cert['non_weight_files_identical']) | {'config.json'}}
            require(verify_cache_identity(cache['dir'], certificate, scope['derived_cache']['certificate_sha256'], artifacts) == scope['derived_cache'], 'Effective base cache changed')
            group = scope['group_PRIVATE']
            backend = NativeGSPOBackend(group['adapter_path'], {'group_size': 4, 'max_tokens': 256}, model_path=model_path,
                contract=load_contract(Path(scope['contract_path'])), seed=7, before_unit=before, cache=cache)
            before('full_parameter_fingerprints_before')
            full_before = utility._fingerprints(backend, dict(backend.tree_flatten(backend.model.parameters())))
            require(len(full_before) == 1942, 'Exact full-model parameter inventory required')
            utility.durable(out/'all_parameters_before_PRIVATE.json', full_before)
            disk = backend.load_arrays(group['adapter_path']); actual = dict(backend.tree_flatten(backend.model.trainable_parameters()))
            require(len(actual) == 504 and set(disk) == set(actual) and all(bool(backend.mx.array_equal(v, actual[k]).item()) for k, v in disk.items()), 'Corresponding 504 pre-rollout adapter tensors differ')
            disk = actual = None
            state, rng, _ = load_optimizer_checkpoint(Path(group['parent_directory'])/'optimizer',
                expected_bindings={'run': group['binding'], 'adapter_sha256': group['adapter_sha256'], 'trainable_signature': backend.trainable_signature()}, load_arrays=backend.load_arrays)
            backend.restore(state, rng)
            require(int(backend.optimizer_state()['step'].item()) == group['parent_optimizer_updates'] == 0, 'Actual inherited Adam0 required')
            from mlx_vlm.models.cache import make_prompt_cache
            result = p.run_resolution(backend, group, load(scope['prior_B2_record_path']), out/'measurements_PRIVATE', utility, before, make_prompt_cache)
            verify_files(scope)
        except BaseException as error:
            pending_error = error
            utility.durable(out/'STOP_PRIVATE.json', {'status': 'STOP_NO_AUTOMATIC_RETRY', 'error_type': type(error).__name__, 'error': str(error),
                'traceback': ''.join(traceback.format_exception(error)), 'generation_calls': 0, 'backward_calls': 0, 'optimizer_updates': 0})
            clear_unwound_exception_frames(error)
            raise
        finally:
            if backend is not None:
                try:
                    if full_before is not None:
                        full_after = utility._fingerprints(backend, dict(backend.tree_flatten(backend.model.parameters())))
                        utility.durable(out/'all_parameters_after_PRIVATE.json', full_after)
                        unchanged = full_before == full_after
                        utility.durable(out/'all_parameter_integrity_SAFE.json', {'all_parameters_unchanged': unchanged, 'tensor_count': len(full_after),
                            'before_digest': canonical(full_before), 'after_digest': canonical(full_after)})
                        require(unchanged, 'Full model parameters changed')
                    verify_files(scope)
                except BaseException as error:
                    utility.durable(out/'exit_integrity_failure_PRIVATE.json', {'error': str(error), 'error_type': type(error).__name__, 'original_failure_preserved': pending_error is not None})
                    clear_unwound_exception_frames(error)
                    if pending_error is None: raise
                finally:
                    cleanup_error = pending_error or sys.exception()
                    if cleanup_error is not None: clear_unwound_exception_frames(cleanup_error)
                    state = rng = disk = actual = None
                    try:
                        backend.close(); gc.collect(); backend.mx.clear_cache(); active = int(backend.mx.get_active_memory())
                        released = backend.model is None and active < 128*1024**2
                        utility.durable(out/'release_SAFE.json', {'model_is_none': backend.model is None, 'active_MLX_bytes': active,
                            'release_verified': released, 'original_failure_preserved': cleanup_error is not None})
                        if cleanup_error is None: require(released, 'Native release failed')
                    except BaseException as error:
                        utility.durable(out/'release_failure_PRIVATE.json', {'error': str(error), 'original_failure_preserved': cleanup_error is not None})
                        clear_unwound_exception_frames(error)
                        if cleanup_error is None: raise
    require(result is not None, 'No completed numerical evidence')
    result.update(scope_sha256=canonical(scope), complete_model_parameters_unchanged=True, native_release_verified=True,
                  budget=scope['budget'], generation_calls=0, backward_calls=0, optimizer_updates=0)
    save(out/'complete_SAFE.json', result)
    print(json.dumps({'status': result['status'], 'receipt_sha256': sha(out/'complete_SAFE.json'), 'B3_status': result['B3_status']}))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('primary-root', 'prior-scope', 'prior-B2-record', 'output-directory', 'scope-output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--review', type=Path); parser.add_argument('--expected-review-sha256'); parser.add_argument('--execute', action='store_true')
    a = parser.parse_args(); scope = prepare(a)
    require(a.scope_output.resolve().is_relative_to(W/'local_data'), 'Private ignored scope required')
    save(a.scope_output, scope)
    if not a.execute:
        print(json.dumps({'status': 'CPU_PREPARED_NO_NATIVE', 'scope_sha256': canonical(scope), 'scope_file_sha256': sha(a.scope_output), 'budget': scope['budget']}))
        return
    result = execute(a, scope)
    if not result['acceptance']['B2_numeric_acceptable']: raise SystemExit(2)


if __name__ == '__main__': main()
