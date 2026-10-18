"""CPU prepare / independently reviewed native saved-TRAIN GSPO path probe.

Default creates a private exact scope, never imports MLX. --execute is reserved
for the parent execution lead after independent review of this exact source.
"""
from __future__ import annotations
import argparse
import fcntl
import gc
import hashlib
import importlib.metadata
import json
import os
import struct
import sys
import time
import traceback
from pathlib import Path

W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W/'src'))
from mri_rl import native_gspo_path_probe_v1 as probe
require, sha, canonical = probe.require, probe.sha, probe.canonical


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    content = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()
    if path.exists():
        require(path.read_bytes() == content, 'Existing immutable scope differs; preserve it')
        return
    with path.open('xb') as handle:
        handle.write(content); handle.flush(); os.fsync(handle.fileno())


def header(path):
    with Path(path).open('rb') as handle:
        size = struct.unpack('<Q', handle.read(8))[0]
        require(0 < size < 8*1024**2, 'Invalid safetensors header')
        result = json.loads(handle.read(size)); result.pop('__metadata__', None)
        return result


def prepare(a):
    R = a.primary_root.resolve(); replay = a.replay_directory.resolve()
    ledger_path, inventory_path = replay/'group_causes_PRIVATE.jsonl', replay/'read_inventory_PRIVATE.json'
    ledger = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    inventory = load(inventory_path)
    chosen = probe.select_first_groups(ledger)
    pool = R/'local_data/c_two_task_pool_v1'; train = pool/'train.verl.jsonl'
    rows = {r['id']: r for r in map(json.loads, train.read_text().splitlines())}
    require(all(r['extra_info']['internal_split'] == 'train' for r in rows.values()), 'Frozen manifest must be TRAIN only')
    from mri_rl.vero_category_sampler import build_category_schedule
    groups = []; bound = {str(ledger_path): sha(ledger_path), str(inventory_path): sha(inventory_path), str(train): sha(train)}
    for selected in chosen:
        arm, seed, index = selected['arm'], selected['seed'], selected['group_index']
        tier = pool/'tier_c_v1'
        run = tier/'calibrated_bundle_v1'/f'{arm}_seed{seed}'/'run' if 'calibrated' in arm else tier/f'{arm}_seed{seed}_run_v1'
        parent = run/f'boundary_{index:04d}'; child = run/f'boundary_{index+1:04d}'
        parent_receipt = load(parent/'receipt.json'); receipt = load(child/'receipt.json'); raw = load(child/'rollout_PRIVATE.json')
        binding = receipt['binding']; experiment = binding['experiment']; conf = binding['configuration']
        require(parent_receipt['binding'] == binding and receipt['parent_receipt_sha256'] == sha(parent/'receipt.json'), 'Exact pre-rollout parent chain mismatch')
        require(parent_receipt['completed_groups'] == index and receipt['completed_groups'] == index+1 and
                raw['group_index'] == index and experiment['arm']['name'] == arm and experiment['seed'] == seed, 'Historical cursor/arm mismatch')
        require(conf['group_size'] == 4 and conf['max_tokens'] == 256, 'K4 saved-response profile required')
        require(experiment['common']['manifest_sha256'] == sha(train), 'Historical TRAIN binding changed')
        schedule = build_category_schedule(list(rows.values()), mode=experiment['arm']['sampling'], seed=seed,
            max_groups=198, specialist_category=experiment['arm'].get('task'))['schedule']
        require(canonical(schedule) == binding['schedule_sha256'] and [r['id'] for r in schedule] == binding['schedule_record_ids'], 'Historical schedule differs')
        row = schedule[index]
        require(row['id'] == raw['record_id'] == receipt['record_id'] == selected['record_id'] and row['ability'] == selected['task'], 'Historical TRAIN occurrence differs')
        require(row['extra_info']['internal_split'] == 'train' and receipt['updated'] and receipt['advantages'] == selected['advantages'], 'Saved accepted GSPO group/advantages differ')
        require(sha(parent/'receipt.json') == inventory[str(parent/'receipt.json')] and
                sha(child/'receipt.json') == selected['receipt_sha256'] == inventory[str(child/'receipt.json')] and
                sha(child/'rollout_PRIVATE.json') == selected['rollout_sha256'] == receipt['files_sha256']['rollout_PRIVATE.json'], 'Frozen replay lineage digest differs')
        for base, rec in [(parent, parent_receipt), (child, receipt)]:
            # Bind all checkpoint/raw files, not just adapter bytes. No arrays are loaded on CPU.
            bound[str(base/'receipt.json')] = sha(base/'receipt.json')
            for name, expected in rec['files_sha256'].items():
                path = base/name; require(sha(path) == expected, 'Historical artifact changed: ' + name)
                bound[str(path)] = expected
        bound[str(row['images'][0])] = sha(row['images'][0])
        require(bound[str(row['images'][0])] == raw['image_sha256'] == binding['image_sha256'][row['id']], 'Real source image digest differs')
        weights = header(parent/'adapter.safetensors')
        require(len(weights) == 504 and all('lora' in key.lower() and 'vision' not in key.lower() for key in weights), 'Full 504 language LoRA parent inventory required')
        require([len(r['token_ids']) for r in raw['responses']] == selected['lengths'], 'Saved response ID inventory differs')
        optimizer = load(parent/'optimizer/manifest.json')
        require(optimizer['bindings']['run'] == binding and optimizer['bindings']['adapter_sha256'] == sha(parent/'adapter.safetensors'), 'Parent complete optimizer binding differs')
        groups.append({'selection_PRIVATE': selected, 'row': row, 'raw': raw, 'advantages': receipt['advantages'],
            'parent_directory': str(parent), 'parent_receipt_sha256': sha(parent/'receipt.json'),
            'parent_optimizer_updates': parent_receipt['optimizer_updates'], 'binding': binding,
            'adapter_path': str(parent/'adapter.safetensors'), 'adapter_sha256': sha(parent/'adapter.safetensors'),
            'adapter_config_sha256': sha(parent/'adapter_config.json'), 'lengths': selected['lengths']})
    require(len({g['adapter_sha256'] for g in groups}) == 1, 'First groups need different weights: prepare explicit new reviewed reload scope')
    original_path = pool/'u7_visual_controls_v1/probe_plan_PRIVATE.json'; original = load(original_path)
    # Reuse only base/cache/contract metadata. Do not read this plan's evaluation rows.
    require(sha(original['prompt_contract_path']) == original['prompt_contract_sha256'], 'Frozen prompt contract changed')
    lock = pool/'tier_c_v1/remaining20_bundle_v1/remaining20.lock'
    require(lock.is_file(), 'Existing global remaining20 lock required')
    native = (R/'../../work/mlx-venv/lib/python3.13/site-packages').resolve()
    source_paths = [Path(__file__), W/'src/mri_rl/native_gspo_path_probe_v1.py']
    source_paths += [W/'src/mri_rl'/name for name in ('mlx_continuous_gspo.py', 'optimizer_checkpoint.py',
        'early_native_conditioning_probe_v1.py', 'paired_native_gradient_probe_v1.py', 'paced_resources.py',
        'resource_supervisor.py', 'likelihood_loading.py', 'likelihood_cache.py', 'experiment_contract.py',
        'vero_mri_reward.py', 'vero_category_sampler.py')]
    source_paths += [native/'mlx_vlm'/name for name in ('generate/dispatch.py', 'generate/ar.py', 'utils.py',
        'models/cache.py', 'models/qwen3_vl/qwen3_vl.py', 'models/qwen3_vl/language.py',
        'models/qwen3_vl/vision.py', 'models/qwen3_vl/processing_qwen3_vl.py', 'prompt_utils.py',
        'trainer/common/model.py')]
    source_paths += [native/'mlx/nn/utils.py', native/'mlx/core/__init__.pyi']
    bound[str(original_path)] = sha(original_path)
    bound[original['prompt_contract_path']] = original['prompt_contract_sha256']
    bound[str(R/'results/EXACT_QUANT_CACHE_CERTIFICATE.json')] = sha(R/'results/EXACT_QUANT_CACHE_CERTIFICATE.json')
    for name in ('tokenizer.json', 'tokenizer_config.json', 'config.json', 'generation_config.json', 'preprocessor_config.json'):
        path = Path(original['model_path'])/name
        require(path.is_file(), 'Actual tokenizer/processor artifact missing')
        bound[str(path)] = sha(path)
    out = a.output_directory.resolve()
    require(out.is_relative_to(W/'local_data') and not out.exists(), 'Fresh ignored private output required; no automatic retry')
    scope = {'schema': 'native-gspo-path-probe-v1', 'execution_tier': 'B', 'status': 'CPU_PREPARED_REQUIRES_INDEPENDENT_REVIEW',
        'primary_root': str(R), 'worktree': str(W), 'output_directory': str(out), 'groups_PRIVATE': groups,
        'bound_artifacts': bound, 'source_sha256': {str(p): sha(p) for p in source_paths},
        'model_path': original['model_path'], 'contract_path': original['prompt_contract_path'],
        'derived_cache': original['derived_cache'], 'global_lock': str(lock), 'budget': probe.budget(chosen),
        'runtime_versions': {'mlx': '0.32.4', 'mlx-vlm': '0.7.8'}, 'conditions': list(probe.CONDITIONS),
        'supports': {'REAL': list(probe.SUPPORTS), 'ZERO_NORMALIZED_PIXELS': [probe.SUPPORTS[0]]},
        'selection': 'FIRST accepted informative TRAIN sequence and correspondence groups in frozen ledger order; their distinct pre-rollout parent receipts have identical B28 weights and Adam0 but separately restored RNG.',
        'generation_tokens': 0, 'optimizer_updates': 0, 'sealed_test_accessed': False,
        'historical_gen_distribution': 'PARTIAL if raw selected generation logps absent; present KV/full parity does not certify historical sampler distribution.',
        'historical_inputs': 'Full prompt IDs/rendered prompt preserved. Historical pixels/grid/mask fingerprints absent => PARTIAL; compare two actual prepare-input paths now.',
        'B2_logp_paths': 'Primary native-dtype KV direct-logsumexp matches installed ar._step; save F32 stable normalization of the same logits separately. Both KV-vs-F32-full comparisons keep maxabs .002; no extra model forward.',
        'mission_alignment': {'tag': 'ENGINEERING_UNBLOCK', 'main_question': 'Qualify the native image-to-signed-GSPO derivative path required for Vero MRI transfer.',
            'falsifier': 'Any input mismatch, nonfinite logp/gradient, frozen KV/full tolerance failure or state mutation stops this configuration.',
            'isolation': 'Saved TRAIN responses, original checkpoints and advantages; no new generation or optimizer step.',
            'provenance': 'Exact historical receipts/optimizer/RNG/504weight files, bound source and image bytes; all raw evidence private.',
            'baseline': 'ZERO normalized pixels with its own detached preupdate old logps; strict-label derivative projection retains response denominator.',
            'resources': 'One resident model, K4 responses serial, existing lock/PacedGate/AC/no-other-heavy, bounded operation counts and 1800s wall time.'},
        'evidence_tier': 'PROPOSED_NOT_RUN', 'claim': 'Engineering diagnostic only; no training, accuracy recovery, anatomy or generalization claim.'}
    return scope


def verify_scope_files(scope):
    for path, expected in {**scope['bound_artifacts'], **scope['source_sha256']}.items():
        require(sha(path) == expected, 'Reviewed source or historical artifact changed: ' + path)


def verify_review(a, scope):
    require(a.review and a.expected_review_sha256 and sha(a.review) == a.expected_review_sha256, 'Exact independently supplied approval digest required')
    review = load(a.review)
    require(review.get('verdict') in {'PASS', 'PASS_WITH_LIMITATIONS'} and review.get('reviewer_kind') in {'isolated_codex', 'claude_code'} and
            review.get('reviewer_id') and review.get('reviewer_findings') and review.get('scope_sha256') == canonical(scope), 'Independent exact scope not approved')
    require(review.get('verdict') != 'PASS_WITH_LIMITATIONS' or review.get('limitations'), 'Review limitations missing')


def execute(a, scope):
    verify_review(a, scope); verify_scope_files(scope)
    out = Path(scope['output_directory']); out.mkdir(exist_ok=False, parents=True)
    from mri_rl import early_native_conditioning_probe_v1 as utility
    from mri_rl.paced_resources import PacedGate
    from mri_rl.mlx_continuous_gspo import NativeGSPOBackend, is_loading_unit
    from mri_rl.experiment_contract import load_contract
    from mri_rl.likelihood_cache import verify_cache_identity
    from mri_rl.optimizer_checkpoint import load_optimizer_checkpoint, pack_tree
    started = time.monotonic(); gate = PacedGate(out/'resources_PRIVATE.jsonl')
    cache = {'dir': scope['derived_cache']['cache_dir']}
    def before(name):
        require(time.monotonic()-started <= probe.MAX_SECONDS, 'Reviewed total wall budget exhausted')
        sample = gate.before_unit(name, loading=is_loading_unit(name, cache))
        require(time.monotonic()-started <= probe.MAX_SECONDS, 'Reviewed total wall budget exhausted during resource hold')
        return sample
    backend = None; full_before = None; state = rng = disk = actual = None; results = []; pending_error = None
    with Path(scope['global_lock']).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            before('native_model_load')
            require({k: importlib.metadata.version(k) for k in scope['runtime_versions']} == scope['runtime_versions'], 'Runtime version differs')
            certificate = Path(scope['primary_root'])/'results/EXACT_QUANT_CACHE_CERTIFICATE.json'
            cert = load(certificate); model = Path(scope['model_path'])
            artifacts = {str(model/name): sha(model/name) for name in set(cert['non_weight_files_identical']) | {'config.json'}}
            require(verify_cache_identity(cache['dir'], certificate, scope['derived_cache']['certificate_sha256'], artifacts) == scope['derived_cache'], 'Effective cached base changed')
            first = scope['groups_PRIVATE'][0]
            backend = NativeGSPOBackend(first['adapter_path'], {'group_size': 4, 'max_tokens': 256}, model_path=model,
                contract=load_contract(Path(scope['contract_path'])), seed=7, before_unit=before, cache=cache)
            before('all_model_parameter_fingerprints_before')
            full_before = utility._fingerprints(backend, dict(backend.tree_flatten(backend.model.parameters())))
            require(len(full_before) > 504, 'All-model parameter coverage missing')
            utility.durable(out/'all_parameters_before_PRIVATE.json', full_before)
            for i, group in enumerate(scope['groups_PRIVATE']):
                verify_scope_files(scope); before('restore_real_pre_rollout_parent_' + str(i))
                disk = backend.load_arrays(group['adapter_path']); actual = dict(backend.tree_flatten(backend.model.trainable_parameters()))
                require(len(actual) == 504 and set(disk) == set(actual) and all(bool(backend.mx.array_equal(v, actual[k]).item()) for k, v in disk.items()), 'Not exact corresponding 504-tensor pre-rollout adapter')
                del disk, actual
                state, rng, _ = load_optimizer_checkpoint(Path(group['parent_directory'])/'optimizer',
                    expected_bindings={'run': group['binding'], 'adapter_sha256': group['adapter_sha256'], 'trainable_signature': backend.trainable_signature()}, load_arrays=backend.load_arrays)
                backend.restore(state, rng)
                require(int(backend.optimizer_state()['step'].item()) == group['parent_optimizer_updates'], 'Wrong corresponding Adam step')
                results.append(probe.run_group(backend, group, out/f'group_{i}_PRIVATE', utility, before))
                state = rng = None; gc.collect(); backend.mx.clear_cache()
            verify_scope_files(scope)
        except BaseException as error:
            pending_error = error
            utility.durable(out/'STOP_PRIVATE.json', {'status': 'STOP_NO_AUTOMATIC_RETRY', 'error_type': type(error).__name__,
                'error': str(error), 'traceback': ''.join(traceback.format_exception(error)),
                'completed_groups': len(results), 'optimizer_updates': 0, 'elapsed_seconds': time.monotonic()-started})
            probe.clear_unwound_exception_frames(error)
            raise
        finally:
            if backend is not None:
                try:
                    # Complete model inventory before release, including frozen quantized base.
                    if full_before is not None:
                        full_after = utility._fingerprints(backend, dict(backend.tree_flatten(backend.model.parameters())))
                        utility.durable(out/'all_parameters_after_PRIVATE.json', full_after)
                        unchanged = full_after == full_before
                        utility.durable(out/'all_parameter_integrity_SAFE.json', {'all_parameters_unchanged': unchanged,
                            'tensor_count': len(full_after), 'before_digest': canonical(full_before), 'after_digest': canonical(full_after)})
                        require(unchanged, 'Full model parameter preservation failed')
                    verify_scope_files(scope)
                except BaseException as error:
                    if not (out/'STOP_PRIVATE.json').exists():
                        utility.durable(out/'STOP_PRIVATE.json', {'status': 'STOP_INTEGRITY_FAILURE_NO_RETRY',
                            'error_type': type(error).__name__, 'error': str(error), 'completed_groups': len(results), 'optimizer_updates': 0})
                    else:
                        utility.durable(out/'exit_integrity_failure_PRIVATE.json', {'error_type': type(error).__name__,
                            'error': str(error), 'traceback': ''.join(traceback.format_exception(error)), 'original_failure_preserved': pending_error is not None})
                    probe.clear_unwound_exception_frames(error)
                    if pending_error is None:
                        raise
                finally:
                    cleanup_error = pending_error or sys.exception()
                    if cleanup_error is not None: probe.clear_unwound_exception_frames(cleanup_error)
                    # Executing CLI frame cannot be cleared; drop its native leaves explicitly.
                    state = rng = disk = actual = None
                    try:
                        backend.close(); gc.collect(); backend.mx.clear_cache()
                        active = int(backend.mx.get_active_memory())
                        released = backend.model is None and active < 128*1024**2
                        utility.durable(out/'release_SAFE.json', {'model_is_none': backend.model is None, 'active_MLX_bytes': active,
                            'release_verified': released, 'original_failure_preserved': cleanup_error is not None})
                        if cleanup_error is None: require(released, 'Native model release failed')
                    except BaseException as release_error:
                        utility.durable(out/'release_failure_PRIVATE.json', {'error_type': type(release_error).__name__,
                            'error': str(release_error), 'original_failure_preserved': cleanup_error is not None})
                        probe.clear_unwound_exception_frames(release_error)
                        if cleanup_error is None: raise
    safe = {'schema': scope['schema'], 'status': 'READONLY_TWO_TRAIN_GSPO_PATH_GROUPS_COMPLETED_WITH_HISTORICAL_LIMITATIONS',
        'scope_sha256': canonical(scope), 'groups': len(results), 'optimizer_updates': 0, 'generated_tokens': 0,
        'budget': scope['budget'], 'comparisons': [r['comparisons'] for r in results],
        'historical_generation_distribution': [r['historical_generation_distribution'] for r in results],
        'label_support_status': [r['label_support_status'] for r in results],
        'complete_model_parameters_unchanged': True, 'optimizer_Python_MLX_RNG_preserved': True,
        'evidence_tier': 'OBSERVED_ENGINEERING_DIAGNOSTIC', 'scientific_generalization': 'NOT_ESTABLISHED'}
    save(out/'complete_SAFE.json', safe)
    print(json.dumps({'status': safe['status'], 'safe_receipt_sha256': sha(out/'complete_SAFE.json'), 'optimizer_updates': 0}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--primary-root', type=Path, required=True)
    parser.add_argument('--replay-directory', type=Path, required=True)
    parser.add_argument('--output-directory', type=Path, required=True)
    parser.add_argument('--scope-output', type=Path, required=True)
    parser.add_argument('--review', type=Path)
    parser.add_argument('--expected-review-sha256')
    parser.add_argument('--execute', action='store_true')
    a = parser.parse_args(); scope = prepare(a)
    require(a.scope_output.resolve().is_relative_to(W/'local_data'), 'Ignored private scope output required')
    save(a.scope_output, scope)
    if not a.execute:
        print(json.dumps({'status': 'CPU_PREPARED_NO_NATIVE', 'scope_sha256': canonical(scope), 'scope_file_sha256': sha(a.scope_output), 'budget': scope['budget']}))
        return
    execute(a, scope)


if __name__ == '__main__':
    main()
