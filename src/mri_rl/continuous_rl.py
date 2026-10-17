"""Persistent-state, completed-group RL driver (backend injected, no model imports).

This is the versioned Metal/LoRA diagnostic profile, not official Vero parity.
The historical reset runner is untouched. Native callers must provide a scoped
preflight/resource authorizer and keep output under ignored local_data. Backends
own one optimizer and expose native serialization; this driver owns transition,
strict reward selection, cursor, crash-consistent bundle and duplicate protection.
"""
from __future__ import annotations
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from mri_rl.optimizer_checkpoint import save_optimizer_checkpoint, load_optimizer_checkpoint, pack_tree
from mri_rl.mlx_gspo_smoke import advantages_from_rewards, task_group_is_informative, overlong_penalty
from mri_rl.vero_mri_reward import compute_score_from_data_source

PROFILE = {
    'profile': 'continuous_mlx_diagnostic_v1',
    'optimizer': {'name': 'Adam', 'learning_rate': 1e-5, 'betas': [0.9, 0.999],
                  'eps': 1e-8, 'bias_correction': False},
    'scheduler': {'type': 'constant'}, 'reward_profile': 'direct_vero',
    'temperature': 1.0, 'group_size': 2, 'max_tokens': 256,
    'clip_low': .0003, 'clip_high': .0004, 'log_ratio_max': 10.,
    'precision': '4bit-base-native-LoRA', 'lora_rank': 4, 'lora_alpha': 8,
    'active_group_rule': 'all-full-format_task-varies_equal-overlong',
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def write_json(path, value):
    with Path(path).open('x') as f:
        f.write(canonical(value) + '\n'); f.flush(); os.fsync(f.fileno())


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def state_fingerprint(backend, state, path):
    layout, arrays = pack_tree(state)
    backend.save_arrays(path, arrays)
    with Path(path).open('rb') as handle: os.fsync(handle.fileno())
    return hashlib.sha256((canonical(layout) + digest(path)).encode()).hexdigest()


def _step(backend):
    value = backend.optimizer_state()['step']
    value = value.item() if hasattr(value, 'item') else value
    require(type(value) is int and value >= 0, 'Invalid optimizer counter')
    return value


def _binding(schedule, bindings, config):
    require(isinstance(bindings, dict) and bindings, 'Exact experiment bindings required')
    require(schedule and len({r['id'] for r in schedule}) == len(schedule), 'Unique frozen schedule required')
    images = {}
    for row in schedule:
        extra = row['extra_info']
        require(row['data_source'] == 'OpenNeuro_ds007045' and extra.get('internal_split') == 'train'
                and extra.get('patient_lineage') == 'BIDS_SUBJECT_VERIFIED' and extra.get('patient_key'),
                'Training-only verified patient schedule required')
        require(row['ability'] == extra['reward_type'] and len(row['images']) == 1,
                'Task route or image inventory mismatch')
        images[row['id']] = digest(row['images'][0])
    return {'experiment': bindings, 'configuration': config,
            'schedule_sha256': hashlib.sha256(canonical(schedule).encode()).hexdigest(),
            'schedule_record_ids': [row['id'] for row in schedule],
            'image_sha256': images, 'driver_source_sha256': digest(__file__)}


def inspect_boundaries(output, expected_binding):
    """Hash and lineage validation before backend/model construction; no tensors decoded."""
    paths = sorted(Path(output).glob('boundary_*'))
    previous = None; last = None
    for index, path in enumerate(paths):
        require(path.name == f'boundary_{index:04d}' and path.is_dir() and not path.is_symlink(),
                'Boundary gap or invalid path')
        receipt = json.loads((path / 'receipt.json').read_text())
        require(receipt.get('schema') == 'continuous-rl-boundary-v1' and
                type(receipt.get('updated')) is bool, 'Invalid completed receipt schema')
        require(receipt['binding'] == expected_binding, 'Resume binding mismatch')
        require(receipt['completed_groups'] == index and receipt['parent_receipt_sha256'] == previous,
                'Boundary cursor/parent chain mismatch')
        require(type(receipt['optimizer_updates']) is int and 0 <= receipt['optimizer_updates'] <= index,
                'Boundary update counter mismatch')
        for name, sha in receipt['files_sha256'].items():
            require(name in {'adapter.safetensors', 'adapter_config.json', 'optimizer/manifest.json', 'optimizer/state.safetensors',
                             'rollout_PRIVATE.json', 'rollout_progress_PRIVATE.json',
                             'optimizer_before.safetensors', 'optimizer_after.safetensors'},
                    'Unexpected bundle inventory')
            p = path / name
            require(p.is_file() and not p.is_symlink() and digest(p) == sha, 'Boundary file digest mismatch')
        require({'adapter.safetensors', 'optimizer/manifest.json', 'optimizer/state.safetensors'} <= set(receipt['files_sha256']),
                'Incomplete committed boundary')
        manifest = json.loads((path / 'optimizer/manifest.json').read_text())
        require(manifest['completed_groups'] == index and manifest['optimizer_updates'] == receipt['optimizer_updates'],
                'Optimizer/group counters differ')
        require(manifest['bindings']['adapter_sha256'] == receipt['files_sha256']['adapter.safetensors']
                and manifest['bindings']['run'] == expected_binding, 'Optimizer/adapter binding differs')
        if last is not None:
            require(receipt['optimizer_updates'] == last['optimizer_updates'] + int(receipt['updated']),
                    'Duplicate or skipped optimizer update')
            require(index <= len(expected_binding['schedule_record_ids']) and
                    receipt['record_id'] == receipt['schedule_record_id'] ==
                    expected_binding['schedule_record_ids'][index-1], 'Recorded group differs')
            require({'rollout_PRIVATE.json', 'optimizer_before.safetensors',
                     'optimizer_after.safetensors'} <= set(receipt['files_sha256']),
                    'Completed group is missing rollout/state evidence')
            if not receipt['updated']:
                require(receipt['files_sha256']['adapter.safetensors'] == last['files_sha256']['adapter.safetensors'] and
                        receipt['optimizer_before_fingerprint'] == receipt['optimizer_after_fingerprint'],
                        'Skipped boundary changed weights or optimizer')
        else:
            require(receipt['optimizer_updates'] == 0 and not receipt['updated'] and
                    receipt['record_id'] is None and receipt['schedule_record_id'] is None,
                    'Invalid initial boundary')
        previous = digest(path / 'receipt.json'); last = {**receipt, 'path': path}
    return last


def _publish(output, work, backend, binding, completed, updates, parent, evidence):
    validator = getattr(backend, 'validate_state', None)
    require(callable(validator), 'Backend must validate complete finite parameters and optimizer state')
    validity = validator()
    require(validity.get('parameters_finite') is True and validity.get('optimizer_finite') is True and
            validity.get('second_moments_nonnegative') is True, 'Invalid resulting model/optimizer state')
    adapter = work / 'adapter.safetensors'; backend.save_adapter(adapter)
    with adapter.open('rb') as f: os.fsync(f.fileno())
    state_binding = {'run': binding, 'adapter_sha256': digest(adapter),
                     'trainable_signature': backend.trainable_signature()}
    save_optimizer_checkpoint(work / 'optimizer', optimizer_state=backend.optimizer_state(),
        rng_state=backend.rng_state(), bindings=state_binding, completed_groups=completed,
        optimizer_updates=updates, save_arrays=backend.save_arrays)
    files = {str(p.relative_to(work)): digest(p) for p in work.rglob('*') if p.is_file()}
    # Backend partial rollout logs are not automatically admitted as a completed bundle.
    allowed = {'adapter.safetensors', 'adapter_config.json', 'optimizer/manifest.json', 'optimizer/state.safetensors',
               'rollout_PRIVATE.json', 'rollout_progress_PRIVATE.json',
               'optimizer_before.safetensors', 'optimizer_after.safetensors'}
    require(set(files) <= allowed, 'Unexpected backend output in atomic bundle')
    receipt = {'schema': 'continuous-rl-boundary-v1', 'binding': binding,
               'completed_groups': completed, 'optimizer_updates': updates,
               'parent_receipt_sha256': parent, 'files_sha256': files, **evidence,
               'implementation': 'Metal/LoRA continuous diagnostic; not native Vero parity',
               'state_validation': validity, 'science_claim': False}
    for p in work.rglob('*'):
        if p.is_file():
            with p.open('rb') as handle: os.fsync(handle.fileno())
    write_json(work / 'receipt.json', receipt); sync_dir(work)
    destination = output / f'boundary_{completed:04d}'
    require(not destination.exists(), 'Boundary already committed; duplicate update refused')
    os.rename(work, destination); sync_dir(output)
    return {**receipt, 'path': destination}


def _grades(row, raw, config):
    require(raw.get('record_id') == row['id'], 'Rollout record mismatch')
    responses = raw.get('responses', [])
    require(len(responses) == config['group_size'], 'Incomplete rollout group')
    grades = []
    for response in responses:
        tokens = response.get('token_ids', [])
        require(isinstance(response.get('text'), str) and tokens and len(tokens) <= config['max_tokens']
                and all(type(t) is int and t >= 0 for t in tokens), 'Invalid response tokens/text')
        grade = compute_score_from_data_source(row['data_source'], response['text'],
            row['reward_model']['ground_truth'], row['extra_info'])
        grade['overlong'] = overlong_penalty(len(tokens), config['max_tokens'], 'direct_vero')
        grade['total'] = grade['score'] + grade['overlong']
        require(all(math.isfinite(v) for v in grade.values()), 'Nonfinite reward')
        grades.append(grade)
    return grades


def run(*, output, schedule, bindings, backend_factory, authorize, max_new_groups,
        resume=False, starting_adapter=None, config=None, max_optimizer_updates=None):
    """Run at most a fixed number of attempted groups, with one backend/optimizer.

    authorize(context) MUST independently validate exact scope and resources;
    called before backend construction and each group. Backend factory receives
    (current_adapter_path, explicit_profile). Backends must evaluate native lazy
    arrays before returning state/save operations. Fixture injection is explicit.
    """
    require(callable(authorize), 'Independent authorizer is required')
    require(max_optimizer_updates is None or (type(max_optimizer_updates) is int and max_optimizer_updates >= 1),
            'Optimizer update budget must be a positive integer')
    require(type(max_new_groups) is int and 1 <= max_new_groups <= 3, 'Bounded 1–3 group stage required')
    config = json.loads(canonical(PROFILE if config is None else config))
    for key in PROFILE:
        if key not in {'group_size', 'max_tokens'}:
            require(config.get(key) == PROFILE[key], 'Historical optimizer/reward/profile change requires new version')
    require(2 <= config['group_size'] <= 8 and 16 <= config['max_tokens'] <= 512, 'Invalid bounded rollout configuration')
    binding = _binding(schedule, bindings, config); output = Path(output)
    if output.exists() and not resume:
        raise FileExistsError('Existing run requires explicit resume; preserve evidence')
    if resume:
        require(output.is_dir(), 'Missing run for resume')
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.writer.lock').open('a') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('Another continuous RL writer is active') from None
        current = inspect_boundaries(output, binding)
        require(not resume or current is not None, 'No complete boundary to resume')
        cursor = current['completed_groups'] if current else 0
        require(cursor <= len(schedule), 'Cursor exceeds frozen schedule')
        if cursor == len(schedule):
            return {'completed_groups': cursor, 'optimizer_updates': current['optimizer_updates'],
                    'status': 'SCHEDULE_COMPLETE', 'boundary': str(current['path'])}
        if current and max_optimizer_updates is not None and current['optimizer_updates'] >= max_optimizer_updates:
            return {'completed_groups': cursor, 'optimizer_updates': current['optimizer_updates'],
                    'status': 'UPDATE_BUDGET_REACHED', 'boundary': str(current['path'])}
        adapter = current['path'] / 'adapter.safetensors' if current else starting_adapter
        context = {'phase': 'load', 'binding': binding, 'cursor': cursor,
                   'max_new_groups': min(max_new_groups, len(schedule)-cursor),
                   'starting_adapter': str(adapter) if adapter else None,
                   'boundary_receipt_sha256': digest(current['path'] / 'receipt.json') if current else None}
        authorize(context)  # Must precede construction/import of any native backend.
        backend = backend_factory(adapter, config)
        try:
            if current:
                expected = {'run': binding, 'adapter_sha256': digest(adapter),
                            'trainable_signature': backend.trainable_signature()}
                state, rng, _ = load_optimizer_checkpoint(current['path'] / 'optimizer',
                    expected_bindings=expected, load_arrays=backend.load_arrays)
                backend.restore(state, rng)  # Restore AFTER setup, before any stochastic action.
            else:
                require(_step(backend) == 0, 'Fresh run requires initial optimizer')
                work = Path(tempfile.mkdtemp(prefix='.pending-', dir=output))
                current = _publish(output, work, backend, binding, 0, 0, None,
                                   {'updated': False, 'record_id': None, 'schedule_record_id': None})
            for index in range(cursor, min(cursor+max_new_groups, len(schedule))):
                # Total budget is enforced before any further rollout; a skip never consumes it.
                if max_optimizer_updates is not None and current['optimizer_updates'] >= max_optimizer_updates:
                    break
                authorize({**context, 'phase': 'group', 'cursor': index,
                           'boundary_receipt_sha256': digest(current['path'] / 'receipt.json'),
                           'starting_adapter': str(current['path'] / 'adapter.safetensors')})
                row = schedule[index]; before_step = _step(backend)
                expected_image = binding['image_sha256'][row['id']]
                require(digest(row['images'][0]) == expected_image, 'Frozen image changed before rollout')
                require(before_step == current['optimizer_updates'], 'Optimizer counter lost continuity')
                work = Path(tempfile.mkdtemp(prefix='.pending-', dir=output))
                before_state = state_fingerprint(backend, backend.optimizer_state(), work / 'optimizer_before.safetensors')
                raw = backend.rollout(row, index, work)
                write_json(work / 'rollout_PRIVATE.json', raw)
                require(raw.get('image_sha256') == expected_image and digest(row['images'][0]) == expected_image,
                        'Rollout image differs from frozen input')
                post_rollout = getattr(backend, 'after_rollout', None)
                if post_rollout is not None: post_rollout()
                authorize({**context, 'phase': 'post_rollout', 'cursor': index,
                           'boundary_receipt_sha256': digest(current['path'] / 'receipt.json')})
                require(digest(row['images'][0]) == expected_image, 'Frozen image changed after rollout')
                grades = _grades(row, raw, config)
                advantages = advantages_from_rewards([g['total'] for g in grades])
                active = task_group_is_informative(grades); update_evidence = {}
                if active:
                    require(max(advantages)-min(advantages) > 1e-12, 'Informative group has zero advantage')
                    update_evidence = backend.update(row, raw, advantages)
                    errors = update_evidence.get('old_current_logp_max_abs_errors', [])
                    losses = update_evidence.get('loss_terms', [])
                    norm = update_evidence.get('gradient_l2', float('nan'))
                    require(len(errors) == len(losses) == config['group_size'] and
                            all(math.isfinite(x) and 0 <= x <= 1e-5 for x in errors) and
                            all(math.isfinite(x) for x in losses) and math.isfinite(norm) and norm > 0,
                            'Update lacks finite gradient/loss and on-policy alignment evidence')
                require(_step(backend) == before_step + int(active), 'Optimizer step count mismatch')
                after_state = state_fingerprint(backend, backend.optimizer_state(), work / 'optimizer_after.safetensors')
                if not active: require(before_state == after_state, 'Inactive group changed optimizer state')
                # Verify byte-stable weights even for a skip; backend serializes deterministically.
                check = work / 'verify_adapter.safetensors'; backend.save_adapter(check)
                changed = digest(check) != current['files_sha256']['adapter.safetensors']; check.unlink()
                require(changed == active, 'Adapter update/skip evidence mismatch')
                current = _publish(output, work, backend, binding, index+1, before_step+int(active),
                    digest(current['path'] / 'receipt.json'), {'updated': active, 'record_id': row['id'],
                    'schedule_record_id': schedule[index]['id'], 'grades': grades, 'advantages': advantages,
                    'optimizer_before_fingerprint': before_state, 'optimizer_after_fingerprint': after_state,
                    'update_evidence': update_evidence})
            budget_hit = max_optimizer_updates is not None and current['optimizer_updates'] >= max_optimizer_updates
            return {'status': 'UPDATE_BUDGET_REACHED' if budget_hit else
                              'SCHEDULE_COMPLETE' if current['completed_groups'] == len(schedule) else 'BOUNDED_STAGE_COMPLETE',
                    'completed_groups': current['completed_groups'], 'optimizer_updates': current['optimizer_updates'],
                    'boundary': str(current['path']), 'exact_resume_profile': config['profile'],
                    'native_runtime_validation': 'REQUIRES_SEPARATE_EVIDENCE', 'science_claim': False}
        finally:
            backend.close()
