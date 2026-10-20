#!/usr/bin/env python3
"""Saved-only sanitized summary; no model/native imports or new computations."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def summarize(base):
    scope_file = base / 'scope_v4_PRIVATE.json'
    assert digest(scope_file) == '0dc854bcd5834ecb8f27fb38a867304084f7c2772f2be00a5c1ce08e55670d1f'
    scope = read(scope_file)
    out = Path(scope['output'])
    closed = read(base / 'actual_exit_v4_PRIVATE.json')
    assert closed['returncode'] == 0
    assert closed['closed_stdout_sha256'] == digest(base / 'native_stdout_v4_PRIVATE.log')
    complete = read(out / 'complete_PRIVATE.json')
    assert complete['status'] == 'ACTUAL_MARGIN_VJP_READONLY_COMPLETE'
    assert complete['scope_sha256'] == hashlib.sha256(json.dumps(scope, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    assert complete['source_sha256'] == scope['source_sha256']
    for paths in (scope['source_sha256'], scope['bound_artifact_sha256']):
        assert all(digest(Path(path)) == expected for path, expected in paths.items())
    assert complete['actual_calls'] == {'backward_calls': 4, 'full_TF_forwards': 11, 'teacherforced_positions': 2868}
    assert all(complete[key] == 0 for key in ('optimizer_updates', 'generation_calls', 'parameter_perturbations'))
    integrity = read(out / 'before_restore_integrity_PRIVATE.json')
    assert integrity['unchanged'] and len(integrity['all_parameters_before']) == 1942
    assert integrity['all_parameters_before'] == integrity['all_parameters_after']
    assert integrity['state_before'] == integrity['state_after']
    assert len(integrity['state_before']['arrays']) == 1011
    restored = read(out / 'restored_state_PRIVATE.json')
    assert restored['unchanged'] and restored['initial'] == restored['restored'] == integrity['state_before']
    release = read(out / 'release_SAFE.json')
    assert release['release_verified'] and release['model_is_none'] and release['active_MLX_bytes'] < 128 * 1024**2
    scores = {name: read(out / (name + '_scores_PRIVATE.json')) for name in
              ('F_REAL', 'T_REAL', 'DALLAS_REAL', 'BLANK_ZERO_NORMALIZED_PIXELS')}
    assert all(s['common_prefix_alignment']['max_abs_error'] <= .002 for s in scores.values())
    assert all(scores[name]['reference_alignment']['per_label_logps_aligned'] for name in ('F_REAL', 'T_REAL', 'DALLAS_REAL'))
    assert scores['BLANK_ZERO_NORMALIZED_PIXELS']['blank_value_oracle_alignment']['per_label_logps_aligned']
    events = [json.loads(line) for line in (out / 'resources_PRIVATE.jsonl').read_text().splitlines()]
    samples = [e['sample'] for e in events]
    assert min(s['battery_percent'] for s in samples) >= 60
    assert all(s['other_heavy_processes'] == 0 for s in samples)
    analysis = complete['analysis']
    result = {
        'schema': 'class-margin-v2-Adam60-actual-native-directions-v1',
        'status': 'ACTUAL_NATIVE_READONLY_COMPLETE_STATE_UNCHANGED',
        'scope_file_sha256': digest(scope_file), 'scope_sha256': complete['scope_sha256'],
        'private_complete_sha256': digest(out / 'complete_PRIVATE.json'),
        'external_exit_sha256': digest(base / 'actual_exit_v4_PRIVATE.json'),
        'closed_stdout_sha256': closed['closed_stdout_sha256'],
        'human_authorization_sha256': closed['human_sha256'],
        'source_commit_at_launch': 'da5276a8d03961e71165ef22cead9c7ce86be3d6',
        'actual_calls': complete['actual_calls'], 'generation_calls': 0,
        'parameter_perturbations': 0, 'optimizer_updates': 0, 'training_runs': 0,
        'external_wall_seconds': closed['external_wall_seconds'],
        'native_wall_seconds': complete['wall_seconds'],
        'actual_optimizer_profile': analysis['actual_optimizer_profile'],
        'original_correct_margins': {name: scores[name]['margins'] for name in ('F_REAL', 'T_REAL', 'DALLAS_REAL')},
        'conditions': {name: {'loss_values': row['loss_values'], 'versions': row['versions']}
                       for name, row in analysis['conditions'].items()},
        'gradient_comparisons': analysis['gradient_comparisons'],
        'state_integrity': {'all1942parameters_unchanged_before_restore': True,
                            'all1011optimizer_RNG_array_fingerprints_and_layout_unchanged': True,
                            'restored_state_exact': True, 'release': release},
        'resource_summary': {
            'samples': len(events), 'actions': dict(Counter(e['action'] for e in events)),
            'supervisor_levels': dict(Counter(e['supervisor_level'] for e in events)),
            'reasons': dict(Counter(reason for e in events for reason in e['reasons'])),
            'minimum_battery_percent': min(s['battery_percent'] for s in samples),
            'maximum_MLX_peak_GiB': max(s['mlx_peak_gib'] or 0 for s in samples),
            'minimum_available_GiB': min(s['available_gib'] for s in samples),
            'minimum_pressure_free_percent': min(s['pressure_free_percent'] for s in samples),
            'initial_swap_GiB': samples[0]['swap_used_gib'], 'final_swap_GiB': samples[-1]['swap_used_gib'],
            'thermal_states': sorted({s['thermal_state'] for s in samples})},
        'interpretation': [
            'V2 has nonzero parameter gradient and differs from blank/swapped controls on this ONE TRAIN pair; substantial shared/prior components remain.',
            'V2 real Euclidean and retained-Adam directions favor T1 but weaken FLAIR; the no-class-tradeoff criterion is NOT_MET.',
            'Dallas auxiliary increment is negative despite positive total Adam direction from old momentum; negative-transfer risk remains.',
            'T1 starts negative and its first-order Adam preview remains negative; no observed correct-class crossing or accuracy improvement.',
            'All directions are local dot products, not evaluated finite steps, trained checkpoint predictions or recall changes.',
            'Only mean-margin gradients measured; sum/first directional effects UNKNOWN. Adam preview is auxiliary-only, not joint GSPO+aux.'
        ],
        'training_readiness': 'NO_CLASS_PROTECTION_CRITERION_NOT_MET_AND_NO_TRAINING_AUTHORIZATION',
        'accuracy_improvement_measured': False, 'holdout_or_final_access': False,
        'independent_actual_review': 'SEPARATE_SAVED_ONLY_REVIEW_REQUIRED',
        'raw_patient_images_text_tokens_gradients_optimizer_RNG_in_public_output': False
    }
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.base)
    with args.output.open('x') as f:
        json.dump(result, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')
    print(json.dumps({'status': result['status'], 'public_sha256': digest(args.output),
                      'optimizer_updates': 0, 'accuracy_improvement_measured': False}))
