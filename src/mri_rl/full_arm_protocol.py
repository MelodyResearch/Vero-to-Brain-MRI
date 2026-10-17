"""Compile a finite two-task arm budget from measured, frozen training inputs.

This is a CPU configuration compiler, not a trainer or launch approval. Input tokens
include visual placeholders expanded by the exact processor; visual_tokens are a
subset, so counting them twice would inflate the budget. No MRI values are loaded.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

TASK_SOURCES = {'mri_sequence': 'OpenNeuro_ds007045',
                'mri_visit_correspondence': 'OpenNeuro_ds004856'}
BINDINGS = ('cache_certificate_sha256', 'starting_adapter_sha256', 'prompt_contract_sha256',
            'native_backend_sha256', 'reward_source_sha256')
GROUP_SIZE, MAX_TOKENS, WORST_CASE_GROUP_EQUIVALENTS = 4, 256, 64


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def compile_plan(measurements, bindings):
    """Freeze numeric budgets only after every exact training input was tokenized.

    The global cost cap is common across specialists and mixtures. Whole groups
    reserve prompt+maximum-generation cost before launching; actual response lengths
    are debited afterwards. Unspent final reservation is reported, not concealed.
    """
    require(measurements.get('schema') == 'frozen-training-token-measurements-v1'
            and measurements.get('split') == 'train', 'Exact training token receipt required')
    require(_sha(measurements.get('manifest_sha256')) and _sha(measurements.get('processor_sha256')),
            'Frozen manifest/processor hashes required')
    require(set(bindings) == set(BINDINGS) and all(_sha(bindings[k]) for k in BINDINGS),
            'Exact common start, cache, prompt, backend and reward hashes required')
    rows = measurements.get('rows', [])
    require(isinstance(rows, list) and bool(rows), 'No measured training inputs')
    seen = set()
    for row in rows:
        require(row.get('route') in TASK_SOURCES and row.get('source') == TASK_SOURCES[row['route']],
                'Task/source mismatch')
        key = row.get('record_sha256')
        require(_sha(key) and key not in seen, 'Duplicate or unbound source record')
        seen.add(key)
        inp, visual = row.get('input_tokens'), row.get('visual_tokens')
        require(type(inp) is int and type(visual) is int and 0 < visual < inp <= 2048,
                'Invalid measured native prompt/visual-token counts')
    counts = dict(Counter(row['route'] for row in rows))
    require(set(counts) == set(TASK_SOURCES), 'Both heterogeneous tasks must be present')
    minimum, maximum = min(r['input_tokens'] for r in rows), max(r['input_tokens'] for r in rows)
    cap = WORST_CASE_GROUP_EQUIVALENTS * GROUP_SIZE * (maximum + MAX_TOKENS)
    arms = [
        {'name': 'S_sequence', 'sampling': 'specialist', 'task': 'mri_sequence', 'reward': 'raw'},
        {'name': 'S_correspondence', 'sampling': 'specialist', 'task': 'mri_visit_correspondence', 'reward': 'raw'},
        {'name': 'P_raw', 'sampling': 'proportional', 'reward': 'raw'},
        {'name': 'U_raw', 'sampling': 'uniform', 'reward': 'raw'},
        {'name': 'P_calibrated', 'sampling': 'proportional', 'reward': 'medgrpo_eq3'},
        {'name': 'U_calibrated', 'sampling': 'uniform', 'reward': 'medgrpo_eq3'},
    ]
    return {
        'schema': 'two-task-finite-arm-plan-v1', 'status': 'COMPILED_NOT_LAUNCH_AUTHORIZED',
        'research_tag': 'PRIMARY_VERO_TRANSFER', 'evidence_tier': 'PROPOSED',
        'official_vero_commit': '025ca3c800fc95224c86323c1f7fd33dceed3090',
        'seeds': [7, 11, 13], 'common_bindings': dict(bindings),
        'measurements_sha256': hashlib.sha256(json.dumps(measurements, sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode()).hexdigest(),
        'manifest_sha256': measurements['manifest_sha256'],
        'processor_sha256': measurements['processor_sha256'], 'training_rows': counts,
        'task_weights': {'uniform': {t: 0.5 for t in TASK_SOURCES},
                         'proportional': {t: counts[t] / len(rows) for t in TASK_SOURCES}},
        'group_size': GROUP_SIZE, 'max_tokens': MAX_TOKENS, 'temperature': 1.0,
        'microbatch_responses': 1, 'rl_arms': arms,
        'references': ['common_B28_start', 'same_C_data_SFT_from_common_B28', 'historical_A0'],
        'budget': {
            'unit': 'sum_per_response(native_input_tokens + generated_tokens)',
            'text_visual_token_cap': cap, 'worst_case_group_equivalents': WORST_CASE_GROUP_EQUIVALENTS,
            'input_tokens_min': minimum, 'input_tokens_max': maximum,
            'maximum_attempted_groups': cap // (GROUP_SIZE * minimum),
            'maximum_optimizer_updates': cap // (GROUP_SIZE * minimum),
            'whole_group_reservation': True, 'actual_cost_debit': True,
            'exact_FLOP_match_claimed': False,
        },
        'active_group_rule': 'all-full-format_task-varies_equal-overlong',
        'advantage_profile': 'sample_std_plus_1e-6; transform-before-advantages',
        'medgrpo': {'equation': 'sigmoid(3*(accuracy-training_p50)/training_IQR)',
                    'statistics_source': 'common-initializer training-only calibration rollouts',
                    'zero_IQR': 'BLOCK_CALIBRATED_ARMS; no invented floor',
                    'format_and_overlong': 'unchanged; 0.8*transformed_accuracy+0.2*format+overlong',
                    'mechanism': 'measure all support; >=3 distinct active reward levels required for nonlinear claim'},
        'required_before_launch': ['independent_task_QC_and_hidden_target_review',
            'source_and_pair_cluster_split_validation', 'exact_token_measurement_receipt',
            'native_driver_token_budget_and_repeated_occurrence_support',
            'calibrated_reward_hook_and_training_statistics',
            'scoped_native_gradient_optimizer_resume_verification', 'AC_and_actual_resource_checks'],
        'optimizer_updates_executed': 0, 'final_evaluation_executed': False,
    }


def can_start_group(used, input_tokens, plan):
    require(type(used) is int and used >= 0 and type(input_tokens) is int,
            'Integer costs required')
    budget = plan['budget']
    require(budget['input_tokens_min'] <= input_tokens <= budget['input_tokens_max'],
            'Unmeasured input cost')
    reserve = plan['group_size'] * (input_tokens + plan['max_tokens'])
    return used + reserve <= budget['text_visual_token_cap']


def charge_group(input_tokens, generated_lengths, plan):
    require(type(input_tokens) is int and plan['budget']['input_tokens_min'] <= input_tokens
            <= plan['budget']['input_tokens_max'], 'Unmeasured input cost')
    require(len(generated_lengths) == plan['group_size'] and all(type(n) is int and
            0 < n <= plan['max_tokens'] for n in generated_lengths), 'Complete bounded group required')
    return plan['group_size'] * input_tokens + sum(generated_lengths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--measurements', type=Path, required=True)
    parser.add_argument('--bindings', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plan = compile_plan(json.loads(args.measurements.read_text()), json.loads(args.bindings.read_text()))
    with args.output.open('x') as handle:
        json.dump(plan, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'status': plan['status'], 'token_cap': plan['budget']['text_visual_token_cap'],
                      'max_groups': plan['budget']['maximum_attempted_groups'], 'rl_arms': len(plan['rl_arms'])}))


if __name__ == '__main__':
    main()
