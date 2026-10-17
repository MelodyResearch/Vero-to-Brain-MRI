"""Finite-budget compiler fixtures; no real model or task admission is implied."""
import pytest
from mri_rl.full_arm_protocol import compile_plan, can_start_group, charge_group


def measurements():
    return {'schema': 'frozen-training-token-measurements-v1',
            'split': 'train', 'manifest_sha256': 'a' * 64,
            'processor_sha256': 'b' * 64,
            'rows': [
                {'route': 'mri_sequence', 'source': 'OpenNeuro_ds007045',
                 'record_sha256': '1' * 64, 'input_tokens': 300, 'visual_tokens': 154},
                {'route': 'mri_sequence', 'source': 'OpenNeuro_ds007045',
                 'record_sha256': '2' * 64, 'input_tokens': 300, 'visual_tokens': 154},
                {'route': 'mri_visit_correspondence', 'source': 'OpenNeuro_ds004856',
                 'record_sha256': '3' * 64, 'input_tokens': 700, 'visual_tokens': 550}]}


def bindings():
    return {name: c * 64 for name, c in zip(
        ('cache_certificate_sha256', 'starting_adapter_sha256', 'prompt_contract_sha256',
         'native_backend_sha256', 'reward_source_sha256'), 'cdef0')}


def test_budget_uses_actual_frozen_input_cost_and_complete_arm_list():
    plan = compile_plan(measurements(), bindings())
    assert plan['budget']['text_visual_token_cap'] == 64 * 4 * (700 + 256)
    assert plan['budget']['maximum_attempted_groups'] == (64 * 4 * 956) // (4 * 300)
    assert plan['seeds'] == [7, 11, 13]
    assert len(plan['rl_arms']) == 6
    assert plan['task_weights']['proportional'] == {'mri_sequence': 2 / 3, 'mri_visit_correspondence': 1 / 3}
    assert plan['status'] == 'COMPILED_NOT_LAUNCH_AUTHORIZED'
    assert plan['optimizer_updates_executed'] == 0


@pytest.mark.parametrize('mutation', ['test', 'duplicate', 'source', 'visual'])
def test_rejects_unusable_training_measurements(mutation):
    data = measurements()
    if mutation == 'test': data['split'] = 'test'
    if mutation == 'duplicate': data['rows'][1]['record_sha256'] = data['rows'][0]['record_sha256']
    if mutation == 'source': data['rows'][-1]['source'] = 'OpenNeuro_ds007045'
    if mutation == 'visual': data['rows'][0]['visual_tokens'] = 301
    with pytest.raises(ValueError): compile_plan(data, bindings())


def test_whole_group_reservation_charges_actual_cost_and_never_partial_group():
    plan = compile_plan(measurements(), bindings())
    cap = plan['budget']['text_visual_token_cap']
    assert can_start_group(cap - 4 * 956, 700, plan)
    assert not can_start_group(cap - 4 * 956 + 1, 700, plan)
    assert charge_group(700, [10, 20, 30, 40], plan) == 4 * 700 + 100
    with pytest.raises(ValueError): charge_group(700, [10], plan)
    with pytest.raises(ValueError): charge_group(700, [10, 20, 30, 257], plan)


def test_no_blind_budget_without_measured_inputs_or_exact_start():
    data = measurements(); data['rows'] = []
    with pytest.raises(ValueError): compile_plan(data, bindings())
    bad = bindings(); bad['starting_adapter_sha256'] = 'unverified'
    with pytest.raises(ValueError): compile_plan(measurements(), bad)
