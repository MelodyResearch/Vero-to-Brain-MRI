"""CPU counterexamples for silent, unreachable rollout length rewards."""
import importlib.util
import pytest

def api():
    spec = importlib.util.find_spec('mri_rl.root_cause_rollout_contract_v1')
    assert spec is not None, 'A versioned rollout/reward reachability validator is required'
    from mri_rl.root_cause_rollout_contract_v1 import validate_length_contract
    return validate_length_contract

def test_historical_256_enabled_penalty_is_rejected():
    with pytest.raises(ValueError, match='unreachable'):
        api()(rollout_cap=256, penalty_max_tokens=18432, buffer_tokens=2048, mode='enabled')

def test_exact_onset_is_zero_and_first_negative_token_must_be_reachable():
    with pytest.raises(ValueError, match='unreachable'):
        api()(rollout_cap=24, penalty_max_tokens=32, buffer_tokens=8, mode='enabled')
    x = api()(rollout_cap=25, penalty_max_tokens=32, buffer_tokens=8, mode='enabled')
    assert x['first_negative_token'] == 25 and x['reachable'] is True
    assert x['training_authorized'] is False

def test_disabled_penalty_must_be_explicit_not_silently_mislabeled_enabled():
    x = api()(rollout_cap=256, penalty_max_tokens=18432, buffer_tokens=2048, mode='disabled_explicit')
    assert x['reachable'] is False and x['effective_penalty_enabled'] is False
    with pytest.raises(ValueError, match='mode'):
        api()(rollout_cap=256, penalty_max_tokens=18432, buffer_tokens=2048, mode='auto')

@pytest.mark.parametrize('cap,max_tokens,buffer', [(True, 32, 8), (0, 32, 8), (32, 32, 0), (32, 32, 33)])
def test_invalid_budget_inputs_fail_closed(cap, max_tokens, buffer):
    with pytest.raises(ValueError):
        api()(rollout_cap=cap, penalty_max_tokens=max_tokens, buffer_tokens=buffer, mode='enabled')
