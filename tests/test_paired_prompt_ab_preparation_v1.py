import pytest
from mri_rl.paired_prompt_ab_preparation_v1 import proposal,refuse_native_execution


def test_equal_finite_budget_and_exposure():
    d=proposal()
    assert d['A']['sequence_groups']==[['F']*4,['T']*4]
    assert d['B']['sequence_groups']==[['F','F','T','T']]*2
    assert d['per_arm']['generated_response_tokens_cap']==4096
    assert d['per_arm']['charged_prompt_plus_returned_cap']==9040
    assert d['per_arm']['attempted_groups']==d['per_arm']['maximum_updates']==4
    assert d['per_arm']['minimum_updates']==0
    assert d['native_per_group_response_loss_weight']==.25
    assert d['offline_fixed_eight_block_response_loss_weight']==.125
    assert d['native_execution_authorized'] is False
    assert d['reward_or_auxiliary_change'] is False
    assert d['all_four_attempts_required_for_complete_exposure'] is True


def test_no_permission_boolean_can_turn_preparation_into_native_execution():
    for supplied in ({},{'approved':True,'training_authorized':True}):
        with pytest.raises(ValueError,match='PREPARATION_ONLY'):
            refuse_native_execution(supplied)
