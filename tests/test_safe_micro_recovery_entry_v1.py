import copy
import pytest
from mri_rl import safe_micro_recovery_entry_v1 as e
from mri_rl import safe_micro_pilot_entry_v3 as old
from test_safe_micro_pilot_entry_v2 import s,grants


def scope(tmp_path):
 result=s(tmp_path)
 result.update(schema=e.SCHEMA,scope_kind=e.KIND,action=e.ACTION,resource_policy=e.MODEL_POLICY,budgets=copy.deepcopy(e.BUDGETS),external_guardian_policy=e.GUARDIAN_POLICY,recovery_contract=e.RECOVERY,adoption=e.ADOPTION,spent_before_recovery=e.SPENT)
 return result


def test_exact_new_human_adoption_is_required(tmp_path):
 x=scope(tmp_path);h,r=grants(x);h['action']=e.ACTION
 with pytest.raises(ValueError):e.authorize(x,h,r,consumed=lambda _:pytest.fail('adoption not granted'))
 h['adopt_A_from_incomplete_transaction']=True
 auth=e.authorize(x,h,r,consumed=lambda _:None)
 assert auth.scope['adoption']['old_journal_commit_written'] is False


def test_no_regeneration_no_A_update_and_original_cost_not_refunded():
 assert e.BUDGETS['2']['generated_responses']==0 and e.BUDGETS['2']['optimizer_updates']==0
 assert e.BUDGETS['3']['generated_responses']==0 and e.BUDGETS['3']['scorer_forwards']==16
 assert e.BUDGETS['3']['optimizer_updates']==1
 assert e.BUDGETS['4']['generated_responses']==72
 assert e.SPENT['optimizer_updates']==1
 assert e.RECOVERY['max_model_load_attempts']==5 and e.RECOVERY['historical_loads_spent']==2
 assert e.SPENT['generated_response_tokens']+sum(x['generated_response_tokens'] for x in e.BUDGETS.values())<=23040
 assert e.SPENT['teacherforced_input_positions']+sum(x['teacherforced_input_positions'] for x in e.BUDGETS.values())<=29880
 assert e.SPENT['backward_calls']+sum(x['backward_calls'] for x in e.BUDGETS.values())<=32


def test_old_compound_scope_is_rejected(tmp_path):
 x=scope(tmp_path);x['action']=old.ACTION;h,r=grants(x);h['adopt_A_from_incomplete_transaction']=True
 with pytest.raises(ValueError):e.authorize(x,h,r,consumed=lambda _:pytest.fail('old grant reused'))
