"""Synthetic admission/sequence tests; no model, image or host sampling."""
import copy,json
from datetime import datetime,timezone,timedelta
from pathlib import Path
import pytest
from mri_rl import safe_micro_pilot_entry_v1 as e


def scope(tmp_path):
 return {'schema':e.SCHEMA,'scope_kind':e.KIND,'action':e.ACTION,'implemented_by':e.IMPLEMENTED_BY,'source_sha256':{},'file_sha256':{},'input_sha256':'b'*64,'output':str(tmp_path/'pilot'),'resource_policy':e.MODEL_POLICY,'budgets':copy.deepcopy(e.BUDGETS),'parent':e.PARENT,'stage_order':[2,3,4],'evaluation':'FROZEN_DEV24_GREEDY_ONLY','automatic_retry':False,'automatic_progression_on_actual_PASS_only':True}


def grants(s):
 now=datetime.now(timezone.utc)
 h={'authorized':True,'action':e.ACTION,'scope_sha256':e.digest(s),'included_stages':[2,3,4],'human_instruction_sha256':'c'*64,'approved_by':'actual_test_researcher','issued_at_utc':now.isoformat(),'expires_at_utc':(now+timedelta(hours=2)).isoformat()}
 rv={'verdict':'PASS_WITH_LIMITATIONS','scope_sha256':e.digest(s),'source_sha256':{},'input_sha256':s['input_sha256'],'reviewer_id':'isolated_test_reviewer','reviewer_kind':'isolated_codex'}
 return h,rv


def test_exact_combined_scope_single_use_and_class_issued_phase(tmp_path):
 s=scope(tmp_path);h,rv=grants(s);used=[]
 token=e.authorize(s,h,rv,consumed=used.append)
 assert used==['c'*64]
 phase=token.for_phase(3)
 assert phase.scope['execution_stage']=='MICRO_TRAIN' and phase.scope['resource_stage']=='B_MICRO'
 assert phase.scope['budget']['optimizer_updates_per_arm']==1
 phase.check('boundary')
 with pytest.raises(ValueError):token.for_phase(1)
 with pytest.raises(ValueError):e.Authorization(object(),s,datetime.now(timezone.utc),None)


@pytest.mark.parametrize('what',['stage1_grant','missing_eval','policy_lowered','budget_increased','review_self','review_inputs','expired'])
def test_no_old_permission_or_unreviewed_budget(tmp_path,what):
 s=scope(tmp_path);h,rv=grants(s)
 if what=='stage1_grant':h['action']='safe_micro_v1_stage1_cache_only_then_stop'
 if what=='missing_eval':h['included_stages']=[2,3]
 if what=='policy_lowered':s['resource_policy']={**e.MODEL_POLICY,'load_available_gib_min':9.};h,rv=grants(s)
 if what=='budget_increased':s['budgets']['3']['optimizer_updates_per_arm']=8;h,rv=grants(s)
 if what=='review_self':rv['reviewer_id']=e.IMPLEMENTED_BY
 if what=='review_inputs':rv['input_sha256']='d'*64
 if what=='expired':h['expires_at_utc']=(datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat()
 with pytest.raises(ValueError):e.authorize(s,h,rv,consumed=lambda _:pytest.fail('consumed invalid permission'))


def test_exact_phase_action_and_input_source_pins_cannot_mutate(tmp_path):
 p=tmp_path/'bound.py';p.write_text('value=1\n');s=scope(tmp_path);s['source_sha256']={str(p):e.H(p)};h,rv=grants(s);rv['source_sha256']=s['source_sha256']
 t=e.authorize(s,h,rv,consumed=lambda _:None).for_phase(2);t.check('before_load')
 p.write_text('value=2\n')
 with pytest.raises(ValueError,match='changed'):t.check('before_load')


def test_budget_arithmetic_greedy_only_and_no_cache_operations():
 b=e.BUDGETS
 assert b['2']['generated_response_tokens']==512 and b['2']['teacherforced_input_positions']==2808
 assert b['3']['generated_response_tokens']==4096 and b['3']['backward_calls']==32
 assert b['4']['greedy_calls']==72 and b['4']['candidate_calls']==0
 assert sum(x['generated_response_tokens'] for x in b.values())==23040
 assert sum(x['teacherforced_input_positions'] for x in b.values())==29880
 assert sum(x['optimizer_updates'] for x in b.values())==2
 assert all(x['cache_quantizations']==0 for x in b.values())


def test_parent_chain_stops_at_partial_and_failure(tmp_path):
 s=scope(tmp_path);h,rv=grants(s);t=e.authorize(s,h,rv,consumed=lambda _:None);calls=[]
 def step(stage):
  calls.append(stage);return {'status':'PARTIAL' if stage==2 else 'PASS','release_verified':True,'actual_exit_code':0}
 result=e.run_order(t,step)
 assert calls==[2] and result['status']=='STOPPED' and result['automatic_retry'] is False
 calls.clear()
 def failing(stage):
  calls.append(stage);return {'status':'PASS','release_verified':True,'actual_exit_code':1 if stage==3 else 0}
 e.run_order(t,failing);assert calls==[2,3]


def test_zero_update_micro_is_valid_terminal_and_goes_to_all24_eval(tmp_path):
 s=scope(tmp_path);h,rv=grants(s);t=e.authorize(s,h,rv,consumed=lambda _:None);calls=[]
 def step(stage):calls.append(stage);return {'status':'PASS','release_verified':True,'actual_exit_code':0,'actual_updates':0}
 result=e.run_order(t,step)
 assert calls==[2,3,4] and result['status']=='COMPLETE'


def test_missing_release_never_advances(tmp_path):
 s=scope(tmp_path);h,rv=grants(s);t=e.authorize(s,h,rv,consumed=lambda _:None);calls=[]
 def step(stage):calls.append(stage);return {'status':'PASS','release_verified':False,'actual_exit_code':0}
 assert e.run_order(t,step)['status']=='STOPPED' and calls==[2]


def test_predecessor_uses_pure_subscope_no_expired_synthetic_permission(tmp_path):
 from mri_rl.safe_micro_witness_v1 import digest as fdigest
 s=scope(tmp_path);base=Path(s['output']);terminal=base/'stage2_PRIVATE';terminal.mkdir(parents=True);w=base/'stage2_exit_witness_PRIVATE';w.mkdir()
 for name in ('stdout','stderr'):(w/(name+'_PRIVATE.log')).write_text('')
 ps=e.phase_scope(s,2)
 (terminal/'terminal_PRIVATE.json').write_text(json.dumps({'scope_sha256':e.digest(ps),'status':'PASS','release_verified':True}))
 (w/'actual_exit_PRIVATE.json').write_text(json.dumps({'scope_file_sha256':'f'*64,'native_exit_code':0,'child_reaped':True,'post_child_global_lock':{'no_open_holders':True},'stdout_sha256':fdigest(w/'stdout_PRIVATE.log'),'stderr_sha256':fdigest(w/'stderr_PRIVATE.log')}))
 e._prior(s,3,'f'*64)


def test_native_F32_runtime_sources_are_pinned_not_just_package_versions(tmp_path):
 p=tmp_path/'ar.py';p.write_text('verified_F32_path=1\n');runtime={'runtime_source_sha256':{str(p):e.H(p)}}
 pins=e.bind_runtime_sources({},runtime)
 assert pins==runtime['runtime_source_sha256']
 p.write_text('verified_F32_path=2\n')
 with pytest.raises(ValueError,match='runtime'):e.bind_runtime_sources({},runtime)
 with pytest.raises(ValueError):e.bind_runtime_sources({}, {'runtime_source_sha256':{}})
