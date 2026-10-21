"""New authorization identity for a resource-only revision; synthetic CPU."""
import copy
from datetime import datetime,timezone,timedelta
import pytest
from mri_rl import safe_micro_pilot_entry_v1 as old
from mri_rl import safe_micro_pilot_entry_v2 as e


def s(tmp_path):return {'schema':e.SCHEMA,'scope_kind':e.KIND,'action':e.ACTION,'implemented_by':e.IMPLEMENTED_BY,'source_sha256':{},'file_sha256':{},'input_sha256':'b'*64,'output':str(tmp_path/'pilot'),'resource_policy':e.MODEL_POLICY,'budgets':copy.deepcopy(e.BUDGETS),'parent':e.PARENT,'stage_order':[2,3,4],'evaluation':'FROZEN_DEV24_GREEDY_ONLY','automatic_retry':False,'automatic_progression_on_actual_PASS_only':True,'external_guardian_policy':e.GUARDIAN_POLICY}


def grants(scope):
 now=datetime.now(timezone.utc);h={'authorized':True,'action':e.ACTION,'scope_sha256':e.digest(scope),'included_stages':[2,3,4],'human_instruction_sha256':'c'*64,'approved_by':'actual_test_researcher','issued_at_utc':now.isoformat(),'expires_at_utc':(now+timedelta(hours=2)).isoformat()};review={'verdict':'PASS','scope_sha256':e.digest(scope),'source_sha256':{},'input_sha256':scope['input_sha256'],'reviewer_id':'independent_test','reviewer_kind':'isolated_codex'};return h,review


def test_old_grant_rejected_new_class_compatible_with_unchanged_native(tmp_path):
 scope=s(tmp_path);h,r=grants(scope);used=[];auth=e.authorize(scope,h,r,consumed=used.append)
 assert isinstance(auth.for_phase(2),old.Authorization) and used==['c'*64]
 assert e.BUDGETS==old.BUDGETS and e.MODEL_POLICY['load_available_gib_min']==9 and e.MODEL_POLICY['load_pressure_free_percent_min']==50
 h['action']=old.ACTION
 with pytest.raises(ValueError):e.authorize(scope,h,r,consumed=lambda _:pytest.fail('oldgrant consumed'))


def test_resource_revision_does_not_relax_swap_or_allow_watchdog_bypass(tmp_path):
 scope=s(tmp_path);scope['external_guardian_policy']={**e.GUARDIAN_POLICY,'swap_net_cap_GiB':100};h,r=grants(scope)
 with pytest.raises(ValueError):e.authorize(scope,h,r,consumed=lambda _:pytest.fail('mutated guardian consumed'))
 assert e.MODEL_POLICY['swap_cumulative_net_growth_cap_gib']==old.MODEL_POLICY['swap_cumulative_net_growth_cap_gib']==1.5


def test_guardian_stop_blocks_next_stage_even_with_successful_child_exit():
 result={'status':'PASS','release_verified':True};witness={'native_exit_code':0,'guardian_stop':True,'post_child_global_lock':{'no_open_holders':True},'child_peak_RSS_GiB':1.,'wall_seconds':1.}
 got=e.gate_witness_result(result,witness)
 assert got['status']!='PASS' and not got['release_verified'] and got['guardian_stop']
