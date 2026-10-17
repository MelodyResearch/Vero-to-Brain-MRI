"""EARLY analysis provenance guard tests; no actual/model outputs published."""
import copy,importlib.util
from pathlib import Path
import pytest


def mod():
 p=Path(__file__).parents[1]/'scripts/analyze_c40_early_visual_probe_v1.py';assert p.is_file(),'EARLY analyzer missing'
 s=importlib.util.spec_from_file_location('early_analysis',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def fixture():
 m=mod();scope={'kind':'C40_early_natural_boundary_likelihood_v1','checkpoint':'B28','weight_updates':0,'original_plan_sha256':m.ORIGINAL_PLAN_SHA,
  'early_plan_sha256':'PLAN','boundary_contract_ready':True,'boundary_contract_sha256':'CONTRACT','boundary_controller_source_sha256':'CONTROLLER','boundary_stage_id':'rl:P_raw:13'}
 lease={'schema':'natural-arm-boundary-handoff-lease-v1','contract_sha256':'CONTRACT','controller_source_sha256':'CONTROLLER','boundary_stage_id':'rl:P_raw:13','original_parent_exited':True,'no_model_children':True}
 status={'status':'ACTUAL_EARLY_PROBE_COMPLETED','checkpoint':'B28','completed':10,'output_sha256':'OUTPUT','lease_sha256':'LEASE','weight_updates':0,'queue_resume_safe':True,'critical_safety_condition':False,'native_children_released':True}
 return m,scope,lease,status


def test_exact_early_scope_and_actual_lease_status_are_bound():
 m,s,l,x=fixture();m.validate_early_bindings(s,l,x,'B28','PLAN','OUTPUT','LEASE')
 for changed in ('scope_kind','contract','lease','status','output','fake_all21'):
  ss,ll,xx=copy.deepcopy(s),copy.deepcopy(l),copy.deepcopy(x)
  if changed=='scope_kind':ss['kind']='C40_visual_likelihood_v1'
  elif changed=='contract':ll['contract_sha256']='OTHER'
  elif changed=='lease':xx['lease_sha256']='OTHER'
  elif changed=='status':xx['completed']=9
  elif changed=='output':xx['output_sha256']='OTHER'
  else:ss['original21training_ready']=True
  with pytest.raises(ValueError):m.validate_early_bindings(ss,ll,xx,'B28','PLAN','OUTPUT','LEASE')


def test_changed_external_artifact_hash_rejected(tmp_path):
 m=mod();p=tmp_path/'file';p.write_text('complete');h=m.sha(p);m.require_hash(p,h);p.write_text('changed')
 with pytest.raises(ValueError):m.require_hash(p,h)
