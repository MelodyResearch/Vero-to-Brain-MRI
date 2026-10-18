import importlib.util
from pathlib import Path
import pytest

def module():
 p=Path(__file__).resolve().parents[1]/'scripts/run_adaptive_post_eval_v1.py';s=importlib.util.spec_from_file_location('post_eval',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_only_exact_terminal_and_preserved_parent_are_eligible():
 m=module();p={'schema':'adaptive-continuation-focused-CPU-proof-v1','verdict':'PASS_WITH_LIMITATIONS','preserved_initial_parent_state':True,'parent_native_step':52,'completed_groups':7,'new_optimizer_updates':4,'absolute_completed_groups':121,'absolute_optimizer_updates':56,'terminal_boundary_receipt_sha256':'receipt','final_adapter_sha256':'adapter'};r={'completed_groups':7,'optimizer_updates':4,'absolute_completed_groups':121,'absolute_optimizer_updates':56};t={**r,'status':'UPDATE_BUDGET_REACHED'};m.validate_child(p,t,r,'receipt','adapter')
 with pytest.raises(ValueError):m.validate_child(p,t,r,'other-best-dev','adapter')
 with pytest.raises(ValueError):m.validate_child({**p,'preserved_initial_parent_state':False},t,r,'receipt','adapter')
 with pytest.raises(ValueError):m.validate_child(p,{**t,'absolute_optimizer_updates':4},r,'receipt','adapter')

def test_child_eval_keeps_all_inputs_and_uses_all_new_predictions(tmp_path):
 m=module()
 class Entry:
  @staticmethod
  def scope_identity(s):return 'child-identity'
 old={'evaluation_contract_sha256':'frozen8c52','checkpoint':'P_calibrated13','native_sources':{'actual':'unchanged'},'parent_checkpoint':{'optimizer_updates':52}}
 child={'adapter':{'sha256':'updated'},'absolute_optimizer_updates':56};s=m.select_child_scope(Entry,old,child,tmp_path);assert s['checkpoint']=='P_calibrated13_adaptive' and s['adapter']==child['adapter'] and s['maximum_new_updates']==0;assert s['candidate_forwards']==75 and s['new_greedy_generations']==41 and s['native_sources']==old['native_sources'] and s['evaluation_contract_sha256']==old['evaluation_contract_sha256'];assert old['checkpoint']=='P_calibrated13'
