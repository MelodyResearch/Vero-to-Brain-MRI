import importlib.util
from pathlib import Path
import pytest

def module():
 p=Path(__file__).resolve().parents[1]/'scripts/run_adaptive_parent_eval_v1.py';s=importlib.util.spec_from_file_location('parent_eval',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_parent_identity_and_counters_reject_wrong_checkpoint():
 m=module();r={'completed_groups':114,'optimizer_updates':52};p={'verdict':'PASS_WITH_LIMITATIONS','stage_id':'rl:P_calibrated:13','full_arm_completed':True,'completed_groups':114,'optimizer_updates':52,'final_adapter_sha256':m.PARENT_SHA,'terminal_boundary_receipt_sha256':m.RECEIPT_SHA};m.validate_parent(r,p,m.PARENT_SHA)
 with pytest.raises(ValueError):m.validate_parent(r,p,'wrong')
 with pytest.raises(ValueError):m.validate_parent({**r,'optimizer_updates':51},p,m.PARENT_SHA)
 with pytest.raises(ValueError):m.validate_parent(r,{**p,'full_arm_completed':False},m.PARENT_SHA)

def test_selector_preserves_inputs_and_requires_full_new_parent_evaluation(tmp_path):
 m=module()
 class Entry:
  @staticmethod
  def scope_identity(s):return 'unique-new-parent-identity'
 old={'entry_sha256':m.ENTRY_SHA,'evaluation_contract_sha256':m.EVAL_SHA,'adapter':{'sha256':'old'},'checkpoint':'post2','maximum_new_updates':2,'scientific_inputs':{'pixels':'unchanged'}};parent={'adapter':{'sha256':m.PARENT_SHA}}
 s=m.select_scope(Entry,old,parent,tmp_path);assert s['adapter']==parent['adapter'] and s['checkpoint']=='P_calibrated13';assert s['candidate_forwards']==75 and s['new_greedy_generations']==41 and s['maximum_new_updates']==0;assert s['scientific_inputs']==old['scientific_inputs'] and old['checkpoint']=='post2';assert s['adapter_selector_sha256']==m.sha(m.__file__)
