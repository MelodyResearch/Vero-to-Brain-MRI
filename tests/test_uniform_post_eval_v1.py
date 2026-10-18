import importlib.util,copy
from pathlib import Path
import pytest
W=Path(__file__).parents[1]
def mod():
 p=W/'scripts/run_uniform_post_eval_v1.py';assert p.exists(),'Uniform terminal selector missing';s=importlib.util.spec_from_file_location('uniform_post_test',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_terminal_proof_continues_actual_Adam56_and_absolute121():
 m=mod();proof={'schema':'uniform-continuation-focused-CPU-proof-v1','verdict':'PASS_WITH_LIMITATIONS','preserved_initial_parent_state':True,'parent_native_step':56,'parent_absolute_groups':121,'completed_groups':5,'new_optimizer_updates':4,'absolute_completed_groups':126,'absolute_optimizer_updates':60,'terminal_boundary_receipt_sha256':'r','final_adapter_sha256':'a'};terminal={'status':'UPDATE_BUDGET_REACHED','completed_groups':5,'optimizer_updates':4,'absolute_completed_groups':126,'absolute_optimizer_updates':60};receipt={k:v for k,v in terminal.items() if k!='status'}
 m.validate_child(proof,terminal,receipt,'r','a')
 for field,value in [('parent_native_step',0),('parent_absolute_groups',7),('absolute_completed_groups',5),('absolute_optimizer_updates',2),('completed_groups',17),('preserved_initial_parent_state',False)]:
  with pytest.raises(ValueError):m.validate_child({**proof,field:value},terminal,receipt,'r','a')

def test_scope_keeps_saved_baseline_and_only_changes_selected_child(tmp_path):
 m=mod()
 class Entry:
  @staticmethod
  def scope_identity(s):return 'identity'
 baseline={'checkpoint':'P_calibrated13_adaptive','adapter':{'sha256':'old'},'native_sources':{'x':'same'},'evaluation_contract_sha256':'8c52','output_directory':'actual_completed_baseline'}
 child={'adapter':{'sha256':'new'}};evidence={'scope_sha256':'022c','outputs_verified':True}
 s=m.select_child_scope(Entry,baseline,child,tmp_path,evidence)
 assert s['kind']=='uniform_category_post_read_only_evaluation_v1' and s['checkpoint']=='uniform_category_continuation'
 assert s['adapter']['sha256']=='new' and s['baseline_evaluation']==evidence and s['candidate_forwards']==75 and s['new_greedy_generations']==41
 assert baseline['adapter']['sha256']=='old' and baseline['output_directory']=='actual_completed_baseline'


def test_early_nonterminal_checkpoint_is_not_chosen(tmp_path):
 m=mod();proof={'schema':'uniform-continuation-focused-CPU-proof-v1','verdict':'PASS_WITH_LIMITATIONS','preserved_initial_parent_state':True,'parent_native_step':56,'parent_absolute_groups':121,'completed_groups':5,'new_optimizer_updates':2,'absolute_completed_groups':126,'absolute_optimizer_updates':58,'terminal_boundary_receipt_sha256':'r','final_adapter_sha256':'a'}
 terminal={'status':'UPDATE_BUDGET_REACHED','completed_groups':5,'optimizer_updates':2,'absolute_completed_groups':126,'absolute_optimizer_updates':58};receipt={k:v for k,v in terminal.items() if k!='status'}
 with pytest.raises(ValueError):m.validate_child(proof,terminal,receipt,'r','a')
