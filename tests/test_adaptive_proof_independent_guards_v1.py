"""Independent synthetic terminal/actual-exit provenance counterexamples."""
import importlib.util,json,sys
from pathlib import Path
import pytest
W=Path(__file__).parents[1]
def mod(name):
 p=W/'scripts'/name;s=importlib.util.spec_from_file_location('independent_'+name.replace('.','_'),p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_zero_debt_worst_case_envelope_cannot_finish_after_one_skip(tmp_path):
 m=mod('verify_adaptive_continuation_v1.py');r={'completed_groups':1,'optimizer_updates':0,'absolute_completed_groups':115,'absolute_optimizer_updates':52};t={**r,'status':'TOKEN_BUDGET_EXHAUSTED','boundary':str(tmp_path/'boundary_0001')}
 with pytest.raises(ValueError):m.terminal_selection(t,r,tmp_path/'boundary_0001')

def test_postselector_cannot_call_three_of_sixteen_groups_schedule_complete():
 m=mod('run_adaptive_post_eval_v1.py');r={'completed_groups':3,'optimizer_updates':1,'absolute_completed_groups':117,'absolute_optimizer_updates':53};p={'schema':'adaptive-continuation-focused-CPU-proof-v1','verdict':'PASS_WITH_LIMITATIONS','preserved_initial_parent_state':True,'parent_native_step':52,'completed_groups':3,'new_optimizer_updates':1,'absolute_completed_groups':117,'absolute_optimizer_updates':53,'terminal_boundary_receipt_sha256':'receipt','final_adapter_sha256':'adapter'}
 with pytest.raises(ValueError):m.validate_child(p,{**r,'status':'SCHEDULE_COMPLETE'},r,'receipt','adapter')

@pytest.mark.parametrize('foreign_scope,missing_review',[(True,False),(False,True)])
def test_actual_exit_must_bind_scope_and_mandatory_review_before_state_reads(tmp_path,monkeypatch,foreign_scope,missing_review):
 m=mod('verify_adaptive_continuation_v1.py');monkeypatch.setattr(m,'W',tmp_path);monkeypatch.setattr(sys,'meta_path',list(sys.meta_path));(tmp_path/'local_data').mkdir()
 s={'kind':'adaptive_accuracy_gate_v1'};scope=tmp_path/'scope';scope.write_text(json.dumps(s));review=tmp_path/'review';review.write_text(json.dumps({**s,'verdict':'PASS_WITH_LIMITATIONS'}));log=tmp_path/'log';log.write_text('actualclosedlog');terminal=tmp_path/'terminal';terminal.write_text(json.dumps({'scope_sha256':m.objsha(s)}));witness=tmp_path/'exit';e={'exit_code':0,'raw_log_sha256':m.sha(log),'scope_file_sha256':'FOREIGN' if foreign_scope else m.sha(scope)}
 if not missing_review:e['review_sha256']=m.sha(review)
 witness.write_text(json.dumps(e));real_load=m.load
 def guarded_load(path):
  if Path(path).name=='spec_NOT_APPROVED_PRIVATE.json':raise RuntimeError('UNBOUND_EXIT_REACHED_STATE_READ')
  return real_load(path)
 monkeypatch.setattr(m,'load',guarded_load)
 monkeypatch.setattr(sys,'argv',['verify','--primary-root',str(tmp_path),'--scope',str(scope),'--review',str(review),'--exit-witness',str(witness),'--native-log',str(log),'--native-complete',str(terminal),'--output',str(tmp_path/'local_data/proof'),'--expected-scope-sha256',m.sha(scope),'--expected-review-sha256',m.sha(review),'--expected-exit-witness-sha256',m.sha(witness),'--reviewer-id','independent fixture'])
 with pytest.raises((ValueError,KeyError)):m.main()
