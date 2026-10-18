import importlib.util,copy,json
from pathlib import Path
import pytest
W=Path(__file__).parents[1]
def mod():
 p=W/'scripts/analyze_uniform_balance_outcomes_v1.py';assert p.exists(),'Adaptive saved-output analyzer missing';s=importlib.util.spec_from_file_location('uniform_analysis_test',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_labels_are_actual_parent_and_child_and_common_prior_is_not_discrimination():
 m=mod();v=m.load_helpers(W.parent/'vero-mri-v2-controls');s=importlib.util.spec_from_file_location('old_fixture',W.parent/'vero-mri-v2-controls/tests/test_analyze_paired_pilot_v2.py');f=importlib.util.module_from_spec(s);s.loader.exec_module(f);rows,base=f.fixture();post=copy.deepcopy(base)
 for r in post.values():r['comparisons']={'mean_t1_minus_flair':1.,'raw_t1_minus_flair':1.};r['prefix_oracle_first_logps']=[-5.,-4.]
 result=m.visual_controls(v,rows,base,post)
 assert set(result['checkpoints'])=={'adaptive_Adam56','uniform_category_continuation'}
 assert result['uniform_minus_parent']['mean']['delta_null_margin']==2.
 assert result['uniform_minus_parent']['mean']['delta_patient_mean_paired_separation']==0.
 assert 'B28' not in json.dumps(result) and 'post2' not in json.dumps(result)

def test_checkpoint_proof_cannot_relabel_reset_as_continuation():
 m=mod();proof={'schema':'uniform-continuation-focused-CPU-proof-v1','verdict':'PASS_WITH_LIMITATIONS','preserved_initial_parent_state':True,'parent_native_step':56,'parent_absolute_groups':121,'completed_groups':3,'new_optimizer_updates':2,'absolute_optimizer_updates':58,'absolute_completed_groups':124,'terminal_boundary_receipt_sha256':'terminal','final_adapter_sha256':'adapter'}
 child={'completed_groups':3,'optimizer_updates':2,'absolute_completed_groups':124,'absolute_optimizer_updates':58,'files_sha256':{'adapter.safetensors':'adapter'}}
 assert m.validate_training_proof(proof,child,'terminal')==(3,2)
 for key,value in [('parent_native_step',0),('preserved_initial_parent_state',False),('absolute_optimizer_updates',2),('new_optimizer_updates',5),('absolute_completed_groups',3)]:
  with pytest.raises(ValueError):m.validate_training_proof({**proof,key:value},child,'terminal')

def test_resource_times_use_monotonic_boundaries_not_file_times():
 m=mod();events=[{'unit_id':unit,'action':'CONTINUE','supervisor_level':'NORMAL','sample':{'monotonic_seconds':time,'available_gib':7.,'pressure_free_percent':40.,'swap_used_gib':3.,'mlx_peak_gib':5.}} for unit,time in [('rollout_group_114',10.),('post_generation',15.),('rollout_group_115',20.),('optimizer_update',25.)]]
 r=m.resource_summary(events);assert r['observed_monotonic_span_seconds']==15. and r['complete_intergroup_intervals_seconds']==[10.] and r['final_group_observed_lower_bound_seconds']==5.
 with pytest.raises(ValueError):m.resource_summary([events[1],events[0]])

def test_strict_greedy_requires_full_boxed_format_not_partial_credit():
 m=mod();labels=('FLAIR','contrast-enhanced T1')
 assert m.strict_prediction('x',labels,lambda x:('FLAIR',1.))==0
 assert m.strict_prediction('x',labels,lambda x:('FLAIR',.5))==-1
 assert m.strict_prediction('x',labels,lambda x:('other',1.))==-1

def test_external_artifact_tamper_rejected(tmp_path):
 m=mod();p=tmp_path/'evidence';p.write_text('{}');item={'path':str(p),'sha256':m.sha(p)};assert m.authenticated(item)=={}
 p.write_text('{"changed":true}')
 with pytest.raises(ValueError):m.authenticated(item)


def test_repeated_hold_samples_do_not_invent_attempted_groups():
 m=mod();events=[]
 for name,t,action in [('rollout_group_114',10.,'HOLD'),('rollout_group_114',25.,'HOLD'),('rollout_group_114',40.,'CONTINUE'),('rollout_group_115',50.,'CONTINUE'),('rollout_group_115',51.,'CONTINUE')]:
  events.append({'unit_id':name,'action':action,'sample':{'monotonic_seconds':t}})
 r=m.resource_summary(events)
 assert r['complete_intergroup_intervals_seconds']==[10.]
 assert r['final_group_observed_lower_bound_seconds']==1.
