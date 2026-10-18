import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest

def mod():
 p=Path(__file__).parents[1]/'scripts/run_c40_direct_auxiliary_v2.py';s=importlib.util.spec_from_file_location('direct_aux',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_finite_indices_preserve_nine_b28_and_cover_exact_u7():
 m=mod();assert m.indices_for('B28')==[9] and m.indices_for('U_raw_seed7')==list(range(10))
 with pytest.raises(ValueError):m.indices_for('A0')

def test_tagged_shape_result_preserves_legacy_fail_and_original_candidates():
 m=mod();scores={'different':{'original':1},'same':{'original':2}};result={'legacy_oracle_gate':'FAIL','verdict':'PASS_WITH_LIMITATIONS','candidate_scores':scores,'legacy_evidence_PRIVATE':{'oracle':[-1.,-2.],'errors':[.12]},'shape_matched_control':{'first_label_candidate_logps':[-1.,-2.1],'maximum_candidate_first_label_error':0.}}
 api=SimpleNamespace(LABELS={'mri_visit_correspondence':['different','same']},bound_candidate_functions=lambda x:(None,lambda s:{'mean_t1_minus_flair':1.}),task_comparisons=lambda task,x:x)
 record={'record_id':'private','category':'mri_visit_correspondence'}
 row=m.normalize_score(api,record,result)
 assert row['candidate_scores']==scores and row['legacy_oracle_gate']=='FAIL' and row['frozen_experiment_completed'] is False
 assert row['legacy_oracle_evidence_PRIVATE']==result['legacy_evidence_PRIVATE'] and row['prefix_oracle_first_logps']==[-1.,-2.1]
 result['verdict']='NOT_VERIFIED'
 with pytest.raises(ValueError):m.normalize_score(api,record,result)

def test_resume_never_reruns_or_accepts_duplicate_identity_or_record(tmp_path):
 import json
 m=mod();p=tmp_path/'x';scope={'checkpoint':'B28','run_identity':'R','indices':[9]};record={'record_id':'row'}
 row={'record_id':'row','record_index':9,'run_identity':'R','checkpoint':'B28','weight_updates':0}
 p.write_text(json.dumps(row)+'\n');assert m.completed(p,scope,[{}]*9+[record])==1
 p.write_text((json.dumps(row)+'\n')*2)
 with pytest.raises(ValueError):m.completed(p,scope,[{}]*9+[record])
 row['run_identity']='BAD';p.write_text(json.dumps(row)+'\n')
 with pytest.raises(ValueError):m.completed(p,scope,[{}]*9+[record])
