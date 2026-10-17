import importlib.util,json
from pathlib import Path
from types import SimpleNamespace
import pytest

def mod():
 p=Path(__file__).parents[1]/'src/mri_rl/c40_shape_validation_v2.py';s=importlib.util.spec_from_file_location('v2_validation',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def fixture_score():
 state={}
 def score(backend,record,index):
  first=[[-.0047,-5.37],[-.0047,-5.37]];oracle=[-.0041,-5.50];errors=[.13];prefix_scores=state.get('prefix_override',[[-.2],[-.2]])
  candidates={'a':{'target_token_logps':[-.2,-.0047]},'b':{'target_token_logps':[-.2,-5.37]}}
  parts={'a':{'target_ids':[1,2,3]},'b':{'target_ids':[1,2,4]}};P=115;prompt_list=list(range(115));label_first=[7,8];pixels=None
  state['candidates']=candidates;state['frame']=__import__('sys')._getframe()
  raise ValueError('Independent prefix oracle disagrees')
 return score,state

def test_targeted_correction_keeps_candidates_and_legacy_failure_and_clears_score_frame(monkeypatch):
 m=mod();score,state=fixture_score();control={'maximum_candidate_first_label_error':0.,'unchanged_tolerance':.002,'control_within_original_tolerance':True}
 monkeypatch.setattr(m,'_load_shape_control',lambda:lambda b,v:control)
 result=m.score_record_v2(SimpleNamespace(score_record=score),None,{'condition':'no_image','category':'mri_visit_correspondence','image':None},9)
 assert result['legacy_oracle_gate']=='FAIL' and result['validation_version']=='C40_no_image_equal_shape_oracle_v2'
 assert result['candidate_scores']==state['candidates'] and result['legacy_evidence_PRIVATE']['oracle']==[-.0041,-5.50]
 assert result['frozen_experiment_completed'] is False and result['threshold']==.002
 assert state['frame'].f_locals=={}

def test_other_tasks_images_faults_and_unequal_targets_are_not_corrected(monkeypatch):
 m=mod();score,state=fixture_score();api=SimpleNamespace(score_record=score)
 for record in ({'condition':'real_image','category':'mri_visit_correspondence','image':'x'},{'condition':'no_image','category':'mri_sequence','image':None}):
  with pytest.raises(ValueError):m.score_record_v2(api,None,record,9)
 def bad(*args):raise ValueError('unrelated error')
 with pytest.raises(ValueError,match='unrelated'):m.score_record_v2(SimpleNamespace(score_record=bad),None,{'condition':'no_image','category':'mri_visit_correspondence','image':None},9)
 def unequal(b,r,i):
  parts={'a':{'target_ids':[1]},'b':{'target_ids':[1,2]}};pixels=None
  raise ValueError('Independent prefix oracle disagrees')
 with pytest.raises(ValueError,match='equal candidate'):m.score_record_v2(SimpleNamespace(score_record=unequal),None,{'condition':'no_image','category':'mri_visit_correspondence','image':None},9)

def test_shape_control_failure_remains_not_verified(monkeypatch):
 m=mod();score,_=fixture_score();monkeypatch.setattr(m,'_load_shape_control',lambda:lambda b,v:{'maximum_candidate_first_label_error':.003,'unchanged_tolerance':.002,'control_within_original_tolerance':False})
 x=m.score_record_v2(SimpleNamespace(score_record=score),None,{'condition':'no_image','category':'mri_visit_correspondence','image':None},9)
 assert x['verdict']=='NOT_VERIFIED' and x['legacy_oracle_gate']=='FAIL'


def test_shape_correction_cannot_hide_disagreeing_common_prefix(monkeypatch):
 m=mod();score,state=fixture_score();state['prefix_override']=[[0.],[.01]]
 monkeypatch.setattr(m,'_load_shape_control',lambda:lambda b,v:{'maximum_candidate_first_label_error':0.,'unchanged_tolerance':.002,'control_within_original_tolerance':True})
 with pytest.raises(ValueError,match='common prefix'):m.score_record_v2(SimpleNamespace(score_record=score),None,{'condition':'no_image','category':'mri_visit_correspondence','image':None},9)
