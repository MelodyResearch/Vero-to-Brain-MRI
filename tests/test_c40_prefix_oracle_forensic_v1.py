import importlib.util
from pathlib import Path
import pytest

def mod():
 p=Path(__file__).parents[1]/'scripts/run_c40_prefix_oracle_forensic_v1.py';s=importlib.util.spec_from_file_location('forensic',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_exact_oracle_failure_captured_from_only_matching_score_frame():
 m=mod()
 def score(backend,record,index):
  first=[[-1.0,-2.0],[-1.01,-2.0]];oracle=[-1.,-2.];errors=[.01];prefix_scores=[[-.5],[-.5]];candidates={'a':{'target_token_logps':[-.5,-1.]}};parts={'a':{'prefix_ids':[1]}};P=4;prompt_list=[1,2,3,4];label_first=[7,8]
  raise ValueError('Independent prefix oracle disagrees')
 x=m.capture_score(score,None,{},9)
 assert x['oracle_gate']=='FAIL' and x['numerics_PRIVATE']['errors']==[.01] and x['numerics_PRIVATE']['first'][1][0]==-1.01

def test_unrelated_errors_are_not_swallowed_and_success_is_not_failed():
 m=mod()
 def score(*a):raise ValueError('other')
 with pytest.raises(ValueError,match='other'):m.capture_score(score,None,{},9)
 assert m.capture_score(lambda *a:{'ok':True},None,{},9)['oracle_gate']=='PASS_REPRODUCTION_NOT_OBSERVED'
