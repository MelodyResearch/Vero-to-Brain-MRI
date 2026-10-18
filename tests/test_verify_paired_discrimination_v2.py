import importlib.util
from pathlib import Path
import pytest
P=Path(__file__).resolve().parents[1]/'scripts/verify_paired_discrimination_v2.py'
def module():
 s=importlib.util.spec_from_file_location('pairedproof',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def plan():
 rows=[]
 for person,level in [('a',45),('a',91),('a',136),('b',45)]:
  for label in ('FLAIR','contrast-enhanced T1'):rows.append({'id':str(len(rows)),'ability':'mri_sequence','extra_info':{'internal_split':'train','patient_key':person,'level':level},'reward_model':{'ground_truth':label}})
 return {'maximum_updates':2,'schedule':[[0,1,2,3],[4,5,6,7]],'rows_PRIVATE':rows,'input_tokens':{r['id']:213 for r in rows},'target_token_cost':{r['id']:35 for r in rows}}
def test_two_step_plan_cost_and_pair_guards():
 m=module();assert m.validate_declared_plan(plan())==[1844,1844]
 p=plan();p['maximum_updates']=3
 with pytest.raises(AssertionError):m.validate_declared_plan(p)
 p=plan();p['rows_PRIVATE'][1]['reward_model']['ground_truth']='FLAIR'
 with pytest.raises(AssertionError):m.validate_declared_plan(p)
 p=plan();p['input_tokens']['0']=214
 with pytest.raises(AssertionError):m.validate_declared_plan(p)
def test_actual_terminal_must_authenticate_final_checkpoint():
 m=module();m.validate_native_terminal({'status':'ACTUAL_TWO_UPDATE_PAIRED_PILOT_COMPLETE_PENDING_ACCURACY','checkpoint_receipt_sha256':'abc'},'abc')
 for d in ({'status':'SCOPE_READY_NOT_RUN','checkpoint_receipt_sha256':'abc'},{'status':'ACTUAL_TWO_UPDATE_PAIRED_PILOT_COMPLETE_PENDING_ACCURACY','checkpoint_receipt_sha256':'wrong'}):
  with pytest.raises(AssertionError):m.validate_native_terminal(d,'abc')
def test_math_source_tampering_is_rejected_before_execution(tmp_path):
 m=module();p=tmp_path/'fake.py';p.write_text('raise RuntimeError("must not execute")')
 with pytest.raises(AssertionError):m.load_approved_math(p)
