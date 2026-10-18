import importlib.util
from pathlib import Path
import pytest

def mod():
 p=Path(__file__).parents[1]/'scripts/run_paired_discrimination_pilot_v2.py';s=importlib.util.spec_from_file_location('paired_entry_v2',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_evaluation_contract_freezes_all40_and25_sequence_candidates_without_outcomes():
 m=mod();rows=[];projection=[];images={}
 for person in range(4):
  for level in (45,91,136):
   for label in ('FLAIR','contrast-enhanced T1'):
    key=f'p{person}-{level}-{label}';row={'id':key,'ability':'mri_sequence','images':[key],'reward_model':{'ground_truth':label},'extra_info':{'patient_key':str(person),'level':level,'question':'Q'}};rows.append(row);images[key]='hash';projection.append({'record_id':key,'category':'mri_sequence','image':key,'prompt':[{'role':'system','content':'S'},{'role':'user','content':'Q'}]})
 for i in range(16):
  key=f'd{i}';rows.append({'id':key,'ability':'mri_visit_correspondence','images':[key],'reward_model':{'ground_truth':'same' if i%2 else 'different'},'extra_info':{'patient_key':str(i//2),'statistical_cluster_key':str(i//4),'question':'D'}});images[key]='hash';projection.append({'record_id':key,'category':'mri_visit_correspondence','image':key,'prompt':[{'role':'system','content':'S'},{'role':'user','content':'D'}]})
 c=m.freeze_evaluation(rows,projection,images,'S')
 assert len(c['greedy_records_PRIVATE'])==40 and len(c['candidate_records_PRIVATE'])==25 and len(c['sequence_pairs_PRIVATE'])==12
 assert c['candidate_records_PRIVATE'][-1]['condition']=='no_image' and c['candidate_records_PRIVATE'][-1]['visible_target_PRIVATE'] is None
 assert c['greedy_max_new_tokens']==256 and c['maximum_candidate_forwards']==75 and c['blank_image_controls']=='NOT_RUN'
 projection.pop()
 with pytest.raises(ValueError):m.freeze_evaluation(rows,projection,images,'S')

def test_feature_completion_requires_actual_release_and_exact48_evidence(tmp_path):
 import json,hashlib
 m=mod();plan={'train_rows':24,'dev_rows':24,'train_people':4,'dev_people':4,'input_sha256':m.FROZEN_INPUT_SHA,'adapter':{'sha256':'B'}};p=tmp_path/'plan';p.write_text(json.dumps(plan));h=hashlib.sha256(p.read_bytes()).hexdigest()
 receipt={'status':'ACTUAL48_VISION_FORWARDS_READONLY_COMPLETE_AND_RELEASED','plan_sha256':h,'adapter_sha256':'B','run_identity':m.canon(plan),'record_sha256':['x']*48,'adapter_tensors_compared_after':504,'optimizer_step':0,'weight_updates':0,'release':{'native_release_verified':True,'model_is_none':True,'active_MLX_bytes':0,'active_limit_bytes':128*1024**2}}
 r=tmp_path/'receipt';r.write_text(json.dumps(receipt));rh=hashlib.sha256(r.read_bytes()).hexdigest()
 assert m.feature_binding(p,h,r,rh)['ready'] is True
 receipt['release']['native_release_verified']=False;r.write_text(json.dumps(receipt))
 with pytest.raises(ValueError):m.feature_binding(p,h,r,hashlib.sha256(r.read_bytes()).hexdigest())


def test_resource_path_exists_before_first_actual_shared_gate_sample(tmp_path):
 import sys
 m=mod();primary=Path(__file__).parents[1].parents[1];sys.path.insert(0,str(primary/'src'))
 from mri_rl.resource_supervisor import ResourceSample
 gate=m.resource_gate(tmp_path/'missing'/'training_output')
 gate.sampler=lambda:ResourceSample(monotonic_seconds=1,available_gib=12,pressure_free_percent=60,power_source='AC',other_heavy_processes=0)
 gate.before_unit('fixture_before_model',loading=True)
 assert gate.path.is_file()


def test_resume_progress_is_review_bound_but_not_a_new_scientific_identity():
 m=mod();base={'adapter':'same','source':'same','starting_journal':None,'resume':False,'ready':True,'starting_outputs_sha256':{'g':None}}
 changed={**base,'resume':True,'starting_journal':{'cursor':2},'starting_outputs_sha256':{'g':'actualprefixSHA'}}
 assert m.scope_identity(base)==m.scope_identity(changed)
 assert m.scope_identity(base)!=m.scope_identity({**changed,'adapter':'different'})
