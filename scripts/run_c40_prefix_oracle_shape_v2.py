"""R2 original B28/index9 causal-shape control; unchanged failed .002 gate.
No completed-probe output is written. Expected oracle failure is captured, not waived.
"""
from __future__ import annotations
import argparse,fcntl,gc,hashlib,importlib.util,json,os,traceback
from pathlib import Path

EARLY_SHA='8cafacbf3a9086160c8e257673c6f933070df208bffc5c32dec38bc0e1e82e06'
KEYS=('first','oracle','errors','prefix_scores','candidates','parts','P','visual','grid','label_first','prompt_list')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(ok,msg):
 if not ok:raise ValueError(msg)
def load(p):return json.loads(Path(p).read_text())
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(x,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 fd=os.open(p.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
def padded_oracle_layout(prompt_ids,prefix_ids,target_length):
 full_length=len(prompt_ids)+target_length-1
 common=list(prompt_ids)+list(prefix_ids)
 require(full_length>=len(common),'Candidate shape shorter than common causal prefix')
 return common+[0]*(full_length-len(common)),len(common)-1

def padded_control(backend,values):
 mx=backend.mx;labels=list(values['parts']);part=values['parts'][labels[0]]
 require(all(len(v['target_ids'])==len(part['target_ids']) for v in values['parts'].values()),'Candidate input shapes differ')
 ids,position=padded_oracle_layout(values['prompt_list'],part['prefix_ids'],len(part['target_ids']))
 require(len(ids)==129 and position==125 and values['pixels'] is None,'Frozen no-image129/position125 control changed')
 backend.before_unit('forensic_shape_matched_oracle')
 inputs=mx.array([ids],dtype=values['prompt_ids'].dtype)
 logits=backend.model(inputs,values['pixels'],mx.ones_like(inputs),**values['kwargs']).logits
 require(int(logits.shape[1])==len(ids),'Shape-control causal logits length mismatch')
 lp=backend.nn.log_softmax(logits[:,position,:].astype(mx.float32),axis=-1)[0,mx.array(values['label_first'])]
 mx.eval(lp);oracle=lp.tolist()
 errors=[abs(a-b) for row in values['first'] for a,b in zip(row,oracle)]
 result={'input_length':len(ids),'score_position':position,'common_prefix_length':position+1,'future_padding_token':0,'future_padding_count':len(ids)-position-1,
  'first_label_candidate_logps':oracle,'candidate_first_label_abs_errors':errors,'maximum_candidate_first_label_error':max(errors),'unchanged_tolerance':.002,
  'control_within_original_tolerance':max(errors)<=.002,'causal_interpretation':'Shape control only; no kernel-cause claim'}
 del logits,lp,inputs;gc.collect();mx.clear_cache()
 return result

def capture_score(score,backend,record,index,capture_path=None):
 try:
  result=score(backend,record,index)
  raise RuntimeError('Original oracle failure not reproduced; no shape control executed')
 except ValueError as exc:
  if str(exc)!='Independent prefix oracle disagrees':raise
  frame=None;tb=exc.__traceback__
  while tb is not None:
   if tb.tb_frame.f_code is score.__code__:frame=tb.tb_frame
   tb=tb.tb_next
  require(frame is not None,'Expected failure not raised by exact score_record code')
  values={k:frame.f_locals[k] for k in KEYS if k in frame.f_locals}
  require(all(k in values for k in ('first','oracle','errors','prefix_scores','candidates','parts','P','label_first')),'Expected numerical locals missing')
  control=padded_control(backend,frame.f_locals)
  x=json.loads(json.dumps({'oracle_gate':'FAIL','exception':str(exc),'unchanged_tolerance':.002,'numerics_PRIVATE':values,'shape_matched_control':control},allow_nan=False))
  if capture_path is not None:save(capture_path,x)
  traceback.clear_frames(exc.__traceback__);del frame,tb,values
  return x
 else:
  x={'oracle_gate':'PASS_REPRODUCTION_NOT_OBSERVED','unchanged_tolerance':.002,'score_result_PRIVATE':result}
  if capture_path is not None:save(capture_path,x)
  return x

def prepare(a):
 primary=a.primary_root.resolve();helper=Path(__file__).with_name('run_c40_early_visual_probe_v1.py');require(sha(helper)==EARLY_SHA,'Reviewed loader helper changed')
 spec=importlib.util.spec_from_file_location('early_forensic_loader',helper);early=importlib.util.module_from_spec(spec);spec.loader.exec_module(early)
 api=early.primary_api(primary);plan=api.validate_plan(a.original_plan)
 require(sha(a.prefix_output)==a.expected_prefix_sha256 and sha(a.failure_status)==a.expected_failure_status_sha256,'Frozen nine-row/failure evidence changed')
 rows=[json.loads(s) for s in a.prefix_output.read_text().splitlines()];require(len(rows)==9,'Exactly nine preserved results required')
 for i,row in enumerate(rows):require(row['record_id']==plan['records_PRIVATE'][i]['record_id'] and row['checkpoint']=='B28','Preserved prefix coverage changed');api.validate_numeric_route(row)
 failure=load(a.failure_status);require(failure['status']=='EARLY_PROBE_FAILED' and failure['checkpoint']=='B28' and failure['native_started'] is True,'Actual failed native B28 status required')
 prior=a.output_dir.parent/'B28_index9_forensic_v1/oracle_capture_PRIVATE.json';require(sha(prior)=='094f846a39f5dad9c6ad6dcd299182391bfdb2da58a00644d7c3d72ddf42d64a','Actual R1 capture changed')
 record=plan['records_PRIVATE'][9];require(record['condition']=='no_image' and record['category']=='mri_visit_correspondence','Exact uncompleted null correspondence record required')
 adapter=plan['checkpoints']['B28'];require(sha(adapter['path'])==adapter['sha256'] and sha(Path(adapter['path']).with_name('adapter_config.json'))==adapter['adapter_config_sha256'],'Frozen B28 changed')
 require(not a.output_dir.exists(),'Forensic output must be new, never append or overwrite')
 lock=Path(api.MANIFEST).parent/'remaining20.lock';require(a.queue_lock.resolve()==lock.resolve(),'Original global queue lock required')
 require(a.output_dir.resolve().is_relative_to(Path(__file__).resolve().parents[1]/'local_data'),'Private worktree output required')
 scope={'kind':'C40_B28_index9_prefix_oracle_shape_control_v2','execution_tier':'B','runner_sha256':sha(__file__),'loader_helper_sha256':EARLY_SHA,'prior_R1_capture_sha256':sha(prior),
  'primary_root':str(primary),'original_plan_sha256':sha(a.original_plan),'sources':api.source_hashes(),'adapter':adapter,'derived_cache':plan['derived_cache'],
  'prefix_output':str(a.prefix_output.resolve()),'prefix_output_sha256':a.expected_prefix_sha256,'preserved_records':9,
  'failure_status':str(a.failure_status.resolve()),'failure_status_sha256':a.expected_failure_status_sha256,'record_index':9,'record_PRIVATE':record,
  'checkpoint':'B28','candidate_forwards':4,'original_unchanged_forwards':3,'shape_control_forwards':1,'shape_control_input_length':129,'shape_control_score_position':125,'future_padding_token':0,'greedy_generations':0,'backward_calls':0,'weight_updates':0,'prefix_tolerance':.002,
  'queue_lock':str(lock.resolve()),'output_dir':str(a.output_dir.resolve()),'expected_failure':'Independent prefix oracle disagrees'}
 return early,api,plan,record,scope

def execute(a,early,api,plan,record,scope):
 early.verify_review(a.review_verdict,a.expected_review_sha256,scope)
 with a.queue_lock.open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  from mri_rl.paced_resources import PacedGate
  from mri_rl.likelihood_cache import verify_cache_identity
  from mri_rl.experiment_contract import load_contract
  a.output_dir.mkdir(parents=True,exist_ok=False);gate=PacedGate(a.output_dir/'resources_PRIVATE.jsonl');gate.before_unit('forensic_before_native_import',loading=True)
  from mri_rl.mlx_continuous_gspo import NativeGSPOBackend,is_loading_unit
  cache={'dir':plan['derived_cache']['cache_dir']};certificate=a.primary_root.resolve()/'results/EXACT_QUANT_CACHE_CERTIFICATE.json';cert=load(certificate);model=Path(plan['model_path'])
  artifacts={str(model/n):sha(model/n) for n in set(cert['non_weight_files_identical'])|{'config.json'}}
  require(verify_cache_identity(cache['dir'],certificate,plan['derived_cache']['certificate_sha256'],artifacts)==plan['derived_cache'],'Certified cache changed')
  def before(name):gate.before_unit(name,loading=is_loading_unit(name,cache))
  backend=None;released=False
  try:
   backend=NativeGSPOBackend(plan['checkpoints']['B28']['path'],{'group_size':2,'max_tokens':256},model_path=model,contract=load_contract(Path(plan['prompt_contract_path'])),seed=7,before_unit=before,cache=cache)
   disk=backend.mx.load(plan['checkpoints']['B28']['path']);actual=dict(backend.tree_flatten(backend.model.trainable_parameters()))
   require(set(disk)==set(actual) and len(actual)==504 and all(bool(backend.mx.array_equal(v,actual[k]).item()) for k,v in disk.items()),'Initial adapter mismatch')
   del disk,actual;gc.collect();backend.mx.clear_cache()
   capture=capture_score(api.score_record,backend,record,9,a.output_dir/'oracle_capture_PRIVATE.json')
   gc.collect();backend.mx.clear_cache();before('forensic_after_capture')
   disk=backend.mx.load(plan['checkpoints']['B28']['path']);actual=dict(backend.tree_flatten(backend.model.trainable_parameters()))
   require(set(disk)==set(actual) and len(actual)==504 and all(bool(backend.mx.array_equal(v,actual[k]).item()) for k,v in disk.items()) and int(backend.optimizer_state()['step'].item())==0,'Read-only adapter/optimizer integrity failed')
   del disk,actual;gc.collect();backend.mx.clear_cache()
  finally:
   if backend is not None:
    backend.close();gc.collect();backend.mx.clear_cache();released=bool(backend.model is None and backend.mx.get_active_memory()<128*1024**2)
  require(released,'Native allocation release not verified')
  require(sha(a.prefix_output)==scope['prefix_output_sha256'],'Preserved nine results changed')
  receipt={'status':'ACTUAL_SHAPE_CONTROL_CAPTURE_COMPLETED','oracle_gate':capture['oracle_gate'],'frozen_experiment_completed':False,'weight_updates':0,'adapter_tensors_exact':504,'optimizer_step':0,'native_children_released':True,'critical_safety_condition':False,'queue_resume_safe':True,'scope_sha256':early.h(scope),'short_oracle_maximum_error':max(capture['numerics_PRIVATE']['errors']),'shape_matched_control':capture['shape_matched_control'],'capture_sha256':sha(a.output_dir/'oracle_capture_PRIVATE.json'),'preserved_prefix_sha256':scope['prefix_output_sha256']}
  save(a.output_dir/'receipt_SAFE.json',receipt);print(json.dumps(receipt,sort_keys=True))

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ('primary-root','original-plan','prefix-output','failure-status','queue-lock','output-dir'):p.add_argument('--'+n,type=Path,required=True)
 for n in ('expected-prefix-sha256','expected-failure-status-sha256'):p.add_argument('--'+n,required=True)
 for n in ('review-verdict','scope-output'):p.add_argument('--'+n,type=Path)
 p.add_argument('--expected-review-sha256');g=p.add_mutually_exclusive_group(required=True);g.add_argument('--prepare-only',action='store_true');g.add_argument('--execute',action='store_true');a=p.parse_args()
 early,api,plan,record,scope=prepare(a)
 if a.prepare_only:require(a.scope_output is not None,'Scope output required');save(a.scope_output,scope);print(json.dumps({'status':'FORENSIC_PREPARED_NOT_RUN','scope_sha256':early.h(scope),'scope_file_sha256':sha(a.scope_output),'candidate_forwards':4,'original_unchanged_forwards':3,'shape_control_forwards':1,'shape_control_input_length':129,'shape_control_score_position':125,'future_padding_token':0,'weight_updates':0}));return
 execute(a,early,api,plan,record,scope)
if __name__=='__main__':main()
