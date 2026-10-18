"""Direct idle-slot read-only auxiliary comparison. No controller or training changes.
B28 index9 only, U7 original10; original B28 nine rows remain immutable evidence.
"""
from __future__ import annotations
import argparse,fcntl,gc,hashlib,importlib.util,json,math,os
from pathlib import Path
WORKTREE=Path(__file__).resolve().parents[1]
EARLY_SHA='8cafacbf3a9086160c8e257673c6f933070df208bffc5c32dec38bc0e1e82e06'
VALIDATION_SHA='bb5a408ced420d752250f0280c74c0140f6723335dfb33be9034ef6f6aac9acd'
def require(ok,msg):
 if not ok:raise ValueError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def h(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def module(path,name,expected):
 require(sha(path)==expected,'Reviewed auxiliary source changed');s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def indices_for(checkpoint):
 require(checkpoint in {'B28','U_raw_seed7'},'Unknown checkpoint')
 return [9] if checkpoint=='B28' else list(range(10))
def normalize_score(api,record,result):
 if result.get('legacy_oracle_gate')=='FAIL':
  require(result['verdict']=='PASS_WITH_LIMITATIONS','Auxiliary shape validation not verified')
  scores=result['candidate_scores'];_,compare=api.bound_candidate_functions(api.LABELS[record['category']])
  row={**record,'candidate_scores':scores,'candidate_order':list(api.LABELS[record['category']]),
   'comparisons':api.task_comparisons(record['category'],compare(scores)),
   'prefix_oracle_first_logps':result['shape_matched_control']['first_label_candidate_logps'],
   'legacy_oracle_gate':'FAIL','legacy_oracle_evidence_PRIVATE':result['legacy_evidence_PRIVATE'],
   'shape_matched_control':result['shape_matched_control'],'legacy_oracle_max_disagreement':max(result['legacy_evidence_PRIVATE']['errors']),
   'validation_version':result['validation_version'] if 'validation_version' in result else 'C40_no_image_equal_shape_oracle_v2'}
 else:
  row=dict(result['legacy_result'] if 'legacy_result' in result else result);row['legacy_oracle_gate']='PASS';row['validation_version']='original_e29_unchanged'
 row.update(auxiliary_version='C40_direct_idle_auxiliary_v2',frozen_experiment_completed=False,weight_updates=0)
 return row

def validate_auxiliary(api,row,record):
 require(all(row.get(k)==v for k,v in record.items()),'Original displayed-image record changed')
 labels=api.LABELS[record['category']];scores=row['candidate_scores'];require(list(scores)==list(labels) or set(scores)==set(labels),'Candidate vocabulary changed')
 for label in labels:
  candidate=scores[label];values=candidate['target_token_logps'];part=candidate['partition']
  expected=api.al.summarize_tokens(values,part)
  require(all(math.isfinite(v) for v in values) and all(candidate[k]==v for k,v in expected.items()),'Candidate token summaries changed')
 _,compare=api.bound_candidate_functions(labels);require(row['comparisons']==api.task_comparisons(record['category'],compare(scores)),'Candidate comparisons changed')
 if row['legacy_oracle_gate']=='PASS':api.validate_numeric_route(row)
 else:
  require(record['condition']=='no_image' and record['category']=='mri_visit_correspondence' and row['shape_matched_control']['maximum_candidate_first_label_error']<=.002 and row['legacy_oracle_max_disagreement']>.002 and row['frozen_experiment_completed'] is False,'Unsupported auxiliary oracle correction')
  require(scores==row['legacy_oracle_evidence_PRIVATE']['candidates'],'Original candidate scores changed')
  require(row['prefix_oracle_first_logps']==row['shape_matched_control']['first_label_candidate_logps'],'Wrong active auxiliary oracle vector')

def completed(path,scope,records):
 if not path.exists():return 0
 raw=path.read_bytes();require(not raw or raw.endswith(b'\n'),'Torn auxiliary output')
 rows=[json.loads(line) for line in raw.splitlines()];require(len(rows)<=len(scope['indices']),'Duplicate/oversized auxiliary output')
 for row,index in zip(rows,scope['indices']):
  require(row['record_index']==index and row['record_id']==records[index]['record_id'] and row['run_identity']==scope['run_identity'] and row['checkpoint']==scope['checkpoint'] and row['weight_updates']==0,'Auxiliary resume identity/order mismatch')
 return len(rows)

def prepare(a):
 primary=a.primary_root.resolve();early=module(WORKTREE/'scripts/run_c40_early_visual_probe_v1.py','direct_early',EARLY_SHA)
 validation=module(WORKTREE/'src/mri_rl/c40_shape_validation_v2.py','direct_shape',VALIDATION_SHA)
 api=early.primary_api(primary);plan=api.validate_plan(a.original_plan);records=plan['records_PRIVATE']
 require(sha(a.legacy_prefix)==a.expected_legacy_prefix_sha256 and sha(a.legacy_scope)==a.expected_legacy_scope_sha256,'Immutable original nine-prefix scope changed')
 prior=load(a.legacy_scope);rows=[json.loads(x) for x in a.legacy_prefix.read_text().splitlines()];require(len(rows)==9,'Exactly nine original B28 results required')
 for index,row in enumerate(rows):
  require(all(row.get(k)==v for k,v in records[index].items()) and row['run_identity']==prior['run_identity'] and row['checkpoint']=='B28' and row['adapter_sha256']==plan['checkpoints']['B28']['sha256'],'Original nine provenance differs');api.validate_numeric_route(row)
 require(sha(a.r2_receipt)==a.expected_r2_receipt_sha256,'Actual R2 evidence changed')
 r2=load(a.r2_receipt);require(r2['status']=='ACTUAL_SHAPE_CONTROL_CAPTURE_COMPLETED' and r2['oracle_gate']=='FAIL' and r2['shape_matched_control']['maximum_candidate_first_label_error']<=.002 and r2['native_children_released'] is True,'Actual bounded R2 support required')
 require(a.output.resolve().is_relative_to(WORKTREE/'local_data') and a.receipt.resolve().is_relative_to(WORKTREE/'local_data'),'Private auxiliary outputs required')
 lock=Path(api.MANIFEST).parent/'remaining20.lock';require(a.queue_lock.resolve()==lock.resolve(),'Original global queue lock required')
 adapter=plan['checkpoints'][a.checkpoint];require(sha(adapter['path'])==adapter['sha256'] and sha(Path(adapter['path']).with_name('adapter_config.json'))==adapter['adapter_config_sha256'],'Frozen adapter/config changed')
 saved,saved_sha=api.saved_predictions(a.checkpoint);indices=indices_for(a.checkpoint)
 scope={'kind':'C40_direct_idle_auxiliary_v2','execution_tier':'B','checkpoint':a.checkpoint,'primary_root':str(primary),
  'original_plan_path':str(a.original_plan.resolve()),'original_plan_sha256':sha(a.original_plan),'runner_sha256':sha(__file__),'validation_helper_sha256':VALIDATION_SHA,'loader_helper_sha256':EARLY_SHA,
  'sources':api.source_hashes(),'adapter':adapter,'derived_cache':plan['derived_cache'],'records_PRIVATE':records,'indices':indices,
  'legacy_prefix':str(a.legacy_prefix.resolve()),'legacy_prefix_sha256':a.expected_legacy_prefix_sha256,'legacy_scope':str(a.legacy_scope.resolve()),'legacy_scope_sha256':a.expected_legacy_scope_sha256,
  'actual_R2_receipt':str(a.r2_receipt.resolve()),'actual_R2_receipt_sha256':a.expected_r2_receipt_sha256,'saved_real_outputs_sha256':saved_sha,
  'output':str(a.output.resolve()),'receipt':str(a.receipt.resolve()),'queue_lock':str(lock.resolve()),'maximum_candidate_forwards':3*len(indices)+1,
  'maximum_shape_control_forwards':1,'new_null_greedy_generations':sum(records[i]['condition']=='no_image' for i in indices),'new_real_greedy_generations':0,
  'blank_image_controls':'NOT_RUN in this finite scope','backward_calls':0,'weight_updates':0,'prefix_tolerance':.002,'legacy_failure_preserved':True,'frozen_experiment_completed':False}
 scope['run_identity']=h(scope);count=completed(a.output,scope,records)
 if count:
  for row,index in zip((json.loads(x) for x in a.output.read_text().splitlines()),indices):validate_auxiliary(api,row,records[index])
 return early,validation,api,plan,scope,saved,count

def execute(a,early,validation,api,plan,scope,saved,count):
 early.verify_review(a.review_verdict,a.expected_review_sha256,scope)
 if a.receipt.exists():
  receipt=load(a.receipt);require(count==len(scope['indices']) and receipt['output_sha256']==sha(a.output) and receipt['scope_sha256']==h(scope),'Completed auxiliary receipt mismatch');print(json.dumps(receipt));return
 with a.queue_lock.open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  from mri_rl.paced_resources import PacedGate
  from mri_rl.likelihood_cache import verify_cache_identity
  from mri_rl.experiment_contract import load_contract
  a.output.parent.mkdir(parents=True,exist_ok=True);gate=PacedGate(a.output.with_suffix('.resources_PRIVATE.jsonl'));gate.before_unit('direct_aux_before_native_import',loading=True)
  from mri_rl.mlx_continuous_gspo import NativeGSPOBackend,is_loading_unit
  cache={'dir':plan['derived_cache']['cache_dir']};certificate=a.primary_root.resolve()/'results/EXACT_QUANT_CACHE_CERTIFICATE.json';cert=load(certificate);model=Path(plan['model_path'])
  artifacts={str(model/n):sha(model/n) for n in set(cert['non_weight_files_identical'])|{'config.json'}}
  require(verify_cache_identity(cache['dir'],certificate,plan['derived_cache']['certificate_sha256'],artifacts)==plan['derived_cache'],'Certified cache changed')
  def before(name):gate.before_unit(name,loading=is_loading_unit(name,cache))
  backend=None;released=False
  def readonly():
   disk=backend.mx.load(scope['adapter']['path']);actual=dict(backend.tree_flatten(backend.model.trainable_parameters()))
   require(set(disk)==set(actual) and len(actual)==504 and all(bool(backend.mx.array_equal(v,actual[k]).item()) for k,v in disk.items()) and int(backend.optimizer_state()['step'].item())==0,'Read-only adapter/optimizer integrity failed')
   del disk,actual;gc.collect();backend.mx.clear_cache()
  try:
   backend=NativeGSPOBackend(scope['adapter']['path'],{'group_size':2,'max_tokens':256},model_path=model,contract=load_contract(Path(plan['prompt_contract_path'])),seed=7,before_unit=before,cache=cache);readonly()
   with a.output.open('a' if a.output.exists() else 'x') as handle:
    fd=os.open(a.output.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
    for index in scope['indices'][count:]:
     record=plan['records_PRIVATE'][index]
     scored=validation.score_record_v2(api,backend,record,index) if record['condition']=='no_image' and record['category']=='mri_visit_correspondence' else api.score_record(backend,record,index)
     row=normalize_score(api,record,scored)
     if record['condition']=='no_image':row.update(backend.generate_record(record,None,256));row['new_greedy_generation']=True
     else:row['response']=saved[record['source_record_id']]['response'];row['new_greedy_generation']=False
     row.update(run_identity=scope['run_identity'],checkpoint=a.checkpoint,record_index=index,adapter_sha256=scope['adapter']['sha256'])
     validate_auxiliary(api,row,record);handle.write(json.dumps(row,allow_nan=False)+'\n');handle.flush();os.fsync(handle.fileno());count+=1
     print(json.dumps({'checkpoint':a.checkpoint,'auxiliary_completed':count,'total':len(scope['indices']),'record_index':index,'legacy_oracle_gate':row['legacy_oracle_gate']}),flush=True)
     gc.collect();backend.mx.clear_cache();before('direct_aux_record_completed')
   readonly()
  finally:
   if backend is not None:backend.close();gc.collect();backend.mx.clear_cache();released=bool(backend.model is None and backend.mx.get_active_memory()<128*1024**2)
  require(released,'Native allocation release not verified');require(sha(a.legacy_prefix)==scope['legacy_prefix_sha256'],'Original nine outputs changed')
  receipt={'status':'ACTUAL_DIRECT_AUXILIARY_COMPLETED','checkpoint':a.checkpoint,'completed_records':count,'output_sha256':sha(a.output),'scope_sha256':h(scope),'adapter_tensors_exact':504,'optimizer_step':0,'weight_updates':0,'native_children_released':True,'legacy_failure_preserved':True,'frozen_experiment_completed':False}
  early.save(a.receipt,receipt);print(json.dumps(receipt,sort_keys=True))

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ('primary-root','original-plan','legacy-prefix','legacy-scope','r2-receipt','queue-lock','output','receipt'):p.add_argument('--'+n,type=Path,required=True)
 for n in ('expected-legacy-prefix-sha256','expected-legacy-scope-sha256','expected-r2-receipt-sha256'):p.add_argument('--'+n,required=True)
 p.add_argument('--checkpoint',choices=['B28','U_raw_seed7'],required=True)
 for n in ('review-verdict','scope-output'):p.add_argument('--'+n,type=Path)
 p.add_argument('--expected-review-sha256');g=p.add_mutually_exclusive_group(required=True);g.add_argument('--prepare-only',action='store_true');g.add_argument('--execute',action='store_true');a=p.parse_args()
 early,validation,api,plan,scope,saved,count=prepare(a)
 if a.prepare_only:require(a.scope_output is not None,'Scope output required');early.save(a.scope_output,scope);print(json.dumps({'status':'DIRECT_AUX_PREPARED_NOT_RUN','checkpoint':a.checkpoint,'records':len(scope['indices']),'completed':count,'scope_file_sha256':sha(a.scope_output),'scope_sha256':h(scope)}));return
 execute(a,early,validation,api,plan,scope,saved,count)
if __name__=='__main__':main()
