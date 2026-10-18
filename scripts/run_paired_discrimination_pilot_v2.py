"""Thin conditional V2 paired-SFT entry and frozen post-two-update evaluation.
No models during prepare. Original study, scorer and data remain unchanged.
"""
from __future__ import annotations
import argparse,fcntl,gc,hashlib,importlib.util,json,os
from collections import defaultdict
from pathlib import Path
W=Path(__file__).resolve().parents[1]
FROZEN_INPUT_SHA='446808a0203ba1e6dd9f45e7f052a7667420b016ffff5a5b88f295d728fc79ab'
PAIRED_SHA='79963d311b4797e8010a5b6c8de81c21f2f0adb1f5088613f2e867afcdfd021e'
LABELS=('FLAIR','contrast-enhanced T1')
def require(ok,msg):
 if not ok:raise ValueError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def canon(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def module(path,name,expected=None):
 if expected:require(sha(path)==expected,'Frozen source differs')
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def save(path,value):
 path=Path(path);data=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode();path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists():require(path.read_bytes()==data,'Preserve earlier artifact; choose new version');return
 with path.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 fd=os.open(path.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)

def feature_binding(plan_path,plan_sha,receipt_path,receipt_sha):
 if not all((plan_path,plan_sha,receipt_path,receipt_sha)):return {'ready':False,'status':'PENDING_ACTUAL_MEAN48_FEATURE_EVIDENCE'}
 require(sha(plan_path)==plan_sha and sha(receipt_path)==receipt_sha,'Actual feature evidence digest differs')
 p=load(plan_path);r=load(receipt_path);release=r['release']
 require(p['input_sha256']==FROZEN_INPUT_SHA and p['train_rows']==p['dev_rows']==24 and p['train_people']==p['dev_people']==4,'Expected fixed feature first stage required')
 require(r['status']=='ACTUAL48_VISION_FORWARDS_READONLY_COMPLETE_AND_RELEASED' and r['plan_sha256']==plan_sha and r['run_identity']==canon(p) and len(r['record_sha256'])==48 and r['adapter_sha256']==p['adapter']['sha256'] and r['adapter_tensors_compared_after']==504 and r['optimizer_step']==r['weight_updates']==0,'Actual feature completion evidence incomplete')
 require(release['native_release_verified'] is True and release['model_is_none'] is True and 0<=release['active_MLX_bytes']<release['active_limit_bytes']==128*1024**2,'Actual feature release not verified')
 return {'ready':True,'plan_path':str(Path(plan_path).resolve()),'plan_sha256':plan_sha,'receipt_path':str(Path(receipt_path).resolve()),'receipt_sha256':receipt_sha,'status':'ACTUAL_FEATURE_STAGE_COMPLETED_INTERPRETATION_REMAINS_SCOPED'}

def freeze_evaluation(rows,projection,images,system):
 require(len(rows)==len(projection)==40 and len({r['id'] for r in rows})==40 and len({r['record_id'] for r in projection})==40,'Frozen full40 coverage required')
 source={r['id']:r for r in rows};require(set(source)=={r['record_id'] for r in projection},'Projection source coverage differs')
 greedy=[];candidates=[];pairs=defaultdict(dict)
 for p in projection:
  row=source[p['record_id']];require(p['category']==row['ability'] and p['image']==row['images'][0] and p['prompt']==[{'role':'system','content':system},{'role':'user','content':row['extra_info']['question']}],'Matched native prompt/image projection differs')
  g={**p,'image_sha256':images[row['id']],'visible_target_PRIVATE':row['reward_model']['ground_truth']};greedy.append(g)
  if row['ability']=='mri_sequence':
   e=row['extra_info'];key=(e['patient_key'],e['level']);label=row['reward_model']['ground_truth'];require(label in LABELS and label not in pairs[key],'Invalid paired sequence truth');pairs[key][label]=row['id']
   candidates.append({**g,'source_record_id':row['id'],'condition':'real_image','candidate_labels':list(LABELS)})
 require(len(candidates)==24 and len(pairs)==12 and len({k[0] for k in pairs})==4 and all(set(v)==set(LABELS) for v in pairs.values()),'Expected4people/3levels/opposite24sequence images')
 require(len({json.dumps(r['prompt'],sort_keys=True) for r in candidates})==1,'Null must match every sequence query')
 null={**candidates[0],'record_id':'V2_SEQUENCE_NULL','source_record_id':None,'image':None,'image_sha256':None,'visible_target_PRIVATE':None,'condition':'no_image'};candidates.append(null)
 return {'schema':'paired-pilot-evaluation-contract-v2','greedy_records_PRIVATE':greedy,'candidate_records_PRIVATE':candidates,
  'sequence_pairs_PRIVATE':[{'patient_PRIVATE':key[0],'level':key[1],'negative_id':v[LABELS[0]],'positive_id':v[LABELS[1]]} for key,v in sorted(pairs.items())],
  'greedy_max_new_tokens':256,'temperature':0,'maximum_candidate_forwards':75,'candidate_score_definitions':['mean label logp PRIMARY','raw label logp sum','independent first-label logp'],
  'displayed_truth':'Always label of displayed MRI; same-patient/level replacement is the paired opposite-class image already in these24, no duplicate calls.',
  'dev_units':{'sequence_people':4,'Dallas_matched_cycles':4},'blank_image_controls':'NOT_RUN','sealed_final100_used':False,'selection_after_outcomes':False}

def reusable_baseline(candidate_records,prefix,scope):
 require(len(prefix)==9,'Original nine B28 rows required');result={}
 for record in candidate_records:
  matches=[(i,r) for i,r in enumerate(prefix) if r['category']=='mri_sequence' and r['condition']==record['condition'] and r['image_sha256']==record['image_sha256'] and r['prompt']==record['prompt'] and r['visible_target_PRIVATE']==record['visible_target_PRIVATE'] and (record['condition']=='no_image' or r['source_record_id']==record['source_record_id'])]
  require(len(matches)<=1,'Duplicate reusable candidate')
  if matches:
   i,row=matches[0];require(row['run_identity']==scope['run_identity'] and row['checkpoint']=='B28' and row['adapter_sha256']==scope['adapter']['sha256'],'Original candidate provenance differs');result[record['record_id']]={'index':i,'original_record_id':row['record_id']}
 require(len(result)==5 and sum(next(r for r in candidate_records if r['record_id']==k)['condition']=='no_image' for k in result)==1,'Exactly4real+1null sequence candidates must be reused')
 return result

def read_rows(path):return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
def checkpoint_state(run,plan,paired):
 first=Path(run)/'boundary_0000/receipt.json'
 if not first.exists():return None
 from mri_rl.continuous_rl import inspect_boundaries
 binding=load(first)['binding'];require(binding['experiment']['pilot_plan_sha256']==paired.identity(plan) and binding['experiment']['objective']=='paired_mean_label_discrimination_v2','Different pilot journal binding')
 current=inspect_boundaries(Path(run),binding)
 require(current['completed_groups']<=2 and current['optimizer_updates']==current['completed_groups'],'Unexpected pilot update/cursor')
 return {'cursor':current['completed_groups'],'optimizer_updates':current['optimizer_updates'],'receipt_sha256':sha(current['path']/'receipt.json'),'adapter':str(current['path']/'adapter.safetensors'),'adapter_sha256':sha(current['path']/'adapter.safetensors'),'adapter_config_sha256':sha(current['path']/'adapter_config.json'),'feature_proof_sha256':binding['experiment']['feature_proof_sha256']}

def scope_identity(scope):
 return canon({k:v for k,v in scope.items() if k not in {'starting_journal','resume','ready','starting_outputs_sha256','run_identity'}})

def prepare(a):
 R=a.primary_root.resolve();paired=module(W/'src/mri_rl/paired_discrimination_pilot_v2.py','paired_entry_core',PAIRED_SHA);sft,al,common=paired.primary_modules(R)
 early=module(W/'scripts/run_c40_early_visual_probe_v1.py','paired_entry_loader','8cafacbf3a9086160c8e257673c6f933070df208bffc5c32dec38bc0e1e82e06');api=early.primary_api(R)
 original_path=R/'local_data/c_two_task_pool_v1/u7_visual_controls_v1/probe_plan_PRIVATE.json';original=api.validate_plan(original_path);plan=load(a.pilot_plan)
 require(plan['objective_source_sha256']==PAIRED_SHA and plan['maximum_updates']==2 and plan['input_tokens'] is not None,'Measured frozen pilot configuration required')
 frozen_path=W/'local_data/v2_cpu_representation_v1/frozen_inputs_PRIVATE.json';require(sha(frozen_path)==FROZEN_INPUT_SHA,'Frozen train/dev source pack changed');frozen=load(frozen_path)
 projection=R/'local_data/c_two_task_pool_v1/evaluation_v1/internal_dev_native_system_matched.jsonl';require(sha(projection)=='b556b70ecd5570bf8f0a814b5a1654a0b60553f1025a123e6e56bfce69187199','Frozen native dev projection differs')
 from mri_rl.experiment_contract import load_contract
 contract=load_contract(Path(original['prompt_contract_path']));evaluation=freeze_evaluation(frozen['dev_PRIVATE'],read_rows(projection),frozen['images_PRIVATE'],contract.system_instruction)
 for row in evaluation['greedy_records_PRIVATE']:require(sha(row['image'])==row['image_sha256'],'Frozen dev image changed')
 prefix_path=W/'local_data/c40_early_visual_v1/B28_outputs_RECOVERY_V6_PRIVATE.jsonl';oldscope_path=W/'local_data/c40_early_visual_v1/B28_scope_RECOVERY_V6_READY_FOR_REVIEW_PRIVATE.json'
 require(sha(prefix_path)=='c59c5362a32a591eb1c7bf5d9858721b97267722d124bd4cb8075bb83ce93c30' and sha(oldscope_path)=='cf88824ba112fa3f639a001bf0414d4ad66b50b6eb55d7d0ab47e84c7f849ea3','Exact reusable B28 candidates changed')
 prefix=read_rows(prefix_path);oldscope=load(oldscope_path)
 for row in prefix:api.validate_numeric_route(row)
 reuse=reusable_baseline(evaluation['candidate_records_PRIVATE'],prefix,oldscope)
 saved_common,saved_common_sha=api.saved_predictions('B28');require(set(saved_common)>={r['record_id'] for r in evaluation['greedy_records_PRIVATE']},'Authenticated common B28 answers must cover all40')
 evaluation.update(original_plan_sha256=sha(original_path),native_projection_sha256=sha(projection),B28_reused_candidate_map_PRIVATE=reuse,B28_prefix_path=str(prefix_path),B28_prefix_sha256=sha(prefix_path),B28_prefix_scope_sha256=sha(oldscope_path),B28_new_candidate_records=20,B28_new_candidate_forwards=60,B28_new_greedy_generations=0,B28_saved_greedy_sha256=saved_common_sha,post2_new_candidate_records=25,post2_new_candidate_forwards=75,post2_new_real_greedy_generations=40,post2_new_null_greedy_generations=1)
 save(a.eval_contract,evaluation)
 feature=feature_binding(a.feature_plan,a.feature_plan_sha256,a.feature_completion,a.feature_completion_sha256)
 state=checkpoint_state(a.run_directory,plan,paired);adapter=original['checkpoints']['B28'] if a.checkpoint=='B28' else ({'path':state['adapter'],'sha256':state['adapter_sha256'],'adapter_config_sha256':state['adapter_config_sha256']} if state and state['cursor']==2 else None)
 ready=feature['ready'] and (a.mode=='train' or adapter is not None)
 if state and feature['ready']:require(state['feature_proof_sha256']==feature['receipt_sha256'],'Checkpoint feature-proof binding differs')
 for path in (a.run_directory,a.output_directory,a.eval_contract):require(path.resolve().is_relative_to(W/'local_data'),'Private worktree destinations required')
 scope={'execution_tier':'B','kind':'paired_discrimination_pilot_'+a.mode+'_v2','entry_sha256':sha(__file__),'paired_source_sha256':PAIRED_SHA,'primary_root':str(R),'sources':plan['sources'],'native_sources':api.source_hashes(),'pilot_plan_path':str(a.pilot_plan.resolve()),'pilot_plan_sha256':sha(a.pilot_plan),'evaluation_contract_path':str(a.eval_contract.resolve()),'evaluation_contract_sha256':sha(a.eval_contract),'feature_evidence':feature,'ready':ready,'checkpoint':a.checkpoint,'adapter':adapter,'common_B28':original['checkpoints']['B28'],'derived_cache':original['derived_cache'],'contract_sha256':sha(original['prompt_contract_path']),'run_directory':str(a.run_directory.resolve()),'output_directory':str(a.output_directory.resolve()),'resume':a.resume,'starting_journal':state,'maximum_new_updates':2-(state['cursor'] if state else 0) if a.mode=='train' else 0,'max_new_steps':2,'seed':7,'greedy_max_tokens':256,'temperature':0,'candidate_forwards':(60 if a.checkpoint=='B28' else 75) if a.mode=='eval' else 16,'new_greedy_generations':(0 if a.checkpoint=='B28' else 41) if a.mode=='eval' else 0,'sealed_final100_used':False,'scientific_claim':'Exploratory prespecified two-update intervention on repeatedly examined internal dev; no independent generalization claim'}
 scope['run_identity']=scope_identity(scope)
 progress={name:(sha(a.output_directory/name) if (a.output_directory/name).exists() else None) for name in ('greedy_PRIVATE.jsonl','candidates_PRIVATE.jsonl','eval_complete_SAFE.json')}
 scope['starting_outputs_sha256']=progress
 if a.mode=='eval':
  grows=durable_prefix(a.output_directory/'greedy_PRIVATE.jsonl',scope['run_identity'],evaluation['greedy_records_PRIVATE'])
  crows=durable_prefix(a.output_directory/'candidates_PRIVATE.jsonl',scope['run_identity'],evaluation['candidate_records_PRIVATE'],api.validate_numeric_route)
  if progress['eval_complete_SAFE.json'] is not None:
   require(len(grows)==40 and len(crows)==25,'Completion requires full coverage');validate_completion(load(a.output_directory/'eval_complete_SAFE.json'),scope,evaluation,a.output_directory/'greedy_PRIVATE.jsonl',a.output_directory/'candidates_PRIVATE.jsonl')
 return paired,early,api,original,plan,evaluation,scope

def durable_prefix(path,identity,records,validator=None):
 require(all(isinstance(r,dict) and 'record_id' in r for r in records),'Frozen record metadata required for resume')
 if not path.exists():return []
 raw=path.read_bytes();require(not raw or raw.endswith(b'\n'),'Torn evaluation output');rows=[json.loads(x) for x in raw.splitlines()];require(len(rows)<=len(records),'Oversized evaluation output')
 for row,expected in zip(rows,records):
  json.dumps(row,allow_nan=False)
  require(row['run_identity']==identity and row['weight_updates']==0 and all(row.get(k)==v for k,v in expected.items()),'Evaluation resume metadata/identity mismatch')
  if validator is not None:validator(row)
  else:require(isinstance(row.get('response'),str),'Complete greedy response required')
 return rows

def validate_completion(receipt,scope,evaluation,greedy_path,candidate_path):
 require(receipt['status']=='ACTUAL_FIXED40_GREEDY_AND25_SEQUENCE_CANDIDATES_COMPLETE' and receipt['checkpoint']==scope['checkpoint'] and receipt['run_identity']==scope['run_identity'] and receipt['adapter_sha256']==scope['adapter']['sha256'] and receipt['native_release_verified'] is True and receipt['weight_updates']==0,'Completed evaluation identity/release differs')
 require(receipt['greedy_records']==40 and receipt['candidate_records']==25 and receipt['greedy_sha256']==sha(greedy_path) and receipt['candidate_sha256']==sha(candidate_path) and receipt['evaluation_contract_sha256']==scope['evaluation_contract_sha256'],'Completed evaluation hashes/counts/contract differ')

def resource_gate(output_directory):
 from mri_rl.paced_resources import PacedGate
 output_directory.mkdir(parents=True,exist_ok=True)
 return PacedGate(output_directory/'resources_PRIVATE.jsonl')

def execute(a,paired,early,api,original,plan,evaluation,scope):
 require(scope['ready'],'Actual feature evidence/selected checkpoint not ready')
 early.verify_review(a.review,a.expected_review_sha256,scope)
 from mri_rl.paced_resources import PacedGate
 from mri_rl.mlx_continuous_gspo import NativeGSPOBackend,is_loading_unit
 from mri_rl.experiment_contract import load_contract
 cache={'dir':original['derived_cache']['cache_dir']};gate=resource_gate(a.output_directory)
 def before(name):gate.before_unit(name,loading=is_loading_unit(name,cache))
 contract=load_contract(Path(original['prompt_contract_path']))
 if a.mode=='train':
  result=paired.run_pilot(primary=a.primary_root.resolve(),plan=plan,output=a.run_directory,adapter=original['checkpoints']['B28']['path'],model_path=Path(original['model_path']),contract=contract,cache=cache,before_unit=before,authorize=lambda event:early.verify_review(a.review,a.expected_review_sha256,scope),feature_proof=a.feature_completion,feature_proof_sha256=a.feature_completion_sha256,resume=a.resume,max_new_steps=2)
  state=checkpoint_state(a.run_directory,plan,paired);require(result['status']=='SCHEDULE_COMPLETE' and state['cursor']==state['optimizer_updates']==2,'Two-update pilot not complete')
  save(a.output_directory/'train_complete_SAFE.json',{'status':'ACTUAL_TWO_UPDATE_PAIRED_PILOT_COMPLETE_PENDING_ACCURACY','scope_sha256':canon(scope),'checkpoint':state,'result':result,'evaluation_contract_sha256':sha(a.eval_contract),'science_claim':False});print(json.dumps({'status':'ACTUAL_TWO_UPDATE_PAIRED_PILOT_COMPLETE_PENDING_ACCURACY','checkpoint_receipt_sha256':state['receipt_sha256']}));return
 lockpath=Path(api.MANIFEST).parent/'remaining20.lock';a.output_directory.mkdir(parents=True,exist_ok=True)
 with lockpath.open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  for name,expected in scope['starting_outputs_sha256'].items():require((sha(a.output_directory/name) if (a.output_directory/name).exists() else None)==expected,'Evaluation output changed after scoped review')
  from mri_rl.likelihood_cache import verify_cache_identity
  certpath=a.primary_root.resolve()/'results/EXACT_QUANT_CACHE_CERTIFICATE.json';cert=load(certpath);model=Path(original['model_path']);artifacts={str(model/n):sha(model/n) for n in set(cert['non_weight_files_identical'])|{'config.json'}}
  require(verify_cache_identity(cache['dir'],certpath,original['derived_cache']['certificate_sha256'],artifacts)==original['derived_cache'],'Common effective model cache differs')
  greedy_path=a.output_directory/'greedy_PRIVATE.jsonl';candidate_path=a.output_directory/'candidates_PRIVATE.jsonl';grows=durable_prefix(greedy_path,scope['run_identity'],evaluation['greedy_records_PRIVATE']);crows=durable_prefix(candidate_path,scope['run_identity'],evaluation['candidate_records_PRIVATE'],api.validate_numeric_route);backend=None;released=False
  if len(grows)==40 and len(crows)==25 and (a.output_directory/'eval_complete_SAFE.json').exists():
   complete=load(a.output_directory/'eval_complete_SAFE.json');validate_completion(complete,scope,evaluation,greedy_path,candidate_path);print(json.dumps(complete));return
  prefix=read_rows(evaluation['B28_prefix_path']);saved,_=api.saved_predictions('B28')
  try:
   gate.before_unit('paired_eval_before_native',loading=True)
   backend=NativeGSPOBackend(scope['adapter']['path'],{'group_size':2,'max_tokens':256},model_path=model,contract=contract,seed=7,before_unit=before,cache=cache)
   def append(path,row):
    with path.open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
   for record in evaluation['greedy_records_PRIVATE'][len(grows):]:
    generated={'response':saved[record['record_id']]['response'],'reused_saved_common_B28_greedy':True} if a.checkpoint=='B28' else backend.generate_record(record,record['image_sha256'],256)
    append(greedy_path,{**record,**generated,'run_identity':scope['run_identity'],'weight_updates':0})
   for index,record in enumerate(evaluation['candidate_records_PRIVATE'][len(crows):],len(crows)):
    reuse=evaluation['B28_reused_candidate_map_PRIVATE'].get(record['record_id']) if a.checkpoint=='B28' else None
    if reuse:
     old=prefix[reuse['index']];row={**old,**record,'reuse_provenance':{'prefix_sha256':evaluation['B28_prefix_sha256'],**reuse,'original_run_identity':old['run_identity']}}
    else:row=api.score_record(backend,record,index)
    if record['condition']=='no_image':row['response']=prefix[reuse['index']]['response'] if reuse else backend.generate_record(record,None,256)['response']
    row.update(run_identity=scope['run_identity'],weight_updates=0);api.validate_numeric_route(row,backend.processor.tokenizer);append(candidate_path,row);gc.collect();backend.mx.clear_cache();before('paired_eval_record_saved')
   disk=backend.mx.load(scope['adapter']['path']);actual=dict(backend.tree_flatten(backend.model.trainable_parameters()));require(len(actual)==504 and set(actual)==set(disk) and all(bool(backend.mx.array_equal(v,actual[k]).item()) for k,v in disk.items()) and int(backend.optimizer_state()['step'].item())==0,'Read-only evaluation weights/optimizer changed');del disk,actual
  finally:
   if backend is not None:backend.close();gc.collect();backend.mx.clear_cache();released=backend.model is None and backend.mx.get_active_memory()<128*1024**2
  require(released,'Evaluation native release not verified')
  receipt={'status':'ACTUAL_FIXED40_GREEDY_AND25_SEQUENCE_CANDIDATES_COMPLETE','checkpoint':a.checkpoint,'scope_sha256':canon(scope),'run_identity':scope['run_identity'],'adapter_sha256':scope['adapter']['sha256'],'greedy_records':40,'candidate_records':25,'greedy_sha256':sha(greedy_path),'candidate_sha256':sha(candidate_path),'native_release_verified':True,'weight_updates':0,'evaluation_contract_sha256':sha(a.eval_contract)};save(a.output_directory/'eval_complete_SAFE.json',receipt);print(json.dumps(receipt,sort_keys=True))

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('primary-root','pilot-plan','run-directory','eval-contract','output-directory'):p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--mode',choices=['train','eval'],required=True);p.add_argument('--checkpoint',choices=['B28','post2'],default='post2');p.add_argument('--resume',action='store_true')
 for name in ('feature-plan','feature-completion','scope-output','review'):p.add_argument('--'+name,type=Path)
 for name in ('feature-plan-sha256','feature-completion-sha256','expected-review-sha256'):p.add_argument('--'+name)
 g=p.add_mutually_exclusive_group(required=True);g.add_argument('--prepare-only',action='store_true');g.add_argument('--execute',action='store_true');a=p.parse_args();values=prepare(a)
 if a.prepare_only:require(a.scope_output is not None,'Scope output required');save(a.scope_output,values[-1]);print(json.dumps({'status':'SCOPE_PREPARED_NOT_RUN' if values[-1]['ready'] else 'CONDITIONAL_PREVIEW_NOT_READY','scope_sha256':sha(a.scope_output),'ready':values[-1]['ready']}));return
 execute(a,*values)
if __name__=='__main__':main()
