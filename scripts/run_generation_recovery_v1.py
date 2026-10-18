"""Finite read-only Adam60 generation controls. Preparation reads receipts/config/source only; no weights or cache shards."""
from __future__ import annotations
import argparse,fcntl,gc,hashlib,importlib.util,json,math,os,sys,time,traceback
from pathlib import Path
W=Path(__file__).resolve().parents[1]
_helper_spec=importlib.util.spec_from_file_location('generation_recovery_pure_helpers',W/'src/mri_rl/generation_recovery_v1.py')
controls=importlib.util.module_from_spec(_helper_spec);_helper_spec.loader.exec_module(controls)
require=controls.require;sha=controls.sha;canon=controls.canon
ENTRY_SHA='d0202667c4a580c4ea34cdd22a31a2238fd2de8ad81f26277600f351b5dd152c'
EVAL_SHA='8c52a40f572393f4a01c6fbd2eb84d044bc14ecd151ae00ef56d9e7600a2539b'
BASE_SCOPE_SHA='860fde061f6dfe106878b1c5bfb50850ccae541460afbf4bb17c2e8fec060a6f'
PROOF_SHA='9dcad56048539502c09b3f0307db2f4412405a045da0fe4912b9596d31eed6e0'
ADAPTER_SHA='1741ff2feb2f569cfe062b92cb79b62f61f83463e6683a928d2c458a5f547310'
RECEIPT_SHA='3c2567723e7bc90a65464c8c3645f49ad91c38a0ad886961c508bc1f8a0e3879'
load=lambda p:json.loads(Path(p).read_text())
def module(path,name,expected):
 require(sha(path)==expected,'Reviewed dependency changed');s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def dependencies(primary):
 v=Path(primary)/'.worktrees/vero-mri-v2-controls';entry=module(v/'scripts/run_paired_discrimination_pilot_v2.py','recovery_frozen_evaluator',ENTRY_SHA);early=module(v/'scripts/run_c40_early_visual_probe_v1.py','recovery_native_loader','8cafacbf3a9086160c8e257673c6f933070df208bffc5c32dec38bc0e1e82e06');return entry,early,early.primary_api(Path(primary))

def timed_row(call,field):
 start=time.monotonic();row=call();elapsed=time.monotonic()-start
 require(math.isfinite(elapsed) and elapsed>=0,"Invalid monotonic duration")
 return {**row,field:elapsed}

def files(output):
 result=controls.output_paths(output);result['completion']=Path(output)/'complete_SAFE.json';result['failure']=Path(output)/'failure_PRIVATE.json';return result

def prepare(primary,output):
 primary=Path(primary).resolve();output=Path(output).resolve();require(output.is_relative_to(W/'local_data'),'Private recovery output required')
 entry,early,api=dependencies(primary);basepath=W/'local_data/uniform_post_eval_v1/scope_READY_REQUIRES_REVIEW_PRIVATE.json';require(sha(basepath)==BASE_SCOPE_SHA,'Prespecified Adam60 standard evaluation scope changed');base=load(basepath);proofpath=W/'local_data/uniform_balance_continuation_v1/independent_continuation_CPU_proof_PRIVATE.json';require(sha(proofpath)==PROOF_SHA,'Actual Adam60 proof changed');proof=load(proofpath);child=base['uniform_child'];receiptpath=Path(child['receipt_path']);require(sha(receiptpath)==RECEIPT_SHA and proof['terminal_boundary_receipt_sha256']==RECEIPT_SHA and proof['final_adapter_sha256']==ADAPTER_SHA==base['adapter']['sha256'] and proof['absolute_optimizer_updates']==60 and proof['verdict']=='PASS_WITH_LIMITATIONS','Exact independently verified Adam60 required')
 # Deliberately no adapter/shard hashing during preparation while original SFT may be active.
 evaluation_path=Path(base['evaluation_contract_path']);require(sha(evaluation_path)==EVAL_SHA,'Frozen40/25 evaluation contract changed');evaluation=load(evaluation_path);plans=controls.control_records(evaluation)
 sources={'runner':sha(__file__),'helper':sha(W/'src/mri_rl/generation_recovery_v1.py'),'entry':ENTRY_SHA,'loader':sha(Path(early.__file__)),**api.source_hashes()};require(base['native_sources']==api.source_hashes(),'Baseline native source closure changed')
 baseline_root=Path(base['output_directory']);baseline={'scope':{'path':str(basepath),'sha256':BASE_SCOPE_SHA}}
 for key,name in (('eval_receipt','eval_complete_SAFE.json'),('exit_receipt','actual_exit_PRIVATE.json'),('closed_log','native_stdout_PRIVATE.log'),('greedy_output','greedy_PRIVATE.jsonl'),('candidate_output','candidates_PRIVATE.jsonl')):
  path=baseline_root/name;baseline[key]={'path':str(path),'sha256':sha(path) if path.exists() else None}
 ready=all(v['sha256'] is not None for v in baseline.values())
 if ready:
  complete=load(baseline['eval_receipt']['path']);exitw=load(baseline['exit_receipt']['path']);require(exitw['exit_code']==0 and exitw['raw_log_sha256']==baseline['closed_log']['sha256'] and exitw['scope_file_sha256']==BASE_SCOPE_SHA,'Actual Adam60 baseline exit/log differs');entry.validate_completion(complete,base,evaluation,Path(baseline['greedy_output']['path']),Path(baseline['candidate_output']['path']))
 scope={'kind':'generation_recovery_inference_v1','execution_tier':'B','ready':ready,'checkpoint':'actual_uniform_Adam60','adapter':base['adapter'],'Adam60_proof_sha256':PROOF_SHA,'Adam60_receipt_sha256':RECEIPT_SHA,'evaluation_contract_sha256':EVAL_SHA,'evaluation_contract_path':str(evaluation_path),'baseline_evidence':baseline,'derived_cache':base['derived_cache'],'native_sources':sources,'output_directory':str(output),'controls':{'original_512':{'max_tokens':512,'prompt_change':False,'greedy_real':40,'greedy_null':1,'candidates':'EXACT_ADAM60_BASELINE_REFERENCE'},'short_reason_256':{'max_tokens':256,'instruction_append':controls.SHORT_INSTRUCTION,'greedy_real':40,'greedy_null':1,'new_candidate_records':25,'candidate_forwards':75}},'records_sha256':canon(plans),'temperature':0,'strict_parser':'UNCHANGED_FULL_VERO_TAGS_ONE_BOX','prefix_oracle_tolerance':.002,'maximum_new_generated_tokens':41*512+41*256,'maximum_new_candidate_forwards':75,'optimizer_updates':0,'backward_calls':0,'resource_policy':__import__('mri_rl.paced_resources',fromlist=['PACED_POLICY']).PACED_POLICY,'global_lock':str(Path(api.MANIFEST).parent/'remaining20.lock')}
 scope['run_identity']=canon(scope);paths=files(output);scope['starting_outputs_sha256']={k:sha(v) if v.exists() else None for k,v in paths.items()}
 if paths['failure'].exists():require(False,'Prior failed configuration requires a separate explicit review; no blind retry')
 for name,plan in plans.items():
  key='original' if name=='original_512' else 'short';controls.durable_prefix(paths[key+'_greedy'],plan['greedy'],scope['run_identity'],lambda r,c=plan['max_tokens']:controls.validate_greedy(r,c));null=next(r for r in plan['candidates'] if r['condition']=='no_image');controls.durable_prefix(paths[key+'_null'],[null],scope['run_identity'],lambda r,c=plan['max_tokens']:controls.validate_greedy(r,c))
  if name=='short_reason_256':controls.durable_prefix(paths['short_candidates'],plan['candidates'],scope['run_identity'],api.validate_numeric_route)
 if paths['completion'].exists():controls.validate_completion(load(paths['completion']),scope,{k:v for k,v in paths.items() if k not in {'completion','failure'}})
 return scope,plans,(entry,early,api,base,evaluation)

def execute(primary,output,review,review_sha):
 scope,plans,modules=prepare(primary,output);require(scope['ready'],'Actual complete Adam60 baseline is required');entry,early,api,base,evaluation=modules;early.verify_review(review,review_sha,scope)
 with Path(scope['global_lock']).open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fresh,plans,modules=prepare(primary,output);require(fresh==scope,'Reviewed inputs/progress changed');paths=files(output)
  if paths['completion'].exists():return load(paths['completion'])
  output=Path(output);output.mkdir(parents=True,exist_ok=True)
  from mri_rl.paced_resources import PacedGate
  from mri_rl.mlx_continuous_gspo import NativeGSPOBackend,is_loading_unit
  from mri_rl.experiment_contract import load_contract
  from mri_rl.likelihood_cache import verify_cache_identity
  from mri_rl.resource_supervisor import sample_macos
  gate=PacedGate(output/'resources_PRIVATE.jsonl');cache={'dir':scope['derived_cache']['cache_dir']}
  def before(name):gate.before_unit(name,loading=is_loading_unit(name,cache))
  before('generation_recovery_before_native');require(sha(scope['adapter']['path'])==ADAPTER_SHA,'Adam60 weights changed before execution');certpath=Path(primary)/'results/EXACT_QUANT_CACHE_CERTIFICATE.json';cert=load(certpath);model=Path(base['primary_root'])/'local_data/models/qwen3vl8b-base';artifacts={str(model/n):sha(model/n) for n in set(cert['non_weight_files_identical'])|{'config.json'}};require(verify_cache_identity(cache['dir'],certpath,scope['derived_cache']['certificate_sha256'],artifacts)==scope['derived_cache'],'Certified effective model cache differs')
  original=load(Path(primary)/'local_data/c_two_task_pool_v1/u7_visual_controls_v1/probe_plan_PRIVATE.json');backend=None;released=False;error=None
  try:
   backend=NativeGSPOBackend(scope['adapter']['path'],{'group_size':2,'max_tokens':512},model_path=model,contract=load_contract(Path(original['prompt_contract_path'])),seed=7,before_unit=before,cache=cache)
   for name,plan in plans.items():
    key='original' if name=='original_512' else 'short';cap=plan['max_tokens'];grows=controls.durable_prefix(paths[key+'_greedy'],plan['greedy'],scope['run_identity'],lambda r:controls.validate_greedy(r,cap))
    for record in plan['greedy'][len(grows):]:
     generated=timed_row(lambda:backend.generate_record(record,record['image_sha256'],cap),'generation_wall_seconds');controls.validate_greedy(generated,cap);controls.append(paths[key+'_greedy'],{**record,**generated,'run_identity':scope['run_identity'],'weight_updates':0});before('recovery_greedy_saved')
    null=next(r for r in plan['candidates'] if r['condition']=='no_image');nrows=controls.durable_prefix(paths[key+'_null'],[null],scope['run_identity'],lambda r:controls.validate_greedy(r,cap))
    if not nrows:
     generated=timed_row(lambda:backend.generate_record(null,None,cap),'generation_wall_seconds');controls.validate_greedy(generated,cap);controls.append(paths[key+'_null'],{**null,**generated,'run_identity':scope['run_identity'],'weight_updates':0});before('recovery_null_saved')
    if name=='short_reason_256':
     crows=controls.durable_prefix(paths['short_candidates'],plan['candidates'],scope['run_identity'],api.validate_numeric_route)
     for index,record in enumerate(plan['candidates'][len(crows):],len(crows)):
      row=timed_row(lambda:api.score_record(backend,record,index),'candidate_wall_seconds');row.update(run_identity=scope['run_identity'],weight_updates=0);api.validate_numeric_route(row,backend.processor.tokenizer);controls.append(paths['short_candidates'],row);gc.collect();backend.mx.clear_cache();before('recovery_candidate_saved')
   disk=backend.mx.load(scope['adapter']['path']);actual=dict(backend.tree_flatten(backend.model.trainable_parameters()));require(len(actual)==504 and set(disk)==set(actual) and all(bool(backend.mx.array_equal(v,actual[k]).item()) for k,v in disk.items()) and int(backend.optimizer_state()['step'].item())==0,'Read-only recovery changed weights/optimizer');del disk,actual
  except BaseException as exc:
   error={'exception_type':type(exc).__name__,'message_sha256':hashlib.sha256(str(exc).encode()).hexdigest()};traceback.clear_frames(exc.__traceback__);exc.__traceback__=None
  finally:
   if backend is not None:backend.close();gc.collect();backend.mx.clear_cache();released=backend.model is None and backend.mx.get_active_memory()<128*1024**2
  if error is not None or not released:
   entry.save(paths['failure'],{'status':'FAILED_PRESERVED_OUTPUTS','error':error,'native_release_verified':released,'run_identity':scope['run_identity']});raise RuntimeError('Recovery incomplete; preserved failure receipt, no automatic retry')
  receipt={'status':'ACTUAL_GENERATION_RECOVERY_CONTROLS_COMPLETE','run_identity':scope['run_identity'],'scope_sha256':canon(scope),'adapter_sha256':ADAPTER_SHA,'baseline_scope_sha256':BASE_SCOPE_SHA,'evaluation_contract_sha256':EVAL_SHA,'counts':controls.COUNTS,'outputs_sha256':{k:sha(v) for k,v in paths.items() if k not in {'completion','failure'}},'native_release_verified':True,'weight_updates':0,'backward_calls':0};entry.save(paths['completion'],receipt);return receipt

def main():
 p=argparse.ArgumentParser();p.add_argument('--primary-root',type=Path,required=True);p.add_argument('--output-directory',type=Path,required=True);p.add_argument('--scope-output',type=Path);p.add_argument('--review',type=Path);p.add_argument('--expected-review-sha256');p.add_argument('--execute',action='store_true');a=p.parse_args();sys.path.insert(0,str(a.primary_root/'src'))
 if not a.execute:
  scope,plans,_=prepare(a.primary_root,a.output_directory);require(a.scope_output is not None and a.scope_output.resolve().is_relative_to(W/'local_data'),'Private scope output required')
  with a.scope_output.open('x') as f:json.dump(scope,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
  print(json.dumps({'status':'READY_FOR_REVIEW_NOT_RUN' if scope['ready'] else 'NOT_READY_ACTUAL_ADAM60_BASELINE_PENDING','scope_file_sha256':sha(a.scope_output),'ready':scope['ready'],'native_execution':False}));return
 print(json.dumps(execute(a.primary_root,a.output_directory,a.review,a.expected_review_sha256),sort_keys=True))
if __name__=='__main__':main()
