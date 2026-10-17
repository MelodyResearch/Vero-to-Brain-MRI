"""Separately versioned EARLY release of the unchanged original10-record probe.

Only eligibility changes: a reviewed natural-arm boundary lease replaces waiting
for all21 stages. Never signal a process, release the controller lock, mutate the
original cursor or redefine numerical scoring. Explicit primary imports only.
"""
from __future__ import annotations
import argparse,fcntl,gc,hashlib,importlib,json,os,subprocess,sys
from pathlib import Path
from types import FunctionType

WORKTREE=Path(__file__).resolve().parents[1]
OLD_PLAN_SHA='732260275ca977ccce1b59ea5856701f77a54b0150899acfc17aee2a6484b0ab'
HELPER_SHA='e29e6c3c13af834efe6705405727ffa1c51f5599ca2dd19f6b9847ee674de94c'
CHECKPOINTS=('B28','U_raw_seed7')
EXECUTION_STATE={'native_started':False,'native_children_released':True}

def require(ok,msg):
 if not ok:raise ValueError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)
def h(x):return hashlib.sha256(canonical(x).encode()).hexdigest()
def save(path,value):
 path=Path(path);content=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
 if path.exists():require(path.read_bytes()==content,'Existing frozen artifact differs; preserve it')
 else:
  path.parent.mkdir(parents=True,exist_ok=True)
  with path.open('xb') as f:f.write(content);f.flush();os.fsync(f.fileno())
  fd=os.open(path.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)

def primary_api(primary):
 primary=Path(primary).resolve();require(sha(primary/'src/mri_rl/c40_visual_probe.py')==HELPER_SHA,'Original numerical helper changed')
 sys.path.insert(0,str(primary/'src'));api=importlib.import_module('mri_rl.c40_visual_probe')
 require(Path(api.__file__).resolve()==primary/'src/mri_rl/c40_visual_probe.py','Wrong worktree/native package imported')
 return api

def early_plan(original,original_path,sources):
 require(len(original['records_PRIVATE'])==10,'Unchanged original10 required')
 return {'schema':'C40-early-natural-boundary-v1','original_plan_path':str(Path(original_path).resolve()),'original_plan_sha256':OLD_PLAN_SHA,
  'records_PRIVATE':original['records_PRIVATE'],'checkpoints':original['checkpoints'],'derived_cache':original['derived_cache'],
  'likelihood':original['likelihood'],'greedy':original['greedy'],'primary_sources':sources,'early_handler_sha256':sha(__file__),
  'eligibility':'Reviewed actual completed-arm boundary with old queue parent exited and inherited original global lock; not all21 complete.',
  'boundary_stage_id':'rl:P_raw:13','checkpoint_order':list(CHECKPOINTS),'records_per_checkpoint':10,'total_candidate_forwards':60,
  'new_null_greedy_generations':4,'saved_real_greedy_generations_reused':16,'weight_updates':0,'sealed_test_used':False}

def process_identity(pid):
 try:
  start=subprocess.run(['ps','-p',str(pid),'-o','lstart='],capture_output=True,text=True,timeout=3)
  if start.returncode or not start.stdout.strip():return None
  cmd=subprocess.run(['ps','-p',str(pid),'-o','command='],capture_output=True,text=True,timeout=3,check=True)
  return {'pid':int(pid),'lstart':start.stdout.strip(),'command':cmd.stdout.strip()}
 except (OSError,subprocess.SubprocessError):return None

def command_base(argv):
 out=[];i=0;omit={'--lease','--expected-lease-sha256','--lease-lock-fd','--review-verdict','--expected-review-sha256'}
 while i<len(argv):
  if argv[i] in omit:require(i+1<len(argv),'Incomplete runtime argument');i+=2
  else:out.append(argv[i]);i+=1
 return out

def normalized_command_base(argv,c,executable):
 lexical=c['original_queue_argv'][0]
 require(executable==lexical,'Actual interpreter is not reviewed lexical virtualenv')
 current=list(argv);require(current and current[0] in {lexical,c['observed_python_executable_path']},'Uncertified interpreter argv0')
 if current[0]==c['observed_python_executable_path']:current[0]=lexical
 return command_base(current)

def validate_contract(path,primary):
 c=load(path)
 require(c['schema']=='natural-arm-boundary-handoff-contract-v1' and c['primary_root']==str(primary) and
  c['boundary_stage_id']=='rl:P_raw:13' and Path(c['handler_source_path']).resolve()==Path(__file__).resolve() and
  c['handler_source_sha256']==sha(__file__),'Boundary contract/handler/source differs')
 for pkey,hkey in (('controller_source_path','controller_source_sha256'),('original_manifest_path','original_manifest_sha256'),
                  ('original_executor_path','original_executor_sha256'),('original_executor_review_path','original_executor_review_sha256')):
  require(sha(c[pkey])==c[hkey],'Original controller/executor/review source changed')
 require(Path(c['queue_lock_path']).resolve()==Path(c['queue_cursor_path']).resolve().parent/'remaining20.lock','Original global lock/cursor path mismatch')
 return c

def validate_lease(path,expected,c,lockpath,fd):
 require(path is not None and Path(path).is_file() and sha(path)==expected,'Exact external runtime lease digest required')
 lease=load(path)
 require(lease.get('schema')=='natural-arm-boundary-handoff-lease-v1' and lease.get('contract_sha256')==c['contract_sha256'] and
  lease.get('controller_source_sha256')==c['controller_source_sha256'] and lease.get('boundary_stage_id')==c['boundary_stage_id'] and
  lease.get('original_parent_exited') is True and lease.get('no_model_children') is True and
  lease.get('original_parent_identity')==c['expected_old_parent_identity'],'Boundary lease/controller/old-parent contract differs')
 identity=lease['controller_identity'];require(os.getppid()==identity['pid'] and process_identity(identity['pid'])=={k:identity[k] for k in ('pid','lstart','command')},'Live controller identity differs')
 require(process_identity(c['expected_old_parent_identity']['pid']) is None,'Original queue parent still exists')
 require(type(fd) is int and fd>=0,'Inherited original global lock descriptor required')
 actual=os.fstat(fd);expected_stat=Path(lockpath).stat();lock=lease['queue_lock']
 require(lock['path']==str(Path(lockpath).resolve()) and (actual.st_dev,actual.st_ino)==(expected_stat.st_dev,expected_stat.st_ino)==(lock['st_dev'],lock['st_ino']),'Inherited descriptor is not original global lock')
 fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB) # Same inherited open description; NEVER unlock the controller's lock.
 require(sha(c['queue_cursor_path'])==lease['queue_cursor_sha256'],'Original cursor changed after handoff')
 manifest=load(c['original_manifest_path']);spec=next(s for s in manifest['stages'] if s['id']==c['boundary_stage_id'])
 terminal=lease['terminal_summary'];exit_path=Path(spec['exit_receipt']);witness=load(exit_path)
 require(sha(exit_path)==lease['arm_exit_receipt_sha256'] and witness['exit_code']==0 and witness['stage_id']==spec['id'] and
  sha(spec['raw_log'])==lease['closed_native_log_sha256']==witness['raw_log_sha256'],'Actual boundary native exit/log differs')
 boundary=Path(spec['output'])/f"boundary_{terminal['completed_groups']:04d}";receipt=load(boundary/'receipt.json')
 require(sha(boundary/'receipt.json')==lease['completed_boundary_receipt_sha256']==terminal['boundary_receipt_sha256'] and
  terminal['full_budget_terminal_verified'] is True and terminal['native_terminal_status'] in {'TOKEN_BUDGET_EXHAUSTED','SCHEDULE_COMPLETE'} and
  receipt['completed_groups']==terminal['completed_groups'] and receipt['optimizer_updates']==terminal['optimizer_updates'],'Actual full-arm terminal boundary differs')
 require(lease['diagnostic_command_base_sha256']==[h(x) for x in c['diagnostic_commands']],'Diagnostic command contract differs')
 current=normalized_command_base(sys.orig_argv,c,sys.executable);require(h(current) in lease['diagnostic_command_base_sha256'],'Current diagnostic command not in reviewed contract')
 return lease

def verify_review(path,expected,scope):
 require(path and Path(path).is_file() and sha(path)==expected,'Independent approval digest required')
 v=load(path);require(v.get('verdict') in {'PASS','PASS_WITH_LIMITATIONS'} and v.get('reviewer_kind') in {'isolated_codex','claude_code'} and
  v.get('reviewer_id') and v.get('reviewer_findings') and all(v.get(k)==value for k,value in scope.items()),'Independent exact EARLY scope not approved')
 require(v['verdict']!='PASS_WITH_LIMITATIONS' or v.get('limitations'),'Approval limitations missing')

def prepare(a):
 primary=a.primary_root.resolve()
 for private in (a.plan,a.output,a.scope_output,a.status_output):
  if private is not None:require(private.resolve().is_relative_to(WORKTREE/'local_data') or private.resolve().is_relative_to(primary/'local_data'),'All early artifacts must remain under ignored private data')
 api=primary_api(primary);original=api.validate_plan(a.original_plan);sources=api.source_hashes()
 plan=early_plan(original,a.original_plan,sources);save(a.plan,plan)
 require(a.checkpoint in CHECKPOINTS and (a.output.resolve().is_relative_to(primary/'local_data') or a.output.resolve().is_relative_to(WORKTREE/'local_data')),'Exact private checkpoint output required')
 adapter=plan['checkpoints'][a.checkpoint];require(sha(adapter['path'])==adapter['sha256'] and sha(Path(adapter['path']).with_name('adapter_config.json'))==adapter['adapter_config_sha256'],'Exact adapter/config changed')
 contract=None
 if a.boundary_contract and a.boundary_contract.is_file():contract=validate_contract(a.boundary_contract,primary)
 identity=h({'early_plan':sha(a.plan),'checkpoint':a.checkpoint,'adapter':adapter,'sources':sources,'boundary_contract':sha(a.boundary_contract) if contract else None})
 fn=api.old.completed_prefix;prefix=FunctionType(fn.__code__,{**fn.__globals__,'validate_numeric_record':api.validate_numeric_route},fn.__name__,fn.__defaults__,fn.__closure__)
 completed=prefix(a.output,plan['records_PRIVATE'],identity);_,saved_sha=api.saved_predictions(a.checkpoint)
 scope={'execution_tier':'B','kind':'C40_early_natural_boundary_likelihood_v1','early_plan_sha256':sha(a.plan),'original_plan_sha256':OLD_PLAN_SHA,
  'checkpoint':a.checkpoint,'adapter':adapter,'run_identity':identity,'sources':sources,'early_handler_sha256':sha(__file__),
  'derived_cache':plan['derived_cache'],'output':str(a.output.resolve()),'existing_output_sha256':sha(a.output) if a.output.exists() else None,
  'completed_records':completed,'new_record_ids':[r['record_id'] for r in plan['records_PRIVATE'][completed:]],'forwards_per_record':3,'max_new_forwards':3*(10-completed),
  'new_greedy_generations':sum(r['condition']=='no_image' for r in plan['records_PRIVATE'][completed:]),'saved_real_image_outputs_sha256':saved_sha,
  'weight_updates':0,'fixed_prefix':plan['likelihood']['prefix'],'fixed_suffix':plan['likelihood']['suffix'],'prefix_tolerance':.002,
  'eligibility':'EARLY reviewed natural-arm boundary; original21 training may remain incomplete','boundary_stage_id':'rl:P_raw:13',
  'boundary_contract_sha256':sha(a.boundary_contract) if contract else None,'boundary_controller_source_sha256':contract['controller_source_sha256'] if contract else None,
  'boundary_contract_ready':bool(contract),'primary_root':str(primary),'status_output':str(a.status_output.resolve()) if a.status_output else None,
  'conditioning_probe':{'enabled':bool(a.conditioning_helper),'helper_path':str(a.conditioning_helper.resolve()) if a.conditioning_helper else None,
   'helper_sha256':sha(a.conditioning_helper) if a.conditioning_helper else None,'record_id':next(r['record_id'] for r in plan['records_PRIVATE'] if r['condition']=='real_image') if a.conditioning_helper else None,
   'extra_generations':1 if a.conditioning_helper else 0,'maximum_generation_tokens':32 if a.conditioning_helper else 0,
   'teacher_forcing_calls':2 if a.conditioning_helper else 0,'backward_calls':2 if a.conditioning_helper else 0,'optimizer_updates':0}}
 return api,original,plan,scope,contract

def execute(a,api,original,plan,scope,contract):
 require(contract is not None and scope['boundary_contract_ready'],'Reviewed static boundary contract required')
 verify_review(a.review_verdict,a.expected_review_sha256,scope)
 c={**contract,'contract_sha256':sha(a.boundary_contract)};validate_lease(a.lease,a.expected_lease_sha256,c,contract['queue_lock_path'],a.lease_lock_fd)
 if scope['completed_records']==10 and not a.conditioning_helper:return {'status':'EARLY_PROBE_ALREADY_COMPLETE','completed':10,'output_sha256':sha(a.output),'checkpoint':a.checkpoint}
 # Only now import native-facing components; source modules themselves import Metal lazily.
 from mri_rl.paced_resources import PacedGate
 from mri_rl.likelihood_cache import verify_cache_identity
 from mri_rl.experiment_contract import load_contract
 a.output.parent.mkdir(parents=True,exist_ok=True);gate=PacedGate(a.output.with_suffix('.resources.jsonl'));gate.before_unit('early_probe_before_native_import',loading=True)
 from mri_rl.mlx_continuous_gspo import NativeGSPOBackend,is_loading_unit
 cache={'dir':original['derived_cache']['cache_dir']};certificate=a.primary_root.resolve()/'results/EXACT_QUANT_CACHE_CERTIFICATE.json';cert=load(certificate);model=Path(original['model_path'])
 artifacts={str(model/name):sha(model/name) for name in set(cert['non_weight_files_identical'])|{'config.json'}}
 require(verify_cache_identity(cache['dir'],certificate,original['derived_cache']['certificate_sha256'],artifacts)==original['derived_cache'],'Certified base cache changed')
 def before(name):gate.before_unit(name,loading=is_loading_unit(name,cache))
 backend=None;completed=scope['completed_records'];saved,_=api.saved_predictions(a.checkpoint)
 try:
  EXECUTION_STATE.update(native_started=True,native_children_released=False)
  backend=NativeGSPOBackend(plan['checkpoints'][a.checkpoint]['path'],{'group_size':2,'max_tokens':256},model_path=model,contract=load_contract(Path(original['prompt_contract_path'])),seed=7,before_unit=before,cache=cache)
  with a.output.open('a' if a.output.exists() else 'x') as handle:
   fd=os.open(a.output.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
   for index,record in enumerate(plan['records_PRIVATE'][completed:],completed):
    result=api.score_record(backend,record,index) # EXACT frozen causal gather/P-1/prefix oracle, never copied.
    if record['condition']=='no_image':result.update(backend.generate_record(record,None,256));result['new_greedy_generation']=True
    else:result['response']=saved[record['source_record_id']]['response'];result['new_greedy_generation']=False
    result.update(run_identity=scope['run_identity'],checkpoint=a.checkpoint,adapter_sha256=plan['checkpoints'][a.checkpoint]['sha256'],early_release=True)
    api.validate_numeric_route(result,backend.processor.tokenizer);handle.write(json.dumps(result,allow_nan=False)+'\n');handle.flush();os.fsync(handle.fileno());completed+=1
    print(json.dumps({'checkpoint':a.checkpoint,'completed':completed,'total':10,'early_release':True,'weight_updates':0}),flush=True)
    gc.collect();backend.mx.clear_cache();before('early_probe_record_completed')
  if a.conditioning_helper:
   require(sha(a.conditioning_helper)==scope['conditioning_probe']['helper_sha256'],'Reviewed conditioning helper changed')
   import importlib.util
   sp=importlib.util.spec_from_file_location('early_conditioning_bound',a.conditioning_helper);module=importlib.util.module_from_spec(sp);sp.loader.exec_module(module)
   record=next(r for r in plan['records_PRIVATE'] if r['record_id']==scope['conditioning_probe']['record_id'])
   module.run_conditioning_probe(backend,record,a.output.with_name(a.output.stem+'_conditioning_PRIVATE'),before)
  disk=backend.mx.load(plan['checkpoints'][a.checkpoint]['path']);actual=dict(backend.tree_flatten(backend.model.trainable_parameters()))
  require(set(disk)==set(actual) and all(bool(backend.mx.array_equal(v,actual[k]).item()) for k,v in disk.items()) and int(backend.optimizer_state()['step'].item())==0,'Read-only adapter/optimizer integrity failed')
 finally:
  if backend is not None:
   backend.close()
   EXECUTION_STATE['native_children_released']=bool(backend.model is None and backend.mx.get_active_memory()<128*1024**2)
 return {'status':'ACTUAL_EARLY_PROBE_COMPLETED','checkpoint':a.checkpoint,'completed':completed,'output_sha256':sha(a.output),'weight_updates':0,'lease_sha256':a.expected_lease_sha256}

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('primary-root','original-plan','plan','output'):p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--checkpoint',choices=CHECKPOINTS,required=True)
 for name in ('boundary-contract','scope-output','lease','review-verdict','status-output','conditioning-helper'):p.add_argument('--'+name,type=Path)
 for name in ('expected-lease-sha256','expected-review-sha256'):p.add_argument('--'+name)
 p.add_argument('--lease-lock-fd',type=int);g=p.add_mutually_exclusive_group(required=True);g.add_argument('--prepare-only',action='store_true');g.add_argument('--execute',action='store_true');a=p.parse_args()
 try:
  require(not a.execute or a.status_output is not None,'Exact private status output required for boundary controller')
  api,original,plan,scope,contract=prepare(a)
  if a.prepare_only:
   require(a.scope_output is not None,'Private scope output required');save(a.scope_output,scope)
   print(json.dumps({'status':'EARLY_SCOPE_PREPARED_NOT_RUN','scope_sha256':h(scope),'scope_file_sha256':sha(a.scope_output),'boundary_contract_ready':bool(contract),'records':10,'forwards':scope['max_new_forwards']}));return
  result=execute(a,api,original,plan,scope,contract)
  require(EXECUTION_STATE['native_children_released'],'Native allocation release not verified')
  result.update(queue_resume_safe=True,critical_safety_condition=False,native_children_released=True)
  save(a.status_output,result);print(json.dumps(result,sort_keys=True))
 except BaseException as exc:
  if a.execute and a.status_output is not None:
   started=EXECUTION_STATE['native_started'];critical=True if 'stopped at safe boundary' in str(exc) else None if started else False
   save(a.status_output,{'status':'EARLY_PROBE_FAILED','checkpoint':a.checkpoint,'failure_type':type(exc).__name__,
    'queue_resume_safe':not started and critical is False,'critical_safety_condition':critical,'native_children_released':EXECUTION_STATE['native_children_released'],
    'native_started':started,'weight_updates':0})
  raise

if __name__=='__main__':main()
