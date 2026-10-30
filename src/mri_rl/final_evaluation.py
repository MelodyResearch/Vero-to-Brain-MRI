"""Final read-only Parent evaluation. Import/validation never imports MLX.

The protocol is fixed; missing private inputs, exact independent scope review or
fresh sealed16 confirmation block execution. A local receipt cannot authenticate
human consent: the operator must verify the real researcher message separately.
No optimizer step, TRAIN rollout, TF scoring, checkpoint write or retry exists here.
"""
from __future__ import annotations
import argparse,collections,fcntl,gc,hashlib,itertools,json,math,os,random,shutil,signal,sys,time
from pathlib import Path
PARENT='c70b426b58191446545ba6333f7595e96f4fdb4cc689ddbaf5206e276b329db6'
LABELS=('FLAIR','contrast-enhanced T1')
BUDGET={'model_loads':1,'greedy_generations':32,'returned_tokens':8192,'teacher_forced_scores':0,'backwards':0,'optimizer_updates':0,'operation_start_seconds':1800}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def require(v,msg):
 if not v:raise ValueError(msg)
def read(p):return json.loads(Path(p).read_text())
def validate_protocol(p):
 require(p.get('schema')=='FINAL_PARENT_SEALED16_V1' and p.get('adapter_sha256')==PARENT,'Fixed Parent identity required')
 require(p.get('budget')==BUDGET,'No training or expanded evaluation permitted')
 require((p.get('patients'),p.get('images'),p.get('prompt'),p.get('numerical_path'),p.get('parser'),p.get('temperature'))==(16,32,'concise','F32','strict',0),'Frozen input/readout required')
 require(p.get('include_all_quality_flags') is True and p.get('post_result_model_selection') is False,'No output-based exclusions or selection')
def validate_rows(rows,excluded):
 require(len(rows)==32,'All32 inputs required')
 people=collections.defaultdict(list)
 for r in rows:
  require(r['truth'] in LABELS and r['patient'] not in excluded,'Labels and patient exclusions required')
  require(len(r['image_sha256'])==64,'Image SHA required');people[r['patient']].append(r['truth'])
 require(len(people)==16 and all(sorted(v)==sorted(LABELS) for v in people.values()),'Exactly16 complete modality pairs required')
 require(len({r['image_sha256'] for r in rows})==32 and len({r['row_PRIVATE']['id'] for r in rows})==32,'Unique images and record identities required')
def summarize(rows,flagged):
 validate_rows(rows,[]);c=[[0]*3 for _ in LABELS];by=collections.defaultdict(list);pairs=collections.defaultdict(list)
 for r in rows:
  t=LABELS.index(r['truth']);p=r['prediction'];p=p if type(p) is int and p in (0,1) else 2
  c[t][p]+=1;by[r['patient']].append(int(t==p));pairs[r['patient']].append(p)
 scores=[sum(v)/len(v) for _,v in sorted(by.items())];rng=random.Random(42)
 boot=sorted(sum(rng.choice(scores) for _ in scores)/len(scores) for _ in range(10000))
 correct=c[0][0]+c[1][1];recall=[c[i][i]/16 for i in (0,1)]
 return {'cases':32,'patients':16,'correct':correct,'accuracy':correct/32,'balanced_accuracy':sum(recall)/2,'class_recall':recall,
  'confusion':c,'confusion_columns':[*LABELS,'INVALID'],'invalid':sum(x[2] for x in c),'invalid_rate':sum(x[2] for x in c)/32,
  'fully_correct_pairs':sum(all(v) for v in by.values()),'constant_label_pairs':sum(len(set(v))==1 and v[0]!=2 for v in pairs.values()),
  'patient_scores_PRIVATE':dict(zip(sorted(by),scores)),'patient_cluster_bootstrap95':[boot[249],boot[9749]],
  'uncertainty':'10000 fixed-seed42 patient-cluster percentile draws, descriptive; no clinical generalization claim',
  'quality_flagged_patients':len(flagged),'all_flagged_images_in_primary_denominator':True,
  'flagged_correct':sum(sum(by[k]) for k in flagged),'flagged_images':2*len(flagged)}
def check_execution_permission(protocol,scope,approval,scope_sha):
 validate_protocol(protocol)
 require(scope.get('protocol_sha256') and scope.get('budget')==BUDGET,'Exact reviewed bounded scope required')
 require(scope.get('inputs_ready') is True and scope.get('sealed_manifest_identity_verified') is True,'Restored original input manifest and isolation verification required')
 require(approval.get('operation')=='FINAL_SEALED16_READONLY' and approval.get('scope_file_sha256')==scope_sha and
  approval.get('adapter_sha256')==PARENT and len(approval.get('researcher_message_sha256',''))==64,'Fresh actual researcher sealed16 confirmation required; flags are not consent')
def inspect_inputs(path):
 v=read(path);rows=v['records_PRIVATE'];validate_rows(rows,v['excluded_patient_keys_PRIVATE'])
 require(len(set(v['excluded_patient_keys_PRIVATE']))==54,'Original54-person exclusion inventory required')
 flags=v['quality_flag_patient_keys_PRIVATE'];require(len(set(flags))==4 and set(flags)<=set(r['patient'] for r in rows),'Original four quality flags required')
 for r in rows:
  row=r['row_PRIVATE'];require(row['extra_info']['patient_key']==r['patient'] and row['reward_model']['ground_truth']==r['truth'] and row['ability']=='mri_sequence','Source/label binding differs')
  require(len(row['images'])==1 and sha(row['images'][0])==r['image_sha256'],'Actual image bytes differ')
 return v

def check_output_binding(scope,output):
 require('output' in scope and Path(output).resolve()==Path(scope['output']).resolve(),'Output differs from reviewed once-only scope')
def create_owned(cls,resident,*args,**kwargs):
 obj=cls.__new__(cls);resident[0]=obj;obj.__init__(*args,**kwargs);return obj
def deadline_sample(deadline,sampler,pid,now=time.time):
 if now()>=deadline:raise TimeoutError('FINAL_SCOPE_DEADLINE_EXHAUSTED; request owned child release')
 return sampler(pid)
def registry(runtime,scope_sha):
 return Path(runtime['global_lock']).resolve().parent/'final_evaluation_claims'/scope_sha

def validate_payload_locations(scope,runtime,keys):
 prefixes={'Parent_checkpoint.':'parent_checkpoint','base_cache.':'cache_dir','original_model_metadata.':'model'}
 for key in keys:
  if key in ('cache_certificate','evaluation_contract'):expected=Path(runtime['cache_certificate' if key=='cache_certificate' else 'contract'])
  else:
   parts=[(prefix,root) for prefix,root in prefixes.items() if key.startswith(prefix)]
   require(len(parts)==1,'Unknown runtime payload name')
   prefix,root=parts[0];expected=Path(runtime[root])/key[len(prefix):]
  require(Path(scope['payload_paths'][key]).resolve()==expected.resolve(),'Hash proof does not bind the actual runtime payload location')

def verify_derivation(protocol,scope,inputs):
 original=scope['original_sealed_files']
 for key,digest in protocol['sealed_source_sha256'].items():require(sha(original[key])==digest,'Original sealed source missing or changed')
 raw=[json.loads(line) for line in Path(original['historical_row_manifest_sha256']).read_text().splitlines() if line.strip()]
 seal=read(original['historical_seal_sha256']);plan=read(original['historical_plan_sha256'])
 require(len(raw)==32 and len(plan['files_PRIVATE'])==32 and seal['manifest_sha256']==protocol['sealed_source_sha256']['historical_row_manifest_sha256'],'Original source coverage differs')
 require(seal['plan_sha256']==protocol['sealed_source_sha256']['historical_plan_sha256'] and seal['model_output_rows']==0 and seal['training_forbidden'] is True,'Sealed original input contract differs')
 by={r['id']:r for r in raw};require(len(by)==32,'Original record duplication')
 def without_paths(row):
  import copy
  v=copy.deepcopy(row);v['images']=['IMAGE_PATH']
  for msg in v.get('prompt',[]):
   for item in msg.get('content',[]):
    if item.get('type')=='image':item['image']='IMAGE_PATH'
  return v
 for r in inputs['records_PRIVATE']:
  row=r['row_PRIVATE'];require(row['id'] in by and without_paths(row)==without_paths(by[row['id']]),'Only image location rebinding is allowed; original row changed')
  require(r['image_sha256']==seal['render_sha256_by_record_PRIVATE'][row['id']],'Original rendered image binding differs')
 require(sha(scope['historical_exclusion_file'])==protocol['historical_exclusion_sha256'],'Original exclusion inventory changed')
 excluded=read(scope['historical_exclusion_file']);expected={hashlib.sha256(x['subject_PRIVATE'].encode()).hexdigest()[:12] for x in excluded['excluded']}
 require(set(inputs['excluded_patient_keys_PRIVATE'])==expected,'Exclusions not derived from original inventory')
 flags=read(scope['original_quality_flag_map']);require(sha(scope['original_quality_flag_map'])==scope['quality_flag_map_sha256'],'Original reviewed quality map changed')
 require(set(flags['patient_keys_PRIVATE'])==set(inputs['quality_flag_patient_keys_PRIVATE']),'Quality flags differ from independently authenticated original map')
 for key,digest in protocol['payload_sha256'].items():require(key in scope['payload_paths'] and sha(scope['payload_paths'][key])==digest,'Parent/model/config/processor/cache payload differs')
 from mri_rl.historical_runtime import verify_source
 verify_source(scope['historical_runtime_root'],scope)
 required={str(q.resolve()) for q in Path(__file__).parent.glob('*.py')}
 required.add(str((Path(__file__).resolve().parents[2]/'results/SIMPLIFICATION_MANIFEST.json').resolve()))
 required|={str((Path(__file__).resolve().parents[2]/x).resolve()) for x in ['configs/final_native_environment_v1.json','scripts/evaluate.py']}
 require(required<=set(scope['file_sha256']),'Complete local native dependency/source pins required')
 from importlib import metadata
 import platform
 env=read(Path(__file__).resolve().parents[2]/'configs/final_native_environment_v1.json')
 require(platform.python_version()==env['python_version'] and platform.machine()==env['machine'],'Native interpreter/platform differs')
 for name,expected_version in env['packages'].items():
  try:actual=metadata.version(name)
  except metadata.PackageNotFoundError:actual=None
  require(actual==expected_version,'Native environment version differs: '+name)

def _validated(args):
 protocol=read(args.protocol);validate_protocol(protocol)
 require(args.inputs and args.scope and args.review and args.approval and args.runtime and args.output,'Inputs, exact scope, independent review, fresh confirmation and runtime required')
 scope=read(args.scope);h=sha(args.scope);review=read(args.review);approval=read(args.approval)
 check_execution_permission(protocol,scope,approval,h)
 require(scope['protocol_sha256']==sha(args.protocol) and scope['input_manifest_sha256']==sha(args.inputs) and scope['runtime_sha256']==sha(args.runtime),'Exact protocol/inputs/runtime changed')
 require(review.get('scope_file_sha256')==h and review.get('reviewer_kind')=='isolated_codex' and review.get('verdict')=='PASS','Independent final execution review required')
 for p,digest in scope['file_sha256'].items():require(sha(p)==digest,'Frozen source/input differs')
 require(scope['resources']==protocol['resources'],'Resource contract changed')
 check_output_binding(scope,args.output)
 inputs=inspect_inputs(args.inputs);runtime=read(args.runtime)
 validate_payload_locations(scope,runtime,protocol['payload_sha256'])
 verify_derivation(protocol,scope,inputs)
 return protocol,scope,h,inputs,runtime

def _native(args,legacy):
 protocol,scope,h,inputs,runtime=_validated(args);out=Path(args.output)
 require(not (out/'native_claim_PRIVATE.json').exists(),'No retry or repeated counting')
 publish_once=legacy.module('shared_rollout_gspo_core_v1').publish_once
 attempt=read(registry(runtime,h).with_suffix('.json'));require(attempt['scope']==h and time.time()<attempt['deadline_unix'],'Original fixed execution window expired')
 publish_once(registry(runtime,h).with_suffix('.native.json'),{'scope':h,'started_unix':time.time(),'model_loads_reserved':1})
 publish_once(out/'native_claim_PRIVATE.json',{'scope':h,'deadline_unix':attempt['deadline_unix'],'model_loads_reserved':1})
 lock=open(runtime['global_lock'],'a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 Controller=legacy.module('safe_micro_dynamic_resources_v1').Controller
 sample_macos=legacy.module('resource_supervisor').sample_macos
 os_pressure_level=legacy.module('accuracy_recovery_gpu_cache_v3').os_pressure_level
 controller=Controller();red=[False];signal.signal(signal.SIGUSR1,lambda *_:red.__setitem__(0,True))
 def gate(name):
  require(time.time()<attempt['deadline_unix'] and not red[0],'Window exhausted or guardian release request')
  s=sample_macos();r=controller.decide(s,os_pressure_level(),shutil.disk_usage(out).free/1024**3,time.monotonic())
  require(s.power_source=='AC','Final read-only scope requires AC; no earlier battery exception reused')
  with (out/'resources_PRIVATE.jsonl').open('a') as f:f.write(json.dumps({'boundary':name,'action':r['action'],'color':r['color'],'rss':s.process_rss_gib,'mlx_peak':s.mlx_peak_gib,'swap':s.swap_used_gib})+'\n')
  require(r['action']=='CONTINUE','Resource boundary HOLD/STOP: '+str(r['reasons']))
 resident=[None]
 try:
  gate('before_native_load')
  PilotV2Backend=legacy.module('accuracy_pilot_v2_native').PilotV2Backend
  generate_one_v2=legacy.module('accuracy_pilot_v2_native').generate_one_v2
  call_limits=legacy.module('accuracy_pilot_v2_native').call_limits
  _configuration=legacy.module('safe_micro_pilot_native_v1')._configuration
  load_contract=legacy.module('experiment_contract').load_contract
  adam52_parent_anchor=legacy.module('accuracy_recovery_v1').adam52_parent_anchor
  parent_disk_snapshot=legacy.module('accuracy_recovery_v1').parent_disk_snapshot
  restore_parent=legacy.module('accuracy_stage_native_v1').restore_parent
  full_identity=legacy.module('safe_micro_recovery_native_v1').full_identity
  CallCounter=legacy.module('safe_micro_pilot_native_v2').CallCounter
  detach_containers=legacy.module('safe_micro_restore_v1').detach_containers
  condition_row=legacy.module('accuracy_pilot_v2_core').condition_row
  strict_label=legacy.module('accuracy_pilot_v2_core').strict_label
  run_with_release=legacy.module('paired_z045_continuation_entry_v1').run_with_release
  parent=Path(runtime['parent_checkpoint']);require(sha(parent/'adapter.safetensors')==PARENT,'Selected Parent changed')
  anchor=adam52_parent_anchor(parent,protocol['parent_receipt_sha256']);disk=parent_disk_snapshot(parent)
  def operation():
   b=create_owned(PilotV2Backend,resident,parent/'adapter.safetensors',_configuration(runtime),model_path=Path(runtime['model']),contract=load_contract(Path(runtime['contract'])),seed=13,before_unit=gate,cache={'dir':runtime['cache_dir']});resident[0]=b
   b.configure_stage(grader=None,expected_images={r['row_PRIVATE']['id']:r['image_sha256'] for r in inputs['records_PRIVATE']},prompt_lengths={'mri_sequence':233},work=out/'unused_rollouts_PRIVATE',binding={'scope':h})
   b.configure_v2(condition='concise',prompt_lengths={'mri_sequence':233});restore_parent(b,parent,anchor)
   before=full_identity(b);rng=detach_containers(b.rng_state(),copy_leaf=b.copy_leaf);counter=CallCounter(call_limits(32),out/'calls_PRIVATE.json');answers=[]
   for i,r in enumerate(inputs['records_PRIVATE']):
    row=condition_row(r['row_PRIVATE'],'concise');a=generate_one_v2(b,row,r['image_sha256'],lengths={'mri_sequence':233},counter=counter,before=gate)
    answer={**r,'prediction':strict_label(a['response']['text'])};publish_once(out/('row_%02d_PRIVATE.json'%i),{'answer_PRIVATE':a,'prediction':answer['prediction']});answers.append(answer)
   random.setstate(rng['python'])
   for i,v in enumerate(rng['mlx']):b.mx.random.state[i][:]=v
   b.mx.eval(b.mx.random.state);require(full_identity(b)==before,'Read-only full model/Adam/RNG integrity failed')
   result={'status':'ACTUAL_FINAL_EVALUATION_COMPLETE','metrics_PRIVATE':summarize(answers,inputs['quality_flag_patient_keys_PRIVATE']),'scope_sha256':h,'calls':counter.used,'optimizer_updates':0}
   del rng;gc.collect();return result
  result=run_with_release(operation,resident,out);resident[0]=None;require(parent_disk_snapshot(parent)==disk,'Parent files changed')
  publish_once(out/'complete_PRIVATE.json',result)
 finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',default=str(Path(__file__).resolve().parents[2]/'configs/final_evaluation_v1.json'))
 for k in ['inputs','scope','review','approval','runtime','output']:p.add_argument('--'+k)
 p.add_argument('--export-runtime',metavar='NEW_DIRECTORY',help='CPU-only exact source export from local pinned Git objects');p.add_argument('--validate-only',action='store_true');p.add_argument('--execute',action='store_true');p.add_argument('--child',action='store_true');a=p.parse_args(argv);started_unix=time.time();legacy=None
 try:
  validate_protocol(read(a.protocol))
  if a.export_runtime:
   require(not a.execute and not a.child,'Source export cannot execute a model')
   from mri_rl.historical_runtime import export_source,COMMIT
   files=export_source(a.export_runtime)
   print(json.dumps({'status':'HISTORICAL_SOURCE_EXPORTED_CPU_ONLY','commit':COMMIT,'files':len(files),'GPU_operations':0}));return 0
  if not a.execute:
   if not a.inputs:print(json.dumps({'status':'NOT_RUN_INPUT_METADATA_NOT_LOCATED_AND_CONFIRMATION_REQUIRED','GPU_operations':0}));return 2
   inspect_inputs(a.inputs);print(json.dumps({'status':'INPUTS_VALID_NOT_EXECUTION_AUTHORIZED','GPU_operations':0}));return 0
  _,validated_scope,_,_,_=_validated(a)
  from mri_rl.historical_runtime import activate
  legacy=activate(validated_scope)
  if a.child:_native(a,legacy);return 0
  witness_child=legacy.module('safe_micro_watchdog_v2').witness_child
  sample_macos=legacy.module('resource_supervisor').sample_macos
  os_pressure_level=legacy.module('accuracy_recovery_gpu_cache_v3').os_pressure_level
  runtime=read(a.runtime);out=Path(a.output);require(not out.exists(),'One-shot output already exists')
  deadline=started_unix+1800;require(time.time()<deadline,'Execution window exhausted during validation')
  publish_once=legacy.module('shared_rollout_gspo_core_v1').publish_once
  claim=registry(runtime,sha(a.scope)).with_suffix('.json');claim.parent.mkdir(parents=True,exist_ok=True)
  publish_once(claim,{'scope':sha(a.scope),'started_unix':started_unix,'deadline_unix':deadline,'output':str(out.resolve())})
  out.mkdir(parents=True);cmd=[sys.executable,'-m','mri_rl.final_evaluation','--execute','--child']
  for k in ['protocol','inputs','scope','review','approval','runtime','output']:cmd+=['--'+k,str(getattr(a,k))]
  def lock_status():
   with open(runtime['global_lock'],'a+b') as f:
    try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:return {'status':'OCCUPIED'}
    fcntl.flock(f,fcntl.LOCK_UN);return {'status':'FREE'}
  r=witness_child(cmd,out/'witness_PRIVATE',scope_sha256=sha(a.scope),sampler=lambda pid:deadline_sample(deadline,sample_macos,pid),level=os_pressure_level,disk_free_gib=lambda:shutil.disk_usage(out).free/1024**3,post_sampler=lambda:{'no_model_in_supervisor':True},lock_inspector=lock_status)
  print(json.dumps(r));return 0 if r['native_exit_code']==0 and not r['guardian_stop'] and r['post_child_global_lock']['status']=='FREE' else 1
 except (ValueError,KeyError,FileNotFoundError) as e:print(json.dumps({'status':'NOT_RUN_OR_STOP','error':str(e)}));return 2
 finally:
  if legacy is not None:legacy.close()
if __name__=='__main__':raise SystemExit(main())
