"""Fresh combined Stage2 -> Stage3 -> Stage4 authority. CPU prepare is inert.

Stage4 in THIS version means DEV24 greedy-only, not the older Small-Pilot stage.
No consumed cache grant, old compound scope, or CPU result can authorize this run.
Each native stage is one child with a durable exit/release gate before progression.
"""
from __future__ import annotations
import argparse,json,os,time
from datetime import datetime,timezone
from pathlib import Path
from mri_rl import accuracy_recovery_v1 as a
from mri_rl import safe_micro_resources_v1 as resources
from mri_rl.safe_micro_entry_v1 import _closure

ROOT=Path(__file__).resolve().parents[2]
SCHEMA='safe-micro-pilot-stages234-scope-v1'
KIND='FRESH_EXACT_MODEL_LOAD_MICRO_AND_DEV24_GREEDY_SCOPE'
ACTION='safe_micro_pilot_v1_load_forward_then_joint_AB_one_update_then_DEV24_greedy'
IMPLEMENTED_BY='Codex/root'
PARENT=dict(a.FROZEN_PARENT)
MODEL_POLICY=dict(resources.MODEL_POLICY)
PHASES={2:'SINGLE_LOAD_FORWARD',3:'MICRO_TRAIN',4:'DEV24_EVAL'}
BUDGETS={
 '2':{'generated_response_tokens':512,'generated_responses':2,'scorer_forwards':6,'teacherforced_input_positions':2808,'backward_calls':0,'optimizer_updates':0,'model_loads':1,'cache_quantizations':0,'window_seconds':900},
 '3':{'generated_response_tokens':4096,'generated_responses':16,'scorer_forwards':48,'teacherforced_input_positions':27072,'backward_calls':32,'optimizer_updates':2,'optimizer_updates_per_arm':1,'model_loads':1,'cache_quantizations':0,'window_seconds':1800},
 '4':{'generated_response_tokens':18432,'generated_responses':72,'greedy_calls':72,'candidate_calls':0,'scorer_forwards':0,'teacherforced_input_positions':0,'backward_calls':0,'optimizer_updates':0,'model_loads':3,'cache_quantizations':0,'window_seconds':2400}}
STAGE1_FILE='49dad3571308ed23af9efb1cec6fd78cbeb8e66e7760a724953fb3316d332419'
SHARDS={'model-00001-of-00002.safetensors':'6a332d4977c95fd6b51ac7155e039016d90041da1bf3cdd3e500192ea00fa626','model-00002-of-00002.safetensors':'2c94dc6761ab4641a4e2c612eb3ff5fd62d939c5704f8301025e741650b9e1e8'}
SPEC_KEYS=('runtime_spec','micro_input','dev24','cache_certificate','cache_stage_complete','cache_exit','cache_actual_review','cache_scope','cache_hash_verification')
require=a.require;H=a.file_sha256;digest=a.scope_digest
_ISSUER=object()


def load(path):
 require(not any(x in str(path).lower() for x in ('final100','holdout16','new16')),'Forbidden development input')
 return json.loads(Path(path).read_text())


def metadata_hash(path):
 require(Path(path).suffix not in ('.safetensors','.png','.gz','.npy','.npz','.bin'),'Metadata/source only during preparation')
 return H(path)


def _output(path):
 root=a.canonical_project_root();p=Path(path);base=root/'local_data/safe_micro_v1'
 require(p.is_absolute() and p==p.resolve() and p.is_relative_to(base) and p!=base,'Canonical private pilot output required')
 require(not base.is_symlink() and not (root/'local_data').is_symlink(),'No private root symlinks')
 return p


def _validate_inputs(inputs,files):
 runtime,micro,dev=inputs['runtime_spec'],inputs['micro_input'],inputs['dev24']
 parent=Path(runtime['parent_checkpoint']);rp=parent/'receipt.json';files[str(rp)]=metadata_hash(rp)
 require(files[str(rp)]==PARENT['receipt_sha256'] and runtime['parent_receipt_sha256']==PARENT['receipt_sha256'],'Original Adam52 receipt required')
 receipt=load(rp);require(receipt['completed_groups']==114 and receipt['optimizer_updates']==52,'Wrong original parent counters')
 for name in ('adapter_config.json','optimizer/manifest.json'):
  p=parent/name;files[str(p)]=metadata_hash(p);require(files[str(p)]==receipt['files_sha256'][name],'Parent metadata changed')
 for name in ('contract','calibration'):
  p=Path(runtime[name]);files[str(p)]=metadata_hash(p);require(files[str(p)]==runtime['file_sha256'][name],'Runtime metadata changed')
 source=Path(micro['source_stage_input_PRIVATE']);files[str(source)]=metadata_hash(source)
 require(files[str(source)]==micro['source_stage_input_sha256'],'Frozen source schedule changed')
 original=load(source)
 require(micro['units_PRIVATE']==original['units_PRIVATE'][:2] and micro['planned_image_bindings_PRIVATE']==original['planned_image_bindings_PRIVATE'][:4],'First complete round only')
 from mri_rl.safe_micro_training_v1 import check_units,micro_budget
 check_units(micro['units_PRIVATE']);require(micro['prompt_lengths']=={'mri_sequence':213,'mri_visit_correspondence':405},'Task prompt caps changed')
 budget=micro_budget(micro['units_PRIVATE'],micro['prompt_lengths'])['total']
 require(budget['generated_response_tokens']==4096 and budget['teacherforced_input_positions']==27072 and budget['optimizer_updates']==2,'Micro budget mismatch')
 rows=[c['row'] for u in micro['units_PRIVATE'] for c in u['contexts']]
 require(len(rows)==4 and all(x['extra_info']['internal_split']=='train' for x in rows),'Exactly four TRAIN contexts required')
 seq=rows[:2];require(len({x['extra_info']['patient_key'] for x in seq})==1,'One paired TRAIN patient required')
 evsrc=Path(dev['source_EVAL40_PRIVATE']);files[str(evsrc)]=metadata_hash(evsrc);require(files[str(evsrc)]==dev['source_EVAL40_sha256'],'Frozen evaluation source changed')
 allseq=[x for x in load(evsrc)['records_PRIVATE'] if x['condition']=='real_image' and x['row_PRIVATE']['ability']=='mri_sequence']
 require(dev['records_PRIVATE']==allseq and len(allseq)==24,'All24 existing real sequence DEV rows required')
 from collections import Counter
 labels=Counter(x['row_PRIVATE']['reward_model']['ground_truth'] for x in allseq)
 require(sorted(labels.values())==[12,12],'Balanced12+12 required')
 def people(records):return {x['extra_info']['patient_key'] for x in records}
 require(len(people([x['row_PRIVATE'] for x in allseq]))==4 and not people(seq)&people([x['row_PRIVATE'] for x in allseq]),'TRAIN/DEV patient membership must be disjoint')
 require(all(x['row_PRIVATE']['extra_info']['internal_split']=='internal_dev' for x in allseq),'Only existing DEV allowed')
 cert=inputs['cache_certificate'];require(cert['units']==750 and cert['tensors_checked']==1438 and {n:v['sha256'] for n,v in cert['shards'].items()}==SHARDS,'Historical cache certificate differs')
 require(runtime['cache_dir']==str(Path(runtime['cache_certificate']).parent/'cache'),'Exact certified cache directory required')
 ex=inputs['cache_exit'];require(ex['native_exit_code']==0 and ex['child_reaped'] and ex['scope_file_sha256']==STAGE1_FILE,'Actual cache child exit0 required')
 require(inputs['cache_stage_complete']['status']=='PHASE_A_CACHE_ONLY_COMPLETE_STOP','Actual cache release return required')
 require(inputs['cache_actual_review']['verdict'] in ('PASS','PASS_WITH_LIMITATIONS'),'Actual cache independent review required')
 return {'sequence_patients':1,'sequence_images':2,'Dallas_rows':2,'DEV_patients':4,'DEV_sequence_rows':24,'DEV_class_counts':[12,12]}


def bind_runtime_sources(sources,runtime):
 pins=runtime.get('runtime_source_sha256');require(isinstance(pins,dict) and pins,'Frozen numerical runtime sources required')
 result=dict(sources)
 for path,h in pins.items():
  require(Path(path).is_absolute() and a.sha(h) and metadata_hash(path)==h,'Frozen F32 runtime source changed')
  require(path not in result or result[path]==h,'Conflicting runtime source binding');result[path]=h
 return result


def prepare(specpath,*,allow_started=False):
 spec=load(specpath);require(spec.get('schema')=='safe-micro-pilot-stages234-spec-v1','Exact combined spec required')
 inputs={};files={str(Path(specpath).resolve()):metadata_hash(specpath)}
 require(set(spec['file_sha256'])==set(SPEC_KEYS),'Exact metadata inventory required')
 for key in SPEC_KEYS:
  p=Path(spec[key]);h=metadata_hash(p);require(h==spec['file_sha256'][key],'Bound input changed: '+key);files[str(p)]=h;inputs[key]=load(p)
 require(inputs['runtime_spec']['cache_certificate']==spec['cache_certificate'],'Runtime cache certificate misbound')
 exposure=_validate_inputs(inputs,files)
 for key in ('cache_stage_complete','cache_exit'):
  # Closed native logs remain pinned by the exit witness and are read under execution; metadata only here.
  require(Path(spec[key]).is_file(),'Missing cache witness')
 out=_output(spec['output']);require(allow_started or not out.exists(),'Fresh combined output required; no rerun')
 sources=_closure({str(ROOT/p):metadata_hash(ROOT/p) for p in ('src/mri_rl/safe_micro_pilot_entry_v1.py','src/mri_rl/safe_micro_pilot_native_v1.py','scripts/run_safe_micro_pilot_v1.py')})
 sources=bind_runtime_sources(sources,inputs['runtime_spec'])
 # Bind third-party runtime identity from the already authenticated cache scope; execution verifies the installed fingerprint before native import.
 result={'schema':SCHEMA,'scope_kind':KIND,'action':ACTION,'implemented_by':IMPLEMENTED_BY,'source_sha256':sources,'file_sha256':files,'input_sha256':a.identity(files),'output':str(out),'resource_policy':MODEL_POLICY,'budgets':BUDGETS,'parent':PARENT,'stage_order':[2,3,4],'evaluation':'FROZEN_DEV24_GREEDY_ONLY','automatic_retry':False,'automatic_progression_on_actual_PASS_only':True,'exposure':exposure,'global_operation_start_window_seconds':5100,'cache_recovery_operations':0,'comparison':'Single joint step per arm; grouping only differs;0.5sequence+0.5Dallas, no inactive renormalization. DEV24 engineering-only.'}
 return {'status':'PREPARED_PENDING_EXACT_REVIEW_AND_FRESH_HUMAN','executable_scope':result,'scope_sha256':digest(result),'native_calls':0}


class Authorization:
 def __init__(self,issuer,scope,expires,parent=None):
  require(issuer is _ISSUER,'Only exact admission may issue authorization')
  self.scope=json.loads(json.dumps(scope));self.sha=digest(scope);self.expires=expires;self.parent=parent;self.start=time.monotonic() if parent is None else parent.start;self.verified={}
 def check(self,phase):
  require(datetime.now(timezone.utc)<self.expires,'Human permission expired')
  require(time.monotonic()-self.start<5100,'Combined operation-start deadline exhausted')
  require(digest(self.scope)==self.sha,'Exact authorization scope changed')
  for p,h in {**self.scope['source_sha256'],**self.scope['file_sha256']}.items():
   path=Path(p);require(path.is_file(),'Bound file missing');s=path.stat();stamp=(s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_ino,s.st_dev)
   if self.verified.get(p)!=stamp or any(x in phase for x in ('admission','model_load','complete','restore')):
    require(metadata_hash(p)==h,'Bound source/input changed');self.verified[p]=stamp
  return True
 def for_phase(self,stage):
  require(self.parent is None and stage in PHASES,'Only2/3/4 can be derived from compound authority')
  self.check('phase_admission')
  return Authorization(_ISSUER,phase_scope(self.scope,stage),self.expires,self)


def phase_scope(scope,stage):
 require(stage in PHASES,'Only model stages2/3/4 permitted')
 return {**scope,'stage':stage,'compound_scope_sha256':digest(scope),'action':ACTION+'_stage'+str(stage),'execution_stage':PHASES[stage],'resource_stage':'B_MICRO','output':str(Path(scope['output'])/('stage'+str(stage)+'_PRIVATE')),'budget':scope['budgets'][str(stage)],'disk_required_bytes':6*1024**3}


def authorize(scope,human,review,*,consumed):
 require(scope.get('schema')==SCHEMA and scope.get('scope_kind')==KIND and scope.get('action')==ACTION and scope.get('implemented_by')==IMPLEMENTED_BY,'Fresh combined scope required')
 require(scope.get('resource_policy')==MODEL_POLICY and scope.get('budgets')==BUDGETS and scope.get('parent')==PARENT and scope.get('stage_order')==[2,3,4] and scope.get('evaluation')=='FROZEN_DEV24_GREEDY_ONLY','Exact conservative policy/budgets/parent required')
 require(human.get('authorized') is True and human.get('action')==ACTION and human.get('scope_sha256')==digest(scope) and human.get('included_stages')==[2,3,4] and a.sha(human.get('human_instruction_sha256')) and human.get('approved_by'),'Fresh explicit stages2+3+4 human permission required')
 try:
  issued,expires=(datetime.fromisoformat(human[k].replace('Z','+00:00')) for k in ('issued_at_utc','expires_at_utc'));now=datetime.now(timezone.utc)
  valid=issued.tzinfo is not None and expires.tzinfo is not None and issued<=now<expires and (expires-issued).total_seconds()<=86400 and (now-issued).total_seconds()<=1800
 except (KeyError,TypeError,ValueError,AttributeError):valid=False
 require(valid,'Fresh bounded permission times required')
 require(review.get('verdict') in ('PASS','PASS_WITH_LIMITATIONS') and review.get('scope_sha256')==digest(scope) and review.get('source_sha256')==scope['source_sha256'] and review.get('input_sha256')==scope['input_sha256'] and review.get('reviewer_kind') in ('isolated_codex','claude_code') and review.get('reviewer_id') and review['reviewer_id']!=IMPLEMENTED_BY,'Independent exact source/input/scope review required')
 token=Authorization(_ISSUER,scope,expires);token.check('admission');require(callable(consumed),'Once-only ledger required');consumed(human['human_instruction_sha256']);return token


def run_order(authorization,run_phase):
 results={}
 for stage in (2,3,4):
  authorization.check('stage_start');result=run_phase(stage);results[str(stage)]=result
  if result.get('actual_exit_code')!=0 or result.get('status')!='PASS' or result.get('release_verified') is not True:
   return {'status':'STOPPED','stages':results,'automatic_retry':False,'failed_stage':stage}
 return {'status':'COMPLETE','stages':results,'automatic_retry':False}


def _credentials(args,*,launch_record=None,consumed):
 for name in ('scope','human','review'):
  require(metadata_hash(getattr(args,name))==getattr(args,'expected_'+name+'_sha256'),'Exact credential FILE hash changed: '+name)
 scope=load(args.scope);prepared=prepare(args.spec,allow_started=launch_record is not None)
 require(scope==prepared['executable_scope'],'Exact current preparation differs')
 human,review=load(args.human),load(args.review)
 if launch_record is None:return authorize(scope,human,review,consumed=consumed),human
 # Reauthenticate the same exact grant for a later child. Only the root launch,
 # not each later stage, has to begin within the fresh1800s admission interval.
 now=datetime.now(timezone.utc);issued=datetime.fromisoformat(human['issued_at_utc'].replace('Z','+00:00'));started=datetime.fromisoformat(launch_record['started_utc'])
 require(launch_record['scope_sha256']==digest(scope) and launch_record['human_instruction_sha256']==human['human_instruction_sha256'] and issued<=started<=now and (started-issued).total_seconds()<=1800,'Child lacks exact fresh compound launch')
 # authorize() is reused without relaxing public admission: validate its genuine
 # issue time relative to the real launch instant, then enforce actual expiry now.
 token=_authorize_later_child(scope,human,review,consumed,launch_record)
 return token,human


def _authorize_later_child(scope,human,review,consumed,launch):
 # All non-time checks are identical to primary admission; no credentials are
 # modified or invented. This is a separate explicit resumed-phase verifier.
 require(scope.get('schema')==SCHEMA and scope.get('scope_kind')==KIND and scope.get('action')==ACTION and scope.get('implemented_by')==IMPLEMENTED_BY,'Wrong child compound scope')
 require(scope.get('resource_policy')==MODEL_POLICY and scope.get('budgets')==BUDGETS and scope.get('parent')==PARENT and scope.get('stage_order')==[2,3,4] and scope.get('evaluation')=='FROZEN_DEV24_GREEDY_ONLY','Child policy/budget changed')
 require(human.get('authorized') is True and human.get('scope_sha256')==digest(scope) and human.get('action')==ACTION and human.get('included_stages')==[2,3,4] and a.sha(human.get('human_instruction_sha256')) and human.get('approved_by'),'Exact current child grant required')
 issued,expires=(datetime.fromisoformat(human[k].replace('Z','+00:00')) for k in ('issued_at_utc','expires_at_utc'))
 require(issued<=datetime.now(timezone.utc)<expires and 0<(expires-issued).total_seconds()<=86400,'Child grant expired/unbounded')
 require(review.get('verdict') in ('PASS','PASS_WITH_LIMITATIONS') and review.get('scope_sha256')==digest(scope) and review.get('source_sha256')==scope['source_sha256'] and review.get('input_sha256')==scope['input_sha256'] and review.get('reviewer_kind') in ('isolated_codex','claude_code') and review.get('reviewer_id') and review['reviewer_id']!=IMPLEMENTED_BY,'Exact child independent review required')
 token=Authorization(_ISSUER,scope,expires);token.start=launch['started_monotonic'];require(type(token.start) in (int,float) and token.start<=time.monotonic(),'Invalid compound clock');token.check('child_admission');consumed(human['human_instruction_sha256']);return token


def _prior(scope,stage,expected_scope_file_sha256=None):
 if stage==2:return
 from mri_rl.safe_micro_witness_v1 import digest as fdigest
 base=Path(scope['output']);old=stage-1;p=base/('stage'+str(old)+'_PRIVATE')/'terminal_PRIVATE.json';record=load(p)
 previous_scope=phase_scope(scope,old)
 require(record['scope_sha256']==digest(previous_scope) and record['status']=='PASS' and record['release_verified'] is True,'Previous stage not native PASS/released')
 witness=base/('stage'+str(old)+'_exit_witness_PRIVATE');exit=load(witness/'actual_exit_PRIVATE.json')
 require(a.sha(expected_scope_file_sha256) and exit.get('scope_file_sha256')==expected_scope_file_sha256 and exit['native_exit_code']==0 and exit['child_reaped'] and exit['post_child_global_lock'].get('no_open_holders') is True,'Previous exact child scope/exit/release differs')
 for name in ('stdout','stderr'):require(fdigest(witness/(name+'_PRIVATE.log'))==exit[name+'_sha256'],'Previous stage closed log changed')


def execute_child(args):
 stage=args.child_stage;scope=load(args.scope);base=Path(scope['output']);launch=load(base/'launch_identity_PRIVATE.json')
 token,human=_credentials(args,launch_record=launch,consumed=a.consume_once(base/('stage'+str(stage)+'_permission_PRIVATE')))
 _prior(scope,stage,args.expected_scope_sha256);phase=token.for_phase(stage)
 # Installed runtime and global lock are checked before any native import.
 from mri_rl.accuracy_recovery_gpu_cache_v2 import installed_fingerprint
 spec=load(args.spec);cache_scope=load(spec['cache_scope'])
 require(installed_fingerprint()==cache_scope['installed'],'Pinned native runtime differs')
 runtime=load(spec['runtime_spec']);lockpath=Path(runtime['global_lock']);require(lockpath.is_file() and not lockpath.is_symlink(),'Existing global lock required')
 import fcntl
 from mri_rl.shared_rollout_gspo_core_v1 import publish_once
 fd=os.open(lockpath,os.O_RDWR)
 try:
  try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:raise ValueError('Global lock held; no force or retry') from None
  gate=resources.make_resource_gate('B_MICRO',spec,phase.scope,phase);gate('native_model_load_preflight')
  # Fresh disk cache identity plus parent restore are performed under the lock.
  from mri_rl.continuous_rl_pilot import cache_binding_for
  from types import SimpleNamespace
  binding=cache_binding_for(SimpleNamespace(cache_dir=Path(runtime['cache_dir']),cache_certificate=Path(runtime['cache_certificate']),cache_certificate_sha256=H(runtime['cache_certificate']),model=Path(runtime['model'])))
  require({n:v['sha256'] for n,v in binding['shards'].items()}==SHARDS,'Cache shard payload mismatch')
  phase.check('before_native_import')
  from mri_rl.safe_micro_pilot_native_v1 import run_phase
  result=run_phase(stage,spec=spec,scope=phase.scope,authorization=phase,resource_gate=gate)
  record={**result,'scope_sha256':phase.sha,'compound_scope_sha256':token.sha,'stage':stage}
  publish_once(Path(phase.scope['output'])/'terminal_PRIVATE.json',record)
  return 0 if record['status']=='PASS' and record['release_verified'] else 2
 finally:os.close(fd)


def execute_chain(args):
 from dataclasses import asdict
 from mri_rl.safe_micro_witness_v1 import witness_child,inspect_lock
 from mri_rl.resource_supervisor import sample_macos
 from mri_rl.shared_rollout_gspo_core_v1 import publish_once
 import sys
 # Only the pinned native interpreter can dispatch native children. The reviewed
 # CLI offers no arbitrary command/backend, retry, stage-skip or threshold option.
 scope=load(args.scope);base=Path(scope['output']);token,human=_credentials(args,consumed=a.consume_once(base.with_name(base.name+'_permission_PRIVATE')))
 base.mkdir(exist_ok=False);root=a.canonical_project_root();python=root/'work/accuracy-recovery-venv-v1/bin/python'
 require(Path(sys.executable).resolve()==python.resolve(),'Pinned production interpreter required')
 publish_once(base/'launch_identity_PRIVATE.json',{'scope_sha256':token.sha,'human_instruction_sha256':human['human_instruction_sha256'],'started_utc':datetime.now(timezone.utc).isoformat(),'started_monotonic':token.start,'automatic_retry':False})
 spec=load(args.spec);runtime=load(spec['runtime_spec'])
 def run(stage):
  _prior(scope,stage,args.expected_scope_sha256)
  argv=[str(python),'-I',str(ROOT/'scripts/run_safe_micro_pilot_v1.py'),'--child-stage',str(stage)]
  for name in ('spec','scope','human','review'):argv+=['--'+name,str(getattr(args,name))]
  for name in ('scope','human','review'):argv+=['--expected-'+name+'-sha256',getattr(args,'expected_'+name+'_sha256')]
  witness=witness_child(argv,base/('stage'+str(stage)+'_exit_witness_PRIVATE'),scope_sha256=args.expected_scope_sha256,sampler=lambda:asdict(sample_macos()),lock_inspector=lambda:inspect_lock(runtime['global_lock']))
  path=base/('stage'+str(stage)+'_PRIVATE')/'terminal_PRIVATE.json'
  result=load(path) if path.exists() else {'status':'FAIL','release_verified':False}
  expected=token.for_phase(stage)
  require(not path.exists() or result['scope_sha256']==expected.sha,'Child terminal scope mismatch')
  if witness['post_child_global_lock'].get('no_open_holders') is not True:result={**result,'release_verified':False}
  return {**result,'actual_exit_code':witness['native_exit_code'],'exact_child_peak_RSS_GiB':witness['child_peak_RSS_GiB'],'wall_seconds':witness['wall_seconds']}
 result=run_order(token,run);publish_once(base/'compound_complete_PRIVATE.json',result)
 print(json.dumps({'status':result['status'],'automatic_retry':False,'stages_completed':list(result['stages'])}));return 0 if result['status']=='COMPLETE' else 1


def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path,required=True);mode=p.add_mutually_exclusive_group();mode.add_argument('--launch',action='store_true');mode.add_argument('--child-stage',type=int,choices=(2,3,4));p.add_argument('--scope-output',type=Path)
 for name in ('scope','human','review'):p.add_argument('--'+name,type=Path);p.add_argument('--expected-'+name+'-sha256')
 args=p.parse_args(argv)
 if args.launch or args.child_stage:
  require(all(getattr(args,n,None) and getattr(args,'expected_'+n+'_sha256',None) for n in ('scope','human','review')),'Explicit exact execution artifacts required')
  return execute_child(args) if args.child_stage else execute_chain(args)
 result=prepare(args.spec)
 if args.scope_output:
  from mri_rl.shared_rollout_gspo_core_v1 import publish_once
  p=_output(args.scope_output);publish_once(p,result['executable_scope'])
 print(json.dumps({k:v for k,v in result.items() if k!='executable_scope'}));return 0
