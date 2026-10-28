"""Fresh Q -> saved-A adoption -> B -> DEV24 recovery authority.

Stage4 in THIS version means DEV24 greedy-only, not the older Small-Pilot stage.
No consumed cache grant, old compound scope, or CPU result can authorize this run.
Each native stage is one child with a durable exit/release gate before progression.
"""
from __future__ import annotations
import argparse,json,os,time
from datetime import datetime,timezone
from pathlib import Path
from mri_rl import accuracy_recovery_v1 as a
from mri_rl import safe_micro_dynamic_resources_v1 as resources
from mri_rl.safe_micro_entry_v1 import _closure
from mri_rl.safe_micro_pilot_entry_v1 import Authorization as HistoricalAuthorization
from mri_rl.safe_micro_watchdog_v2 import POLICY as GUARDIAN_POLICY

ROOT=Path(__file__).resolve().parents[2]
SCHEMA='safe-micro-recovery-QBE-scope-v1'
KIND='FRESH_EXACT_RESTORE_SALVAGE_A_B_DEV24_SCOPE'
ACTION='safe_micro_recovery_v1_restore_qualify_adopt_saved_A_B_once_DEV24_greedy'
IMPLEMENTED_BY='Codex/root'
RECOVERY={'max_automatic_recoveries':2,'max_controller_launches':3,'max_model_load_attempts':5,'historical_loads_spent':2,'lifetime_total_load_cap':7,'wait_samples_before_restart':3,'wait_counts_against_original_deadlines':True,'incomplete_operation':'INDETERMINATE_NO_REPLAY','completed_results':'HASH_VERIFIED_REUSE_ONLY','old_scope_retry':False}
PARENT=dict(a.FROZEN_PARENT)
MODEL_POLICY=dict(resources.POLICY)
PHASES={2:'SINGLE_LOAD_FORWARD',3:'MICRO_TRAIN',4:'DEV24_EVAL'}
ADOPTION={'label':'SALVAGED_FROM_INCOMPLETE_TRANSACTION','adopt_saved_A_after_Q':True,'old_journal_commit_written':False,'no_A_update_repeated':True,'no_shared_rollout_regenerated':True,'B_parent':'ORIGINAL_ADAM52_FROM_DISK','old_fatal_scope_retried':False}
SPENT={'generated_response_tokens':673,'scorer_forwards':30,'teacherforced_input_positions':9655,'backward_calls':8,'optimizer_updates':1,'model_loads':2}
BUDGETS={
 '2':{'generated_response_tokens':0,'generated_responses':0,'scorer_forwards':0,'teacherforced_input_positions':0,'backward_calls':0,'optimizer_updates':0,'model_loads':1,'cache_quantizations':0,'tiny_array_alias_fixtures':3,'array_leaf_copies_max':10000,'full_snapshots_max':2,'full_restores_max':2,'window_seconds':900},
 '3':{'generated_response_tokens':0,'generated_responses':0,'scorer_forwards':16,'teacherforced_input_positions':9024,'backward_calls':16,'optimizer_updates':1,'optimizer_updates_per_arm':1,'model_loads':1,'cache_quantizations':0,'window_seconds':1800},
 '4':{'generated_response_tokens':18432,'generated_responses':72,'greedy_calls':72,'candidate_calls':0,'scorer_forwards':0,'teacherforced_input_positions':0,'backward_calls':0,'optimizer_updates':0,'model_loads':3,'cache_quantizations':0,'window_seconds':2400}}
STAGE1_FILE='49dad3571308ed23af9efb1cec6fd78cbeb8e66e7760a724953fb3316d332419'
SHARDS={'model-00001-of-00002.safetensors':'6a332d4977c95fd6b51ac7155e039016d90041da1bf3cdd3e500192ea00fa626','model-00002-of-00002.safetensors':'2c94dc6761ab4641a4e2c612eb3ff5fd62d939c5704f8301025e741650b9e1e8'}
SPEC_KEYS=('previous_spec','previous_actual','salvage_qualification','runtime_spec','micro_input','dev24','cache_certificate','cache_stage_complete','cache_exit','cache_actual_review','cache_scope','cache_hash_verification','previous_scope','previous_stop','previous_no_native')
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
 from mri_rl import safe_micro_pilot_entry_v3 as previous
 spec=load(specpath);require(spec.get('schema')=='safe-micro-recovery-QBE-spec-v1','Exact fresh recovery spec required')
 require(set(spec['file_sha256'])==set(SPEC_KEYS),'Exact recovery metadata inventory required')
 files={str(Path(specpath).resolve()):metadata_hash(specpath)};inputs={}
 for key in SPEC_KEYS:
  p=Path(spec[key]);h=metadata_hash(p);require(h==spec['file_sha256'][key],'Recovery input changed: '+key);files[str(p)]=h;inputs[key]=load(p)
 prior=previous.prepare(spec['previous_spec'],allow_started=True)['executable_scope']
 require(prior==inputs['previous_scope'] and metadata_hash(spec['previous_scope'])=='da7ef34ccf171922d601bb3160c99758c0873a518e58ea981f39e8f86012c279','Exact closed prior run required')
 require(inputs['previous_actual']['status']=='STAGE2_PASS_STAGE3_FAIL_POST_A_UPDATE_PARENT_ROLLBACK' and inputs['previous_actual']['total_actual']['optimizer_updates']==1,'Saved one-update failure required')
 q=inputs['salvage_qualification']
 require(q['status']=='CPU_QUALIFIED_SALVAGE_PENDING_NATIVE_Q_AND_HUMAN' and q['label']=='SALVAGED_FROM_INCOMPLETE_TRANSACTION' and q['original_outer_journal_committed'] is False,'A not CPU-qualified for prospective salvage')
 require(q['parent_receipt_sha256']==PARENT['receipt_sha256'] and a.sha(q['common_commit_sha256']) and a.sha(q['A_receipt_sha256']),'Salvage parent/common/receipt bindings absent')
 require(set(q['A_checkpoint_files'])=={'adapter.safetensors','adapter_config.json','optimizer/manifest.json','optimizer/state.safetensors'},'Complete A payload/config binding required')
 require(q['A_checkpoint_files']['adapter_config.json']==metadata_hash(Path(inputs['runtime_spec']['parent_checkpoint'])/'adapter_config.json'),'A adapter configuration differs from original parent')
 require(spec['previous_output']==prior['output'] and all(spec[k]==load(spec['previous_spec'])[k] for k in ('runtime_spec','micro_input','dev24','cache_scope','cache_certificate')),'Data/runtime/previous output changed')
 out=_output(spec['output']);require(allow_started or not out.exists(),'Fresh recovery output required; old fatal run cannot restart')
 for path,h in prior['file_sha256'].items():
  require(path not in files or files[path]==h,'Conflicting inherited metadata');files[path]=h
 sources=_closure({str(ROOT/p):metadata_hash(ROOT/p) for p in ('src/mri_rl/safe_micro_recovery_entry_v1.py','src/mri_rl/safe_micro_recovery_native_v1.py','scripts/run_safe_micro_recovery_v1.py','src/mri_rl/safe_micro_watchdog_v2.py')})
 for path,h in prior['source_sha256'].items():
  require(metadata_hash(path)==h and (path not in sources or sources[path]==h),'Historical run source changed');sources[path]=h
 sources=bind_runtime_sources(sources,inputs['runtime_spec'])
 result={'schema':SCHEMA,'scope_kind':KIND,'action':ACTION,'implemented_by':IMPLEMENTED_BY,'source_sha256':sources,'file_sha256':files,'input_sha256':a.identity(files),'output':str(out),'resource_policy':MODEL_POLICY,'external_guardian_policy':GUARDIAN_POLICY,'budgets':BUDGETS,'parent':PARENT,'stage_order':[2,3,4],'evaluation':'FROZEN_DEV24_GREEDY_ONLY','automatic_retry':False,'automatic_progression_on_actual_PASS_only':True,'global_operation_start_window_seconds':5100,'recovery_contract':RECOVERY,'adoption':ADOPTION,'spent_before_recovery':SPENT,'salvage_qualification_FILE_sha256':metadata_hash(spec['salvage_qualification']),'cache_recovery_operations':0,'comparison':'Adopt exact savedA only afterQ+explicit human grant. B freshdiskAdam52/common16/no newrollout; only grouping differs. DEV24engineeringonly.'}
 return {'status':'PREPARED_PENDING_EXACT_REVIEW_AND_FRESH_HUMAN','executable_scope':result,'scope_sha256':digest(result),'native_calls':0}


class Authorization(HistoricalAuthorization):
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
  if self.parent is not None:
   stage=self.scope['stage'];clock=load(Path(self.scope['output']).parent/('stage'+str(stage)+'_clock_PRIVATE.json'))
   require(time.monotonic()-clock['started_monotonic']<BUDGETS[str(stage)]['window_seconds'],'Cumulative stage deadline exhausted including pauses')
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
 require(scope.get('adoption')==ADOPTION and scope.get('spent_before_recovery')==SPENT and scope.get('resource_policy')==MODEL_POLICY and scope.get('external_guardian_policy')==GUARDIAN_POLICY and scope.get('budgets')==BUDGETS and scope.get('recovery_contract')==RECOVERY and scope.get('parent')==PARENT and scope.get('stage_order')==[2,3,4] and scope.get('evaluation')=='FROZEN_DEV24_GREEDY_ONLY','Exact conservative policy/budgets/parent required')
 require(human.get('adopt_A_from_incomplete_transaction') is True and human.get('authorized') is True and human.get('action')==ACTION and human.get('scope_sha256')==digest(scope) and human.get('included_stages')==[2,3,4] and a.sha(human.get('human_instruction_sha256')) and human.get('approved_by'),'Fresh explicit stages2+3+4 human permission required')
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
 require(scope.get('adoption')==ADOPTION and scope.get('spent_before_recovery')==SPENT and scope.get('resource_policy')==MODEL_POLICY and scope.get('external_guardian_policy')==GUARDIAN_POLICY and scope.get('budgets')==BUDGETS and scope.get('recovery_contract')==RECOVERY and scope.get('parent')==PARENT and scope.get('stage_order')==[2,3,4] and scope.get('evaluation')=='FROZEN_DEV24_GREEDY_ONLY','Child policy/budget changed')
 require(human.get('adopt_A_from_incomplete_transaction') is True and human.get('authorized') is True and human.get('scope_sha256')==digest(scope) and human.get('action')==ACTION and human.get('included_stages')==[2,3,4] and a.sha(human.get('human_instruction_sha256')) and human.get('approved_by'),'Exact current child grant required')
 issued,expires=(datetime.fromisoformat(human[k].replace('Z','+00:00')) for k in ('issued_at_utc','expires_at_utc'))
 require(issued<=datetime.now(timezone.utc)<expires and 0<(expires-issued).total_seconds()<=86400,'Child grant expired/unbounded')
 require(review.get('verdict') in ('PASS','PASS_WITH_LIMITATIONS') and review.get('scope_sha256')==digest(scope) and review.get('source_sha256')==scope['source_sha256'] and review.get('input_sha256')==scope['input_sha256'] and review.get('reviewer_kind') in ('isolated_codex','claude_code') and review.get('reviewer_id') and review['reviewer_id']!=IMPLEMENTED_BY,'Exact child independent review required')
 token=Authorization(_ISSUER,scope,expires);token.start=launch['started_monotonic'];require(type(token.start) in (int,float) and token.start<=time.monotonic(),'Invalid compound clock');token.check('child_admission');consumed(human['human_instruction_sha256']);return token


def _prior(scope,stage,expected_scope_file_sha256=None):
 if stage==2:return
 from mri_rl.safe_micro_witness_v1 import digest as fdigest
 base=Path(scope['output']);p=base/('stage'+str(stage-1)+'_verified_PRIVATE.json');rec=load(p)
 require(rec['scope_file_sha256']==expected_scope_file_sha256,'Predecessor scope mismatch')
 terminal=base/('stage'+str(stage-1)+'_PRIVATE')/'terminal_PRIVATE.json';result=load(terminal)
 require(result['scope_sha256']==digest(phase_scope(scope,stage-1)) and fdigest(terminal)==rec['terminal_sha256'] and result['status']=='PASS' and result['release_verified'] is True,'Previous stage incomplete')
 witness=Path(rec['exit_path']);exit=load(witness)
 require(fdigest(witness)==rec['exit_sha256'] and exit['native_exit_code']==0 and exit['child_reaped'] and exit['post_child_global_lock'].get('no_open_holders') is True,'Previous exit/lock invalid')
 for name in ('stdout','stderr'):require(fdigest(witness.parent/(name+'_PRIVATE.log'))==exit[name+'_sha256'],'Closed log changed')


class RecoveryResourceGate(resources.ResourceGate):
 def claim_model_load(self):
  from mri_rl.shared_rollout_gspo_core_v1 import replace_atomic
  self.authorization.check('recovery_model_load_claim')
  require(self.scope['recovery_contract']==RECOVERY and self.scope['spent_before_recovery']==SPENT,'Load debt changed')
  path=Path(self.spec['output'])/'model_load_ledger_PRIVATE.json'
  rec=load(path) if path.exists() else {'scope':self.scope['compound_scope_sha256'],'claimed':0,'limit':5,'historical_spent':2}
  require(rec['scope']==self.scope['compound_scope_sha256'] and rec['limit']==5 and rec['historical_spent']==2 and type(rec['claimed']) is int and 0<=rec['claimed']<5,'New5/lifetime7 load budget exhausted')
  rec['claimed']+=1;replace_atomic(path,rec)


def execute_child(args):
 import signal
 from mri_rl.safe_micro_dynamic_resources_v1 import ResourcePause
 from mri_rl.safe_micro_journal_v1 import Indeterminate
 stage=args.child_stage;scope=load(args.scope);base=Path(scope['output']);launch=load(base/'launch_identity_PRIVATE.json')
 control=load(base/'control_PRIVATE.json');require(type(args.attempt) is int and 0<=args.attempt<3 and args.attempt==control['attempts'][str(stage)]-1 and not (base/('stage'+str(stage)+'_verified_PRIVATE.json')).exists(),'Unreserved/duplicate child attempt')
 token,human=_credentials(args,launch_record=launch,consumed=a.consume_once(base/('stage%d_attempt%d_permission_PRIVATE'%(stage,args.attempt))))
 _prior(scope,stage,args.expected_scope_sha256);phase=token.for_phase(stage)
 from mri_rl.accuracy_recovery_gpu_cache_v2 import installed_fingerprint
 spec=load(args.spec);require(installed_fingerprint()==load(spec['cache_scope'])['installed'],'Pinned native runtime differs')
 runtime=load(spec['runtime_spec']);lockpath=Path(runtime['global_lock']);require(lockpath.is_file() and not lockpath.is_symlink(),'Existing lock required')
 import fcntl
 from mri_rl.shared_rollout_gspo_core_v1 import publish_once
 fd=os.open(lockpath,os.O_RDWR);gate=None
 try:
  fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
  gate=RecoveryResourceGate('B_MICRO',spec,phase.scope,phase)
  signal.signal(signal.SIGUSR1,lambda *unused:setattr(gate,'red_requested',True))
  gate('native_model_load_preflight')
  from mri_rl.continuous_rl_pilot import cache_binding_for
  from types import SimpleNamespace
  binding=cache_binding_for(SimpleNamespace(cache_dir=Path(runtime['cache_dir']),cache_certificate=Path(runtime['cache_certificate']),cache_certificate_sha256=H(runtime['cache_certificate']),model=Path(runtime['model'])))
  require({n:v['sha256'] for n,v in binding['shards'].items()}==SHARDS,'Cache payload mismatch')
  phase.check('before_native_import')
  from mri_rl.safe_micro_recovery_native_v1 import run_phase
  result=run_phase(stage,spec=spec,scope=phase.scope,authorization=phase,resource_gate=gate)
  record={**result,'scope_sha256':phase.sha,'compound_scope_sha256':token.sha,'stage':stage}
  p=Path(phase.scope['output'])/'terminal_PRIVATE.json'
  if p.exists():require(load(p)==record,'Completed stage terminal changed')
  else:publish_once(p,record)
  return 0 if record['status']=='PASS' and record['release_verified'] else 2
 except ResourcePause as error:
  phaseout=Path(phase.scope['output'])
  require(not list(phaseout.rglob('release_failure_PRIVATE.json')),'Allocator release failure: no automatic recovery')
  for receipt in phaseout.rglob('release_SAFE.json'):require(load(receipt).get('release_verified') is True,'Allocator release not verified: no automatic recovery')
  publish_once(base/('stage%d_attempt%d_pause_PRIVATE.json'%(stage,args.attempt)),{'status':'PAUSED_EXTERNAL_REAP_REQUIRED','allocator_release':'Separate actual session receipts or NOT_MEASURED_NO_MODEL','reason':str(error),'attempt':args.attempt,'stage':stage})
  return 75
 except Indeterminate as error:
  publish_once(base/('stage%d_attempt%d_indeterminate_PRIVATE.json'%(stage,args.attempt)),{'status':'INDETERMINATE_NO_REPLAY','reason':str(error)})
  return 76
 finally:os.close(fd)


def validate_pause_exit(path,scope_file,base,stage,attempt):
 from mri_rl.safe_micro_witness_v1 import digest as fdigest
 path=Path(path);require(path.is_file(),'Prior child exit absent/active: cannot resume')
 w=load(path)
 require(w['scope_file_sha256']==scope_file and w['child_reaped'] and w['native_exit_code']==75 and w['post_child_global_lock'].get('no_open_holders') is True,'Prior child not exact closed cooperative pause')
 require(not w['SIGTERM_sent'] and not w['SIGKILL_after_grace'] and not w['monitor_log_errors'],'Fatal monitor/forced stop cannot resume')
 for name in ('stdout','stderr'):require(fdigest(path.parent/(name+'_PRIVATE.log'))==w[name+'_sha256'],'Prior closed log changed')
 pause=load(Path(base)/('stage%d_attempt%d_pause_PRIVATE.json'%(stage,attempt)))
 require(pause['stage']==stage and pause['attempt']==attempt and pause['status']=='PAUSED_EXTERNAL_REAP_REQUIRED','Missing exact native cooperative pause')
 return True


def wait_nonresident(scope,token,stage,observe,sleep=time.sleep,clock=time.monotonic):
 good=0;start=load(Path(scope['output'])/('stage'+str(stage)+'_clock_PRIVATE.json'))['started_monotonic']
 while True:
  token.check('nonresident_wait')
  require(clock()-start<BUDGETS[str(stage)]['window_seconds'],'Stage cumulative wait deadline exhausted')
  result=observe()
  if result['action']=='STOP':raise resources.ResourceFatal(','.join(result['reasons']))
  good=good+1 if result['action']=='CONTINUE' else 0
  if good>=3:return
  sleep(5.)


def execute_chain(args):
 import fcntl,sys,shutil
 from dataclasses import asdict
 from mri_rl.safe_micro_witness_v1 import inspect_lock
 from mri_rl.safe_micro_watchdog_v2 import witness_child
 from mri_rl.accuracy_recovery_gpu_cache_v3 import os_pressure_level
 from mri_rl.resource_supervisor import sample_macos
 from mri_rl.shared_rollout_gspo_core_v1 import publish_once,replace_atomic
 from mri_rl.safe_micro_journal_v1 import Journal,durable_mkdir
 scope=load(args.scope);base=Path(scope['output']);root=a.canonical_project_root();python=root/'work/accuracy-recovery-venv-v1/bin/python'
 require(Path(sys.executable).resolve()==python.resolve(),'Pinned production interpreter required')
 if args.resume:
  launch=load(base/'launch_identity_PRIVATE.json');token,human=_credentials(args,launch_record=launch,consumed=lambda _:None)
 else:
  token,human=_credentials(args,consumed=a.consume_once(base.with_name(base.name+'_permission_PRIVATE')))
  require(not base.exists(),'Fresh output required');durable_mkdir(base);publish_once(base/'launch_identity_PRIVATE.json',{'scope_sha256':token.sha,'human_instruction_sha256':human['human_instruction_sha256'],'started_utc':datetime.now(timezone.utc).isoformat(),'started_monotonic':token.start})
 fd=os.open(base/'controller.lock',os.O_CREAT|os.O_RDWR,0o600)
 try:
  fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
  controlpath=base/'control_PRIVATE.json'
  control=load(controlpath) if controlpath.exists() else {'launches':0,'recoveries':0,'attempts':{'2':0,'3':0,'4':0},'accepted_pauses':[]}
  require(control['launches']<3,'Controller recovery launch limit exhausted');control['launches']+=1;replace_atomic(controlpath,control)
  spec=load(args.spec);runtime=load(spec['runtime_spec']);controller=resources.Controller()
  def observe():
   s=sample_macos();r=controller.decide(s,os_pressure_level(),shutil.disk_usage(base).free/1024**3,time.monotonic())
   replace_atomic(base/'nonresident_latest_PRIVATE.json',{'sample':asdict(s),**r})
   return r
  for stage in (2,3,4):
   verified=base/('stage'+str(stage)+'_verified_PRIVATE.json')
   if verified.exists():
    # Reauthenticate this completed stage through the same predecessor gate.
    _prior(scope,stage+1,args.expected_scope_sha256);continue
   _prior(scope,stage,args.expected_scope_sha256)
   stageclock=base/('stage'+str(stage)+'_clock_PRIVATE.json')
   if not stageclock.exists():publish_once(stageclock,{'started_monotonic':time.monotonic()})
   while True:
    if control['attempts'][str(stage)]:
     previous=control['attempts'][str(stage)]-1;pw=base/('stage%d_attempt%d_exit_PRIVATE'%(stage,previous))/'actual_exit_PRIVATE.json'
     validate_pause_exit(pw,args.expected_scope_sha256,base,stage,previous)
     tag='%d:%d'%(stage,previous)
     if tag not in control['accepted_pauses']:
      require(control['recoveries']<2,'Recovery budget exhausted');control['recoveries']+=1;control['accepted_pauses'].append(tag);replace_atomic(controlpath,control)
    # Every existing operation journal must be unambiguous BEFORE model loading.
    for bindingpath in (base/('stage'+str(stage)+'_PRIVATE')).rglob('binding_PRIVATE.json'):
     with Journal(bindingpath.parent,load(bindingpath)) as journal:journal.assert_recoverable()
    wait_nonresident(scope,token,stage,observe)
    n=control['attempts'][str(stage)];control['attempts'][str(stage)]+=1;replace_atomic(controlpath,control)
    argv=[str(python),'-I',str(ROOT/'scripts/run_safe_micro_recovery_v1.py'),'--child-stage',str(stage),'--attempt',str(n)]
    for name in ('spec','scope','human','review'):argv+=['--'+name,str(getattr(args,name))]
    for name in ('scope','human','review'):argv+=['--expected-'+name+'-sha256',getattr(args,'expected_'+name+'_sha256')]
    wdir=base/('stage%d_attempt%d_exit_PRIVATE'%(stage,n))
    witness=witness_child(argv,wdir,scope_sha256=args.expected_scope_sha256,sampler=lambda pid:sample_macos(pid=pid),level=os_pressure_level,disk_free_gib=lambda:shutil.disk_usage(base).free/1024**3,post_sampler=lambda:asdict(sample_macos()),lock_inspector=lambda:inspect_lock(runtime['global_lock']))
    require(witness['post_child_global_lock'].get('no_open_holders') is True,'Model lock not released')
    terminal=base/('stage'+str(stage)+'_PRIVATE')/'terminal_PRIVATE.json'
    if witness['native_exit_code']==0 and not witness['SIGTERM_sent'] and not witness['SIGKILL_after_grace'] and not witness['monitor_log_errors']:
     result=load(terminal);require(result['scope_sha256']==digest(phase_scope(scope,stage)) and result['status']=='PASS' and result['release_verified'],'Stage is not PASS/released')
     publish_once(verified,{'scope_file_sha256':args.expected_scope_sha256,'terminal_sha256':H(terminal),'exit_path':str(wdir/'actual_exit_PRIVATE.json'),'exit_sha256':H(wdir/'actual_exit_PRIVATE.json')});break
    validate_pause_exit(wdir/'actual_exit_PRIVATE.json',args.expected_scope_sha256,base,stage,n)
  publish_once(base/'compound_complete_PRIVATE.json',{'status':'COMPLETE','automatic_expansion':False})
  print(json.dumps({'status':'COMPLETE','automatic_expansion':False}));return 0
 finally:os.close(fd)


def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path,required=True);mode=p.add_mutually_exclusive_group();mode.add_argument('--launch',action='store_true');mode.add_argument('--resume',action='store_true');mode.add_argument('--child-stage',type=int,choices=(2,3,4));p.add_argument('--attempt',type=int,default=0);p.add_argument('--scope-output',type=Path)
 for name in ('scope','human','review'):p.add_argument('--'+name,type=Path);p.add_argument('--expected-'+name+'-sha256')
 args=p.parse_args(argv)
 if args.launch or args.resume or args.child_stage:
  require(all(getattr(args,n,None) and getattr(args,'expected_'+n+'_sha256',None) for n in ('scope','human','review')),'Exact artifacts required')
  return execute_child(args) if args.child_stage else execute_chain(args)
 result=prepare(args.spec)
 if args.scope_output:
  from mri_rl.shared_rollout_gspo_core_v1 import publish_once
  publish_once(_output(args.scope_output),result['executable_scope'])
 print(json.dumps({k:v for k,v in result.items() if k!='executable_scope'}));return 0
