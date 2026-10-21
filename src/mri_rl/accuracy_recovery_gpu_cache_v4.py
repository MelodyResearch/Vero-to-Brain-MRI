"""Fresh, one-shot cache-only v4 continuation. No preparation command imports MLX.

Reuse the unchanged v3 converter/assembly/historical certifier and v2 GPU backend.
406 retained units and saved exact unit0 are adopted, never re-quantized. The
original swap baseline/debt is carried, and only this round's power rule changes.
No model construction, inference, generation, backward, or training is authorized.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import fcntl,json,os,shutil,stat,time
from pathlib import Path
from mri_rl import accuracy_recovery_gpu_cache_v3 as v3, accuracy_recovery_gpu_cache_v2 as v2
from mri_rl import accuracy_recovery_cpu_cache_v1 as cc, quant_convert as qc, shared_rollout_gspo_core_v1 as shared

WORK_ROOT_NAME='accuracy-recovery-historical-gpu-cache-v4'
CONSUMED_REL='local_data/accuracy_recovery_v1/historical_gpu_cache_v4_consumed'
ACTION='accuracy_recovery_historical_gpu_cache_v4_fresh_continue343_adopt407_battery20'
SCHEMA='accuracy-recovery-historical-gpu-cache-v4-scope-v1'
KIND='FRESH_EXACT_HISTORICAL_GPU_CACHE_V4_CONTINUATION_SCOPE'
IMPLEMENTED_BY='codex'
BASELINE_SWAP_GIB=13.96966796875
POLICY={**v3.POLICY,'ac_required':False,'battery_percent_min':20.0,
 'power_exception':'THIS ROUND ONLY; actual AC/BATTERY sample preserved; battery<20 STOP, unknown HOLD. Waiver is not GPU authorization.'}
SCOPE={**v3.SCOPE,'name':'VERO_TO_MRI_ACCURACY_RECOVERY_V1/historical_gpu_cache_v4_continuation',
 'work_root_name':WORK_ROOT_NAME,'previous_work_root_name_read_only':v3.WORK_ROOT_NAME,'resource_policy':POLICY,
 'continuation':'Fresh permission after closed v3 preflight STOP/0new units. Adopt406 retained+unit0; no re-quantization of adopted units.'}
STATIC_SHA=shared.identity(SCOPE)
PRODUCTION_CLOSED_PINS={
 'scope':'658542e0c31a39b512a7705a41d08abc5b027e714c5715df32cb47cd960ab4ee',
 'launcher':'dcd5fdefc79fa308b95c7388ae5f548f0baa1058dd6d7a5e96c02aca2e4e0ece',
 'stderr':'45aa47c527dd7bce0ba6c4f8e91eaf46bb0f24ac5372ecc27d80aa21742859ee',
 'stdout':'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
 'resource':'20dbd8edf390f7440ce1bd8c21b2ada197e87d47308bb8ea513aa18d74e6c734',
 'adopt_manifest':'8c1d9044b9df850a93308d9652689843439a739ff0989792c8720983ddb1e53e',
 'adoption':'50db0f94f35ace179d0b6ca60c22f73bea975ce4c9a2d5baae1119a9ee426d7a',
 'human':'5d2cef90c6dd63fa9548862949d8d183eae334d4127d2df0f91cd7cf0038af93',
 'review':'713d17713277a4203cfe1f8a949b839694468629b130f4907facc55ed7438546'}
require=v3._req
INPUT_KEYS=('roots','files','source_shards','inventory_sha256','plan_sha256','expected','installed','previous_attempt','adoption','closed_v3','payload_verification')


def roots_for(repo_root,source_dir):
 r=Path(repo_root)
 return {**v3.canonical_roots(r,source_dir),'work_root':str(r/'work'/WORK_ROOT_NAME),
  'previous_work_root':str(r/'work'/v3.WORK_ROOT_NAME),'consumed_dir':str(r/CONSUMED_REL),
  'previous_consumed_dir':str(r/v3.CONSUMED_REL),'v2_consumed_dir':str(r/v2.CONSUMED_REL)}


def closed_paths(repo_root):
 r=Path(repo_root);d=r/v2.SCOPE_DIR_REL;w=r/'work'/v3.WORK_ROOT_NAME
 prefix='gpu_cache_v3_resource_revised_ACTUAL_EXECUTION_'
 return {'scope':d/'historical_gpu_cache_v3_resource_revised_scope_PRIVATE.json',
  'launcher':d/(prefix+'launcher_record_PRIVATE.json'),'stdout':d/(prefix+'child_stdout_PRIVATE.txt'),
  'stderr':d/(prefix+'child_stderr_PRIVATE.txt'),'resource':w/'v3_resource.jsonl',
  'adopt_manifest':w/'adopt_manifest.json','adoption':w/'adoption/unit0_adoption_PRIVATE.json',
  'human':d/'V3_RESOURCE_REVISED_ACTUAL_HUMAN_AUTHORIZATION_PRIVATE.json',
  'review':d/'HISTORICAL_GPU_CACHE_V3_RESOURCE_REVISED_INDEPENDENT_REVIEW_PRIVATE.json',
  'waiver':r/'local_data/accuracy_stage_v1/safe_handoff/HUMAN_POWER_WAIVER_THIS_ROUND.txt'}


def verify_closed_v3(repo_root,pins=PRODUCTION_CLOSED_PINS,expected=cc.Expected()):
 """Metadata only. Payload verification is explicitly deferred, never fabricated."""
 paths=closed_paths(repo_root);files={}
 for name,path in paths.items():
  require(path.is_file(),'Closed v3 evidence missing: '+name)
  digest=shared.file_sha256(path)
  require(name not in pins or digest==pins[name],'Closed v3 evidence digest differs: '+name)
  files['closed_'+name]={'path':str(path),'sha256':digest}
 old=json.loads(paths['scope'].read_text());launch=json.loads(paths['launcher'].read_text());child=launch.get('child',{})
 require(launch.get('scope_file_sha256')==files['closed_scope']['sha256'] and launch.get('scope_digest')==shared.scope_digest(old), 'Closed v3 scope/launcher binding differs')
 require(launch.get('human_file_sha256')==files['closed_human']['sha256'],'Closed v3 human/launcher binding differs')
 require(old['roots']==v3.canonical_roots(repo_root,old['roots']['source_dir']),'Closed v3 roots are not canonical')
 require(child.get('native_exit_code')==1 and child.get('timed_out') is False and
  child.get('stdout_sha256')==files['closed_stdout']['sha256'] and child.get('stderr_sha256')==files['closed_stderr']['sha256'],'Closed v3 child/log evidence differs')
 previous=Path(old['roots']['work_root'])
 require(not any((previous/n).exists() for n in ('v3_ledger.json','v3_active.lock','v3_rss.jsonl')) and not list(previous.glob('v3_failure_*.json')),'Closed v3 must be preflight-only, no ledger/backend progress')
 rows=[json.loads(l) for l in paths['resource'].read_text().splitlines() if l.strip()]
 require(rows and all(x.get('boundary')=='preflight' for x in rows) and rows[-1].get('decision')=='STOP' and
  all(x.get('baseline_swap_gib')==BASELINE_SWAP_GIB for x in rows),'Closed v3 is not the pinned preflight STOP with original baseline')
 require(old['previous_attempt']['swap_baseline_gib']==BASELINE_SWAP_GIB,'Original swap debt must not reset')
 manifest=json.loads(paths['adopt_manifest'].read_text());adoption=json.loads(paths['adoption'].read_text())
 require(manifest['adopted_retained_units']==expected.retained and manifest['adopted_saved_units']==1 and
  manifest['adoption_record_sha256']==files['closed_adoption']['sha256'] and adoption==old['adoption'],'Closed v3 adoption metadata differs')
 require(len(list((previous/'units').glob('unit_*.safetensors')))==expected.retained+1,'Closed v3 unit filename count differs')
 require(len((previous/'units/receipts.jsonl').read_text().splitlines())==expected.retained,'Closed v3 retained receipt count differs')
 require(paths['waiver'].stat().st_size>0,'This-round human power waiver empty')
 return old,files,{'scope_sha256':shared.scope_digest(old),'new_gpu_unit_attempts':0,'new_underlying_quantizations':0,
  'swap_baseline_gib':BASELINE_SWAP_GIB,'human_instruction_sha256':json.loads(paths['human'].read_text()).get('human_instruction_sha256')}


class ResourceController(v3.ResourceController):
 def __init__(self,**kw):
  require(kw.get('baseline_swap_gib')==BASELINE_SWAP_GIB,'Original swap baseline/debt is required')
  super().__init__(policy=POLICY,**kw)
 def _decide(self,s,level,required,now):
  decision,reasons,growth,trend=super()._decide(s,level,required,now)
  reasons=[r for r in reasons if r!='not_ac']
  battery=getattr(s,'battery_percent',None)
  if not v3._valid_number(battery,0.,100.):reasons.append('battery_unknown_or_invalid')
  elif battery<20.:reasons.append('battery_below_20');decision='STOP'
  if decision=='STOP':return decision,reasons,growth,trend
  return ('HOLD' if reasons else 'CONTINUE'),reasons,growth,trend


def default_entries():return [Path(__file__),Path(__file__).resolve().parents[2]/'scripts/rebuild_accuracy_recovery_historical_gpu_cache_v4.py']


def prepare_scope_v4(*,repo_root,scope_output,closed_pins=PRODUCTION_CLOSED_PINS,expected=cc.Expected(),installed=v2.installed_fingerprint):
 old,files,closed=verify_closed_v3(repo_root,closed_pins,expected)
 output=v2.canon_path(scope_output);r=Path(repo_root)
 require(output.parent==r/v2.SCOPE_DIR_REL and not output.exists(),'Fresh canonical private scope path required')
 roots=roots_for(r,old['roots']['source_dir'])
 inst=installed();require(inst==old['installed'],'Installed runtime differs from closed v3')
 inputs={k:old[k] for k in ('source_shards','inventory_sha256','plan_sha256','expected','previous_attempt','adoption')}
 inputs.update(roots=roots,files={**old['files'],**files},installed=inst,closed_v3=closed,payload_verification='DEFERRED_TO_EXECUTE_UNDER_GLOBAL_LOCK')
 require(inputs['expected']==asdict(expected),'Inventory expectation differs')
 for path,digest in old['source_sha256'].items():require(shared.file_sha256(path)==digest,'Closed v3 source changed: '+path)
 closure={**old['source_sha256'],**v2.local_closure(default_entries())}
 # Hash source/config metadata only, never shards or units.
 for p,digest in closure.items():require(shared.file_sha256(p)==digest,'Closed source changed: '+p)
 for binding in inputs['files'].values():require(shared.file_sha256(binding['path'])==binding['sha256'],'Bound metadata changed')
 scope={'schema':SCHEMA,'scope_kind':KIND,'action':ACTION,'implemented_by':IMPLEMENTED_BY,'experiment':'VERO_TO_MRI_ACCURACY_RECOVERY_V1',
  'static_scope_sha256':STATIC_SHA,'limits':SCOPE,**inputs,'source_sha256':closure,'input_sha256':shared.identity(inputs)}
 shared.publish_once(output,scope)
 return {'scope_path':str(output),'scope_file_sha256':shared.file_sha256(output),'scope_sha256':shared.scope_digest(scope),
  'input_sha256':scope['input_sha256'],'source_files_in_closure':len(closure),'new_pending_units':expected.quantized-1,'native_imports':0}


class Admission(v3.Admission3):
 def already_consumed(self):
  reason=super().already_consumed()
  instruction=self.human['human_instruction_sha256']
  if reason:return reason
  if instruction==self.scope['closed_v3']['human_instruction_sha256'] or any((Path(self.scope['roots'][key])/('authorization_consumed_'+instruction+'.json')).exists() for key in ('v2_consumed_dir','previous_consumed_dir')):
   return 'Old human instruction already consumed/closed; fresh authorization required'
  return None


def admit_v4(*,scope_path,scope_sha256,human_path,human_sha256,review_path,review_sha256,repo_root,now,
 expected=cc.Expected(),installed=v2.installed_fingerprint,closed_pins=PRODUCTION_CLOSED_PINS):
 scope=v2._load_pinned(scope_path,scope_sha256,'Fresh v4 scope');human=v2._load_pinned(human_path,human_sha256,'Fresh human');review=v2._load_pinned(review_path,review_sha256,'Independent review')
 require(scope.get('schema')==SCHEMA and scope.get('scope_kind')==KIND and scope.get('action')==ACTION and v2._flag_free(scope),'Not an exact v4 scope')
 require(scope.get('limits')==SCOPE and scope.get('static_scope_sha256')==STATIC_SHA and scope.get('implemented_by')==IMPLEMENTED_BY,'Budget/policy/implementer differs')
 require(scope.get('expected')==asdict(expected) and (expected!=cc.Expected() or scope.get('plan_sha256')==v2.PLAN_SHA256),'Plan/inventory differs')
 roots=scope['roots'];require(roots==roots_for(repo_root,roots.get('source_dir')),'Canonical roots differ')
 require(scope['input_sha256']==shared.identity({k:scope[k] for k in INPUT_KEYS}),'Input digest differs')
 old,closed_files,closed=verify_closed_v3(repo_root,closed_pins,expected)
 require(closed==scope['closed_v3'] and all(scope['files'].get(k)==v for k,v in closed_files.items()),'Closed v3 evidence differs from scope')
 require(scope['adoption']==old['adoption'] and scope['previous_attempt']==old['previous_attempt'],'Adoption/debt differs')
 require(scope['source_sha256']=={**old['source_sha256'],**v2.local_closure(default_entries())},'Exact current source closure differs')
 adm=Admission(scope,human,installed,roots['consumed_dir'],roots['previous_consumed_dir']);adm.check('admission',full=True)
 digest=adm.digest
 require(human.get('authorized') is True and human.get('action')==ACTION and human.get('scope_sha256')==digest and shared.sha(human.get('human_instruction_sha256')) and human.get('approved_by'),'Fresh exact human cache permission required; power waiver alone does not authorize GPU')
 try:issued,expires=(datetime.fromisoformat(human[k].replace('Z','+00:00')) for k in ('issued_at_utc','expires_at_utc'))
 except (KeyError,ValueError,AttributeError):raise v2.ScopeRefused('Missing bounded human times') from None
 require(issued<=now<expires and 0<(expires-issued).total_seconds()<=v2.MAX_AUTH_HOURS*3600,'Human authorization expired/unbounded')
 require(review.get('verdict') in ('PASS','PASS_WITH_LIMITATIONS') and review.get('scope_sha256')==digest and review.get('reviewer_id') and review.get('reviewer_kind') in v2.REVIEWER_KINDS and review.get('reviewer_id')!=human.get('approved_by') and review.get('source_sha256')==scope['source_sha256'] and review.get('input_sha256')==scope['input_sha256'],'Independent exact scope/source/input review required')
 reason=adm.already_consumed();require(reason is None,reason or '')
 return adm


def execute(*,scope_path,scope_sha256,human_path,human_sha256,review_path,review_sha256,repo_root=None,expected=cc.Expected(),
 installed=v2.installed_fingerprint,closed_pins=PRODUCTION_CLOSED_PINS,backend_factory=v2.make_gpu_backend,
 sampler=None,os_level=None,rss_gib=cc._ps_rss_gib,clock=time.monotonic,sleep=time.sleep,now=lambda:datetime.now(timezone.utc),disk_usage=shutil.disk_usage):
 """Future single execution only. Lazy backend factory is the first native import."""
 adm=admit_v4(scope_path=scope_path,scope_sha256=scope_sha256,human_path=human_path,human_sha256=human_sha256,review_path=review_path,review_sha256=review_sha256,
  repo_root=repo_root or v2.repo_root_from_git(),now=now(),expected=expected,installed=installed,closed_pins=closed_pins)
 scope=adm.scope;roots=scope['roots'];files=scope['files'];work=Path(roots['work_root']);old_units=Path(roots['previous_work_root'])/'units'
 lock=Path(roots['global_lock']);require(lock.exists() and stat.S_ISREG(os.lstat(lock).st_mode) and v2.canon_path(lock)==lock,'Canonical existing global lock required')
 fd=os.open(lock,os.O_RDWR)
 try:
  try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:raise v2.ScopeRefused('Global model lock held; do not force/remove it') from None
  require(not work.exists(),'Fresh v4 destination required; no retry/resume')
  adm.check('after_global_lock',full=True);adm.consume()
  work.mkdir(parents=True);shared.publish_once(work/'v4_ledger.json',{'scope_sha256':adm.digest,'human_instruction_sha256':adm.human['human_instruction_sha256'],'started_utc':now().isoformat(),'old_swap_baseline_gib':BASELINE_SWAP_GIB})
  backend=None;plan=None;t0=clock()
  ctrl=ResourceController(sampler=sampler,os_level=os_level,clock=clock,sleep=sleep,log_path=work/'v4_resource.jsonl',baseline_swap_gib=BASELINE_SWAP_GIB);ctrl.start_window()
  try:
   ctrl.preflight()
   plan,hist,proofs=v2.load_pins(files['plan']['path'],scope['plan_sha256'],files['historical_certificate']['path'],files['historical_certificate']['sha256'],files['historical_proof']['path'],files['historical_proof']['sha256'],expected)
   pending=[u for u in plan['units'] if u['action']=='quantize' and u['index']!=0]
   require(len(pending)==expected.quantized-1 and 2+2*len(pending)<=v3.MAX_CUMULATIVE_UNDERLYING,'343+343 new/cumulative688 cap differs')
   v2.disk_gate(work,v2.disk_estimate_bytes(plan,pending),usage=disk_usage)
   plan,qmap=cc.prepare(files['plan']['path'],scope['plan_sha256'],roots['source_dir'],files['quant_map']['path'],{k:v['sha256'] for k,v in scope['source_shards'].items()},expected)
   retained=v2.authenticate_retained(plan,proofs,old_units)
   record=scope['adoption'];ev=v3.verify_saved_unit(plan,proofs,plan['units'][0],old_units/record['file'],roots['source_dir'])
   require(all(record[k]==v for k,v in ev.items()),'Saved unit0 differs; never requantize')
   require(len(retained)==expected.retained,'Retained count differs')
   units=work/'units';units.mkdir()
   for _,src in retained.values():shutil.copyfile(src,units/src.name)
   shutil.copyfile(old_units/record['file'],units/record['file'])
   for name in ('receipts.jsonl','plan_binding.json'):shutil.copyfile(old_units/name,units/name)
   shared.publish_once(work/'unit0_adoption_PRIVATE.json',record)
   # Exact historical payload verification at the inherited seam; no correctness threshold changes.
   v2.authenticate_retained(plan,proofs,units);v3.verify_saved_unit(plan,proofs,plan['units'][0],units/record['file'],roots['source_dir'])
   ctrl.boundary('before_native_backend');adm.check('before_native_backend',full=True)
   backend=backend_factory(proofs,clock=clock,t0=t0,rss_gib=rss_gib,rss_log=work/'v4_rss.jsonl',work_root=work,disk_usage=disk_usage,
    disk_need=lambda u:int(u['nbytes']*qc.QUANT_OUT_FRACTION),max_attempts=len(pending),max_quantize=len(pending),max_module=len(pending),max_underlying=2*len(pending),phase_check=adm.check)
   result=v3.convert_v3(plan,roots['source_dir'],units,backend,ctrl,adopted=frozenset({0}));require(result['complete'],'Incomplete conversion STOP')
   for unit in pending:v2.check_unit_file(plan,unit,proofs[unit['index']],units/('unit_%04d.safetensors'%unit['index']))
   done={**qc.load_done(plan,units),0:{'output_file':record['file'],'adopted':True}}
   ctrl.boundary('pre_assembly',enforce_window=False);v3.assemble_v3(plan,units,work/'cache',done)
   from mri_rl.quant_cache import finalize_cache
   finalize_cache(plan,units,work/'cache',roots['source_dir'],qmap)
   certificate=v3.verify_historical_v3(plan,units,work/'cache',roots['source_dir'],qmap,hist,expected,done)
   ctrl.boundary('post_verify',enforce_window=False)
   certificate['gpu_reconstruction']={'device':'gpu','scope_sha256':adm.digest,'retained_units_adopted':len(retained),'adopted_saved_units':1,
    'new_gpu_unit_attempts':backend.attempts,'new_direct_quantize_ops':backend.quantize_ops,'new_module_path_quantizations':backend.module_path_checks,
    'new_underlying_quantizations':backend.underlying_quantizations,'previous_attempt_underlying_quantizations':2,
    'cumulative_underlying_quantizations':2+backend.underlying_quantizations,'cumulative_underlying_quantizations_max':688,
    'original_swap_baseline_gib':BASELINE_SWAP_GIB,'model_loaded':False,'training':False,'elapsed_seconds':clock()-t0,
    'adopted_unit_evidence_missing':record.get('evidence_missing'),'power_sample_not_falsified':True}
  except BaseException as error:
   shared.publish_once(work/'v4_failure_PRIVATE.json',{'error_type':type(error).__name__,'error':str(error)[:1000],'automatic_retry':False,'rerun_allowed':False,
    'new_gpu_unit_attempts':getattr(backend,'attempts',0),'new_underlying_quantizations':getattr(backend,'underlying_quantizations',0),'last_resource_sample':ctrl.last,'elapsed_seconds':clock()-t0})
   adm.outcome('FAILED',str(error));raise
  finally:
   if backend is not None:backend.release()
  shared.publish_once(work/'certificate.json',certificate);adm.outcome('SUCCEEDED',str(certificate['gpu_reconstruction']))
  return {'certificate':certificate}
 finally:os.close(fd)


def main(argv=None):
 parser=argparse.ArgumentParser(description='v4 cache-only continuation; CPU prepare, fresh exact human/review required for execute')
 parser.add_argument('command',choices=('prepare-scope','execute'))
 parser.add_argument('--repo-root',type=Path);parser.add_argument('--scope-output',type=Path)
 for name in ('scope','scope-sha256','human-authorization','human-authorization-sha256','independent-review','independent-review-sha256'):parser.add_argument('--'+name)
 args=parser.parse_args(argv);root=args.repo_root or v2.repo_root_from_git()
 if args.command=='prepare-scope':
  if not args.scope_output:parser.error('--scope-output required')
  result=prepare_scope_v4(repo_root=root,scope_output=args.scope_output)
 else:
  required=(args.scope,args.scope_sha256,args.human_authorization,args.human_authorization_sha256,args.independent_review,args.independent_review_sha256)
  if not all(required):parser.error('Fresh scope/human/review with all exact file SHA256 pins required')
  result=execute(scope_path=args.scope,scope_sha256=args.scope_sha256,human_path=args.human_authorization,human_sha256=args.human_authorization_sha256,
   review_path=args.independent_review,review_sha256=args.independent_review_sha256,repo_root=root)
 print(json.dumps(result,sort_keys=True))
