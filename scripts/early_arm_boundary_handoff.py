"""One-shot completed-arm handoff. Never signal a trainer or edit original cursor.

prepare is read-only except its new contract. arm requires independent static
review and operates only on the exact queue orchestrator at a verified no-child
completed-arm boundary. All model work lives in separately reviewed handlers.
"""
from pathlib import Path
import argparse,ctypes,fcntl,hashlib,importlib.util,json,os,shlex,signal,subprocess,sys,time,uuid

MANIFEST_SHA='f47169577b285d03e21dafe904244d77be83c10edcf730f3b72a7a129cd1bd01'
EXECUTOR_SHA='44c2636f3ee3c65b208cc7588eb15d49025b0c195f73eb86772353b9de042629'
EXEC_REVIEW_SHA='fe2d11c66f841b3cb7e76bbfffd1be98739baabd4fe708bc5e90122c21a0296c'
GATE_SHA='d2c14df7364991f66537bc4554b769ab3d75659e463b510fdde46b72cc8b214b'
def require(ok,why):
 if not ok:raise ValueError(why)
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def objsha(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
 with tmp.open('w') as f:json.dump(x,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,p);fd=os.open(p.parent,os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)

class BSDInfo(ctypes.Structure):
 # macOS SDK sys/proc_info.h struct proc_bsdinfo (136 bytes).
 _fields_=[('prefix',ctypes.c_uint32*12),('comm',ctypes.c_char*16),('name',ctypes.c_char*32),('middle',ctypes.c_uint32*6),('sec',ctypes.c_uint64),('usec',ctypes.c_uint64)]
def birth(pid):
 if sys.platform=='darwin':
  info=BSDInfo();lib=ctypes.CDLL('/usr/lib/libproc.dylib');n=lib.proc_pidinfo(int(pid),3,0,ctypes.byref(info),ctypes.sizeof(info))
  require(n==ctypes.sizeof(info),'Cannot authenticate microsecond process birth');return {'sec':int(info.sec),'usec':int(info.usec)}
 # Linux dummy-test support; no cross-platform production inference claims.
 value=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19];return {'linux_start_ticks':int(value)}
def identity(pid):
 p=subprocess.run(['ps','-ww','-p',str(pid),'-o','pid=,ppid=,lstart=,stat=,command='],capture_output=True,text=True)
 if p.returncode==1 and not p.stdout.strip():return None
 require(p.returncode==0,'Process inventory unavailable')
 fields=p.stdout.strip().split(None,8);require(len(fields)==9,'Unknown ps identity format')
 try:stamp=birth(pid)
 except ValueError:
  # The process may exit after ps captured its row but before proc_pidinfo.
  # Only a second successful absence observation converts this to None.
  confirm=subprocess.run(['ps','-p',str(pid),'-o','pid='],capture_output=True,text=True)
  if confirm.returncode==1 and not confirm.stdout.strip():return None
  raise
 return {'pid':int(fields[0]),'ppid':int(fields[1]),'lstart':' '.join(fields[2:7]),'status':fields[7],'command':fields[8],'birth':stamp}
def same(a,b):return bool(a and b and all(a.get(k)==b.get(k) for k in ('pid','lstart','command','birth')))
def table():
 query=subprocess.Popen(['ps','-axo','pid=,ppid=,stat=,command='],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
 query_pid=query.pid;text,error=query.communicate();require(query.returncode==0,'Process inventory query failed');out={}
 for line in text.splitlines():
  parts=line.strip().split(None,3)
  # Exclude only this exact already-reaped query, not ps-named processes.
  if len(parts)==4 and int(parts[0])!=query_pid:out[int(parts[0])]={'pid':int(parts[0]),'ppid':int(parts[1]),'status':parts[2],'command':parts[3]}
 return out
def descendants(pid):
 rows=table();found=set();parents={pid}
 while True:
  new={p for p,v in rows.items() if v['ppid'] in parents and p not in found}
  if not new:break
  found|=new;parents=new
 return [rows[p] for p in sorted(found) if not rows[p]['status'].startswith('Z')]
def alive(expected):
 now=identity(expected['pid']);return same(now,expected) and not now['status'].startswith('Z')
def guarded_signal(expected,sig):
 require(same(identity(expected['pid']),expected),'PID reused or command changed; signal refused')
 os.kill(expected['pid'],sig)
def quiet_cursor(cursor,stage):
 x=load(cursor);states=x.get('stages',{});require(states.get(stage,{}).get('state') in {'RUNNING','DONE'},'Target cursor is not a normally completing stage')
 require(not x.get('halted'),'Queue already halted; no automatic takeover')
 require(not any(k!=stage and v.get('state')=='RUNNING' for k,v in states.items()),'Next stage already reserved; skip boundary')
 return x

class RealOps:
 identity=staticmethod(identity);descendants=staticmethod(descendants);signal=staticmethod(guarded_signal);alive=staticmethod(alive);sleep=staticmethod(time.sleep)

def retire_at_boundary(expected,ops,verify,check_cursor,timeout=300,before_retire=None):
 """Stop/recheck one known orchestrator, verify while paused, then retire it.

 All failures before retirement CONT only that same process. No descendant signal.
 """
 require(same(ops.identity(expected['pid']),expected),'Wrong parent identity')
 if ops.descendants(expected['pid']):return None,'SKIPPED_CHILD_PRESENT'
 stopped=False
 try:
  ops.signal(expected,signal.SIGSTOP);stopped=True
  until=time.monotonic()+min(timeout,5)
  while True:
   now=ops.identity(expected['pid']);require(same(now,expected),'Parent identity changed after stop')
   if 'T' in now.get('status',''):break
   require(time.monotonic()<until,'Parent did not acknowledge STOP');ops.sleep(.01)
  if ops.descendants(expected['pid']):return None,'SKIPPED_CHILD_RACE'
  check_cursor();summary=verify()
  if ops.descendants(expected['pid']):return None,'SKIPPED_CHILD_RACE'
  check_cursor()
  # Paused validation is reversible until this exact boundary. A validation
  # alarm must never interrupt TERM/CONT or the original-lock handoff.
  if before_retire is not None:before_retire()
  ops.signal(expected,signal.SIGTERM)
  # Darwin can deliver pending TERM and reap a stopped process immediately.
  # CONT only the same surviving identity; absence is successful retirement.
  now=ops.identity(expected['pid'])
  known_zombie=bool(now and now.get('status','').startswith('Z') and now.get('pid')==expected['pid'] and now.get('birth')==expected['birth'])
  if same(now,expected) and not known_zombie:ops.signal(expected,signal.SIGCONT)
  elif now is not None and not known_zombie:require(False,'PID reused after TERM; never signal replacement')
  stopped=False
  until=time.monotonic()+min(timeout,10)
  while ops.alive(expected) and time.monotonic()<until:ops.sleep(.05)
  require(not ops.alive(expected),'Parent did not terminate; handoff refused')
  return summary,'PARENT_RETIRED_WITHOUT_CHILD'
 finally:
  if stopped and same(ops.identity(expected['pid']),expected):ops.signal(expected,signal.SIGCONT)

def original(ex_root):
 sys.path.insert(0,str(Path(ex_root)/'src'));path=Path(ex_root)/'local_data/c_two_task_pool_v1/tier_c_v1/remaining20_bundle_v1/run_remaining20_PRIVATE.py'
 require(sha(path)==EXECUTOR_SHA,'Original reviewed executor changed')
 sp=importlib.util.spec_from_file_location('original_reviewed_queue',path);module=importlib.util.module_from_spec(sp);sp.loader.exec_module(module);return module

def validate_contract(c):
 require(c['schema']=='natural-arm-boundary-handoff-contract-v1' and c['boundary_stage_id']=='rl:P_raw:13','Wrong one-shot handoff contract')
 for key in ('controller_source','original_manifest','original_executor','original_executor_review','handler_source'):
  require(sha(c[key+'_path'])==c[key+'_sha256'],'Pinned source/artifact changed: '+key)
 require(c['controller_source_sha256']==sha(__file__) and c['original_manifest_sha256']==MANIFEST_SHA and c['original_executor_sha256']==EXECUTOR_SHA and c['original_executor_review_sha256']==EXEC_REVIEW_SHA,'Wrong original release')
 m=load(c['original_manifest_path']);require(c['original_queue_argv'][0]==m['lexical_python'],'Wrong lexical original interpreter')
 argv=c['original_queue_argv'];require(c['original_executor_path'] in argv and '--execute' in argv and argv[argv.index('--expected-manifest-sha256')+1]==MANIFEST_SHA,'Not the reviewed original executing queue')
 for flag,value in (('--manifest',c['original_manifest_path']),('--executor-review',c['original_executor_review_path']),('--expected-executor-review-sha256',EXEC_REVIEW_SHA),('--U7-prerequisite-release',c['U7_prerequisite_path']),('--expected-U7-prerequisite-sha256',GATE_SHA)):
  require(flag in argv and argv[argv.index(flag)+1]==value,'Original argv binding differs: '+flag)
 observed=shlex.split(c['expected_old_parent_identity']['command'])
 require(observed[1:]==argv[1:] and observed[0]==c['observed_python_executable_path'],'Parent argv identity differs')
 require(sha(c['U7_prerequisite_path'])==GATE_SHA and c['U7_prerequisite_sha256']==GATE_SHA,'Actual original prerequisite changed')
 require(c['queue_lock_path']==str(Path(c['original_executor_path']).parent/'remaining20.lock') and c['queue_cursor_path']==str(Path(c['original_executor_path']).parent/'remaining20_cursor_PRIVATE.json'),'Not the original global lock/cursor')
 require(0<c['wait_timeout_seconds']<=7200 and 0<c['paused_validation_timeout_seconds']<=300 and 0<c['diagnostic_timeout_seconds']<=3600,'Unbounded one-off wait')
 require(len(c['diagnostic_commands'])==2 and len(c['diagnostic_status_paths'])==2,'Exact two sequential checkpoints required')
 for argv in c['diagnostic_commands']:
  require(c['handler_source_path'] in argv and '--execute' in argv and not any(x in argv for x in ('--lease','--lease-lock-fd','--expected-lease-sha256','--review-verdict','--expected-review-sha256')),'Unexpected handler base argv')
 if c.get('retired_boundary_recovery'):
  r=c['retired_boundary_recovery']
  for key in ('failed_contract','failed_marker','failed_controller_source'):
   require(sha(r[key+'_path'])==r[key+'_sha256'],'Retired recovery evidence changed: '+key)
  old=load(r['failed_contract_path']);marker=load(r['failed_marker_path'])
  require(old['controller_source_sha256']==r['failed_controller_source_sha256'] and marker['contract_sha256']==r['failed_contract_sha256'],'Recovery is not bound to actually armed failed source')
  require(old['expected_old_parent_identity']==c['expected_old_parent_identity'] and old['boundary_stage_id']==c['boundary_stage_id'] and old['original_manifest_sha256']==MANIFEST_SHA,'Recovery changes original boundary')
 return m

def prepare(config,output):
 x=load(config);root=Path(x['primary_root']).absolute();base=root/'local_data/c_two_task_pool_v1/tier_c_v1/remaining20_bundle_v1';parent=identity(x['queue_pid'])
 require(parent and 'run_remaining20_PRIVATE.py' in parent['command'],'Supplied PID is not original orchestrator')
 c={**x,'schema':'natural-arm-boundary-handoff-contract-v1','primary_root':str(root),'boundary_stage_id':'rl:P_raw:13','expected_old_parent_identity':parent,
 'original_queue_argv':shlex.split(parent['command']),'controller_source_path':str(Path(__file__).absolute()),'controller_source_sha256':sha(__file__),
 'original_manifest_path':str(base/'manifest_READY_REQUIRES_REVIEW_PRIVATE.json'),'original_manifest_sha256':MANIFEST_SHA,
 'original_executor_path':str(base/'run_remaining20_PRIVATE.py'),'original_executor_sha256':EXECUTOR_SHA,
 'original_executor_review_path':str(base/'executor_review_PRIVATE.json'),'original_executor_review_sha256':EXEC_REVIEW_SHA,
 'queue_lock_path':str(base/'remaining20.lock'),'queue_cursor_path':str(base/'remaining20_cursor_PRIVATE.json'),
 'U7_prerequisite_path':str(base.parent/'remaining_raw_bundle_v1/U_raw7_actual_prerequisite_release_SAFE.json'),'U7_prerequisite_sha256':GATE_SHA}
 # macOS ps displays the resolved Python Mach-O path; execution must retain
 # lexical venv argv[0] to preserve sys.prefix and installed packages.
 lexical=load(c['original_manifest_path'])['lexical_python'];c['original_queue_argv'][0]=lexical
 probe='import sys,os,subprocess,json;print(json.dumps(dict(executable=sys.executable,prefix=sys.prefix,ps=subprocess.check_output(["ps","-ww","-p",str(os.getpid()),"-o","command="],text=True))))'
 evidence=json.loads(subprocess.check_output([lexical,'-c',probe],text=True));require(evidence['executable']==lexical and evidence['prefix']==str(Path(lexical).parent.parent),'Lexical venv CPU identity invalid')
 c['observed_python_executable_path']=evidence['ps'].strip().split(' -c ',1)[0];c['lexical_python_CPU_probe']=evidence
 c['handler_source_sha256']=sha(c['handler_source_path']);validate_contract(c);require(not Path(output).exists(),'Preserve existing contract');save(output,c)
 return {'status':'PREPARED_NOT_ARMED','contract_sha256':sha(output),'parent_identity':parent}


def prepare_recovery(config,output):
 x=load(config);r=x['retired_boundary_recovery'];require(sha(r['failed_contract_path'])==r['failed_contract_sha256'],'Wrong failed contract')
 old=load(r['failed_contract_path']);c={**old,**x,'controller_source_path':str(Path(__file__).absolute()),'controller_source_sha256':sha(__file__)}
 require(identity(c['expected_old_parent_identity']['pid']) is None,'Retired original parent still exists; no recovery')
 require(not Path(output).exists(),'Preserve existing recovery contract');validate_contract(c);save(output,c)
 return {'status':'RETIRED_RECOVERY_PREPARED_NOT_EXECUTED','contract_sha256':sha(output)}

def timeout_alarm(signum,frame):raise TimeoutError('Bounded paused-phase validation expired')
def cheap_terminal(ex,st):
 if not Path(st['exit_receipt']).is_file():return False
 witness=load(st['exit_receipt']);require(witness.get('exit_code')==0,'Current arm failed; no boundary takeover')
 require(witness['stage_id']==st['id'] and witness['launcher_sha256']==st['launcher_sha256'] and witness['review_sha256']==st['review_sha256'] and witness['raw_log_sha256']==sha(st['raw_log']),'Actual closed arm exit/log binding differs')
 r=ex.terminal_json(st['raw_log']);require(r['status'] in {'TOKEN_BUDGET_EXHAUSTED','SCHEDULE_COMPLETE'},'Not actual full-budget terminal')
 return True

def restart(c,lockfd):
 require(not descendants(os.getpid()),'Live child remains; original queue not restarted')
 require(sha(c['original_executor_path'])==EXECUTOR_SHA and sha(c['original_manifest_path'])==MANIFEST_SHA,'Original restart release changed')
 os.set_inheritable(lockfd,False) # kernel closes/relinquishes the lease atomically on exec
 os.execv(c['original_queue_argv'][0],c['original_queue_argv'])

def arm(contract_path,expected_contract,review_path,expected_review,approval_index,expected_index):
 require(sha(contract_path)==expected_contract and sha(review_path)==expected_review and sha(approval_index)==expected_index,'Exact independent static artifacts required')
 c=load(contract_path);m=validate_contract(c);review=load(review_path)
 require(review.get('verdict') in {'PASS','PASS_WITH_LIMITATIONS'} and review.get('reviewer_kind')=='isolated_codex' and review.get('reviewer_id')
  and review.get('controller_source_sha256')==sha(__file__) and review.get('boundary_contract_sha256')==expected_contract and review.get('diagnostic_approval_index_sha256')==expected_index,'No applicable independent handoff review')
 ex=original(c['primary_root']);require(ex.verify_config(m)['status']=='ALL20_CONFIGS_VERIFIED_NO_MODELS_STARTED','Original configs changed')
 st=next(s for s in m['stages'] if s['id']==c['boundary_stage_id']);expected=c['expected_old_parent_identity'];out=Path(c['handoff_output_dir']);out.mkdir(parents=True,exist_ok=True)
 with (out/'armed_SAFE.json').open('x') as marker:
  json.dump({'contract_sha256':expected_contract,'review_sha256':expected_review,'controller_identity':identity(os.getpid())},marker);marker.flush();os.fsync(marker.fileno())
 summary=None
 if c.get('retired_boundary_recovery'):
  r=c['retired_boundary_recovery']
  require(identity(expected['pid']) is None,'Original parent is not absent; recovery never signals')
  require(sha(st['exit_receipt'])==r['actual_exit_receipt_sha256'] and sha(st['raw_log'])==r['actual_closed_log_sha256'],'Actual completed boundary differs')
  require(sha(c['queue_cursor_path'])==r['expected_cursor_sha256'],'Recovery cursor changed')
  require(cheap_terminal(ex,st),'Actual completed boundary missing');quiet_cursor(c['queue_cursor_path'],st['id'])
  summary=ex.verify_rl_terminal(st,st['exit_receipt'])
 else:
  deadline=time.monotonic()+c['wait_timeout_seconds'];summary=None
  while time.monotonic()<deadline:
   if not same(identity(expected['pid']),expected):save(out/'handoff_status_SAFE.json',{'status':'SKIPPED_PARENT_CHANGED_NO_SIGNALS'});return
   if cheap_terminal(ex,st):break
   time.sleep(1.)
  else:save(out/'handoff_status_SAFE.json',{'status':'SKIPPED_TIMEOUT_NO_SIGNALS'});return
  # All validation failures/timeout here resume the old queue; signals never target a child.
  prior_handler=signal.signal(signal.SIGALRM,timeout_alarm);signal.setitimer(signal.ITIMER_REAL,c['paused_validation_timeout_seconds'])
  try:
   summary,status=retire_at_boundary(expected,RealOps(),lambda:ex.verify_rl_terminal(st,st['exit_receipt']),lambda:quiet_cursor(c['queue_cursor_path'],st['id']),before_retire=lambda:signal.setitimer(signal.ITIMER_REAL,0))
  finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,prior_handler)
  if summary is None:save(out/'handoff_status_SAFE.json',{'status':status,'queue_left_running':True});return
 lock=open(c['queue_lock_path'],'a');lockfd=lock.fileno();os.set_inheritable(lockfd,False)
 until=time.monotonic()+10
 while True:
  try:fcntl.flock(lockfd,fcntl.LOCK_EX|fcntl.LOCK_NB);break
  except BlockingIOError:
   if time.monotonic()>=until:save(out/'handoff_status_SAFE.json',{'status':'OTHER_QUEUE_OWNS_LOCK_NO_DIAGNOSTIC'});return
   time.sleep(.1)
 safe_to_resume=True;lease_sha=None
 try:
  require(not descendants(os.getpid()),'Unexpected controller child before lease')
  cursor=quiet_cursor(c['queue_cursor_path'],st['id']);cursor_sha=sha(c['queue_cursor_path']);info=os.fstat(lockfd)
  if c.get('retired_boundary_recovery'):
   require(cursor_sha==c['retired_boundary_recovery']['expected_cursor_sha256'] and identity(expected['pid']) is None,'Recovery state changed before exclusive lease')
  save(out/'original_cursor_PRIVATE.json',cursor);save(out/'original_exit_PRIVATE.json',load(st['exit_receipt']))
  lease={'schema':'natural-arm-boundary-handoff-lease-v1','lease_id':str(uuid.uuid4()),'contract_sha256':expected_contract,'controller_source_sha256':sha(__file__),
   'controller_identity':identity(os.getpid()),'original_parent_identity':expected,'original_parent_exited':not alive(expected),'boundary_stage_id':st['id'],
   'arm_exit_receipt_sha256':sha(st['exit_receipt']),'closed_native_log_sha256':sha(st['raw_log']),'completed_boundary_receipt_sha256':summary['boundary_receipt_sha256'],
   'terminal_summary':summary,'queue_cursor_sha256':cursor_sha,'queue_lock':{'path':c['queue_lock_path'],'st_dev':info.st_dev,'st_ino':info.st_ino},
   'diagnostic_command_base_sha256':[objsha(v) for v in c['diagnostic_commands']],'no_model_children':True,'model_child_pids':[]}
  require(lease['original_parent_exited'],'Original parent still alive');leasepath=out/'lease_PRIVATE.json';save(leasepath,lease);lease_sha=sha(leasepath)
  approvals=load(approval_index)
  for i,base in enumerate(c['diagnostic_commands']):
   require(sha(c['queue_cursor_path'])==cursor_sha,'Original queue cursor unexpectedly changed')
   a=approvals[str(i)];require(sha(a['review_path'])==a['review_sha256'],'Diagnostic independent review changed')
   argv=base+['--lease',str(leasepath),'--expected-lease-sha256',lease_sha,'--lease-lock-fd',str(lockfd),'--review-verdict',a['review_path'],'--expected-review-sha256',a['review_sha256']]
   safe_to_resume=False
   with (out/f'diagnostic_{i}_stdout_PRIVATE.log').open('xb') as log:
    child=subprocess.Popen(argv,cwd=c['primary_root'],stdout=log,stderr=subprocess.STDOUT,pass_fds=(lockfd,))
    try:code=child.wait(timeout=c['diagnostic_timeout_seconds'])
    except subprocess.TimeoutExpired:
     # Never kill a potentially healthy native process. Its inherited FD retains
     # the global lock until it exits; no queue restart or second child occurs.
     save(out/'handoff_status_SAFE.json',{'status':'DIAGNOSTIC_TIMEOUT_CHILD_LEFT_UNSIGNALLED_LOCK_INHERITED','child_pid':child.pid,'queue_resume_safe':False});raise
   require(not descendants(os.getpid()),'Diagnostic descendant remains; no next model or queue restart')
   status_path=Path(c['diagnostic_status_paths'][i]);status=load(status_path) if status_path.exists() else {}
   safe_to_resume=bool(status.get('queue_resume_safe') is True and status.get('critical_safety_condition') is False and status.get('native_children_released') is True)
   require(code==0 and safe_to_resume,'Diagnostic failed/unknown safety; preserve intentional halt')
  save(out/'handoff_status_SAFE.json',{'status':'EARLY_DIAGNOSTICS_EXITED_QUEUE_RESTARTING','lease_sha256':lease_sha,'original_cursor_unchanged':sha(c['queue_cursor_path'])==cursor_sha})
 finally:
  if safe_to_resume and not descendants(os.getpid()):restart(c,lockfd)
  else:save(out/'handoff_status_SAFE.json',{'status':'HALTED_AFTER_DIAGNOSTIC_FAILURE_PENDING_REVIEW','queue_resume_safe':False,'lease_sha256':lease_sha})



def main():
 p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
 a=sub.add_parser('prepare');a.add_argument('--config',required=True);a.add_argument('--output',required=True)
 a=sub.add_parser('prepare-recovery');a.add_argument('--config',required=True);a.add_argument('--output',required=True)
 a=sub.add_parser('arm')
 for flag in ('contract','expected-contract-sha256','review','expected-review-sha256','approval-index','expected-approval-index-sha256'):a.add_argument('--'+flag,required=True)
 a=p.parse_args()
 if a.command=='prepare-recovery':print(json.dumps(prepare_recovery(a.config,a.output),sort_keys=True))
 elif a.command=='prepare':print(json.dumps(prepare(a.config,a.output),sort_keys=True))
 else:arm(a.contract,a.expected_contract_sha256,a.review,a.expected_review_sha256,a.approval_index,a.expected_approval_index_sha256)
if __name__=='__main__':main()
