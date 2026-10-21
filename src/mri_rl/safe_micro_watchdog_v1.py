"""Independent non-MLX parent guardian for its ONE owned native child.

Samples global pressure/swap/free memory/disk/battery and exact child RSS during
atomic load. Cannot query another process's MLX allocator: those readings remain
unknown until the child reports at boundaries. Sampling/termination are best
effort; kernel/GPU work may be uninterruptible. No promise against OOM or panic.
"""
from __future__ import annotations
import hashlib,json,math,os,platform,signal,subprocess,time
from pathlib import Path
from dataclasses import asdict,is_dataclass
from mri_rl.shared_rollout_gspo_core_v1 import publish_once,replace_atomic
from mri_rl.safe_micro_witness_v1 import digest

POLICY={'sample_interval_seconds':2.,'available_immediate_stop_below_GiB':2.,'available_sustained_below_GiB':4.,'sustained_low_seconds':30.,
 'pressure_free_stop_at_or_below':15.,'OS_CRITICAL':4,'RSS_stop_above_GiB':14.,'original_swap_baseline_GiB':13.96966796875,'swap_net_cap_GiB':1.5,
 'swap_absolute_cap_GiB':16.,'disk_forecast_plus_reserve_GiB':26.,'battery_floor':20.,'essential_unknown_timeout_seconds':120.,'SIGTERM_grace_seconds':10.,
 'stop_target':'ONLY exact Popen-owned child PID, never app/other process/group','MLX_external':'UNAVAILABLE; child12GiB gate and after-call allocator observations remain required',
 'ordinary_swap_rate':'trend only; no stop from rate alone','monitor_log_failure':'STOP exact owned child; preserve exit if destination still writable','automatic_retry':False}


def finite(x):return type(x) in (int,float) and math.isfinite(x) and x>=0


class Guardian:
 def __init__(self):self.low_since=None;self.unknown_since=None
 def observe(self,s,level,disk,now):
  reasons=[];values={} if s is None else {k:getattr(s,k,None) for k in ('available_gib','pressure_free_percent','swap_used_gib','process_rss_gib','battery_percent')}
  essential=(bool(values) and all(finite(v) for v in values.values()) and 0<=values['pressure_free_percent']<=100 and 0<=values['battery_percent']<=100 and type(level) is int and level in (1,2,4) and finite(disk))
  if not essential:
   self.unknown_since=now if self.unknown_since is None else self.unknown_since
   if now-self.unknown_since>=120.:reasons.append('essential_monitor_unknown_timeout')
  else:self.unknown_since=None
  get=lambda name:values.get(name)
  available=get('available_gib');free=get('pressure_free_percent');rss=get('process_rss_gib');swap=get('swap_used_gib');battery=get('battery_percent')
  if level==4:reasons.append('OS_CRITICAL')
  if finite(free) and free<=15.:reasons.append('pressure_free_at_or_below15')
  if finite(available):
   if available<2.:reasons.append('available_below2GiB')
   if available<4.:
    self.low_since=now if self.low_since is None else self.low_since
    if now-self.low_since>=30.:reasons.append('sustained_available_below4GiB')
   else:self.low_since=None
  if finite(rss) and rss>14.:reasons.append('exact_child_RSS_above14GiB')
  if finite(swap):
   if swap-13.96966796875>1.5:reasons.append('original_baseline_swap_net_cap')
   if swap>16.:reasons.append('swap_absolute_cap')
  if finite(battery) and battery<20.:reasons.append('battery_below20')
  if finite(disk) and disk<26.:reasons.append('disk_below_forecast_plus_reserve')
  return reasons


def append_monitor(path,row):
 with Path(path).open('a') as stream:
  stream.write(json.dumps(row,sort_keys=True)+'\n');stream.flush();os.fsync(stream.fileno())


def emergency_reap_owned(child):
 # Prove ownership BEFORE signaling, including the interrupt-after-wait4
 # window in which returncode has not yet been assigned but PID was reaped.
 try:pid,status,usage=os.wait4(child.pid,os.WNOHANG)
 except ChildProcessError:return
 if pid:child.returncode=os.waitstatus_to_exitcode(status);return
 try:os.kill(child.pid,signal.SIGTERM)
 except ProcessLookupError:pass
 start=time.monotonic()
 while True:
  pid,status,usage=os.wait4(child.pid,os.WNOHANG)
  if pid:child.returncode=os.waitstatus_to_exitcode(status);return
  if time.monotonic()-start>=10.:
   try:os.kill(child.pid,signal.SIGKILL)
   except ProcessLookupError:pass
  time.sleep(.1)


def witness_child(argv,output,*,scope_sha256,sampler,level,disk_free_gib,post_sampler,lock_inspector,interval=2.):
 output=Path(output);output.mkdir(parents=True,exist_ok=False);started=time.time();t0=time.monotonic();guardian=Guardian();stopped=False;stop_reasons=[];term_time=None;killed=False;samples=0;peak=0.;next_sample=t0;monitor_errors=[]
 publish_once(output/'launch_PRIVATE.json',{'scope_file_sha256':scope_sha256,'argv_sha256':hashlib.sha256(json.dumps(argv).encode()).hexdigest(),'started_unix':started,'guardian_policy':POLICY,'automatic_retry':False})
 with (output/'stdout_PRIVATE.log').open('xb') as stdout,(output/'stderr_PRIVATE.log').open('xb') as stderr:
  child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr)
  try:publish_once(output/'child_pid_PRIVATE.json',{'pid':child.pid,'parent_pid':os.getpid()})
  except BaseException:
   emergency_reap_owned(child);raise
  try:
   while True:
    pid,status,usage=os.wait4(child.pid,os.WNOHANG)
    if pid:child.returncode=os.waitstatus_to_exitcode(status);break
    now=time.monotonic()
    if now>=next_sample:
     sample=pressure=disk=None;errors={}
     for name,fn in (('sample',lambda:sampler(child.pid)),('level',level),('disk',disk_free_gib)):
      try:
       value=fn()
       if name=='sample':sample=value
       elif name=='level':pressure=value
       else:disk=value
      except Exception as error:errors[name]={'type':type(error).__name__,'message':str(error)}
     observed=time.monotonic();reasons=guardian.observe(sample,pressure,disk,observed)
     raw=None if sample is None else asdict(sample) if is_dataclass(sample) else dict(vars(sample)) if hasattr(sample,'__dict__') else {'status':'MALFORMED_SAMPLE'}
     if sample is not None and finite(getattr(sample,'process_rss_gib',None)):peak=max(peak,sample.process_rss_gib)
     try:append_monitor(output/'independent_monitor_PRIVATE.jsonl',{'monotonic':observed,'child_pid':child.pid,'sample':raw,'OS_pressure_level':pressure,'disk_free_GiB':disk,'errors':errors,'stop_reasons':reasons,'external_MLX_active_peak':'NOT_OBSERVABLE'})
     except Exception as error:
      reasons.append('monitor_log_write_failure');monitor_errors.append({'type':type(error).__name__,'message':str(error)})
     samples+=1;next_sample=observed+interval
     if reasons and not stopped:
      stopped=True;stop_reasons=reasons;term_time=observed
      # Do not call Popen.poll/terminate: poll can reap an exiting child and
      # steal the exact wait4 resource/exit witness. Our unreaped PID cannot be
      # reused by another process. Only this Popen-owned child is signaled.
      try:os.kill(child.pid,signal.SIGTERM)
      except ProcessLookupError:pass
    if stopped and not killed and time.monotonic()-term_time>=10.:
     try:os.kill(child.pid,signal.SIGKILL)
     except ProcessLookupError:pass
     killed=True
    time.sleep(min(.1,max(.001,next_sample-time.monotonic())))
  except BaseException:
   if child.returncode is None:emergency_reap_owned(child)
   raise
  stdout.flush();stderr.flush();os.fsync(stdout.fileno());os.fsync(stderr.fileno())
 unit=1024**3 if platform.system()=='Darwin' else 1024**2
 result={'schema':'safe-micro-independent-guardian-exit-v1','scope_file_sha256':scope_sha256,'native_exit_code':child.returncode,'child_pid':child.pid,'child_reaped':True,
 'started_unix':started,'finished_unix':time.time(),'wall_seconds':time.monotonic()-t0,'stdout_sha256':digest(output/'stdout_PRIVATE.log'),'stderr_sha256':digest(output/'stderr_PRIVATE.log'),
 'child_peak_RSS_GiB':usage.ru_maxrss/unit,'peak_RSS_provenance':'os.wait4 exact child; platform-native units','independent_sampled_child_peak_RSS_GiB':peak,
 'guardian_samples':samples,'monitor_log_errors':monitor_errors,'guardian_stop':stopped,'guardian_stop_reasons':stop_reasons,'stop_target_pid':child.pid if stopped else None,'SIGKILL_after_grace':killed,
 'post_child_resource_sample':{'status':'PENDING'},'post_child_global_lock':{'status':'PENDING'},'post_child_inspection_complete':False,'post_child_inspection_errors':{},
 'allocator_release_proof':'SEPARATE_CHILD_RECEIPT_OR_NOT_MEASURED','parent_Adam_RNG_integrity':'NOT_ATTESTED_BY_EXIT_WITNESS','automatic_retry':False,'automatic_next_stage':False}
 publish_once(output/'actual_exit_PRIVATE.json',result)
 for name,observe in (('post_child_resource_sample',post_sampler),('post_child_global_lock',lock_inspector)):
  try:result[name]=observe()
  except Exception as error:
   result[name]={'status':'UNAVAILABLE','error_type':type(error).__name__};result['post_child_inspection_errors'][name]={'type':type(error).__name__,'message':str(error)}
 result['post_child_inspection_complete']=True;replace_atomic(output/'actual_exit_PRIVATE.json',result);return result
