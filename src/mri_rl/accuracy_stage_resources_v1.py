"""This-round model-stage resource gate, source-only until exact stage authorization.

Distinct from the cache-only 5.5GiB/RSS8 policy: model load8GiB/free35%,
resident unit6GiB/free30%, RSS/MLX18GiB. The original swap baseline/debt is
carried. Actual BATTERY samples are retained; no global power policy is edited.
Samples/limits are boundary checks, not an interruptible hard runtime guarantee.
"""
from __future__ import annotations
import fcntl,json,math,os,shutil,stat,time
from pathlib import Path
from mri_rl import accuracy_recovery_gpu_cache_v3 as v3, accuracy_recovery_gpu_cache_v2 as v2
from mri_rl import accuracy_recovery_cpu_cache_v1 as cc, resource_supervisor

BASELINE_SWAP_GIB=13.96966796875
MODEL_POLICY={**v3.POLICY,'continue_available_gib_min':6.,'continue_pressure_free_percent_min':30.,
 'rss_envelope_gib':18.,'mlx_active_gib_stop_above':18.,'ac_required':False,'battery_percent_min':20.,
 'hold_seconds_per_boundary_max':300.,'operation_start_window_seconds':10800.,
 'load_available_gib_min':8.,'load_pressure_free_percent_min':35.,'disk_reserve_gib':20.,
 'power_exception':'This round only: real battery>=20 allowed, battery<20 STOP, unknown HOLD'}


def require(condition,message):
 if not condition:raise ValueError(message)


class ResourceController(v3.ResourceController):
 def __init__(self,**kwargs):
  baseline=kwargs.pop('baseline_swap_gib',BASELINE_SWAP_GIB)
  require(baseline==BASELINE_SWAP_GIB,'Original swap baseline/debt cannot reset')
  super().__init__(policy=dict(MODEL_POLICY),baseline_swap_gib=baseline,**kwargs)
  self.start_window()
 def _decide(self,s,level,required,now):
  decision,reasons,growth,trend=super()._decide(s,level,required,now)
  reasons=[('process_rss_above_18gib' if reason=='process_rss_above_8gib' else reason) for reason in reasons if reason!='not_ac']
  battery=getattr(s,'battery_percent',None)
  if not v3._valid_number(battery,0.,100.):reasons.append('battery_unknown_or_invalid')
  elif battery<20.:decision='STOP';reasons.append('battery_below_20')
  return ('STOP' if decision=='STOP' else 'HOLD' if reasons else 'CONTINUE'),reasons,growth,trend
 def _deadline(self):
  if self.clock()-self.t0>=MODEL_POLICY['operation_start_window_seconds']:
   raise v2.WindowExceeded('Global10800s operation-start deadline exhausted (holds/cache/loading included)')
 def boundary(self,name,**kwargs):
  # The cache has been required before model materialization; quantized_model_ready
  # is resident. Only the actual native_model_load boundary gets the load floor.
  loading='native_model_load' in str(name)
  self.policy['continue_available_gib_min']=8. if loading else 6.
  self.policy['continue_pressure_free_percent_min']=35. if loading else 30.
  self._deadline();result=super().boundary(name,**kwargs);self._deadline()
  return result


def assert_global_lock_held(path):
 """Entry owns the flock. Verify it remains held, without creating/unlinking it.

 Advisory check cannot authenticate another process's owner; the exact entry's
 own enclosing flock is authoritative. Never take over an absent/unheld lock.
 """
 path=Path(path)
 require(path.is_absolute() and path.exists() and stat.S_ISREG(os.lstat(path).st_mode) and not path.is_symlink(),'Existing canonical global flock file required')
 fd=os.open(path,os.O_RDWR)
 try:
  try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return
  fcntl.flock(fd,fcntl.LOCK_UN)
  raise ValueError('Global flock is not held by the enclosing authorized entry')
 finally:os.close(fd)


def make_resource_gate(spec,scope,authorization):
 """Return only callable(name)->None. Constructing it does not sample this host.

 Runtime must contain global_lock, and the authorized scope must bind its output
 and this-round policy. Output resources live alongside the fresh run directory.
 Optional scope.resource_policy.disk_required_bytes reserves additional output
 space above the inherited20GiB floor. No model/native imports occur here.
 """
 require(callable(getattr(authorization,'check',None)) and getattr(authorization,'scope',None)==scope,'Exact enclosing stage authorization required')
 policy=scope.get('resource_policy',{})
 require(policy.get('this_round_only') is True and policy.get('battery_percent_min')==20 and policy.get('global_wall_seconds')==10800,'Exact this-round power/global resource contract required')
 output=Path(scope['output']);require(output.is_absolute(),'Absolute authorized output required')
 runtime=json.loads(Path(spec['runtime_spec']).read_text());lock=Path(runtime['global_lock'])
 need=policy.get('disk_required_bytes',0)
 require(type(need) is int and need>=0,'Nonnegative integer output disk forecast required')
 ctrl=ResourceController(sampler=lambda:resource_supervisor.sample_macos(),os_level=lambda:v3.os_pressure_level(),
  log_path=output.with_name(output.name+'_resources_PRIVATE.jsonl'))
 def guards(name):
  authorization.check('resources_'+str(name))
  assert_global_lock_held(lock)
  try:free=shutil.disk_usage(output.parent).free
  except OSError as error:raise cc.ResourceStop('Disk sensor unavailable',['disk_unknown']) from error
  if not (isinstance(free,(int,float)) and not isinstance(free,bool) and math.isfinite(free) and free>=need+MODEL_POLICY['disk_reserve_gib']*1024**3):
   raise cc.ResourceStop('Free disk below output forecast plus20GiB reserve',['disk_reserve'])
 def before(name):
  guards(name);ctrl.boundary(name);guards(name)  # repeat authority/flock/disk after any HOLD
  return None
 return before
