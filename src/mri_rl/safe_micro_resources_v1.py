"""Resource classes only: audited cache A, conservative model B; five separate permissions.

No import here initializes MLX or samples the host. Model limits/setters are only
used by a separately authorized loader. All limits are sampled at safe boundaries,
not an interruptible guarantee. No old source, global gate, or scope is changed.
"""
from __future__ import annotations
import json,math,os,shutil,time
from pathlib import Path
from mri_rl import accuracy_recovery_gpu_cache_v4 as v4, accuracy_recovery_gpu_cache_v3 as v3, accuracy_recovery_gpu_cache_v2 as v2
from mri_rl import accuracy_recovery_cpu_cache_v1 as cc, resource_supervisor
from mri_rl.accuracy_stage_resources_v1 import assert_global_lock_held

BASELINE_SWAP_GIB=13.96966796875
CACHE_POLICY=dict(v4.POLICY)
WINDOWS={'CACHE_ONLY':1800.,'SINGLE_LOAD_FORWARD':900.,'MICRO_TRAIN':1800.,'SMALL_PILOT':1800.,'DEV24_EVAL':2400.}
cache_controller=v4.ResourceController
MODEL_MEMORY_LIMIT_BYTES=12*1024**3
MODEL_CACHE_LIMIT_BYTES=256*1024**2
MODEL_POLICY={**v4.POLICY,'continue_available_gib_min':6.,'continue_pressure_free_percent_min':35.,
 'load_available_gib_min':10.,'load_pressure_free_percent_min':40.,'rss_envelope_gib':14.,'mlx_active_gib_stop_above':12.,
 'operation_start_window_seconds':1800.,'hold_seconds_per_boundary_max':120.,'disk_reserve_gib':20.,
 'memory_limit_gib':12.,'mlx_memory_limit_bytes':MODEL_MEMORY_LIMIT_BYTES,'mlx_cache_limit_bytes':MODEL_CACHE_LIMIT_BYTES,
 'coupled_swap_rate_gib_per_min_min':.25,'coupled_swap_interval_seconds_min':5.,'coupled_swap_consecutive_intervals':2,
 'coupled_swap_net_growth_gib_min':.25,'coupled_swap_available_gib_below':8.,'coupled_swap_free_percent_below':40.,
 'coupled_swap_rule':'HOLD only after2 consecutive >=5s intervals rate>=.25GiB/min AND original-baseline net growth>=.25GiB AND (available<8GiB OR free<40%). Hard net1.5/absolute16GiB STOP unchanged.'}


def require(condition,message):
 if not condition:raise ValueError(message)


class ModelController(v4.ResourceController):
 def __init__(self,window_seconds=1800.,**kwargs):
  require(window_seconds in (900.,1800.,2400.),'Unknown finite operation window')
  baseline=kwargs.pop('baseline_swap_gib',BASELINE_SWAP_GIB)
  require(baseline==BASELINE_SWAP_GIB,'Original swap debt must not reset')
  v3.ResourceController.__init__(self,policy=dict(MODEL_POLICY),baseline_swap_gib=baseline,**kwargs)
  self.policy['operation_start_window_seconds']=window_seconds
  self.swap_warning_streak=0;self.peak_mlx_active_gib=0.;self.peak_mlx_reported_gib=0.;self.start_window()
 def _decide(self,s,level,required,now):
  previous_time=self.last_swap_time
  decision,reasons,growth,trend=super()._decide(s,level,required,now)
  reasons=[('process_rss_above_14gib' if reason=='process_rss_above_8gib' else reason) for reason in reasons]
  timestamp=getattr(s,'monotonic_seconds',None)
  interval=timestamp-previous_time if v3._valid_number(timestamp,0.,None) and v3._valid_number(previous_time,0.,None) else None
  sustained=interval is not None and interval>=5. and v3._valid_number(trend,.25,None)
  self.swap_warning_streak=self.swap_warning_streak+1 if sustained else 0
  coupled=((v3._valid_number(s.available_gib,0.,None) and s.available_gib<8.) or
           (v3._valid_number(s.pressure_free_percent,0.,100.) and s.pressure_free_percent<40.))
  if self.swap_warning_streak>=2 and growth is not None and growth>=.25 and coupled:
   reasons.append('coupled_sustained_swap_growth')
   if decision!='STOP':decision='HOLD'
  for attr,target in (('mlx_active_gib','peak_mlx_active_gib'),('mlx_peak_gib','peak_mlx_reported_gib')):
   value=getattr(s,attr,None)
   if v3._valid_number(value,0.,None):setattr(self,target,max(getattr(self,target),float(value)))
  return decision,reasons,growth,trend
 def _sample(self,*args,**kwargs):
  # Reuse exact inherited sampling/decision mechanics; append the actual model
  # memory endpoints and observed peaks to the same durable row before writing.
  path=self.log_path;self.log_path=None
  try:result=super()._sample(*args,**kwargs)
  finally:self.log_path=path
  s=result[0]
  self.last.update(mlx_active_gib=getattr(s,'mlx_active_gib',None),mlx_reported_peak_gib=getattr(s,'mlx_peak_gib',None),
   peak_process_rss_gib=self.peak_rss_gib,peak_mlx_active_gib=self.peak_mlx_active_gib,peak_mlx_reported_gib=self.peak_mlx_reported_gib,
   coupled_swap_warning_streak=self.swap_warning_streak)
  if path:
   with Path(path).open('a') as stream:stream.write(json.dumps(self.last,sort_keys=True)+'\n');stream.flush();os.fsync(stream.fileno())
  return result
 def boundary(self,name,**kwargs):
  loading='native_model_load' in str(name)
  self.policy['continue_available_gib_min']=10. if loading else 6.
  self.policy['continue_pressure_free_percent_min']=40. if loading else 35.
  def deadline():
   if self.clock()-self.t0>=self.policy['operation_start_window_seconds']:raise v2.WindowExceeded('Authorized phase operation-start deadline exhausted; holds/loading included')
  deadline();result=super().boundary(name,**kwargs);deadline();return result


def _model_authority(authorization):
 require(callable(getattr(authorization,'check',None)) and isinstance(getattr(authorization,'scope',None),dict),'Exact model B authorization required')
 require(authorization.scope.get('resource_stage')=='B_MICRO' and authorization.scope.get('resource_policy')==MODEL_POLICY,'This is not the separately authorized model B resource contract')
 require(authorization.scope.get('execution_stage') in set(WINDOWS)-{'CACHE_ONLY'} and authorization.scope.get('action'),'Explicit separately authorized model phase/action required')
 authorization.check('model_B_resource_contract')


def configure_authorized_mlx(mx,authorization):
 """Set on an existing runtime only. The old native constructor overwrites these
 limits, so the reviewed local loader must also bind its constructor namespace
 to these values; calling this before the old constructor is insufficient.
 """
 _model_authority(authorization)
 mx.set_memory_limit(MODEL_MEMORY_LIMIT_BYTES);mx.set_cache_limit(MODEL_CACHE_LIMIT_BYTES)


def release_evidence(mx,model,authorization):
 """Actual existing-runtime observation after the caller drops its model refs.
 This is not parameter/Adam integrity proof, nor evidence of all Python refs.
 """
 _model_authority(authorization)
 peak=mx.get_peak_memory();mx.clear_cache();active=mx.get_active_memory()
 valid=type(active) in (int,float) and math.isfinite(active) and active>=0
 return {'schema':'safe-micro-model-release-v1','model_reference_is_none':model is None,
  'active_bytes_after_release':active,'mlx_reported_peak_bytes':peak,
  'release_verified':model is None and valid and active<128*1024**2,
  'limitations':'Observed allocator bytes and supplied actual model reference only; source/input/parameter/Adam/RNG integrity remains separately required.'}


class ResourceGate:
 def __init__(self,stage,spec,scope,authorization):
  require(stage in ('A_CACHE','B_MICRO') and callable(getattr(authorization,'check',None)) and getattr(authorization,'scope',None)==scope,'Exact stage resource authorization required')
  policy=CACHE_POLICY if stage=='A_CACHE' else MODEL_POLICY
  require(scope.get('resource_stage')==stage and scope.get('resource_policy')==policy,'Resource-class policy must be exact')
  phase=scope.get('execution_stage')
  require(phase in WINDOWS and scope.get('action') and ((phase=='CACHE_ONLY')==(stage=='A_CACHE')),'Separate execution phase/action must match the resource class')
  self.stage,self.scope,self.authorization=stage,scope,authorization
  self.output=Path(scope['output']);require(self.output.is_absolute(),'Absolute output required')
  runtime=json.loads(Path(spec['runtime_spec']).read_text());self.lock=Path(runtime['global_lock'])
  self.need=scope.get('disk_required_bytes',0);require(type(self.need) is int and self.need>=0,'Disk output forecast must be nonnegative integer')
  cls=cache_controller if stage=='A_CACHE' else ModelController
  self.controller=cls(sampler=lambda:resource_supervisor.sample_macos(),os_level=lambda:v3.os_pressure_level(),
   baseline_swap_gib=BASELINE_SWAP_GIB,log_path=self.output.with_name(self.output.name+'_resources_PRIVATE.jsonl'),
   **({'window_seconds':WINDOWS[phase]} if stage=='B_MICRO' else {}))
  if self.controller.t0 is None:self.controller.start_window()
 def _guards(self,name):
  self.authorization.check(self.stage+'_resources_'+str(name));assert_global_lock_held(self.lock)
  try:free=shutil.disk_usage(self.output.parent).free
  except OSError as error:raise cc.ResourceStop('Disk sensor unknown',['disk_unknown']) from error
  if not v3._valid_number(free,0.,None) or free<self.need+20.*1024**3:
   raise cc.ResourceStop('Disk below forecast plus20GiB reserve',['disk_reserve'])
 def __call__(self,name):
  self._guards(name);self.controller.boundary(name);self._guards(name)
  return None
 def peaks(self):
  return {'peak_process_rss_gib':self.controller.peak_rss_gib,
   'peak_mlx_active_gib':getattr(self.controller,'peak_mlx_active_gib',None),
   'peak_mlx_reported_gib':getattr(self.controller,'peak_mlx_reported_gib',None)}


def make_resource_gate(stage,spec,scope,authorization):
 """Return a callable boundary gate; construction does not sample resources."""
 return ResourceGate(stage,spec,scope,authorization)
