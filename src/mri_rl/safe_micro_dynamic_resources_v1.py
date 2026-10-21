"""This micro pilot ONLY. OS-color admission, explicit last-resort protections.
No fixed9/10GiB admission, no120s ordinary-HOLD failure, no swap-only STOP.
RED exits a resident model; waiting/recovery happens in the non-MLX parent.
"""
import json,math,time,shutil
from pathlib import Path
from dataclasses import asdict
from mri_rl.resource_supervisor import sample_macos
from mri_rl.accuracy_recovery_gpu_cache_v3 import os_pressure_level
from mri_rl.shared_rollout_gspo_core_v1 import replace_atomic

POLICY={'version':'MICRO_COLOR_V1','normal':1,'warning':2,'critical':4,'sample_seconds':2.,'parent_wait_seconds':5.,
 'available_emergency_below_GiB':2.,'available_sustained_below_GiB':4.,'sustained_seconds':30.,'pressure_free_emergency_at_or_below_percent':15.,
 'RSS_limit_GiB':14.,'MLX_limit_GiB':12.,'MLX_cache_bytes':256*1024**2,'disk_min_GiB':26.,'battery_min_percent':20.,
 'swap_standalone_stop':False,'old_effective15_4697_and_absolute16_caps':'RETIRED_THIS_SCOPE_ONLY; logged, not standalone stops',
 'coupled_rate_GiB_per_min':.25,'coupled_min_interval_seconds':5.,'coupled_streak':2,'coupled_available_below_GiB':4.,'coupled_free_below_percent':20.,'coupled_disk_below_GiB':30.,
 'unknown_action':'WAIT_NONRESIDENT_OR_PAUSE_RELEASE_RESIDENT','warning_action':'CONTINUE_SERIAL_LOW_CACHE; coupled sustained swap plus severe memory or disk risk is emergency STOP',
 'recovery_good_samples':3,'compound_seconds':5100.,'model_load_attempts_max':7,'original_planned_model_loads':5,'extra_recovery_loads_max':2,
 'recoveries_max':2,'resident_wait_on_RED':False,'no_oom_guarantee':True}

class ResourcePause(RuntimeError):pass
class ResourceFatal(RuntimeError):pass

def valid(x,hi=None):return type(x) in (int,float) and math.isfinite(x) and x>=0 and (hi is None or x<=hi)

class Controller:
 def __init__(self):self.low_since=None;self.prev=None;self.streak=0
 def decide(self,s,level,disk,now):
  stop=[];wait=[];color={1:'GREEN',2:'YELLOW',4:'RED'}.get(level,'UNKNOWN') if type(level) is int else 'UNKNOWN'
  vals={k:getattr(s,k,None) for k in ('available_gib','pressure_free_percent','swap_used_gib','process_rss_gib','battery_percent')}
  if any(not valid(v,100 if k in ('pressure_free_percent','battery_percent') else None) for k,v in vals.items()) or not valid(disk) or color=='UNKNOWN':wait.append('essential_sensor_unknown')
  av,free,swap,rss,battery=(vals[k] for k in ('available_gib','pressure_free_percent','swap_used_gib','process_rss_gib','battery_percent'))
  if valid(rss) and rss>14:stop.append('RSS_above14GiB')
  mlx=getattr(s,'mlx_active_gib',None)
  if mlx is not None and (not valid(mlx) or mlx>12):stop.append('MLX_invalid_or_above12GiB')
  if valid(av):
   if av<2:stop.append('available_below2GiB')
   if av<4:
    self.low_since=now if self.low_since is None else self.low_since
    if now-self.low_since>=30:stop.append('available_below4GiB_for30s')
   else:self.low_since=None
  if valid(free,100) and free<=15:stop.append('free_at_or_below15percent')
  if valid(disk) and disk<26:stop.append('disk_below26GiB')
  if valid(battery,100) and battery<20:stop.append('battery_below20percent')
  if getattr(s,'thermal_state',None)=='THROTTLED':wait.append('thermal_throttled')
  heavy=getattr(s,'other_heavy_processes',None)
  if type(heavy) is not int or heavy<0:wait.append('other_heavy_unknown')
  elif heavy:wait.append('other_heavy_process')
  if getattr(s,'power_source',None) not in ('AC','BATTERY'):wait.append('power_unknown')
  rate=None
  if valid(swap):
   if self.prev and now-self.prev[0]>=5:
    rate=(swap-self.prev[1])*60/(now-self.prev[0]);self.streak=self.streak+1 if rate>=.25 else 0;self.prev=(now,swap)
   elif self.prev is None:self.prev=(now,swap)
  if self.streak>=2 and ((valid(av) and av<4 and valid(free,100) and free<20) or (valid(disk) and disk<30)):
   stop.append('coupled_swap_deterioration')
  action='STOP' if stop else 'PAUSE_RELEASE' if color=='RED' else 'WAIT' if wait else 'CONTINUE'
  return {'color':'RED' if stop else color,'action':action,'reasons':stop+(['OS_CRITICAL'] if color=='RED' else [])+wait,'swap_rate_GiB_per_min':rate}

class ResourceGate:
 def __init__(self,stage,spec,scope,authorization):
  if scope.get('resource_policy')!=POLICY or stage!='B_MICRO':raise ValueError('Exact dynamic scope required')
  self.stage,self.scope,self.authorization=stage,scope,authorization
  self.red_requested=False;self.spec=spec
  self.output=Path(scope['output']);self.output.mkdir(parents=True,exist_ok=True)
  runtime=json.loads(Path(spec['runtime_spec']).read_text());self.lock=Path(runtime['global_lock']);self.controller=Controller();self.peak_rss=0.;self.peak_mlx=None
 def __call__(self,name):
  from mri_rl.accuracy_stage_resources_v1 import assert_global_lock_held
  self.authorization.check('resources_'+str(name));assert_global_lock_held(self.lock)
  if self.red_requested:raise ResourcePause('Independent guardian requested RED/unknown release')
  s=sample_macos();r=self.controller.decide(s,os_pressure_level(),shutil.disk_usage(self.output).free/1024**3,time.monotonic())
  if valid(s.process_rss_gib):self.peak_rss=max(self.peak_rss,s.process_rss_gib)
  if valid(s.mlx_peak_gib):self.peak_mlx=max(self.peak_mlx or 0,s.mlx_peak_gib)
  row={'boundary':name,**r,'sample':asdict(s),'monotonic':time.monotonic()}
  with (self.output/'dynamic_resources_PRIVATE.jsonl').open('a') as f:
   import os
   f.write(json.dumps(row,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
  if r['action']=='STOP':raise ResourceFatal(','.join(r['reasons']))
  if r['action'] in ('WAIT','PAUSE_RELEASE'):raise ResourcePause(','.join(r['reasons']))
 def claim_model_load(self):
  self.authorization.check('model_load_claim')
  path=Path(self.spec['output'])/'model_load_ledger_PRIVATE.json'
  record=json.loads(path.read_text()) if path.exists() else {'scope':self.scope['compound_scope_sha256'],'claimed':0,'limit':7}
  if record['scope']!=self.scope['compound_scope_sha256'] or record['limit']!=7 or not 0<=record['claimed']<7:raise ResourceFatal('Model load budget exhausted or changed')
  record['claimed']+=1;replace_atomic(path,record)
 def peaks(self):return {'peak_process_rss_gib':self.peak_rss,'peak_mlx_reported_gib':self.peak_mlx,'MLX_unobserved_is_not_zero':True}

def make_resource_gate(stage,spec,scope,authorization):return ResourceGate(stage,spec,scope,authorization)
