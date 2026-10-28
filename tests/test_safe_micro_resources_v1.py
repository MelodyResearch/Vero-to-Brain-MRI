"""Synthetic samples/fake MLX only; no host sampling or GPU initialization."""
import copy,json,sys
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
from mri_rl import safe_micro_resources_v1 as r
from mri_rl import accuracy_recovery_gpu_cache_v4 as v4, accuracy_recovery_gpu_cache_v3 as v3, accuracy_recovery_cpu_cache_v1 as cc

class Clock:
 def __init__(self):self.t=0.;self.waits=[]
 def __call__(self):return self.t
 def sleep(self,v):self.t+=v;self.waits.append(v)

def sample(**kwargs):
 return NS(**{**dict(monotonic_seconds=0.,available_gib=11.,pressure_free_percent=45.,swap_used_gib=14.,process_rss_gib=.2,
  power_source='BATTERY',battery_percent=60.,other_heavy_processes=0,thermal_state='SENSOR_UNAVAILABLE',mlx_active_gib=None,mlx_peak_gib=None),**kwargs})

def ctl(tmp_path,samples):
 clock=Clock();iterator=iter(samples);last=[None]
 def get():
  try:last[0]=next(iterator)
  except StopIteration:pass
  return last[0]
 return r.ModelController(sampler=get,os_level=lambda:1,clock=clock,sleep=clock.sleep,log_path=tmp_path/'r.jsonl'),clock


def test_stage_A_exact_old_policy_and_B_declared_caps(tmp_path,monkeypatch):
 assert r.CACHE_POLICY==v4.POLICY and r.cache_controller is v4.ResourceController
 assert r.MODEL_POLICY['operation_start_window_seconds']==1800 and r.MODEL_POLICY['memory_limit_gib']==12.
 assert r.MODEL_MEMORY_LIMIT_BYTES==12*1024**3 and r.MODEL_CACHE_LIMIT_BYTES==256*1024**2
 assert v3.POLICY['rss_envelope_gib']==8. and v4.POLICY['continue_available_gib_min']==5.5

 runtime=tmp_path/'runtime.json';runtime.write_text(json.dumps({'global_lock':str(tmp_path/'remaining20.lock')}))
 scope={'resource_stage':'B_MICRO','resource_policy':copy.deepcopy(r.MODEL_POLICY),'execution_stage':'SINGLE_LOAD_FORWARD','action':'SYNTHETIC_LOAD_ACTION','output':str(tmp_path/'micro')};token=auth(scope)
 calls=[]
 monkeypatch.setattr(r.resource_supervisor,'sample_macos',lambda:(calls.append('sample') or sample()))
 monkeypatch.setattr(r.v3,'os_pressure_level',lambda:1)
 monkeypatch.setattr(r.shutil,'disk_usage',lambda path:NS(free=25*1024**3))
 monkeypatch.setattr(r,'assert_global_lock_held',lambda path:calls.append('lock'))
 gate=r.make_resource_gate('B_MICRO',{'runtime_spec':str(runtime)},scope,token)
 assert calls==[] and gate('native_model_load') is None
 assert calls==['lock','sample','lock'] and gate.peaks()['peak_process_rss_gib']==.2
 row=json.loads((tmp_path/'micro_resources_PRIVATE.jsonl').read_text().splitlines()[-1]);assert row['power_source']=='BATTERY'
 with pytest.raises(ValueError):r.make_resource_gate('A_CACHE',{'runtime_spec':str(runtime)},scope,token)
 assert not any(n=='mlx' or n.startswith('mlx.') for n in sys.modules)


def test_B_load_and_resident_thresholds_and_real_peak_log(tmp_path):
 c,clock=ctl(tmp_path,[sample(available_gib=9.76,pressure_free_percent=40),sample(available_gib=10,pressure_free_percent=40)])
 c.boundary('native_model_load');assert clock.waits==[5.]
 c,_=ctl(tmp_path,[sample(available_gib=6,pressure_free_percent=35,process_rss_gib=13,mlx_active_gib=11,mlx_peak_gib=11.5)])
 c.boundary('response_gradient');assert c.last['peak_process_rss_gib']==13 and c.last['peak_mlx_active_gib']==11
 assert c.last['mlx_reported_peak_gib']==11.5
 for kw in ({'process_rss_gib':14.1},{'mlx_active_gib':12.1},{'pressure_free_percent':15}):
  c,_=ctl(tmp_path,[sample(**kw)])
  with pytest.raises(cc.ResourceStop):c.boundary('unit')


def test_battery_unknown_hold_limit_and_original_swap_debt(tmp_path):
 c,_=ctl(tmp_path,[sample(battery_percent=20,swap_used_gib=14.912)]);c.boundary('unit');assert c.last['power_source']=='BATTERY'
 assert c.baseline_swap==13.96966796875
 for kw in ({'battery_percent':19.9},{'swap_used_gib':15.47},{'swap_used_gib':16.01}):
  c,_=ctl(tmp_path,[sample(**kw)])
  with pytest.raises(cc.ResourceStop):c.boundary('unit')
 c,clock=ctl(tmp_path,[sample(battery_percent=None)])
 with pytest.raises(cc.ResourceStop,match='hold timeout'):c.boundary('unit')
 assert clock.t==120.


def test_coupled_sustained_swap_growth_holds_only_after_two_intervals(tmp_path):
 c,_=ctl(tmp_path,[sample(monotonic_seconds=0,available_gib=7,swap_used_gib=14.3),sample(monotonic_seconds=5,available_gib=7,swap_used_gib=14.33),sample(monotonic_seconds=10,available_gib=7,swap_used_gib=14.36)])
 assert c.observe('unit')[1]=='CONTINUE' and c.observe('unit')[1]=='CONTINUE'
 assert c.observe('unit')[1]=='HOLD' and 'coupled_sustained_swap_growth' in c.last['reasons']
 c,_=ctl(tmp_path,[sample(monotonic_seconds=t,available_gib=9,pressure_free_percent=45,swap_used_gib=14.3+.006*t) for t in (0,5,10)])
 assert all(c.observe('unit')[1]=='CONTINUE' for _ in range(3))


def test_phase_B_deadline_includes_hold(tmp_path):
 c,clock=ctl(tmp_path,[sample(available_gib=9.7),sample()]);clock.t=1799.
 with pytest.raises(r.v2.WindowExceeded):c.boundary('native_model_load')
 assert clock.t==1804.


def auth(scope):return NS(scope=scope,check=lambda phase:True)


def test_authorized_setters_and_actual_release_receipt_use_existing_runtime_only():
 scope={'resource_stage':'B_MICRO','resource_policy':copy.deepcopy(r.MODEL_POLICY),'execution_stage':'MICRO_TRAIN','action':'SYNTHETIC_MICRO_ACTION'};token=auth(scope);calls=[]
 mx=NS(set_memory_limit=lambda n:calls.append(('memory',n)),set_cache_limit=lambda n:calls.append(('cache',n)),
  clear_cache=lambda:calls.append(('clear',None)),get_active_memory=lambda:1050,get_peak_memory=lambda:10*1024**3)
 r.configure_authorized_mlx(mx,token);assert calls[:2]==[('memory',12*1024**3),('cache',256*1024**2)]
 evidence=r.release_evidence(mx,None,token);assert evidence['release_verified'] and evidence['active_bytes_after_release']==1050
 assert r.release_evidence(mx,object(),token)['release_verified'] is False
 with pytest.raises(ValueError):r.configure_authorized_mlx(mx,auth({**scope,'resource_stage':'A_CACHE'}))
