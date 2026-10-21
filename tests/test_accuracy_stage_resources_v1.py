"""Synthetic resource fixtures only. Never sample this machine or import MLX."""
import json
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
from mri_rl import accuracy_stage_resources_v1 as r
from mri_rl import accuracy_recovery_gpu_cache_v3 as v3, accuracy_recovery_cpu_cache_v1 as cc

class Clock:
 def __init__(self):self.value=0.;self.waits=[]
 def __call__(self):return self.value
 def sleep(self,t):self.waits.append(t);self.value+=t

def sample(**over):
 return NS(**{**dict(monotonic_seconds=0.,available_gib=9.,pressure_free_percent=40.,swap_used_gib=14.,process_rss_gib=.1,
  power_source='BATTERY',battery_percent=60.,other_heavy_processes=0,thermal_state='SENSOR_UNAVAILABLE',mlx_active_gib=None),**over})

def ctl(tmp_path,samples):
 clock=Clock();it=iter(samples);last=[None]
 def sampler():
  try:last[0]=next(it)
  except StopIteration:pass
  return last[0]
 c=r.ResourceController(sampler=sampler,os_level=lambda:1,clock=clock,sleep=clock.sleep,log_path=tmp_path/'resource.jsonl')
 return c,clock


def test_load_and_unit_thresholds_are_distinct_and_no_global_policy_mutation(tmp_path):
 c,_=ctl(tmp_path,[sample(available_gib=8,pressure_free_percent=35)])
 assert c.boundary('A_native_model_load') is not None
 c,_=ctl(tmp_path,[sample(available_gib=6,pressure_free_percent=30)])
 assert c.boundary('A_response_gradient') is not None
 c,_=ctl(tmp_path,[sample(available_gib=7.9,pressure_free_percent=34),sample(available_gib=8,pressure_free_percent=35)])
 c.boundary('native_model_load');assert c.hold_samples==1
 assert v3.POLICY['ac_required'] is True and v3.POLICY['rss_envelope_gib']==8.


def test_real_battery_unknown_and_critical_memory_guards(tmp_path):
 c,_=ctl(tmp_path,[sample(battery_percent=20)]);c.boundary('unit');assert c.last['power_source']=='BATTERY'
 for kw in ({'battery_percent':19.9},{'pressure_free_percent':15},{'process_rss_gib':18.1},{'mlx_active_gib':18.1}):
  c,_=ctl(tmp_path,[sample(**kw)])
  with pytest.raises(cc.ResourceStop):c.boundary('unit')
 c,clock=ctl(tmp_path,[sample(battery_percent=None),sample(battery_percent=20)])
 c.boundary('unit');assert clock.waits==[5.]


def test_original_swap_debt_and_rate_only_behavior(tmp_path):
 c,_=ctl(tmp_path,[sample(swap_used_gib=14.912,monotonic_seconds=0),sample(swap_used_gib=15.0,monotonic_seconds=.01)])
 c.boundary('a');c.boundary('b');assert c.baseline_swap==13.96966796875
 assert c.last['swap_growth_gib_per_min_trend_only']>100.
 for used in (15.47,16.01):
  c,_=ctl(tmp_path,[sample(swap_used_gib=used)])
  with pytest.raises(cc.ResourceStop):c.boundary('unit')


def test_hold_bound_and_global_deadline_include_waits(tmp_path):
 c,clock=ctl(tmp_path,[sample(available_gib=5.9)])
 with pytest.raises(cc.ResourceStop,match='hold timeout'):c.boundary('unit')
 assert clock.value==300.
 c,clock=ctl(tmp_path,[sample(available_gib=5.9),sample()]);clock.value=10799.
 with pytest.raises(r.v2.WindowExceeded):c.boundary('unit')
 assert clock.value==10804.


def factory_fixture(tmp_path,monkeypatch):
 lock=tmp_path/'remaining20.lock';lock.write_text('')
 runtime=tmp_path/'runtime.json';runtime.write_text(json.dumps({'global_lock':str(lock)}))
 spec={'runtime_spec':str(runtime)};scope={'output':str(tmp_path/'run'),'resource_policy':{'this_round_only':True,'battery_percent_min':20,'global_wall_seconds':10800}}
 calls=[];auth=NS(scope=scope,check=lambda name:calls.append(name))
 monkeypatch.setattr(r.resource_supervisor,'sample_macos',lambda:sample())
 monkeypatch.setattr(r.v3,'os_pressure_level',lambda:1)
 monkeypatch.setattr(r.shutil,'disk_usage',lambda path:NS(free=30*1024**3))
 monkeypatch.setattr(r,'assert_global_lock_held',lambda path:None)
 return spec,scope,auth,calls


def test_factory_returns_only_callable_logs_real_power_and_checks_authority(tmp_path,monkeypatch):
 spec,scope,auth,calls=factory_fixture(tmp_path,monkeypatch)
 gate=r.make_resource_gate(spec,scope,auth);assert callable(gate)
 assert gate('native_model_load') is None and len(calls)>=2
 log=tmp_path/'run_resources_PRIVATE.jsonl';rows=[json.loads(l) for l in log.read_text().splitlines()]
 assert rows[-1]['power_source']=='BATTERY' and rows[-1]['baseline_swap_gib']==13.96966796875


def test_factory_cannot_skip_disk_lock_or_exact_authority(tmp_path,monkeypatch):
 spec,scope,auth,_=factory_fixture(tmp_path,monkeypatch)
 monkeypatch.setattr(r.shutil,'disk_usage',lambda path:NS(free=19*1024**3))
 with pytest.raises(cc.ResourceStop,match='disk'):r.make_resource_gate(spec,scope,auth)('unit')
 monkeypatch.setattr(r.shutil,'disk_usage',lambda path:NS(free=30*1024**3))
 def missing(path):raise ValueError('global flock absent')
 monkeypatch.setattr(r,'assert_global_lock_held',missing)
 with pytest.raises(ValueError,match='flock'):r.make_resource_gate(spec,scope,auth)('unit')
 with pytest.raises(ValueError):r.make_resource_gate(spec,{**scope,'output':'foreign'},auth)
