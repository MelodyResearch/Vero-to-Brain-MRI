"""Local9GiB/50% policy using synthetic telemetry, no native imports."""
from types import SimpleNamespace as NS
import copy
import pytest
from mri_rl import safe_micro_resources_v1 as old
from mri_rl import safe_micro_resources_v2 as new


def sample(**kwargs):
 return NS(**{**dict(monotonic_seconds=0.,available_gib=9.5,pressure_free_percent=53.,swap_used_gib=14.579892578125,process_rss_gib=.2,power_source='BATTERY',battery_percent=67.,other_heavy_processes=0,thermal_state='SENSOR_UNAVAILABLE',mlx_active_gib=None,mlx_peak_gib=None),**kwargs})


def decision(tmp_path,**kwargs):
 c=new.ModelController(sampler=lambda:sample(**kwargs),os_level=lambda:1,log_path=tmp_path/'resource.jsonl')
 c.policy['continue_available_gib_min']=new.MODEL_POLICY['load_available_gib_min'];c.policy['continue_pressure_free_percent_min']=new.MODEL_POLICY['load_pressure_free_percent_min']
 return c._decide(sample(**kwargs),1,9.,0.)


def test_only_two_loading_thresholds_change_globally_unchanged(tmp_path):
 assert new.MODEL_POLICY['load_available_gib_min']==9 and new.MODEL_POLICY['load_pressure_free_percent_min']==50
 assert {k:v for k,v in new.MODEL_POLICY.items() if k not in ('load_available_gib_min','load_pressure_free_percent_min')}=={k:v for k,v in old.MODEL_POLICY.items() if k not in ('load_available_gib_min','load_pressure_free_percent_min')}
 assert old.MODEL_POLICY['load_available_gib_min']==10 and old.MODEL_POLICY['load_pressure_free_percent_min']==40
 assert decision(tmp_path,available_gib=9.493,pressure_free_percent=53)[0]=='CONTINUE'
 assert decision(tmp_path,available_gib=9,pressure_free_percent=50)[0]=='CONTINUE'
 assert decision(tmp_path,available_gib=8.999)[0]=='HOLD'
 assert decision(tmp_path,pressure_free_percent=49.9)[0]=='HOLD'


@pytest.mark.parametrize('change',[{'swap_used_gib':15.47},{'process_rss_gib':14.01},{'mlx_active_gib':12.01},{'pressure_free_percent':15},{'battery_percent':19.9}])
def test_hard_stops_remain(tmp_path,change):assert decision(tmp_path,**change)[0]=='STOP'


def test_real_boundary_uses9_50_then_resident6_35(tmp_path):
 c=new.ModelController(sampler=lambda:sample(),os_level=lambda:1,log_path=tmp_path/'r.jsonl');c.boundary('native_model_load')
 assert c.last['decision']=='CONTINUE' and c.last['required_available_gib']==9
 c.boundary('response_gradient');assert c.last['required_available_gib']==6
