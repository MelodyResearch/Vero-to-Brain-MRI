"""A single qualification may use real battery telemetry; training gates stay intact."""
import pytest
from dataclasses import replace
from mri_rl.resource_supervisor import ResourceSample
from mri_rl.paced_resources import decisions as original
from mri_rl.shared_rollout_qualification_power_v1 import decisions


def sample(**changes):
    return ResourceSample(1,available_gib=12,pressure_free_percent=60,swap_used_gib=10,
        power_source='BATTERY',battery_percent=36,other_heavy_processes=0,**changes)


def test_only_qualification_ac_hold_is_removed():
    s=sample();assert original(s,loading=True)[0]=='HOLD'
    assert decisions(s,loading=True)==('CONTINUE',[])
    assert s.power_source=='BATTERY' and s.battery_percent==36


@pytest.mark.parametrize('pct',[None,10,3])
def test_low_or_unknown_battery_stops(pct):
    s=replace(sample(),battery_percent=pct)
    assert decisions(s,loading=True)[0]=='STOP'


@pytest.mark.parametrize('field,value',[('available_gib',7),('other_heavy_processes',1),
    ('thermal_state','THROTTLED'),('pressure_free_percent',20)])
def test_other_protections_preserved(field,value):
    s=replace(sample(),**{field:value})
    assert decisions(s,loading=True)[0]!='CONTINUE'


def test_supervisor_swap_reasons_and_unknown_power_preserved():
    assert decisions(sample(),loading=True,supervisor_reasons=['sustained_swap_growth'])[0]=='HOLD'
    s=replace(sample(),power_source='SENSOR_UNAVAILABLE')
    assert decisions(s,loading=True)[0]=='HOLD'


def test_battery_launch_scope_can_never_be_a_train_scope():
    from mri_rl import shared_rollout_gspo_entry_v1 as entry
    from mri_rl import shared_rollout_gspo_core_v1 as core
    prepared={'invocation':{'adopt_qualification_run':None,'battery_readonly_exception':True},
        'budgets':{m:core.derive_budget({'F':213,'T':213},m) for m in core.MODES},
        'source_sha256':{'fixture':'a'*64},'input_sha256':'b'*64,'output':'/future'}
    assert entry.launch_scope(prepared,'qualify')['budget']['optimizer_updates']==0
    with pytest.raises(ValueError,match='cannot authorize training'):entry.launch_scope(prepared,'train')
