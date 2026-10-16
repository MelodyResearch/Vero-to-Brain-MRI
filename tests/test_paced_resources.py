import pytest
from mri_rl.paced_resources import decisions
from mri_rl.resource_supervisor import ResourceSample


def sample(**kw):
    return ResourceSample(monotonic_seconds=1,available_gib=10,pressure_free_percent=60,power_source='AC',other_heavy_processes=0,**kw)


def test_healthy_unknown_thermal_is_permitted_without_claim():
    assert decisions(sample(),loading=False)==('CONTINUE',[])


def test_first_critical_swap_signal_holds_even_if_old_hysteresis_normal():
    r=ResourceSample(monotonic_seconds=10,available_gib=3.9,pressure_free_percent=18,power_source='AC',other_heavy_processes=0)
    state,reasons=decisions(r,loading=False,supervisor_reasons=['critical_resource_signal'])
    assert state=='HOLD' and 'critical_resource_signal' in reasons


def test_severe_pressure_stops_and_transient_battery_holds():
    r=ResourceSample(monotonic_seconds=10,available_gib=.8,pressure_free_percent=1,power_source='AC',other_heavy_processes=0)
    assert decisions(r,loading=False)[0]=='STOP'
    r=ResourceSample(monotonic_seconds=10,available_gib=12,pressure_free_percent=60,power_source='BATTERY',other_heavy_processes=0)
    assert decisions(r,loading=False)[0]=='HOLD'
