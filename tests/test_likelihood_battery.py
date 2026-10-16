from dataclasses import replace
import json
import pytest
from mri_rl.resource_supervisor import ResourceSample

def healthy(t=0, battery=91, **kw):
    return ResourceSample(t, available_gib=13, pressure_free_percent=74,
        swap_used_gib=5.7, power_source='BATTERY', battery_percent=battery,
        other_heavy_processes=0, **kw)

def test_exception_is_diagnostic_only_and_global_training_stays_ac():
    from mri_rl.likelihood_battery import BatteryDiagnosticGate
    from mri_rl.paced_resources import decisions
    assert decisions(healthy(),loading=True)[0]=='HOLD'
    with pytest.raises(ValueError):BatteryDiagnosticGate('/tmp/unused',runner='training')

@pytest.mark.parametrize('sample', [healthy(battery=74),replace(healthy(),battery_percent=None),
    replace(healthy(),other_heavy_processes=1),replace(healthy(),available_gib=3),
    replace(healthy(),power_source='SENSOR_UNAVAILABLE')])
def test_start_rejects_low_unknown_or_unsafe(tmp_path,sample):
    from mri_rl.likelihood_battery import BatteryDiagnosticGate
    gate=BatteryDiagnosticGate(tmp_path/'r',sampler=lambda:sample,sleep=lambda _:None)
    with pytest.raises(RuntimeError):gate.before_unit('start',loading=True)

def test_initial_two_samples_and_durable_stop_at_65(tmp_path):
    from mri_rl.likelihood_battery import BatteryDiagnosticGate
    samples=iter([healthy(),healthy(31),healthy(300,65)])
    sleeps=[];p=tmp_path/'r'
    gate=BatteryDiagnosticGate(p,sampler=lambda:next(samples),sleep=sleeps.append)
    gate.before_unit('start',loading=True)
    assert sleeps==[31]
    with pytest.raises(RuntimeError,match='battery_stop'):gate.before_unit('next')
    assert json.loads(p.read_text().splitlines()[-1])['action']=='STOP'

@pytest.mark.parametrize('later,reason',[(healthy(120,89),'rapid_battery_drain'),
    (healthy(1200,86),'stage_battery_budget'),
    (replace(healthy(60),swap_used_gib=7.0),'stage_swap_growth')])
def test_runtime_limits(tmp_path,later,reason):
    from mri_rl.likelihood_battery import BatteryDiagnosticGate
    samples=iter([healthy(),healthy(31),later])
    gate=BatteryDiagnosticGate(tmp_path/'r',sampler=lambda:next(samples),sleep=lambda _:None)
    gate.before_unit('start',loading=True)
    with pytest.raises(RuntimeError,match=reason):gate.before_unit('next')

def test_serial_materialization_releases_between_leaves():
    from mri_rl.likelihood_loading import materialize_serial
    events=[]
    materialize_serial([('first',object()),('second',object())],
        evaluate=lambda x:events.append('eval'), collect=lambda:events.append('gc'),
        clear=lambda:events.append('clear'), observe=lambda name:events.append(name))
    assert events==['eval','gc','clear','materialized:1','eval','gc','clear','materialized:2']

def test_serial_failure_does_not_retain_parameter_references():
    import weakref,gc
    from mri_rl.likelihood_loading import materialize_serial
    class Value:pass
    value=Value();reference=weakref.ref(value)
    def fail(_):raise RuntimeError('observed stop')
    error=None
    try:
        materialize_serial([('parameter',value)],evaluate=lambda _:None,collect=lambda:None,clear=lambda:None,observe=fail)
    except RuntimeError as exc:error=exc
    del value;gc.collect()
    assert error is not None
    assert reference() is None
