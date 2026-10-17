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

@pytest.mark.parametrize('sample', [healthy(battery=20),replace(healthy(),battery_percent=None),
    replace(healthy(),other_heavy_processes=1),replace(healthy(),available_gib=3),
    replace(healthy(),power_source='SENSOR_UNAVAILABLE')])
def test_start_rejects_low_unknown_or_unsafe(tmp_path,sample):
    from mri_rl.likelihood_battery import BatteryDiagnosticGate
    gate=BatteryDiagnosticGate(tmp_path/'r',sampler=lambda:sample,sleep=lambda _:None)
    with pytest.raises(RuntimeError):gate.before_unit('start',loading=True)

def test_initial_two_samples_and_durable_low_reserve_stop_at_20(tmp_path):
    from mri_rl.likelihood_battery import BatteryDiagnosticGate
    samples=iter([healthy(),healthy(31),healthy(300,20)])
    sleeps=[];p=tmp_path/'r'
    gate=BatteryDiagnosticGate(p,sampler=lambda:next(samples),sleep=sleeps.append)
    gate.before_unit('start',loading=True)
    assert sleeps==[31]
    with pytest.raises(RuntimeError,match='battery_low_reserve_stop'):gate.before_unit('next')
    assert json.loads(p.read_text().splitlines()[-1])['action']=='STOP'

@pytest.mark.parametrize('later,reason',[(replace(healthy(60),swap_used_gib=7.0),'stage_swap_growth')])
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

def hold_sample(t,swap=5.9,battery=91):
    return replace(healthy(t,battery),available_gib=9,swap_used_gib=swap)

@pytest.mark.parametrize('loading',[True,False])
def test_isolated_rate_warning_admits_without_recovery_sleep(tmp_path,loading):
    from mri_rl.likelihood_battery import BatteryDiagnosticGate
    samples=iter([healthy(),healthy(31),hold_sample(32)])
    sleeps=[];p=tmp_path/'r'
    gate=BatteryDiagnosticGate(p,sampler=lambda:next(samples),sleep=sleeps.append)
    gate.before_unit('start',loading=loading)
    gate.before_unit('next',loading=loading)
    event=json.loads(p.read_text().splitlines()[-1])
    assert event['action']=='WARNING' and event['admitted'] is True
    assert event['phase']==('loading' if loading else 'resident')
    assert event['rate_warning_duration_seconds']==0
    assert sleeps==[31] and gate.initial.swap_used_gib==5.7
    assert len(gate.history)==3

def test_sustained_rate_warning_enters_bounded_recovery(tmp_path):
    from mri_rl.likelihood_battery import BatteryDiagnosticGate
    samples=iter([healthy(),healthy(31),hold_sample(32),hold_sample(62,6.1),hold_sample(77,6.1),hold_sample(92,6.1)])
    sleeps=[];p=tmp_path/'r'
    gate=BatteryDiagnosticGate(p,sampler=lambda:next(samples),sleep=sleeps.append)
    gate.before_unit('start');gate.before_unit('first');gate.before_unit('second')
    events=[json.loads(x) for x in p.read_text().splitlines()]
    assert [e['action'] for e in events]==['CONTINUE','CONTINUE','WARNING','HOLD','HOLD_RECOVERY','CONTINUE']
    assert events[3]['rate_warning_duration_seconds']==30
    assert sleeps==[31,15,15] and len(gate.history)==6
    assert gate.initial.swap_used_gib==5.7

def test_phase_change_does_not_carry_loading_transient_or_reset_budget(tmp_path):
    from mri_rl.likelihood_battery import BatteryDiagnosticGate
    samples=iter([healthy(),healthy(31),hold_sample(32),hold_sample(62,6.1),hold_sample(100,6.8)])
    sleeps=[];p=tmp_path/'r'
    gate=BatteryDiagnosticGate(p,sampler=lambda:next(samples),sleep=sleeps.append)
    gate.before_unit('start',loading=True);gate.before_unit('load',loading=True)
    gate.before_unit('resident')
    event=json.loads(p.read_text().splitlines()[-1])
    assert event['action']=='WARNING' and event['rate_warning_duration_seconds']==0
    assert sleeps==[31] and gate.initial.swap_used_gib==5.7
    with pytest.raises(RuntimeError,match='stage_swap_growth'):gate.before_unit('budget')

@pytest.mark.parametrize('later', [hold_sample(77,6.8),hold_sample(77,battery=20),replace(hold_sample(77),other_heavy_processes=1)])
def test_sustained_hold_stops_immediately_on_hard_deterioration(tmp_path,later):
    from mri_rl.likelihood_battery import BatteryDiagnosticGate
    samples=iter([healthy(),healthy(31),hold_sample(32),hold_sample(62,6.1),later])
    sleeps=[]
    gate=BatteryDiagnosticGate(tmp_path/'r',sampler=lambda:next(samples),sleep=sleeps.append)
    gate.before_unit('start');gate.before_unit('first')
    with pytest.raises(RuntimeError):gate.before_unit('sustained')
    assert sleeps==[31,15]

def test_clean_sample_breaks_rate_streak(tmp_path):
    from mri_rl.likelihood_battery import BatteryDiagnosticGate
    samples=iter([healthy(),healthy(31),hold_sample(32),hold_sample(50),hold_sample(80,6.1)])
    sleeps=[];p=tmp_path/'r'
    gate=BatteryDiagnosticGate(p,sampler=lambda:next(samples),sleep=sleeps.append)
    gate.before_unit('start')
    for name in ['first','clean','later']:gate.before_unit(name)
    assert sleeps==[31]
    assert json.loads(p.read_text().splitlines()[-1])['rate_warning_duration_seconds']==0

def test_rate_only_supervisor_escalation_does_not_create_premature_stop(tmp_path):
    from mri_rl.likelihood_battery import BatteryDiagnosticGate
    samples=iter([healthy(),healthy(31),hold_sample(32),hold_sample(33,6.0),hold_sample(34,6.1)])
    sleeps=[]
    gate=BatteryDiagnosticGate(tmp_path/'r',sampler=lambda:next(samples),sleep=sleeps.append)
    gate.before_unit('start');gate.before_unit('first');gate.before_unit('second')
    gate.before_unit('third')
    event=json.loads((tmp_path/'r').read_text().splitlines()[-1])
    assert event['action']=='WARNING' and event['supervisor_level']=='WARNING'
    assert sleeps==[31]

def test_materializer_phase_inferred_without_raising_headroom_gate(tmp_path):
    from mri_rl.likelihood_battery import BatteryDiagnosticGate
    samples=iter([healthy(),healthy(31),replace(hold_sample(32),available_gib=7)])
    sleeps=[];p=tmp_path/'r'
    gate=BatteryDiagnosticGate(p,sampler=lambda:next(samples),sleep=sleeps.append)
    gate.before_unit('start',loading=True);gate.before_unit('materialized:64')
    event=json.loads(p.read_text().splitlines()[-1])
    assert event['phase']=='loading' and event['action']=='WARNING'
    assert sleeps==[31]

def test_real_pressure_is_never_admitted_as_rate_only_warning(tmp_path):
    from mri_rl.likelihood_battery import BatteryDiagnosticGate
    samples=iter([healthy(),healthy(31),replace(hold_sample(32),pressure_free_percent=14)])
    gate=BatteryDiagnosticGate(tmp_path/'r',sampler=lambda:next(samples),sleep=lambda _:None)
    gate.before_unit('start')
    with pytest.raises(RuntimeError,match='memory_pressure'):gate.before_unit('next')

def test_sustained_recovery_expires_after_three_rechecks(tmp_path):
    from mri_rl.likelihood_battery import BatteryDiagnosticGate
    samples=iter([healthy(),healthy(31),hold_sample(32),hold_sample(62,6.1),
                  hold_sample(77,6.1),hold_sample(92,6.2),hold_sample(107,6.2)])
    sleeps=[]
    gate=BatteryDiagnosticGate(tmp_path/'r',sampler=lambda:next(samples),sleep=sleeps.append)
    gate.before_unit('start');gate.before_unit('first')
    with pytest.raises(RuntimeError,match='warning_hold_not_recovered'):gate.before_unit('sustained')
    assert sleeps==[31,15,15,15]

def test_unattributed_supervisor_warning_cannot_be_reclassified(tmp_path):
    from mri_rl.likelihood_battery import BatteryDiagnosticGate
    samples=iter([healthy(),healthy(31),hold_sample(32)])
    gate=BatteryDiagnosticGate(tmp_path/'r',sampler=lambda:next(samples),sleep=lambda _:None)
    gate.before_unit('start')
    gate.policy.level='WARNING'  # No rate-only episode established in this gate.
    with pytest.raises(RuntimeError,match='supervisor_not_normal'):gate.before_unit('next')

def test_battery_drain_and_mid_charge_no_longer_gate_read_only_inference(tmp_path):
    from mri_rl.likelihood_battery import BatteryDiagnosticGate
    # Start at 60% (previously <75% rejected); drop 40 points in 2 minutes (previously rapid/stage stop).
    samples=iter([healthy(0,60),healthy(31,60),healthy(151,25)])
    p=tmp_path/'r'
    gate=BatteryDiagnosticGate(p,sampler=lambda:next(samples),sleep=lambda _:None)
    gate.before_unit('start',loading=True)
    gate.before_unit('next')
    last=json.loads(p.read_text().splitlines()[-1])
    assert last['action']=='CONTINUE' and last['sample']['battery_percent']==25
