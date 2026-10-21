"""Own tiny child processes only. No GPU, host resource sampling or other apps."""
import sys,json
from types import SimpleNamespace as NS
from mri_rl import safe_micro_watchdog_v1 as w


def sample(**kw):return NS(**{**dict(available_gib=9.5,pressure_free_percent=53.,swap_used_gib=14.58,process_rss_gib=.1,battery_percent=67.,mlx_active_gib=None,mlx_peak_gib=None),**kw})


def test_guardian_does_not_stop_normal_swap_or_soft_load_shortfall():
 g=w.Guardian();assert g.observe(sample(available_gib=8.5),1,100.,0)==[]
 assert g.observe(sample(swap_used_gib=15.4),1,100.,1)==[]
 assert 'OS_CRITICAL' in g.observe(sample(),4,100.,2)


def test_true_risk_and_sustained_low_or_unknown_stops():
 for kw in ({'pressure_free_percent':15},{'process_rss_gib':14.1},{'swap_used_gib':15.47},{'battery_percent':19.9}):assert w.Guardian().observe(sample(**kw),1,100.,0)
 g=w.Guardian();assert g.observe(sample(available_gib=3.9),1,100.,0)==[]
 assert g.observe(sample(available_gib=3.9),1,100.,30)
 g=w.Guardian();assert g.observe(None,None,None,0)==[]
 assert g.observe(None,None,None,120)


def test_normal_child_closed_logs_wait4_and_sensor_failure_unknown(tmp_path):
 out=tmp_path/'normal'
 def bad():raise OSError('synthetic missing sensor')
 result=w.witness_child([sys.executable,'-c','print("tiny")'],out,scope_sha256='a'*64,sampler=lambda pid:sample(),level=lambda:1,disk_free_gib=lambda:100.,post_sampler=bad,lock_inspector=lambda:{'no_open_holders':True},interval=.01)
 assert result['native_exit_code']==0 and result['child_reaped'] and result['guardian_stop'] is False
 assert result['post_child_resource_sample']['status']=='UNAVAILABLE'
 assert result['stdout_sha256']==w.digest(out/'stdout_PRIVATE.log')
 assert json.loads((out/'actual_exit_PRIVATE.json').read_text())==result


def test_critical_monitor_stops_exact_child_without_retry(tmp_path):
 out=tmp_path/'critical'
 result=w.witness_child([sys.executable,'-c','import time;time.sleep(30)'],out,scope_sha256='a'*64,sampler=lambda pid:sample(),level=lambda:4,disk_free_gib=lambda:100.,post_sampler=lambda:{},lock_inspector=lambda:{'no_open_holders':True},interval=.01)
 assert result['guardian_stop'] and result['native_exit_code']!=0 and result['child_reaped']
 assert result['stop_target_pid']==result['child_pid'] and result['automatic_retry'] is False
 assert result['wall_seconds']<10


def test_exit_during_sensor_observation_keeps_wait4_exit_witness(tmp_path):
 import time
 def slow(pid):time.sleep(.1);return sample()
 result=w.witness_child([sys.executable,'-c','pass'],tmp_path/'race',scope_sha256='a'*64,sampler=slow,level=lambda:4,disk_free_gib=lambda:100.,post_sampler=lambda:{},lock_inspector=lambda:{'no_open_holders':True},interval=.01)
 assert result['child_reaped'] and result['native_exit_code']==0 and result['guardian_stop']


# Independent reviewer regression: percentages/OS enums cannot fail open.
import pytest

@pytest.mark.parametrize('change,level',[({'pressure_free_percent':101.},1),({'battery_percent':101.},1),({},True)])
def test_out_of_range_essential_sensor_uses_unknown_timeout(change,level):
    s=NS(**{**dict(available_gib=9.5,pressure_free_percent=53.,swap_used_gib=14.58,process_rss_gib=.1,battery_percent=67.),**change})
    g=w.Guardian()
    assert not g.observe(s,level,100.,0.)
    assert 'essential_monitor_unknown_timeout' in g.observe(s,level,100.,120.)


def test_monitor_log_failure_stops_owned_child_and_preserves_exit(tmp_path,monkeypatch):
 def failed(*args):raise OSError('synthetic monitor log failure')
 monkeypatch.setattr(w,'append_monitor',failed)
 result=w.witness_child([sys.executable,'-c','import time;time.sleep(30)'],tmp_path/'io_failure',scope_sha256='a'*64,sampler=lambda pid:sample(),level=lambda:1,disk_free_gib=lambda:100.,post_sampler=lambda:{},lock_inspector=lambda:{'no_open_holders':True},interval=.01)
 assert result['guardian_stop'] and result['child_reaped'] and result['native_exit_code']!=0
 assert 'monitor_log_write_failure' in result['guardian_stop_reasons'] and result['monitor_log_errors']


def test_parent_interrupt_does_not_leave_its_experiment_child_running(tmp_path,monkeypatch):
 children=[];original=w.subprocess.Popen
 def spawn(*args,**kwargs):
  child=original(*args,**kwargs);children.append(child);return child
 def interrupted(pid):raise KeyboardInterrupt('synthetic parent interrupt')
 monkeypatch.setattr(w.subprocess,'Popen',spawn)
 try:
  with pytest.raises(KeyboardInterrupt):w.witness_child([sys.executable,'-c','import time;time.sleep(30)'],tmp_path/'interrupt',scope_sha256='a'*64,sampler=interrupted,level=lambda:1,disk_free_gib=lambda:100.,post_sampler=lambda:{},lock_inspector=lambda:{'no_open_holders':True},interval=.01)
  assert children[0].returncode is not None
 finally:
  if children and children[0].returncode is None:w.emergency_reap_owned(children[0])


# Independent reviewer regression: never signal a PID already reaped.

def test_emergency_echild_does_not_signal_pid_after_reap(monkeypatch):
    signals=[]
    # An interrupt can arrive after wait4 reaps but before returncode assignment.
    child=NS(pid=123456,returncode=None)
    monkeypatch.setattr(w.os,'kill',lambda pid,sig:signals.append((pid,sig)))
    def no_owned_child(*args):raise ChildProcessError('synthetic already reaped')
    monkeypatch.setattr(w.os,'wait4',no_owned_child)
    try:w.emergency_reap_owned(child)
    except ChildProcessError:pass
    assert signals==[], 'ECHILD means PID is no longer held unreaped; do not signal before checking ownership'
