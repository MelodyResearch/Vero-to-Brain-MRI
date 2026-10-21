from types import SimpleNamespace
from pathlib import Path
import pytest
from mri_rl.safe_micro_dynamic_resources_v1 import Controller, POLICY
from mri_rl.safe_micro_journal_v1 import Journal, Indeterminate


def sample(**kw):
 return SimpleNamespace(available_gib=7.,pressure_free_percent=38.,swap_used_gib=18.,process_rss_gib=.1,battery_percent=80,power_source='AC',thermal_state='SENSOR_UNAVAILABLE',other_heavy_processes=0,mlx_active_gib=None,mlx_peak_gib=None,**kw)


def test_green_yellow_no_fixed_headroom_or_swap_only_stop():
 c=Controller()
 for level,color in ((1,'GREEN'),(2,'YELLOW')):
  result=c.decide(sample(),level,100.,0.)
  assert result['color']==color and result['action']=='CONTINUE'
 assert 'load_available_gib_min' not in POLICY


def test_red_and_hard_danger_override_green():
 for kw in ({'available_gib':1.},{'process_rss_gib':15.},{'mlx_active_gib':13.},{'pressure_free_percent':14.}):
  s=sample();s.__dict__.update(kw)
  assert Controller().decide(s,1,100.,0.)['action']=='STOP'
 assert Controller().decide(sample(),4,100.,0.)['action']=='PAUSE_RELEASE'
 assert Controller().decide(sample(),1,25.,0.)['action']=='STOP'


def test_short_yellow_recovery_and_unknown_are_not_permanent_stop():
 c=Controller()
 assert c.decide(sample(),None,100.,0.)['action']=='WAIT'
 assert c.decide(sample(),None,100.,180.)['action']=='WAIT'
 assert c.decide(sample(),2,100.,181.)['action']=='CONTINUE'


def test_journal_replays_complete_not_operation(tmp_path):
 binding={'scope':'a','input':'b'};calls=[]
 with Journal(tmp_path,binding) as j:
  def op(d):
   calls.append(1);p=d/'state';p.write_text('full')
   import hashlib
   return {'value':{'updates':1},'files':{str(p):hashlib.sha256(p.read_bytes()).hexdigest()}}
  assert j.run('arm_A',op,reservation={'updates':1})=={'updates':1}
 with Journal(tmp_path,binding) as j:
  assert j.run('arm_A',lambda d:pytest.fail('duplicate optimizer'))=={'updates':1}
 assert calls==[1]


def test_journal_incomplete_never_replays(tmp_path):
 with Journal(tmp_path,{'scope':'a'}) as j:
  with pytest.raises(RuntimeError):j.run('generation',lambda d:(_ for _ in ()).throw(RuntimeError('interrupt')),reservation={'responses':16})
 with Journal(tmp_path,{'scope':'a'}) as j:
  with pytest.raises(Indeterminate):j.run('generation',lambda d:pytest.fail('resample'))
  with pytest.raises(Indeterminate):j.assert_recoverable()


def test_pause_before_reservation_and_hash_corruption(tmp_path):
 class Pause(Exception):pass
 with Journal(tmp_path,{'scope':'a'},before=lambda k:(_ for _ in ()).throw(Pause())) as j:
  with pytest.raises(Pause):j.run('unused',lambda d:None)
  assert j.get('unused') is None
 with pytest.raises(ValueError):Journal(tmp_path,{'scope':'different'})


def test_journal_single_writer(tmp_path):
 with Journal(tmp_path,{'scope':'a'}):
  with pytest.raises(BlockingIOError):Journal(tmp_path,{'scope':'a'})


def test_sustained_emergency_not_short_yellow():
 c=Controller();s=sample();s.available_gib=3.
 assert c.decide(s,2,100.,0.)['action']=='CONTINUE'
 assert c.decide(s,2,100.,29.)['action']=='CONTINUE'
 assert c.decide(s,2,100.,30.)['action']=='STOP'


def test_swap_only_is_trend_coupled_emergency_stops():
 c=Controller();s=sample();s.swap_used_gib=14.
 assert c.decide(s,2,100.,0.)['action']=='CONTINUE'
 s.swap_used_gib=14.1
 assert c.decide(s,2,100.,10.)['action']=='CONTINUE'
 s.swap_used_gib=14.2
 assert c.decide(s,2,100.,20.)['action']=='CONTINUE'
 s.swap_used_gib=14.3
 assert c.decide(s,2,29.,30.)['action']=='STOP'


def test_committed_payload_corruption_blocks_replay(tmp_path):
 import hashlib
 with Journal(tmp_path,{'scope':'a'}) as j:
  def op(d):
   p=d/'payload';p.write_text('exact');return {'value':{'complete':True},'files':{str(p):hashlib.sha256(p.read_bytes()).hexdigest()}}
  j.run('row0',op)
  (tmp_path/'row0/payload').write_text('wrong')
  with pytest.raises(ValueError):j.get('row0')


def test_dynamic_authority_preserves_scientific_caps(tmp_path):
 from mri_rl import safe_micro_pilot_entry_v3 as e
 from mri_rl import safe_micro_pilot_entry_v2 as old
 from test_safe_micro_pilot_entry_v2 import s,grants
 scope=s(tmp_path);scope.update(schema=e.SCHEMA,scope_kind=e.KIND,action=e.ACTION,resource_policy=e.MODEL_POLICY,external_guardian_policy=e.GUARDIAN_POLICY,recovery_contract=e.RECOVERY)
 h,r=grants(scope);h['action']=e.ACTION
 token=e.authorize(scope,h,r,consumed=lambda _:None)
 assert token.scope['budgets']==old.BUDGETS
 assert e.RECOVERY['max_model_load_attempts']==7
 scope['recovery_contract']={**e.RECOVERY,'max_model_load_attempts':99};h,r=grants(scope);h['action']=e.ACTION
 with pytest.raises(ValueError):e.authorize(scope,h,r,consumed=lambda _:None)


def test_guardian_cooperative_red_targets_only_owned_child(tmp_path):
 import sys,time
 from mri_rl.safe_micro_watchdog_v2 import witness_child
 ready=tmp_path/'ready'
 code="import signal,sys,time; from pathlib import Path; signal.signal(signal.SIGUSR1,lambda *args:sys.exit(75)); Path(sys.argv[1]).write_text('ready'); time.sleep(10)"
 def observe(pid):
  limit=time.monotonic()+2
  while not ready.exists() and time.monotonic()<limit:time.sleep(.005)
  return sample()
 result=witness_child([sys.executable,'-c',code,str(ready)],tmp_path/'witness',scope_sha256='a'*64,sampler=observe,level=lambda:4,disk_free_gib=lambda:100.,post_sampler=lambda:{},lock_inspector=lambda:{'no_open_holders':True},interval=.01)
 assert result['native_exit_code']==75 and result['child_reaped']
 assert result['cooperative_RED_requested'] and not result['SIGTERM_sent'] and not result['SIGKILL_after_grace']
