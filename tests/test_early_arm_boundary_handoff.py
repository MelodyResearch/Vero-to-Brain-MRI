import fcntl,importlib.util,json,os,signal,subprocess,sys,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
P=Path(__file__).parents[1]/'scripts/early_arm_boundary_handoff.py'
spec=importlib.util.spec_from_file_location('handoff',P);h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
E={'pid':123,'lstart':'today','command':'python queue.py','birth':{'sec':1,'usec':2},'status':'S'}
class Fake:
 def __init__(self,children=None):self.now=dict(E);self.sent=[];self.children=iter(children or [[],[],[]]);self.done=False
 def identity(self,pid):return self.now
 def descendants(self,pid):return next(self.children,[])
 def signal(self,e,s):
  self.sent.append(s)
  if s==signal.SIGSTOP:self.now={**self.now,'status':'T'}
  if s==signal.SIGTERM:self.done=True
  if s==signal.SIGCONT:self.now={**self.now,'status':'S'}
 def alive(self,e):return not self.done
 def sleep(self,x):pass
class Tests(unittest.TestCase):
 def test_identity_birth_reuse(self):self.assertFalse(h.same(E,{**E,'birth':{'sec':1,'usec':3}}))
 def test_identity_command(self):self.assertFalse(h.same(E,{**E,'command':'other'}))
 def test_ps_birth_disappearance_requires_second_confirmed_absence(self):
  row='123 1 Sat Oct 17 09:43:17 2026 S python queue.py\n'
  responses=[subprocess.CompletedProcess([],0,row,''),subprocess.CompletedProcess([],1,'','')]
  with patch.object(h.subprocess,'run',side_effect=responses) as run,patch.object(h,'birth',side_effect=ValueError('proc_pidinfo vanished')):
   self.assertIsNone(h.identity(123));self.assertEqual(run.call_count,2)
 def test_ps_birth_failure_still_present_remains_error(self):
  row='123 1 Sat Oct 17 09:43:17 2026 S python queue.py\n'
  responses=[subprocess.CompletedProcess([],0,row,''),subprocess.CompletedProcess([],0,'123\n','')]
  with patch.object(h.subprocess,'run',side_effect=responses),patch.object(h,'birth',side_effect=ValueError('proc_pidinfo denied')):
   with self.assertRaises(ValueError):h.identity(123)
 def test_ps_birth_confirmation_error_is_not_assumed_absence(self):
  row='123 1 Sat Oct 17 09:43:17 2026 S python queue.py\n'
  responses=[subprocess.CompletedProcess([],0,row,''),subprocess.CompletedProcess([],2,'','query failed')]
  with patch.object(h.subprocess,'run',side_effect=responses),patch.object(h,'birth',side_effect=ValueError('proc_pidinfo unknown')):
   with self.assertRaises(ValueError):h.identity(123)
 def test_wrong_identity_never_signals(self):
  f=Fake();f.now={**E,'command':'other'}
  with self.assertRaises(ValueError):h.retire_at_boundary(E,f,lambda:{},lambda:None)
  self.assertEqual(f.sent,[])
 def test_existing_child_no_signal(self):
  f=Fake([[{'pid':456}]])
  self.assertEqual(h.retire_at_boundary(E,f,lambda:{},lambda:None)[1],'SKIPPED_CHILD_PRESENT');self.assertEqual(f.sent,[])
 def test_child_race_resumes_parent_only(self):
  f=Fake([[],[{'pid':456}]])
  self.assertEqual(h.retire_at_boundary(E,f,lambda:{},lambda:None)[1],'SKIPPED_CHILD_RACE');self.assertEqual(f.sent,[signal.SIGSTOP,signal.SIGCONT])
 def test_verify_failure_resumes(self):
  f=Fake()
  def fail():raise ValueError('bad terminal')
  with self.assertRaises(ValueError):h.retire_at_boundary(E,f,fail,lambda:None)
  self.assertEqual(f.sent,[signal.SIGSTOP,signal.SIGCONT])
 def test_cursor_race_resumes(self):
  f=Fake()
  with self.assertRaises(ValueError):h.retire_at_boundary(E,f,lambda:{},lambda:(_ for _ in ()).throw(ValueError('next reserved')))
  self.assertEqual(f.sent,[signal.SIGSTOP,signal.SIGCONT])
 def test_success_term_then_cont_only_parent(self):
  f=Fake();summary={'verified':True};self.assertEqual(h.retire_at_boundary(E,f,lambda:summary,lambda:None),(summary,'PARENT_RETIRED_WITHOUT_CHILD'))
  self.assertEqual(f.sent,[signal.SIGSTOP,signal.SIGTERM,signal.SIGCONT])
 def test_validation_alarm_disarmed_before_irreversible_term(self):
  f=Fake();events=[];old_signal=f.signal
  def record_signal(expected,sig):events.append(sig);old_signal(expected,sig)
  f.signal=record_signal
  h.retire_at_boundary(E,f,lambda:events.append('verified'),lambda:events.append('cursor'),before_retire=lambda:events.append('alarm_disarmed'))
  assert events.index('verified')<events.index('alarm_disarmed')<events.index(signal.SIGTERM)
  assert events[-3:]==['alarm_disarmed',signal.SIGTERM,signal.SIGCONT]
 def test_failure_before_alarm_disarm_only_resumes_parent(self):
  f=Fake();events=[]
  with self.assertRaises(ValueError):h.retire_at_boundary(E,f,lambda:(_ for _ in ()).throw(ValueError('bad')),lambda:None,before_retire=lambda:events.append('alarm_disarmed'))
  self.assertEqual(events,[]);self.assertEqual(f.sent,[signal.SIGSTOP,signal.SIGCONT])
 def test_darwin_term_can_immediately_reap_parent_without_cont(self):
  f=Fake();previous=f.signal
  def deliver(expected,sig):
   previous(expected,sig)
   if sig==signal.SIGTERM:f.now=None
  f.signal=deliver
  result=h.retire_at_boundary(E,f,lambda:{'done':True},lambda:None)
  self.assertEqual(result,({'done':True},'PARENT_RETIRED_WITHOUT_CHILD'))
  self.assertEqual(f.sent,[signal.SIGSTOP,signal.SIGTERM])
 def test_known_zombie_after_term_is_exited_without_cont(self):
  f=Fake();previous=f.signal
  def deliver(expected,sig):
   previous(expected,sig)
   if sig==signal.SIGTERM:f.now={**E,'status':'Z','command':'<defunct>'}
  f.signal=deliver
  self.assertEqual(h.retire_at_boundary(E,f,lambda:{},lambda:None)[1],'PARENT_RETIRED_WITHOUT_CHILD')
  self.assertEqual(f.sent,[signal.SIGSTOP,signal.SIGTERM])
 def test_pid_replacement_after_term_is_not_signalled(self):
  f=Fake();previous=f.signal
  def deliver(expected,sig):
   previous(expected,sig)
   if sig==signal.SIGTERM:f.now={**E,'birth':{'sec':999,'usec':0}}
  f.signal=deliver
  with self.assertRaises(ValueError):h.retire_at_boundary(E,f,lambda:{},lambda:None)
  self.assertEqual(f.sent,[signal.SIGSTOP,signal.SIGTERM])
 def test_retired_prepare_refuses_live_parent_and_writes_no_contract(self):
  with tempfile.TemporaryDirectory() as d:
   base=Path(d);old=base/'old';old.write_text(json.dumps({'expected_old_parent_identity':E}));config=base/'config';config.write_text(json.dumps({'retired_boundary_recovery':{'failed_contract_path':str(old),'failed_contract_sha256':h.sha(old)}}));output=base/'new'
   with patch.object(h,'identity',return_value=E),patch.object(h,'guarded_signal') as signals:
    with self.assertRaises(ValueError):h.prepare_recovery(config,output)
    signals.assert_not_called();self.assertFalse(output.exists())
 def test_retired_prepare_has_no_signals_and_preserves_previous_contract(self):
  with tempfile.TemporaryDirectory() as d:
   base=Path(d);old=base/'old';old.write_text(json.dumps({'expected_old_parent_identity':E}));before=old.read_bytes();config=base/'config';config.write_text(json.dumps({'retired_boundary_recovery':{'failed_contract_path':str(old),'failed_contract_sha256':h.sha(old)}}));output=base/'new'
   with patch.object(h,'identity',return_value=None),patch.object(h,'validate_contract'),patch.object(h,'guarded_signal') as signals:
    result=h.prepare_recovery(config,output);signals.assert_not_called()
   self.assertEqual(result['status'],'RETIRED_RECOVERY_PREPARED_NOT_EXECUTED');self.assertEqual(before,old.read_bytes())
 def test_no_exit_and_nonzero_exit(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'exit';self.assertFalse(h.cheap_terminal(None,{'exit_receipt':str(p)}));p.write_text('{"exit_code":1}')
   with self.assertRaises(ValueError):h.cheap_terminal(None,{'exit_receipt':str(p)})
 def test_cursor_is_readonly_and_next_running_blocks(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'cursor';p.write_text(json.dumps({'stages':{'target':{'state':'RUNNING'},'next':{'state':'PENDING'}}}));b=p.read_bytes();h.quiet_cursor(p,'target');self.assertEqual(b,p.read_bytes())
   p.write_text(json.dumps({'stages':{'target':{'state':'DONE'},'next':{'state':'RUNNING'}}}))
   with self.assertRaises(ValueError):h.quiet_cursor(p,'target')
 def test_restart_sets_cloexec_and_exact_original_argv(self):
  c={'original_executor_path':'exe','original_manifest_path':'man','original_queue_argv':['lexical-python','exe','--execute']}
  with patch.object(h,'descendants',return_value=[]),patch.object(h,'sha',side_effect=[h.EXECUTOR_SHA,h.MANIFEST_SHA]),patch.object(h.os,'set_inheritable') as inherit,patch.object(h.os,'execv') as exe:
   h.restart(c,8);inherit.assert_called_once_with(8,False);exe.assert_called_once_with('lexical-python',c['original_queue_argv'])
 def test_inherited_lock_retained_without_parent_descriptor(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'global.lock';fd=os.open(p,os.O_CREAT|os.O_RDWR,0o600);fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
   child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(.2)'],pass_fds=(fd,));os.close(fd)
   other=os.open(p,os.O_RDWR)
   try:
    with self.assertRaises(BlockingIOError):fcntl.flock(other,fcntl.LOCK_EX|fcntl.LOCK_NB)
    child.wait(timeout=3);fcntl.flock(other,fcntl.LOCK_EX|fcntl.LOCK_NB)
   finally:
    os.close(other)
    if child.poll() is None:child.terminate();child.wait(timeout=3)
 def test_global_lock_excludes_separate_descriptor(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'lock'
   with p.open('a') as a,p.open('a') as b:
    fcntl.flock(a.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    with self.assertRaises(BlockingIOError):fcntl.flock(b.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
 def test_real_self_inventory_is_not_a_descendant(self):
  self.assertEqual(h.descendants(os.getpid()),[])
 def test_real_owned_dummy_is_still_a_descendant(self):
  child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(20)'])
  try:self.assertIn(child.pid,{x['pid'] for x in h.descendants(os.getpid())})
  finally:child.terminate();child.wait(timeout=5)
 def test_own_dummy_process_stop_failure_resume(self):
  # Only this test-owned child is signalled; never any real queue/model PID.
  child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(20)'])
  try:
   expected=h.identity(child.pid)
   with self.assertRaises(ValueError):h.retire_at_boundary(expected,h.RealOps(),lambda:(_ for _ in ()).throw(ValueError('fixture')),lambda:None,timeout=2)
   self.assertIsNone(child.poll());self.assertNotIn('T',h.identity(child.pid)['status'])
  finally:child.terminate();child.wait(timeout=5)
if __name__=='__main__':unittest.main()
