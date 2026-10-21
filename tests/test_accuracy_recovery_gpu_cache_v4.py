"""CPU fixtures only; fake converter, no MLX/GPU/model. No actual authorization."""
import copy,json
from pathlib import Path
import pytest
from mri_rl import accuracy_recovery_gpu_cache_v4 as v4
from mri_rl import accuracy_recovery_gpu_cache_v3 as v3, accuracy_recovery_gpu_cache_v2 as v2
from tests.test_accuracy_recovery_gpu_cache_v3 import Clock,S,seq,env3,prep3,adopt3
from tests.test_accuracy_recovery_gpu_cache_v2 import installed_fn,now_utc,SHA,write as write2


def controller(tmp_path,sample):
 c=Clock();return v4.ResourceController(sampler=seq(sample),os_level=lambda:1,clock=c,sleep=c.sleep,
   log_path=tmp_path/'resources.jsonl',baseline_swap_gib=13.96966796875),c


def test_local_battery_policy_preserves_real_samples_and_thresholds(tmp_path):
 for power,battery,want in [('BATTERY',20,'CONTINUE'),('BATTERY',19.99,'STOP'),('BATTERY',None,'HOLD'),('AC',90,'CONTINUE'),('AC',None,'HOLD'),('UNKNOWN',80,'HOLD')]:
  c,_=controller(tmp_path,S(power_source=power,battery_percent=battery,swap_used_gib=14.912))
  assert c.observe('probe')[1]==want
  assert c.last['power_source']==power and c.last['battery_percent']==battery
 assert v3.POLICY['ac_required'] is True


def test_carried_swap_debt_and_all_other_guards_remain(tmp_path):
 for kw,level,want in [({'swap_used_gib':15.47},1,'STOP'),({'swap_used_gib':16.01},1,'STOP'),({'pressure_free_percent':15},1,'STOP'),({'available_gib':5.49},1,'HOLD'),({'process_rss_gib':8.1},1,'STOP'),({},4,'STOP')]:
  c,_=controller(tmp_path,S(power_source='BATTERY',battery_percent=80,**kw));c.os_level=lambda:level
  assert c.observe('test')[1]==want
  assert c.baseline_swap==13.96966796875


def fixture4(tmp_path):
 e=env3(tmp_path);adopt3(e);info=prep3(e);scope=json.loads(Path(info['scope_path']).read_text())
 paths=v4.closed_paths(e.repo);scope_path=paths['scope'];scope_path.write_text(json.dumps(scope))
 child={'native_exit_code':1,'timed_out':False,'elapsed_seconds':1.,'stdout_sha256':SHA(paths['stdout']) if paths['stdout'].exists() else ''}
 paths['human'].write_text(json.dumps({'human_instruction_sha256':'c'*64}));paths['review'].write_text('{}')
 paths['stdout'].write_text('');paths['stderr'].write_text('ResourceStop at ctrl.preflight before backend')
 child.update(stdout_sha256=SHA(paths['stdout']),stderr_sha256=SHA(paths['stderr']))
 paths['launcher'].write_text(json.dumps({'scope_file_sha256':SHA(scope_path),'scope_digest':v4.shared.scope_digest(scope),'human_file_sha256':SHA(paths['human']),'child':child}))
 paths['resource'].write_text(json.dumps({'boundary':'preflight','decision':'STOP','reasons':['swap_net_growth_over_cap'],'baseline_swap_gib':13.96966796875})+'\n')
 paths['waiver'].parent.mkdir(parents=True,exist_ok=True);paths['waiver'].write_text('HUMAN FIXTURE battery waiver this round only')
 pins={k:SHA(p) for k,p in paths.items()};return e,pins


def prepare4(e,pins):
 return v4.prepare_scope_v4(repo_root=e.repo,scope_output=e.repo/v2.SCOPE_DIR_REL/'v4.json',closed_pins=pins,expected=e.expected,installed=installed_fn)


def test_prepare_is_metadata_only_and_binds_closed_attempt_and_waiver(tmp_path,monkeypatch):
 e,pins=fixture4(tmp_path);original=Path.open
 def guarded(path,*a,**kw):
  assert not str(path).endswith('.safetensors'),'prepare must not hash source/unit payloads'
  return original(path,*a,**kw)
 monkeypatch.setattr(Path,'open',guarded)
 info=prepare4(e,pins);scope=json.loads(Path(info['scope_path']).read_text())
 assert scope['implemented_by']=='codex' and scope['limits']['new_underlying_quantizations_max']==686
 assert scope['previous_attempt']['swap_baseline_gib']==13.96966796875
 assert 'closed_waiver' in scope['files'] and scope['payload_verification']=='DEFERRED_TO_EXECUTE_UNDER_GLOBAL_LOCK'


def test_closed_attempt_tampering_and_any_native_progress_refuse(tmp_path):
 e,pins=fixture4(tmp_path);paths=v4.closed_paths(e.repo)
 paths['stderr'].write_text('changed')
 with pytest.raises(v2.ScopeRefused):prepare4(e,pins)
 paths['stderr'].write_text('ResourceStop at ctrl.preflight before backend')
 pins['stderr']=SHA(paths['stderr'])
 (Path(e.work3)/'v3_ledger.json').write_text('{}')
 with pytest.raises(v2.ScopeRefused,match='preflight|ledger'):prepare4(e,pins)


def test_fresh_exact_authority_and_once_ledger(tmp_path):
 e,pins=fixture4(tmp_path);info=prepare4(e,pins);scope=json.loads(Path(info['scope_path']).read_text())
 from datetime import timedelta
 t=now_utc();human={'authorized':True,'action':v4.ACTION,'scope_sha256':info['scope_sha256'],'human_instruction_sha256':'b'*64,'approved_by':'HUMAN_FIXTURE','issued_at_utc':(t-timedelta(minutes=1)).isoformat(),'expires_at_utc':(t+timedelta(hours=1)).isoformat()}
 review={'verdict':'PASS','scope_sha256':info['scope_sha256'],'reviewer_id':'INDEPENDENT_FIXTURE','reviewer_kind':'isolated_codex','source_sha256':scope['source_sha256'],'input_sha256':scope['input_sha256']}
 hp,hh=write2(e,'human4.json',human);rp,rh=write2(e,'review4.json',review)
 kw=dict(scope_path=info['scope_path'],scope_sha256=info['scope_file_sha256'],human_path=hp,human_sha256=hh,review_path=rp,review_sha256=rh,repo_root=e.repo,expected=e.expected,installed=installed_fn,closed_pins=pins,now=t)
 adm=v4.admit_v4(**kw);adm.consume()
 with pytest.raises(v2.ScopeRefused,match='consumed'):v4.admit_v4(**kw)
 human['action']=v3.ACTION;hp,hh=write2(e,'old_action.json',human)
 with pytest.raises(v2.ScopeRefused):v4.admit_v4(**{**kw,'human_path':hp,'human_sha256':hh})


def test_converter_reuses_adopted_unit_without_correctness_relaxation(tmp_path):
 e,pins=fixture4(tmp_path)
 from tests.test_accuracy_recovery_gpu_cache_v3 import RecController
 from tests.test_accuracy_recovery_gpu_cache_v2 import GpuMx
 from tests.test_accuracy_recovery_cpu_cache_v1 import FakeNN
 import numpy as np
 mx=GpuMx();b=v2.GpuConvertBackend(e.proofs,mx=mx,np=np,nn=FakeNN(mx),rss_gib=lambda:.5,max_attempts=1,max_quantize=1,max_module=1,max_underlying=2)
 r=v3.convert_v3(e.plan,e.args['source_dir'],e.work3/'units',b,RecController(),adopted=frozenset({0}))
 assert r['complete'] and b.attempts==1 and b.current!=0 and b.underlying_quantizations==2
 assert v4.v3.verify_historical_v3 is v3.verify_historical_v3 and v4.v2.make_gpu_backend is v2.make_gpu_backend


@pytest.mark.parametrize('alter',[0,1])
def test_execute_fake_only_adopts_unit0_and_fails_closed_on_payload_difference(tmp_path,alter):
 from datetime import timedelta
 from tests.test_accuracy_recovery_gpu_cache_v2 import GpuMx
 from tests.test_accuracy_recovery_cpu_cache_v1 import FakeNN,sampler
 import numpy as np
 e,pins=fixture4(tmp_path);info=prepare4(e,pins);scope=json.loads(Path(info['scope_path']).read_text());t=now_utc()
 human={'authorized':True,'action':v4.ACTION,'scope_sha256':info['scope_sha256'],'human_instruction_sha256':'d'*64,'approved_by':'HUMAN_FIXTURE','issued_at_utc':(t-timedelta(minutes=1)).isoformat(),'expires_at_utc':(t+timedelta(hours=1)).isoformat()}
 review={'verdict':'PASS','scope_sha256':info['scope_sha256'],'reviewer_id':'INDEPENDENT_FIXTURE','reviewer_kind':'isolated_codex','source_sha256':scope['source_sha256'],'input_sha256':scope['input_sha256']}
 hp,hh=write2(e,'h4exec.json',human);rp,rh=write2(e,'r4exec.json',review);mx=GpuMx(alter=alter);seen=[];clock=Clock()
 def factory(proofs,**kw):
  b=v2.GpuConvertBackend(proofs,mx=mx,np=np,nn=FakeNN(mx),**kw);seen.append(b);return b
 kw=dict(scope_path=info['scope_path'],scope_sha256=info['scope_file_sha256'],human_path=hp,human_sha256=hh,review_path=rp,review_sha256=rh,
  repo_root=e.repo,expected=e.expected,installed=installed_fn,closed_pins=pins,backend_factory=factory,sampler=sampler(),os_level=lambda:1,rss_gib=lambda:.5,clock=clock,sleep=clock.sleep,now=lambda:t)
 if alter:
  with pytest.raises(v2.HistoricalPayloadMismatch):v4.execute(**kw)
  failure=json.loads((e.repo/'work'/v4.WORK_ROOT_NAME/'v4_failure_PRIVATE.json').read_text())
  assert failure['new_gpu_unit_attempts']==failure['new_underlying_quantizations']==1
  with pytest.raises(v2.ScopeRefused,match='consumed'):v4.execute(**kw)
  return
 result=v4.execute(**kw)
 assert seen[0].attempts==1 and seen[0].current!=0
 assert result['certificate']['gpu_reconstruction']['new_underlying_quantizations']==2
 assert result['certificate']['shards']==e.hist['shards']
 with pytest.raises(v2.ScopeRefused,match='consumed'):v4.execute(**kw)
