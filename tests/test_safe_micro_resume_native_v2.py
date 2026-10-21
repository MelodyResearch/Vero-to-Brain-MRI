"""Synthetic, no model loading: durable A/B milestones and spent-work accounting."""
from pathlib import Path
import json
import pytest
from test_safe_micro_training_v1 import MicroFake, MIXED, units, parent_of
from test_multistep_gspo_v1 import LENGTHS, REWARD
from mri_rl import safe_micro_training_v2 as resume
from mri_rl.safe_micro_journal_v1 import Indeterminate


class Paused(RuntimeError):
    pass


def execute(path, backend, before=lambda name: None):
    return resume.execute_micro(backend=backend, units=units(), prompt_lengths=LENGTHS,
        reward_policy_sha256=REWARD, output=path, before=before, expected_parent=parent_of(backend),
        journal_binding={'scope':'a'*64, 'input':'b'*64})


def test_resume_after_arm_a_never_generates_or_updates_a_again(tmp_path):
    b=MicroFake(MIXED)
    def pause(name):
        if name=='arm_B': raise Paused('safe boundary before claim')
    with pytest.raises(Paused): execute(tmp_path/'run', b, pause)
    assert b.generate_calls==2 and b.apply_calls==1
    before=json.loads((tmp_path/'run/counters_PRIVATE.json').read_text())
    child=MicroFake(MIXED)
    result=execute(tmp_path/'run', child)
    assert child.generate_calls==0 and child.apply_calls==1
    assert result['optimizer_updates']==2
    assert result['spent_total']['optimizer_updates']==2
    assert result['spent_total']['generated_response_tokens']==before['used']['common']['generated_response_tokens']
    assert child.optimizer_step()==52


def test_interrupted_claimed_common_cannot_resample(tmp_path):
    b=MicroFake(MIXED)
    def abort(name):
        if name=='micro_common_score_old_unit_0': raise Paused('during common')
    with pytest.raises(Paused): execute(tmp_path/'run', b, abort)
    assert b.generate_calls==2
    second=MicroFake(MIXED)
    with pytest.raises(Indeterminate): execute(tmp_path/'run', second)
    assert second.generate_calls==0 and second.apply_calls==0


def test_resume_matches_uninterrupted_exact_endpoints(tmp_path):
    base=MicroFake(MIXED); whole=execute(tmp_path/'whole',base)
    b=MicroFake(MIXED)
    def pause(name):
        if name=='arm_A': raise Paused()
    with pytest.raises(Paused): execute(tmp_path/'resumed',b,pause)
    restored=MicroFake(MIXED); actual=execute(tmp_path/'resumed',restored)
    assert actual['spent_total']==whole['spent_total']
    for arm in ('A','B'):
        expected=json.loads((tmp_path/'whole'/('arm_'+arm)/'receipt_PRIVATE.json').read_text())
        got=json.loads((tmp_path/'resumed'/('arm_'+arm)/'receipt_PRIVATE.json').read_text())
        for key in ('policy_sha_after','adam_digest_after','rng_digest_after','counters','effective_groups'):
            assert expected[key]==got[key]


def test_tampered_committed_arm_does_not_replay_update(tmp_path):
    execute(tmp_path/'run',MicroFake(MIXED))
    p=tmp_path/'run/arm_A/checkpoint_PRIVATE/state.npz'
    p.write_bytes(p.read_bytes()+b'x')
    b=MicroFake(MIXED)
    with pytest.raises((ValueError,RuntimeError)): execute(tmp_path/'run',b)
    assert b.generate_calls==0 and b.apply_calls==0


def test_persistent_native_counter_does_not_reset(tmp_path):
    from mri_rl.safe_micro_pilot_native_v2 import CallCounter
    p=tmp_path/'calls.json'; limits={'generations':2,'tokens':5}
    CallCounter(limits,p).charge('generations')
    counter=CallCounter(limits,p); assert counter.used['generations']==1
    counter.charge('generations')
    with pytest.raises(ValueError): counter.charge('generations')
    assert json.loads(p.read_text())['used']['generations']==2


def test_persistent_native_counter_refuses_changed_limits(tmp_path):
    from mri_rl.safe_micro_pilot_native_v2 import CallCounter
    p=tmp_path/'calls.json'; CallCounter({'calls':1},p)
    with pytest.raises(ValueError): CallCounter({'calls':2},p)


def test_resumed_training_matches_original_v1_scientific_endpoints(tmp_path):
    from mri_rl import safe_micro_training_v1 as legacy
    b=MicroFake(MIXED)
    old=legacy.execute_micro(backend=b,units=units(),prompt_lengths=LENGTHS,reward_policy_sha256=REWARD,
        output=tmp_path/'old',before=lambda _:None,expected_parent=parent_of(b))
    new=execute(tmp_path/'new',MicroFake(MIXED))
    assert old['spent_total']==new['spent_total'] and old['optimizer_updates']==new['optimizer_updates']
    for arm in ('A','B'):
        a=json.loads((tmp_path/'old'/('arm_'+arm)/'receipt_PRIVATE.json').read_text())
        z=json.loads((tmp_path/'new'/('arm_'+arm)/'receipt_PRIVATE.json').read_text())
        for k in ('policy_sha_after','adam_digest_after','rng_digest_after','effective_groups','counters'):
            assert a[k]==z[k]


class EvalFake:
    def __init__(self):self.rng=0;self.loaded=[]
    def save_rng(self,directory):
        (Path(directory)/'rng.json').write_text(json.dumps(self.rng))
    def load_rng(self,directory):
        self.rng=json.loads((Path(directory)/'rng.json').read_text());self.loaded.append(self.rng)
    def rng_digest(self):return str(self.rng)


def test_completed_greedy_row_and_rng_are_reused_without_generation(tmp_path,monkeypatch):
    from mri_rl import safe_micro_pilot_native_v2 as native
    from mri_rl.safe_micro_journal_v1 import Journal
    records=[{'id':str(i),'row_PRIVATE':{'id':str(i),'reward_model':{'ground_truth':label}},'image_sha256':'c'*64}
             for i,label in enumerate(('FLAIR','contrast-enhanced T1'))]
    calls=[]
    def generate(backend,row,image_sha,*,counter,**kwargs):
        calls.append(row['id']);counter.charge('generations');backend.rng+=1
        return {'response':{'text':'<think>x</think><answer>\\boxed{'+row['reward_model']['ground_truth']+'}</answer>'}}
    monkeypatch.setattr(native.v1,'generate_one',generate)
    s1=tmp_path/'session1';s1.mkdir();s2=tmp_path/'session2';s2.mkdir()
    counter=native.CallCounter({'generations':2},tmp_path/'calls.json')
    def pause(key):
        if key=='Parent_row_01':raise Paused()
    binding={'scope':'a'*64}
    with Journal(tmp_path/'journal',binding,before=pause) as j:
        with pytest.raises(Paused):native._evaluate_rows(EvalFake(),'Parent',records,j,counter,lambda _:None,tmp_path/'rows',s1)
    (s1/'release_SAFE.json').write_text(json.dumps({'release_verified':True}))
    fresh=EvalFake()
    with Journal(tmp_path/'journal',binding) as j:
        metric=native._evaluate_rows(fresh,'Parent',records,j,native.CallCounter({'generations':2},tmp_path/'calls.json'),lambda _:None,tmp_path/'rows',s2)
    assert calls==['0','1'] and fresh.loaded==[1] and metric['correct']==2


def test_cached_greedy_row_requires_real_previous_allocator_release(tmp_path):
    from mri_rl.safe_micro_pilot_native_v2 import _check_completed_row
    from types import SimpleNamespace
    row={'counters':{'generations':1},'resident_session':str(tmp_path/'missing')}
    with pytest.raises(ValueError,match='allocator release'):_check_completed_row(row,SimpleNamespace(used={'generations':1}))


def test_stage2_actual_dispatch_calls_qualifier_then_release_then_commit(tmp_path,monkeypatch):
    """Exercise real run_phase control flow, using a CPU-only backend stub."""
    from types import SimpleNamespace
    from mri_rl import safe_micro_pilot_native_v2 as native
    from mri_rl import safe_micro_native_v1 as bridge
    from mri_rl import accuracy_stage_native_v1 as stage
    from mri_rl import accuracy_recovery_v1 as recovery
    from mri_rl import experiment_contract, two_task_training, paired_z045_continuation_entry_v1 as continuation
    from mri_rl.safe_micro_pilot_entry_v1 import Authorization
    from mri_rl.safe_micro_journal_v1 import Journal
    events=[]
    class CPUBackend:
        def __init__(self,*args,**kwargs):
            events.append('construct');self.model=object()
            self.mx=SimpleNamespace(clear_cache=lambda:None,get_active_memory=lambda:0)
        def configure_stage(self,**kwargs):pass
        def close(self):events.append('close');self.model=None
    class Gate:
        def __call__(self,name):events.append('boundary:'+name)
        def claim_model_load(self):events.append('load_claim')
        def peaks(self):return {'model_memory':'CPU_MOCK_NOT_MEASURED'}
    monkeypatch.setattr(bridge,'MicroNativeBackend',CPUBackend)
    monkeypatch.setattr(stage,'restore_parent',lambda *a:events.append('restore_parent'))
    monkeypatch.setattr(recovery,'parent_disk_snapshot',lambda *a:{})
    monkeypatch.setattr(recovery,'adam52_parent_anchor',lambda *a:{'receipt_sha256':'c'*64})
    monkeypatch.setattr(recovery,'require_parent_unchanged',lambda *a:True)
    monkeypatch.setattr(experiment_contract,'load_contract',lambda *a:{})
    monkeypatch.setattr(two_task_training,'make_grader',lambda *a:object())
    monkeypatch.setattr(continuation,'verify_images',lambda *a:True)
    monkeypatch.setattr(native.v1,'_configuration',lambda runtime:{})
    out=tmp_path/'stage2_PRIVATE'
    def qualify(backend,micro,dev,directory,before):
        events.append('qualify');assert backend.model is not None
        for name in ('calls_PRIVATE.json','forward_actual_PRIVATE.json','readonly_integrity_PRIVATE.json'):
            (Path(directory)/name).write_text('{}')
        return {'status':'PASS','optimizer_updates':0}
    monkeypatch.setattr(native,'_stage2',qualify)
    paths={k:tmp_path/(k+'.json') for k in ('runtime_spec','micro_input','dev24')}
    calibration=tmp_path/'calibration.json';calibration.write_text('{}')
    paths['runtime_spec'].write_text(json.dumps({'parent_checkpoint':str(tmp_path/'parent'),'parent_receipt_sha256':'c'*64,
        'calibration':str(calibration),'model':str(tmp_path/'model'),'contract':'unused','cache_dir':'unused'}))
    paths['micro_input'].write_text(json.dumps({'units_PRIVATE':units(),'planned_image_bindings_PRIVATE':[], 'prompt_lengths':LENGTHS}))
    paths['dev24'].write_text('{}')
    scope={'execution_stage':'SINGLE_LOAD_FORWARD','output':str(out),'compound_scope_sha256':'a'*64,'input_sha256':'b'*64}
    auth=object.__new__(Authorization);auth.parent=object();auth.scope=scope;auth.sha='d'*64;auth.check=lambda _:True
    gate=Gate();gate.authorization=auth;gate.scope=scope
    result=native.run_phase(2,spec={k:str(p) for k,p in paths.items()},scope=scope,authorization=auth,resource_gate=gate)
    assert result['status']=='PASS' and result['release_verified']
    assert events.count('qualify')==1 and events.count('load_claim')==1
    assert events.index('qualify')<events.index('close')
    releases=list(out.glob('parent_load_resident_*_PRIVATE/qualification_release_SAFE.json'))
    assert len(releases)==1 and json.loads(releases[0].read_text())['release_verified']
    with Journal(out/'native_journal_PRIVATE',native._binding(scope,2)) as journal:
        assert journal.get('qualification')['status']=='PASS'
        assert journal.get('native_complete')['release_verified']
    replay=native.run_phase(2,spec={k:str(p) for k,p in paths.items()},scope=scope,authorization=auth,resource_gate=gate)
    assert replay==result and events.count('qualify')==1 and events.count('load_claim')==1


def test_pause_before_native_finalization_does_not_publish_uncommitted_result(tmp_path):
    from mri_rl.safe_micro_pilot_native_v2 import commit_native_complete
    from mri_rl.safe_micro_journal_v1 import Journal
    def pause(key):
        if key=='native_complete':raise Paused()
    with Journal(tmp_path/'journal',{'scope':'a'*64},before=pause) as journal:
        with pytest.raises(Paused):commit_native_complete(journal,tmp_path,{'status':'PASS','resources':{'peak':1}})
    assert not (tmp_path/'native_complete_PRIVATE.json').exists()
    with Journal(tmp_path/'journal',{'scope':'a'*64}) as journal:
        assert commit_native_complete(journal,tmp_path,{'status':'PASS','resources':{'peak':2}})['resources']['peak']==2
