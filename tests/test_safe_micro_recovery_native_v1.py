"""CPU synthetic recovery checks; never model loading or native qualification."""
from pathlib import Path
import ast,json
from types import SimpleNamespace
import numpy as np
import pytest
from mri_rl import safe_micro_recovery_native_v1 as recovery
from mri_rl import safe_micro_training_v1 as old
from mri_rl import safe_micro_training_v2 as durable
from test_safe_micro_training_v1 import MicroFake,MIXED,units,parent_of
from test_multistep_gspo_v1 import LENGTHS,REWARD


def common(tmp_path,b):
    p=tmp_path/'source';p.mkdir();budget=old.micro_budget(units(),LENGTHS);ledger=old.Ledger(budget,p/'counts.json')
    parent=old._state(b);snap=b.snapshot()
    value=durable._common(b,units(),LENGTHS,REWARD,ledger,parent,p,lambda _:None,None)['value']
    b.restore(snap)
    return durable.restore_common(value,units(),REWARD,parent,None,b),parent,snap


def test_B_body_only_removes_original_final_rollback():
    original,new=recovery.arm_body_nodes()
    assert isinstance(original.body[-1],ast.Try) and original.body[-1].finalbody
    expected=original.body[:-1]+original.body[-1].body
    assert ast.dump(ast.Module(body=new.body,type_ignores=[]))==ast.dump(ast.Module(body=expected,type_ignores=[]))
    assert all('finally' not in ast.unparse(n) for n in new.body)


def test_B_checkpoint_matches_old_formula_but_does_not_rollback_live_model(tmp_path):
    b=MicroFake(MIXED);c,parent,snapshot=common(tmp_path,b)
    p=tmp_path/'new';p.mkdir();ledger=old.Ledger(old.micro_budget(units(),LENGTHS),p/'counts.json')
    actual=recovery.run_B_once(backend=b,units=units(),common=c,ledger=ledger,snapshot=snapshot,parent=parent,output=p,before=lambda _:None)
    assert actual['updated'] and b.optimizer_step()==53 and b.apply_calls==1 and b.generate_calls==2
    oldb=MicroFake(MIXED);q=tmp_path/'old';q.mkdir()
    older=old._arm(arm='B',backend=oldb,units=units(),common=c,ledger=old.Ledger(old.micro_budget(units(),LENGTHS),q/'counts.json'),
        snapshot=oldb.snapshot(),parent=old._state(oldb),output=q,before=lambda _:None,kind=oldb.evidence_kind)
    assert oldb.optimizer_step()==52 and older['counters']==actual['counters']
    a=json.loads((p/'arm_B/receipt_PRIVATE.json').read_text());z=json.loads((q/'arm_B/receipt_PRIVATE.json').read_text())
    for key in ('policy_sha_after','adam_digest_after','rng_digest_after','effective_groups','counters'):
        assert a[key]==z[key]


def test_B_rejects_another_live_parent_or_existing_arm(tmp_path):
    b=MicroFake(MIXED);c,parent,snap=common(tmp_path,b);p=tmp_path/'new';p.mkdir()
    ledger=old.Ledger(old.micro_budget(units(),LENGTHS),p/'counts.json')
    b.step=53
    with pytest.raises(ValueError):recovery.run_B_once(backend=b,units=units(),common=c,ledger=ledger,snapshot=snap,parent=parent,output=p,before=lambda _:None)
    b.restore(snap);(p/'arm_B').mkdir()
    with pytest.raises(ValueError):recovery.run_B_once(backend=b,units=units(),common=c,ledger=ledger,snapshot=snap,parent=parent,output=p,before=lambda _:None)


def test_leaf_copy_copies_trainable_and_state_but_does_not_copy_frozen_base():
    b=object.__new__(recovery.RecoveryBackend)
    b.mx=SimpleNamespace(eval=lambda *a:None);b.copy_observations=0
    frozen=np.array([1.,2.]);trained=np.array([3.,4.])
    tree={'frozen':frozen,'layer':[{'weight':trained}]}
    value=recovery.clone_parameter_tree(tree,{'layer.0.weight'},b.copy_leaf)
    assert value is not tree and value['frozen'] is frozen and value['layer'][0]['weight'] is not trained
    value['layer'][0]['weight'][0]=99
    assert trained[0]==3 and b.copy_observations==1


def test_copy_budget_and_unknown_container_paths_fail_closed():
    b=object.__new__(recovery.RecoveryBackend);b.mx=SimpleNamespace(eval=lambda *a:None);b.copy_observations=10000
    with pytest.raises(ValueError,match='copy'):b.copy_leaf(np.array([1]))
    with pytest.raises(ValueError,match='trainable'):recovery.clone_parameter_tree({'x':np.array([1])},{'missing'},lambda x:x.copy())


def test_cleanup_releases_snapshot_and_active_references_even_on_failure():
    b=SimpleNamespace(_unit_snapshot=object(),_active=object(),_active_modality='F',active=object(),_recorded=object(),_rollout_contexts={1:object()})
    recovery.drop_snapshot_references(b)
    assert b._unit_snapshot is None and b._active is None and b.active is None and not b._rollout_contexts


def test_snapshot_holder_cleanup_releases_cpu_array_storage():
    import gc,weakref
    tensor=np.arange(100,dtype=np.float32);reference=weakref.ref(tensor)
    holder=[{'parameter':tensor}]
    b=SimpleNamespace(_unit_snapshot=holder[0],_active={'pixels':tensor},active=tensor,_rollout_contexts={1:tensor})
    del tensor;assert reference() is not None
    holder[0]=None;recovery.drop_snapshot_references(b);gc.collect()
    assert reference() is None


def dispatch_setup(tmp_path,monkeypatch,stage_id):
    from mri_rl import accuracy_recovery_v1 as parent_module,accuracy_stage_native_v1 as loader
    from mri_rl import experiment_contract,two_task_training,paired_z045_continuation_entry_v1 as entry
    from mri_rl.safe_micro_pilot_entry_v1 import Authorization,PHASES
    events=[];instances=[]
    class CPUBackend(MicroFake):
        def __init__(self,*args,**kwargs):
            super().__init__(MIXED);self.model=object();self.mx=SimpleNamespace(clear_cache=lambda:None,get_active_memory=lambda:0)
            instances.append(self);events.append('construct')
        def configure_stage(self,**kwargs):pass
        def close(self):
            assert getattr(self,'_unit_snapshot',None) is None
            events.append('close');self.model=None
    class Gate:
        def __call__(self,name):events.append('guard:'+name)
        def claim_model_load(self):events.append('load_claim')
        def peaks(self):return {'evidence_kind':'CPU_MOCK_NOT_NATIVE_MEMORY'}
    base=MicroFake(MIXED);p=parent_of(base)
    salvage={**{'parent_'+k:v for k,v in zip(('policy_sha256','adam_digest','rng_digest'),(p['policy'],p['adam'],p['rng']))},'parent_receipt_sha256':None}
    common(tmp_path,base)
    value=json.loads((tmp_path/'source/common_PRIVATE/raw_rollout_and_own_old_PRIVATE.json').read_text())
    monkeypatch.setattr(recovery,'RecoveryBackend',CPUBackend)
    monkeypatch.setattr(recovery,'read_salvage',lambda spec:(salvage,value,{'receipt_sha256':'c'*64},tmp_path/'old_A'))
    monkeypatch.setattr(parent_module,'parent_disk_snapshot',lambda *a:{})
    monkeypatch.setattr(parent_module,'adam52_parent_anchor',lambda *a:{'receipt_sha256':None})
    monkeypatch.setattr(parent_module,'require_parent_unchanged',lambda *a:True)
    monkeypatch.setattr(loader,'restore_parent',lambda backend,*a:events.append('fresh_disk_parent52'))
    monkeypatch.setattr(experiment_contract,'load_contract',lambda *a:{})
    monkeypatch.setattr(two_task_training,'make_grader',lambda *a:object())
    monkeypatch.setattr(entry,'verify_images',lambda *a:True)
    monkeypatch.setattr(recovery.pilot,'_configuration',lambda *a:{})
    root=tmp_path/'new';root.mkdir();out=root/('stage'+str(stage_id)+'_PRIVATE')
    calibration=tmp_path/'calibration.json';calibration.write_text('{}')
    # Same reward hash used when building the saved common bundle.
    original_hash=recovery.shared.file_sha256
    monkeypatch.setattr(recovery.shared,'file_sha256',lambda path:REWARD if Path(path)==calibration else original_hash(path))
    files={k:tmp_path/(k+'.json') for k in ('runtime_spec','micro_input','dev24')}
    files['runtime_spec'].write_text(json.dumps({'parent_checkpoint':str(tmp_path/'parent'),'parent_receipt_sha256':None,
        'calibration':str(calibration),'model':'unused','contract':'unused','cache_dir':'unused'}))
    files['micro_input'].write_text(json.dumps({'units_PRIVATE':units(),'prompt_lengths':LENGTHS,'planned_image_bindings_PRIVATE':[]}))
    files['dev24'].write_text('{}')
    if stage_id!=2:
        q=root/'stage2_PRIVATE';q.mkdir();(q/'terminal_PRIVATE.json').write_text(json.dumps({'status':'PASS','qualification':{'status':'PASS','evidence_kind':'CPU_MOCK'},'release_verified':True}))
    from mri_rl.safe_micro_recovery_entry_v1 import ADOPTION
    scope={'execution_stage':PHASES[stage_id],'output':str(out),'compound_scope_sha256':'a'*64,'input_sha256':'b'*64,'adoption':dict(ADOPTION)}
    auth=object.__new__(Authorization);auth.scope=scope;auth.parent=object();auth.sha='d'*64;auth.check=lambda _:True
    gate=Gate();gate.authorization=auth;gate.scope=scope
    return {**{k:str(v) for k,v in files.items()},'output':str(root)},scope,auth,gate,events,instances


def test_Q_dispatch_qualifies_releases_then_commits_no_generation(tmp_path,monkeypatch):
    spec,scope,auth,gate,events,instances=dispatch_setup(tmp_path,monkeypatch,2)
    def qualification(backend,before):
        events.append('Q');backend._unit_snapshot={'held':np.ones(5)}
        return {'status':'PASS','evidence_kind':'CPU_MOCK','optimizer_updates':0}
    monkeypatch.setattr(recovery,'qualify_restore',qualification)
    result=recovery.run_phase(2,spec=spec,scope=scope,authorization=auth,resource_gate=gate)
    assert result['status']=='PASS' and result['release_verified']
    assert events.index('Q')<events.index('close') and events.count('load_claim')==1
    assert instances[0].generate_calls==instances[0].apply_calls==0
    replay=recovery.run_phase(2,spec=spec,scope=scope,authorization=auth,resource_gate=gate)
    assert replay==result and events.count('Q')==events.count('load_claim')==1


def test_B_dispatch_uses_saved_common_only_and_never_final_parent_rollback(tmp_path,monkeypatch):
    spec,scope,auth,gate,events,instances=dispatch_setup(tmp_path,monkeypatch,3)
    result=recovery.run_phase(3,spec=spec,scope=scope,authorization=auth,resource_gate=gate)
    assert result['status']=='PASS' and result['B']['updated'] and result['optimizer_updates']==1
    assert result['adopted_A_optimizer_updates']==1 and not result['common_generated_again'] and not result['old_scores_refreshed']
    assert instances[0].generate_calls==0 and instances[0].apply_calls==1 and instances[0].optimizer_step()==53
    replay=recovery.run_phase(3,spec=spec,scope=scope,authorization=auth,resource_gate=gate)
    assert replay==result and events.count('load_claim')==1 and instances[0].apply_calls==1


def test_top_level_adoption_boolean_cannot_replace_exact_nested_contract(tmp_path,monkeypatch):
    spec,scope,auth,gate,events,instances=dispatch_setup(tmp_path,monkeypatch,2)
    scope.pop('adoption');scope['adopt_saved_A_after_Q']=True
    with pytest.raises(ValueError,match='adoption'):recovery.run_phase(2,spec=spec,scope=scope,authorization=auth,resource_gate=gate)
    assert not instances and 'load_claim' not in events


def test_load_saved_rng_does_not_copy_or_reinstall_optimizer(tmp_path):
    import random
    from mri_rl.optimizer_checkpoint import pack_tree
    from mri_rl import shared_rollout_gspo_core_v1 as shared
    python_before=random.getstate();target=random.Random(19).getstate()
    rng={'python':target,'mlx':[np.array([7,9],dtype=np.uint32)]}
    layout,arrays=pack_tree(rng)
    p=tmp_path/'rng.safetensors';p.write_bytes(b'CPU_ARRAY_FIXTURE_ONLY')
    (tmp_path/'rng.json').write_text(json.dumps({'layout':layout,'tensors_sha256':shared.file_sha256(p)}))
    b=object.__new__(recovery.RecoveryBackend);b.copy_observations=0
    b.mx=SimpleNamespace(eval=lambda *a:None,array_equal=lambda a,z:np.asarray(np.array_equal(a,z)),random=SimpleNamespace(state=[np.array([0,0],dtype=np.uint32)]))
    b.load_arrays=lambda _:arrays;b.adam_digest=lambda:'a'*64
    b.optimizer_state=lambda:pytest.fail('RNG restore must not request/copy/reinstall optimizer')
    try:
        b.load_rng(tmp_path)
        assert random.getstate()==target and np.array_equal(b.mx.random.state[0],rng['mlx'][0])
        assert b.copy_observations==1
    finally:random.setstate(python_before)
