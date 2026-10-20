"""CPU fake-backend continuation/state/launch-block tests; no model imports."""
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pytest
from mri_rl import paired_z045_continuation_entry_v1 as entry
from mri_rl import paired_z045_continuation_driver_v1 as driver


def fixture(tmp_path):
    path=Path(__file__).with_name('test_uniform_balance_continuation_v1.py')
    spec=importlib.util.spec_from_file_location('old_uniform_fixture',path); old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    B,parent,_,rows=old.fixture(tmp_path)
    # Emit a genuine nested CPU fixture using the existing serialization/state API.
    b=B(None,{});b.state['step']=np.array(60,dtype=np.uint64);b.state['weight']={'m':np.array([.4,-.3]),'v':np.array([.2,.1])}
    out=tmp_path/'adam60';out.mkdir();pending=out/'.pending';pending.mkdir()
    r=driver._publish(out,pending,b,{'experiment':{'adaptive_parent':{'groups':121,'updates':56}}},8,4,'prior',{'updated':True})
    return B,r['path'],driver.digest(r['path']/'receipt.json'),rows


def invoke(B,parent,sha,rows,out,n,resume=False,**kwargs):
    return driver.run(output=out,schedule=rows,bindings={'test':'new-paired-stage'},backend_factory=B,authorize=lambda c:None,
        max_new_groups=n,resume=resume,parent_checkpoint=parent,parent_receipt_sha256=sha,max_optimizer_updates=8,**kwargs)


def test_adam60_global129_and_exact_resume(tmp_path):
    B,parent,sha,rows=fixture(tmp_path)
    invoke(B,parent,sha,rows,tmp_path/'full',3); a=B.instances[-1]
    invoke(B,parent,sha,rows,tmp_path/'split',1); result=invoke(B,parent,sha,rows,tmp_path/'split',2,True); b=B.instances[-1]
    assert result['absolute_completed_groups']==132 and result['absolute_optimizer_updates']==63
    np.testing.assert_array_equal(a.weight,b.weight)
    for key in ('m','v'):np.testing.assert_array_equal(a.state['weight'][key],b.state['weight'][key])
    assert a.rng.random()==b.rng.random()
    initial=json.loads((tmp_path/'split/boundary_0000/receipt.json').read_text())
    assert initial['absolute_completed_groups']==129 and initial['absolute_optimizer_updates']==60


def test_wall_deadline_stops_at_committed_boundary(tmp_path):
    B,parent,sha,rows=fixture(tmp_path)
    now=[0.]
    class Slow(B):
        def rollout(self,*args,**kwargs):
            value=super().rollout(*args,**kwargs);now[0]=100.;return value
    result=invoke(Slow,parent,sha,rows,tmp_path/'wall',3,maximum_seconds=10,clock=lambda:now[0])
    assert result['status']=='WALL_BUDGET_REACHED' and result['completed_groups']==1
    assert not list((tmp_path/'wall').glob('.pending-*'))


def test_metadata_inspection_does_not_read_array_payloads(tmp_path,monkeypatch):
    B,parent,sha,rows=fixture(tmp_path)
    original=Path.open
    def guarded(path,*args,**kwargs):
        assert path.suffix!='.safetensors','CPU metadata must not read weight/Adam payloads'
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',guarded)
    anchor=entry.parent_metadata(parent,sha)
    assert anchor['groups']==129 and anchor['updates']==60 and anchor['array_bytes_verified'] is False


def test_native_backend_missing_fails_before_model_import_even_with_forged_pass():
    before=set(sys.modules)
    with pytest.raises(ValueError,match='F32_NATIVE_QUALIFICATION_REQUIRED'):
        entry.training_preflight({'scope':'fixture'},{'status':'PASS'},{'authorized':True},{'verdict':'PASS'})
    assert not any(name=='mlx' or name.startswith('mlx.') for name in set(sys.modules)-before)


def test_contract_describes_missing_full_training_path():
    missing=entry.numeric_prerequisites()
    assert missing['implementation_status']=='IMPLEMENTED_CPU_NOT_NATIVE_QUALIFIED'
    assert {'onpolicy_generation_logps','KV_full_on_generated_IDS','gradient_path','parent_Adam60_state','complete_response_EOS'}<=set(missing['required_actual_evidence'])
    assert missing['narrow_readonly_proofs_authorize_training'] is False


def test_envelope_is32_8_and_historical_profile_unchanged():
    from mri_rl import uniform_balance_continuation_driver_v1 as old
    assert driver.PROFILE==old.PROFILE
    assert driver.ENVELOPE=={'maximum_attempted_groups':32,'maximum_optimizer_updates':8}


def test_corrupt_parent_metadata_and_nonfinite_wall_rejected_before_backend(tmp_path):
    B,parent,sha,rows=fixture(tmp_path)
    with pytest.raises(ValueError):invoke(B,parent,sha,rows,tmp_path/'bad',1,maximum_seconds=float('nan'))
    n=len(B.instances);(parent/'receipt.json').write_text('{}')
    with pytest.raises(ValueError):entry.parent_metadata(parent,sha)
    assert len(B.instances)==n


def test_interrupted_pending_work_cost_charged_once_after_Adam60(tmp_path):
    from mri_rl.two_task_training import TokenLedger
    B,parent,sha,rows=fixture(tmp_path)
    class Crash(B):
        def rollout(self,row,index,workdir):
            raw=super().rollout(row,index,workdir);raw['group_index']=index;raw['responses']=raw['responses'][:1]
            (Path(workdir)/'rollout_progress_PRIVATE.json').write_text(json.dumps(raw))
            raise RuntimeError('one interrupted CPU group')
    plan={'group_size':2,'max_tokens':256,'budget':{'maximum_attempted_groups':32,'text_visual_token_cap':10000,'input_tokens_min':213,'input_tokens_max':213}}
    def go(cls,resume):
        ledger=TokenLedger(plan)
        result=invoke(cls,parent,sha,rows,tmp_path/'debt',1,resume,ledger=ledger,input_tokens=lambda row:213)
        return result,ledger
    with pytest.raises(RuntimeError,match='interrupted'):go(Crash,False)
    _,first=go(B,True);_,second=go(B,True)
    assert first.used==214+469+428 and second.used==first.used+428
    assert first.imputed==second.imputed==469
    receipt=json.loads((tmp_path/'debt/boundary_0002/receipt.json').read_text())
    assert receipt['absolute_completed_groups']==131 and receipt['absolute_optimizer_updates']==62
    assert len(receipt['ledger']['partial_dirs_included'])==1


def test_future_numeric_human_review_evidence_cannot_substitute_narrow_proofs():
    scope={'scope_kind':'FUTURE_EXACT_TRAIN_LAUNCH_SCOPE','numeric_contract_sha256':'numeric','source_sha256':{'f32_backend.py':'source'},'budget':{'maximum_attempted_groups':32,'maximum_optimizer_updates':8}}
    sh=entry.scope_digest(scope)
    numeric={'policy':'F32_ACTIVATION_GENERATION_SCORER_GSPO_V1','source_sha256':scope['source_sha256'],
             'status':'PASS_WITH_LIMITATIONS','contract_sha256':'numeric','actual_evidence':{k:True for k in entry.numeric_prerequisites()['required_actual_evidence']}}
    human={'authorized':True,'action':'finite_train','scope_sha256':sh,'human_instruction_sha256':'direct-human-evidence'}
    review={'verdict':'PASS_WITH_LIMITATIONS','scope_sha256':sh,'reviewer_id':'independent_fixture','reviewer_kind':'isolated_codex'}
    assert entry.validate_launch_evidence(scope,numeric,human,review)
    for field in numeric['actual_evidence']:
        bad={**numeric,'actual_evidence':{**numeric['actual_evidence'],field:False}}
        with pytest.raises(ValueError):entry.validate_launch_evidence(scope,bad,human,review)
    with pytest.raises(ValueError):entry.validate_launch_evidence(scope,numeric,{**human,'authorized':False},review)
    with pytest.raises(ValueError):entry.validate_launch_evidence(scope,numeric,human,{**review,'scope_sha256':'changed'})
    # Even complete evidence does not create the currently absent backend.
    with pytest.raises(ValueError,match='F32_NATIVE_QUALIFICATION_REQUIRED'):entry.training_preflight(scope,numeric,human,review)


def test_driver_algorithm_body_is_unchanged_outside_declared_envelope_and_boundary_stop():
    import ast
    old=Path(driver.__file__).with_name('uniform_balance_continuation_driver_v1.py').read_text()
    new=Path(driver.__file__).read_text()
    oldtree=ast.parse(old);newtree=ast.parse(new)
    of={n.name:n for n in oldtree.body if isinstance(n,ast.FunctionDef)}
    nf={n.name:n for n in newtree.body if isinstance(n,ast.FunctionDef)}
    for name in of:
        if name!='run':assert ast.dump(of[name])==ast.dump(nf[name]),name
    # The sole driver function change is bounded-start bookkeeping. Reward,
    # advantages, active gate, backend.update and _publish call syntax stay exact.
    def calls(function):
        return [ast.dump(n) for n in ast.walk(function) if isinstance(n,ast.Call) and
                isinstance(n.func,(ast.Name,ast.Attribute)) and
                ((isinstance(n.func,ast.Name) and n.func.id in ('_grades','advantages_from_rewards','task_group_is_informative','_publish','_partial_cost')) or
                 (isinstance(n.func,ast.Attribute) and n.func.attr in ('rollout','update','restore','charge','charge_imputed')))]
    assert calls(of['run'])==calls(nf['run'])


def test_resource_hold_cannot_start_group_after_deadline(tmp_path):
    B,parent,sha,rows=fixture(tmp_path);now=[0.]
    def authorize(context):
        if context['phase']=='group':now[0]=11.
    result=driver.run(output=tmp_path/'held',schedule=rows,bindings={'test':'held'},backend_factory=B,authorize=authorize,
        max_new_groups=3,parent_checkpoint=parent,parent_receipt_sha256=sha,max_optimizer_updates=8,
        maximum_seconds=10,clock=lambda:now[0])
    assert result['status']=='WALL_BUDGET_REACHED' and result['completed_groups']==0
    assert not list((tmp_path/'held').glob('.pending-*'))


def test_resume_cannot_reset_experiment_wall_deadline_or_load_expired_backend(tmp_path):
    B,parent,sha,rows=fixture(tmp_path);wall=[1000.]
    invoke(B,parent,sha,rows,tmp_path/'absolute',1,maximum_seconds=10,wall_clock=lambda:wall[0])
    journal=(tmp_path/'absolute/operation_deadline_PRIVATE.json').read_bytes();n=len(B.instances)
    wall[0]=1011.
    result=invoke(B,parent,sha,rows,tmp_path/'absolute',2,True,maximum_seconds=10,wall_clock=lambda:wall[0])
    assert result['status']=='WALL_BUDGET_REACHED' and result['completed_groups']==1 and len(B.instances)==n
    assert (tmp_path/'absolute/operation_deadline_PRIVATE.json').read_bytes()==journal
    with pytest.raises(ValueError):invoke(B,parent,sha,rows,tmp_path/'absolute',2,True,maximum_seconds=20,wall_clock=lambda:wall[0])


def test_resume_flag_does_not_change_stage_binding_but_profile_change_does():
    a={'resume':False,'output':'same','profile':'fixed'}
    assert entry.stage_spec_binding(a)==entry.stage_spec_binding({**a,'resume':True})
    assert entry.stage_spec_binding(a)!=entry.stage_spec_binding({**a,'profile':'changed'})


def test_failure_cleanup_clears_actual_F32_view_traceback_before_release(tmp_path):
    import weakref,gc
    from types import SimpleNamespace as NS
    from mri_rl.paired_z045_f32_backend_v1 import F32LanguageView
    class Language:
        model=NS(embed_tokens=lambda ids:np.ones((*ids.shape,3),dtype=np.float32))
        def __call__(self,*args,**kwargs):raise RuntimeError('original view failure')
    class B:
        def __init__(self):
            self.model=Language();self.mx=NS(clear_cache=lambda:None,get_active_memory=lambda:0)
        def close(self):self.model=None
    backend=B();ref=weakref.ref(backend.model);resident=[backend]
    def run():F32LanguageView(backend.model,NS(float32=np.float32),lambda n:None,[])(np.array([[1]]))
    with pytest.raises(RuntimeError,match='original view failure'):entry.run_with_release(run,resident,tmp_path)
    gc.collect();assert ref() is None
    assert json.loads((tmp_path/'entry_STOP_PRIVATE.json').read_text())['error']=='original view failure'
    assert json.loads((tmp_path/'release_SAFE.json').read_text())['release_verified']


def test_future_narrow_approval_cannot_run_hardcoded32_8_prepared_budget():
    prepared={'budget':{'maximum_attempted_groups':32,'maximum_optimizer_updates':8,'maximum_seconds':3000,'text_visual_token_cap':70784}}
    assert entry.validate_prepared_budget({'budget':dict(prepared['budget'])},prepared)
    narrowed={'budget':{**prepared['budget'],'maximum_attempted_groups':1,'maximum_optimizer_updates':1}}
    with pytest.raises(ValueError,match='budget'):entry.validate_prepared_budget(narrowed,prepared)


def test_cache_metadata_binds_exact_expected_cache_without_opening_payload(tmp_path,monkeypatch):
    import hashlib
    from mri_rl.likelihood_cache import verify_cache_identity
    cache=tmp_path/'cache';model=tmp_path/'model';cache.mkdir();model.mkdir()
    source={'model_type':'fixture'};cached={**source,'quantization':{'bits':4},'quantization_config':{'bits':4}}
    (cache/'config.json').write_text(json.dumps(cached));(cache/'config_source.json').write_text(json.dumps(source));(model/'config.json').write_text(json.dumps(source))
    shard=cache/'model-1.safetensors';shard.write_bytes(b'SYNTHETIC PAYLOAD NOT A MODEL')
    (cache/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{f't{i}':shard.name for i in range(1438)}}))
    cert={'tensors_checked':1438,'payload_bytes_equal_to_verified_units':True,'shards':{shard.name:{'bytes':shard.stat().st_size,'sha256':entry.digest(shard)}},'config_cached_sha256':entry.digest(cache/'config.json'),'config_source_sha256':entry.digest(model/'config.json'),'non_weight_files_identical':{},'plan_sha256':'fixture','quantized_modules':1}
    certpath=tmp_path/'certificate.json';certpath.write_text(json.dumps(cert));certsha=entry.digest(certpath)
    expected=verify_cache_identity(cache,certpath,certsha,{str(model/'config.json'):entry.digest(model/'config.json')})
    original=Path.open
    def guarded(path,*args,**kwargs):
        assert path.suffix!='.safetensors','CPU cache metadata must not open weights'
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',guarded)
    actual=entry.cache_metadata({'cache_dir':str(cache),'cache_certificate':str(certpath),'cache_certificate_sha256':certsha,'model':str(model)})
    assert actual['expected_binding']==expected
    assert actual['payload_verification']=='DEFERRED_TO_AUTHORIZED_EXECUTE_FULL_SHA_AND_LOADED_PARAMETER_PARITY'


def test_frozen_image_map_joins_membership_and_Dallas_inventory_and_stops_mutation(tmp_path):
    image=tmp_path/'seq.png';image.write_bytes(b'original-sequence');dallas=tmp_path/'Dallas.png';dallas.write_bytes(b'original-Dallas')
    seqsha=entry.digest(image);dallassha=entry.digest(dallas)
    membership=tmp_path/'members.json';membership.write_text(json.dumps([{'record_id_PRIVATE':'seqsource','PNG_sha256':seqsha}]))
    inventory=tmp_path/'inventory.json';inventory.write_text(json.dumps({'image_sha256_by_record':{'dallassource':dallassha}}))
    receipt=tmp_path/'receipt.json';receipt.write_text(json.dumps({'private_image_inventory_sha256':entry.digest(inventory)}))
    spec={'data_membership':str(membership),'original_image_inventory':str(inventory),'original_pool_receipt':str(receipt),'file_sha256':{'data_membership':entry.digest(membership),'original_image_inventory':entry.digest(inventory),'original_pool_receipt':entry.digest(receipt)}}
    rows=[{'id':'s','source_record_id':'seqsource','ability':'mri_sequence','images':[str(image)],'rc_data_v1':{'image_sha256':seqsha}},
          {'id':'d','source_record_id':'dallassource','ability':'mri_visit_correspondence','images':[str(dallas)]}]
    frozen=entry.frozen_image_map(spec,rows);assert frozen=={'s':seqsha,'d':dallassha}
    assert entry.verify_images(rows,frozen)
    image.write_bytes(b'original-sequencE');called=[]
    with pytest.raises(ValueError,match='image'):
        entry.verify_images(rows,frozen);called.append('factory')
    assert called==[]
    # Updating only nested current SHA cannot override authenticated membership.
    rows[0]['rc_data_v1']['image_sha256']=entry.digest(image)
    with pytest.raises(ValueError):entry.frozen_image_map(spec,rows)
