"""CPU state-fault and qualify->train integration tests (fake numpy arrays; no model, MLX or MRI data)."""
import importlib.util
import json
from pathlib import Path
import shutil
import uuid
from types import SimpleNamespace as NS
import pytest
from mri_rl import shared_rollout_gspo_core_v1 as core
from mri_rl import shared_rollout_gspo_entry_v1 as e
from mri_rl import paired_z045_continuation_entry_v1 as inherited
from mri_rl import paired_z045_continuation_driver_v1 as old

spec=importlib.util.spec_from_file_location('state_base',Path(__file__).with_name('test_shared_rollout_gspo_v1.py'))
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
FAULT=(ValueError,RuntimeError,AssertionError)


@pytest.fixture(autouse=True)
def reset():
    f.Fake.instances=[];f.Fake.rollouts=0;f.Fake.log=None;f.Fake.accuracy=f.ACCURACY;f.Fake.fail_update_arm=None;f.Fake.identities=[]


def corrupt(backend,kind):
    if kind=='Adam':backend.state['weight']['m']=backend.state['weight']['m']+.01
    else:backend.rng.random()


@pytest.mark.parametrize('phase',['score_old','arm_gradient'])
@pytest.mark.parametrize('kind',['Adam','RNG'])
@pytest.mark.parametrize('mode',['qualify','train'])
def test_deterministic_scoring_or_backward_cannot_hide_state_mutation_by_restore(tmp_path,phase,kind,mode):
    class Corrupt(f.Fake):
        def score_old(self,bundle):
            result=super().score_old(bundle)
            if phase=='score_old':corrupt(self,kind)
            return result
        def arm_gradient(self,planned,old_logps):
            result=super().arm_gradient(planned,old_logps)
            if phase=='arm_gradient':corrupt(self,kind)
            return result
    with pytest.raises(FAULT,match=r'pre-restore audit'):f.launch(tmp_path,tmp_path/'run',mode=mode,cls=Corrupt)
    assert not list((tmp_path/'run').glob('arm_*_COMMITTED.json'))


def test_single_Adam_step_does_not_authorize_changing_parent_RNG(tmp_path):
    class BadUpdate(f.Fake):
        def apply_update(self,gradient):
            result=super().apply_update(gradient);self.rng.random();return result
    with pytest.raises(FAULT,match='RNG changed'):f.launch(tmp_path,tmp_path/'run',mode='train',cls=BadUpdate)
    assert not (tmp_path/'run/arm_A/boundary_0001').exists()


def test_updated_arm_must_actually_change_adam_and_a_noop_update_is_rejected(tmp_path):
    class Silent(f.Fake):
        def apply_update(self,gradient):
            self.events.append('update');self.arm_updates+=1;self.state['step']=self.state['step']+1;return {'optimizer_calls':1}   # counter moves, moments do not
    with pytest.raises(FAULT):f.launch(tmp_path,tmp_path/'run',mode='train',cls=Silent)


def test_rollout_may_consume_RNG_but_cannot_mutate_Adam_and_must_consume_RNG(tmp_path):
    class BadRollout(f.Fake):
        def generate_shared_rollout(self,rows,workdir):
            result=super().generate_shared_rollout(rows,workdir);corrupt(self,'Adam');return result
    with pytest.raises(ValueError,match='Rollout changed Adam'):f.launch(tmp_path,tmp_path/'run',mode='qualify',cls=BadRollout)
    assert f.Fake.rollouts==1 and not (tmp_path/'run/bundle_COMMITTED.json').exists()
    class NoRng(f.Fake):
        def generate_shared_rollout(self,rows,workdir):
            state=self.rng.getstate();result=super().generate_shared_rollout(rows,workdir);self.rng.setstate(state);return result
    with pytest.raises(ValueError,match='did not consume RNG'):f.launch(tmp_path,tmp_path/'run2',mode='qualify',cls=NoRng)


def test_saved_own_old_is_rederived_not_trusted_on_resume(tmp_path):
    def stop_before_arms(context):
        if context['phase']=='arm':raise PermissionError('stop after own-old commit')
    with pytest.raises(PermissionError):f.launch(tmp_path,tmp_path/'run',mode='qualify',authorize=stop_before_arms)
    path=tmp_path/'run/old_PRIVATE.json';record=json.loads(path.read_text());record['old']['F0'][0]+=.5;path.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='exceeds .002'):f.launch(tmp_path,tmp_path/'run',mode='qualify',resume=True)


# ------------------------------------------------------------------ qualify -> train through prepare, scopes and the real driver
def world(tmp_path,monkeypatch):
    parent,sha_,rows=f.parent_fixture(tmp_path)
    calibration={'mri_sequence':{'p50':.5,'iqr':.2}};cal=tmp_path/'calibration.json';cal.write_text(json.dumps(calibration))
    (tmp_path/'runtime.json').write_text(json.dumps({'parent_checkpoint':str(parent),'parent_receipt_sha256':sha_,'calibration':str(cal),'file_sha256':{'calibration':inherited.digest(cal)}}))
    (tmp_path/'schedule.json').write_text(json.dumps(rows));(tmp_path/'rows.json').write_text(json.dumps(rows))
    config={**old.PROFILE,'group_size':4,'max_tokens':256}
    monkeypatch.setattr(inherited,'parent_metadata',lambda path,sha:{'receipt_sha256':sha,'groups':129,'updates':60,'original_binding':{'configuration':config,
        'experiment':{'parent_experiment':{'arm':{'name':'U_calibrated','reward':'medgrpo_eq3','sampling':'uniform'},'seed':13,'calibration_sha256':inherited.scope_digest(calibration)}}}})
    monkeypatch.setattr(inherited,'source_closure',lambda s:{'/installed/runtime.py':f.sha('runtime')})
    monkeypatch.setattr(inherited,'cache_metadata',lambda s:{'expected_binding':{'cache':'fixture'}})
    return rows


def make_spec(tmp_path,rows,output,**extra):
    spec={'runtime_spec':str(tmp_path/'runtime.json'),'schedule':str(tmp_path/'schedule.json'),'output':str(output),'expected_image_sha256':{r['id']:inherited.digest(r['images'][0]) for r in rows},
          'file_sha256':{'runtime_spec':inherited.digest(tmp_path/'runtime.json'),'schedule':inherited.digest(tmp_path/'schedule.json')},**extra}
    return spec


def signed(tmp_path,prepared,mode,tag,qualification=None):
    scope=e.launch_scope(prepared,mode);binding=core.scope_digest(scope)
    files={'scope':scope,'human':{'authorized':True,'action':core.ACTIONS[mode],'scope_sha256':binding,'human_instruction_sha256':f.sha('h'+tag)},
           'review':{'verdict':'PASS','scope_sha256':binding,'reviewer_id':'r','reviewer_kind':'claude_code','source_sha256':scope['source_sha256'],'input_sha256':scope['input_sha256']}}
    if qualification:files.update({'qualification':qualification['proof'],'qualification_review':qualification['review']})
    paths={}
    for key,value in files.items():paths[key]=tmp_path/f'{tag}_{key}.json';paths[key].write_text(json.dumps(value))
    args=NS(spec=tmp_path/f'{tag}_spec.json',scope=paths['scope'],human=paths['human'],review=paths['review'],expected_scope_sha256=inherited.digest(paths['scope']),
            expected_human_sha256=inherited.digest(paths['human']),expected_review_sha256=inherited.digest(paths['review']))
    if qualification:
        args.qualification=paths['qualification'];args.qualification_review=paths['qualification_review']
        args.expected_qualification_sha256=inherited.digest(paths['qualification']);args.expected_qualification_review_sha256=inherited.digest(paths['qualification_review'])
    return scope,args,files


def test_qualify_then_train_cli_path_shares_immutable_input_hash_and_adopts_through_the_real_driver(tmp_path,monkeypatch):
    rows=world(tmp_path,monkeypatch);root=Path(e.__file__).resolve().parents[2];base_dir=root/'local_data/shared_rollout_gspo'/('pytest-'+uuid.uuid4().hex)
    out_q=base_dir/'qualify';out_t=base_dir/'train'
    try:
        prep_q=e.prepare(make_spec(tmp_path,rows,out_q));prep_t=e.prepare(make_spec(tmp_path,rows,out_t,adopt_qualification_run=str(out_q)))
        assert prep_q['input_sha256']==prep_t['input_sha256'] and prep_q['source_sha256']==prep_t['source_sha256'] and prep_q['parent']==prep_t['parent']
        assert prep_q['output']!=prep_t['output'] and prep_t['invocation']['adopt_qualification_run']==str(out_q) and prep_q['invocation']['adopt_qualification_run'] is None
        assert core.scope_digest(e.launch_scope(prep_q,'qualify'))!=core.scope_digest(e.launch_scope(prep_t,'train'))
        with pytest.raises(ValueError,match='never adopts'):e.launch_scope(prep_t,'qualify')
        # 1) qualification through the REAL driver, bound to the prepared source/input (fake numpy model only)
        scope_q,args_q,_=signed(tmp_path,prep_q,'qualify','q')
        args_q.spec.write_text(json.dumps(make_spec(tmp_path,rows,out_q)))
        assert e._authorized('qualify',args_q)[1]['input_sha256']==prep_q['input_sha256']
        evidence=tmp_path/'q_evidence';evidence.mkdir()
        resident=[NS(close=lambda:None,mx=NS(clear_cache=lambda:None,get_active_memory=lambda:0),model=None)]
        result=inherited.run_with_release(lambda:f.launch(tmp_path,out_q,mode='qualify',bindings_input=prep_q['input_sha256'],bindings_source=prep_q['source_sha256']),resident,evidence)
        core.publish_once(evidence/'complete_PRIVATE.json',result)
        proof=e.qualification_proof(out_q,evidence,prep_q);assert proof['status']=='ACTUAL_SHARED_ROLLOUT_GSPO_READONLY_QUALIFIED' and f.Fake.rollouts==1
        review={'verdict':'PASS','actual_proof_sha256':core.identity(proof),'reviewer_id':'r','reviewer_kind':'isolated_codex'}
        # 2) FRESH train authorization over a different output + adoption target, same immutable inputs
        scope_t,args_t,_=signed(tmp_path,prep_t,'train','t',{'proof':proof,'review':review})
        args_t.spec.write_text(json.dumps(make_spec(tmp_path,rows,out_t,adopt_qualification_run=str(out_q))))
        assert scope_t['input_sha256']==scope_q['input_sha256'] and e._authorized('train',args_t)[1]['output']==str(out_t)
        with pytest.raises(ValueError,match='already consumed'):e._authorized('train',args_t)
        # 3) the real driver adopts: zero new generation / own-old scoring, same bundle + old bytes, inherited costs separate
        trained=f.launch(tmp_path,out_t,mode='train',bindings_input=prep_t['input_sha256'],bindings_source=prep_t['source_sha256'],adopt={'run':out_q,'proof':proof})
        assert f.Fake.rollouts==1 and trained['status']=='SCHEDULE_COMPLETE' and trained['optimizer_updates']==2
        for name in ('bundle_PRIVATE.json','old_PRIVATE.json'):assert (out_q/name).read_bytes()==(out_t/name).read_bytes()
        new,inherited_counters=trained['counters'],trained['inherited_counters']
        assert new['generated_response_tokens']==new['detached_old_forwards']==new['sampler_draws']==0 and inherited_counters['generated_response_tokens']==24 and inherited_counters['detached_old_forwards']==8
        assert json.loads((out_t/'adoption_PRIVATE.json').read_text())['generation_in_this_run'] is False
        # 4) changing a SCIENTIFIC input changes the immutable hash and breaks both the gate and the adoption
        cal2=tmp_path/'calibration2.json';cal2.write_text(json.dumps({'mri_sequence':{'p50':.4,'iqr':.2}}))
        variants=[]
        images=make_spec(tmp_path,rows,out_t,adopt_qualification_run=str(out_q));images['expected_image_sha256']={**images['expected_image_sha256'],rows[0]['id']:f.sha('other image')};variants.append(images)
        extra=make_spec(tmp_path,rows,out_t,adopt_qualification_run=str(out_q));extra['prompt_lengths']={'F':213,'T':213};variants.append(extra)
        for variant in variants:
            changed=e.prepare(variant);assert changed['input_sha256']!=prep_t['input_sha256']
            scope_c=e.launch_scope(changed,'train');bind=core.scope_digest(scope_c)
            human={'authorized':True,'action':core.ACTIONS['train'],'scope_sha256':bind,'human_instruction_sha256':f.sha('x')}
            rev={'verdict':'PASS','scope_sha256':bind,'reviewer_id':'r','reviewer_kind':'claude_code','source_sha256':scope_c['source_sha256'],'input_sha256':scope_c['input_sha256']}
            with pytest.raises(ValueError,match='qualification'):core.authorize_launch('train',scope_c,human,rev,qualification={'proof':proof,'review':review})
            with pytest.raises(ValueError,match='source/input/parent'):f.launch(tmp_path,base_dir/('c'+uuid.uuid4().hex[:6]),mode='train',bindings_input=changed['input_sha256'],bindings_source=prep_t['source_sha256'],adopt={'run':out_q,'proof':proof})
        # a different source closure is equally refused
        with pytest.raises(ValueError,match='source/input/parent'):f.launch(tmp_path,base_dir/'srcchange',mode='train',bindings_input=prep_t['input_sha256'],bindings_source={'/x.py':f.sha('x')},adopt={'run':out_q,'proof':proof})
    finally:
        shutil.rmtree(base_dir,ignore_errors=True);shutil.rmtree(base_dir.parent/'pytest-state-cleanup',ignore_errors=True)


def test_adoption_rejects_unverified_completion_claims(tmp_path,monkeypatch):
    """A proof/prepared boolean cannot substitute for the qualification run's own completion and coverage files."""
    run,evidence=f.qualify(tmp_path,{'F':[1,1,1,1],'T':[0,0,0,0]})                       # arm A never had an admitted group
    forged={'status':'ACTUAL_SHARED_ROLLOUT_GSPO_READONLY_QUALIFIED','blockers':[],'optimizer_updates':0,'source_sha256':{'/s.py':f.sha('s')},'input_sha256':f.sha('i'),
            'bundle_sha256':json.loads((run/'bundle_COMMITTED.json').read_text())['bundle_sha256'],'old_sha256':core.file_sha256(run/'old_PRIVATE.json'),
            'release_witness_sha256':core.file_sha256(evidence/'release_SAFE.json'),'run_identity_sha256':core.file_sha256(run/'run_identity_PRIVATE.json')}
    assert core.qualification_blockers(run)==['ARM_A_NO_EFFECTIVE_FINITE_NONZERO_BACKWARD_PATH']
    before=f.Fake.rollouts
    with pytest.raises(ValueError,match='effective finite nonzero'):f.launch(tmp_path,tmp_path/'train',adopt={'run':run,'proof':forged})
    assert f.Fake.rollouts==before and not (tmp_path/'train/bundle_COMMITTED.json').exists()
    (run/'complete_SAFE.json').unlink()
    assert core.qualification_blockers(run)==['RUN_COMPLETION_EVIDENCE_MISSING']


def test_parent_only_alias_scope_when_neither_arm_updated(tmp_path):
    from mri_rl import shared_rollout_gspo_eval_v1 as ev
    from mri_rl.uniform_balance_continuation_gate_v1 import parent_anchor
    f.Fake.accuracy={'F':[1,1,1,1],'T':[1,1,1,1]};f.launch(tmp_path,tmp_path/'run')
    parent=tmp_path/'adam60/boundary_0008';anchor=parent_anchor(parent,old.digest(parent/'receipt.json'))
    arms={a:ev.arm_checkpoint(tmp_path/'run',a,anchor) for a in 'AB'}
    distinct=1+sum(not v['alias_of_parent'] for v in arms.values())
    assert distinct==1 and ev.eval_budget(distinct)['checkpoints']==1
