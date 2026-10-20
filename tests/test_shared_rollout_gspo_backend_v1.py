"""CPU dual-number/mock-MLX fixtures for the native backend code path and the F32 evaluation entry. Not native evidence."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import numpy as np
import pytest
from mri_rl import shared_rollout_gspo_core_v1 as core
from mri_rl import shared_rollout_gspo_backend_v1 as backend_module
from mri_rl import paired_prompt_groups_v1 as groups


def load(name):
    path=Path(__file__).with_name(name);spec=importlib.util.spec_from_file_location(name[:-3],path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


base=load('test_shared_rollout_gspo_v1.py')


class D:
    """Forward-mode dual number: value plus Jacobian w.r.t. the trainable parameter vector."""
    def __init__(self,value,jac):self.v=np.asarray(value,dtype=float);self.j=np.asarray(jac,dtype=float)
    def __sub__(self,other):
        if not isinstance(other,D):other=D(other,np.zeros_like(self.j))
        return D(self.v-other.v,self.j-other.j)
    def __add__(self,other):return D(self.v+other.v,self.j+other.j)
    def __mul__(self,value):return D(self.v*value,self.j*value)
    __rmul__=__mul__
    def item(self):return float(self.v)
    def tolist(self):return self.v.tolist()


def unary(x,f,df):return D(f(x.v),x.j*np.asarray(df(x.v))[...,None])
def maximum(left,right):
    choose=left.v>=right.v;return D(np.maximum(left.v,right.v),np.where(choose[...,None],left.j,right.j))


MX=NS(stop_gradient=lambda x:D(x.v,np.zeros_like(x.j)),mean=lambda x:D(x.v.mean(),x.j.mean(axis=0)),exp=lambda x:unary(x,np.exp,np.exp),
      minimum=lambda x,b:unary(x,lambda v:np.minimum(v,b),lambda v:(v<b).astype(float)),maximum=maximum,
      clip=lambda x,l,h:unary(x,lambda v:np.clip(v,l,h),lambda v:((v>l)&(v<h)).astype(float)),
      eval=lambda *a:None,array=lambda v,dtype=None:np.asarray(v,dtype=float),abs=lambda x:np.abs(x.v if isinstance(x,D) else x),max=np.max,
      all=np.all,isfinite=lambda x:np.isfinite(x.v if isinstance(x,D) else x),sum=np.sum,float32=np.float32,clear_cache=lambda:None)
NN=NS(value_and_grad=lambda model,fn:(lambda m,tokens,previous,advantage:(lambda loss:(loss,{'w':loss.j}))(fn(m,tokens,previous,advantage))))
tree_map=lambda f,*trees:{k:f(*(t[k] for t in trees)) for k in trees[0]}


def jacobian(tokens,modality):
    scale=1. if modality=='F' else -.5
    return np.array([[scale*token/10,1.-.1*i] for i,token in enumerate(tokens)])


def fake_logps(calls):
    def logps(backend,tokens,inputs):
        calls.append((inputs['modality'],tuple(tokens)));return D(-1.-.1*np.arange(len(tokens)),jacobian(tokens,inputs['modality']))
    return logps


def backend(monkeypatch,bundle,calls):
    monkeypatch.setattr(backend_module,'f32_response_logps_observed',fake_logps(calls))
    b=backend_module.SharedRolloutBackend.__new__(backend_module.SharedRolloutBackend)
    limits=core.counter_limits(core.derive_budget(base.LENGTHS,'train'))
    b.mx=MX;b.nn=NN;b.tree_map=tree_map;b.tree_flatten=lambda tree:list(tree.items());b.trainable_signature=lambda:{'w':{}}
    b.model=NS(eval=lambda:None);b.before_unit=lambda name:None;b.counters=core.Counters(limits);b.rows={'F':{},'T':{}}
    b._verify_image=lambda modality:None;b._active=None;b._active_modality=None;b._recorded=None;b._arm_updates=0;b.prepared=[]
    def prepare(modality):
        b.prepared.append(modality);return {'modality':modality,'prompt_length':213,'inputs':{'modality':modality},'prompt':'generic question','prompt_ids':[1],'fingerprints_sha256':base.sha('fp'+modality)}
    b._prepare_context=prepare
    return b


def own_old(bundle):
    return {r['response_id']:(-1.-.1*np.arange(len(r['token_ids']))).tolist() for r in bundle.raw['responses']}


def test_arm_gradient_runs_unchanged_loss_ast_per_response_context_with_one_eighth_weight(monkeypatch):
    bundle=base.bundle();calls=[];b=backend(monkeypatch,bundle,calls);old=own_old(bundle)
    for arm in ('A','B'):
        planned=core.arm_plan(bundle,arm);gradient,evidence=b.arm_gradient(planned,old)
        expected=np.zeros(2)
        for item in planned.plan.items:
            if item.weight==0:continue
            tokens=item.response.token_ids;previous=np.asarray(old[item.response.response_id])
            coefficients=groups.response_surrogate(previous,previous,item.advantage)['d_loss_d_token_logps']
            expected+=item.weight*(np.asarray(coefficients)@jacobian(tokens,item.response.context.modality))
        np.testing.assert_allclose(gradient['w'],expected,rtol=1e-5,atol=1e-6)         # native tree maps scale in F32
        assert evidence['backward_calls']==8 and evidence['gradient_l2']==pytest.approx(float(np.linalg.norm(expected)),rel=1e-4,abs=1e-6)
        assert [r['weight'] for r in evidence['responses']]==[.125]*8 and evidence['native_loss_AST_sha256']
    # each response was scored only under ITS OWN context, never under the other image
    assert all(modality==(m:=next(r.context.modality for r in bundle.responses if r.token_ids==tokens and r.context.modality==modality)) for modality,tokens in calls)
    assert {(m,t) for m,t in calls}=={(r.context.modality,r.token_ids) for r in bundle.responses}
    used=b.counters.used
    assert used['backward_calls']==16 and used['gradient_forwards']==16 and used['scorer_vision_embedding_calls']==16 and used['teacherforced_input_positions']==16*(213+2)


def test_old_scoring_uses_own_context_and_charges_exactly_eight_forwards(monkeypatch):
    bundle=base.bundle();calls=[];b=backend(monkeypatch,bundle,calls)
    result=b.score_old(bundle)
    assert set(result)==set(core.RESPONSE_IDS) and b.counters.used['detached_old_forwards']==8 and b.counters.used['teacherforced_input_positions']==8*(213+2)
    assert calls==[(r.context.modality,r.token_ids) for r in bundle.responses]
    assert b.prepared==['F','T']                     # one context resident at a time; no repeated preparation inside a context
    with pytest.raises(ValueError,match='exceeds .002'):core.parity_report(bundle,result)       # the synthetic scorer deliberately differs from the saved rollout logps


def test_contexts_are_prepared_and_released_separately_and_reproduce_bundle_fingerprints(monkeypatch):
    bundle=base.bundle();b=backend(monkeypatch,bundle,[])
    b.activate('F');b.activate('F');b.activate('T');b.activate('F')
    assert b.prepared==['F','T','F'] and b._active_modality=='F'            # switch releases + re-prepares; never reuse another image's inputs
    b.bind_bundle(bundle)
    b.prepared.clear();b._active=None;b._active_modality=None
    bad=backend_module.SharedRolloutBackend.__new__(backend_module.SharedRolloutBackend)
    b._prepare_context=lambda m:{'modality':m,'prompt_length':213,'inputs':{},'prompt':'generic','prompt_ids':[1],'fingerprints_sha256':base.sha('other')}
    with pytest.raises(ValueError,match='differ from the immutable bundle'):b.activate('F')
    with pytest.raises(ValueError):b.activate('X')


def test_inactive_arm_needs_no_backward(monkeypatch):
    bundle=base.bundle({'F':[1,1,1,1],'T':[0,0,0,0]});calls=[];b=backend(monkeypatch,bundle,calls)
    gradient,evidence=b.arm_gradient(core.arm_plan(bundle,'A'),own_old(bundle))
    assert gradient is None and evidence['backward_calls']==0 and evidence['gradient_l2']==0. and calls==[] and b.counters.used['backward_calls']==0


def test_only_one_optimizer_call_per_arm_restore_resets_and_zero_gradient_refused(monkeypatch):
    from mri_rl.class_margin_aux_v2 import ClassMarginAuxBackend
    monkeypatch.setattr(ClassMarginAuxBackend,'transaction_restore',lambda self,snapshot:None)
    bundle=base.bundle();b=backend(monkeypatch,bundle,[]);updates=[]
    b.optimizer=NS(update=lambda model,gradient:updates.append(gradient),state={});b.model=NS(trainable_parameters=lambda:{})
    with pytest.raises(ValueError,match='Zero gradient'):b.apply_update({'w':np.zeros(2)})
    result=b.apply_update({'w':np.array([1.,-2.])});assert result['optimizer_calls']==1 and len(updates)==1
    with pytest.raises(ValueError,match='second optimizer update'):b.apply_update({'w':np.array([1.,0.])})
    b.transaction_restore(None);b.apply_update({'w':np.array([1.,0.])});assert len(updates)==2


def test_rollout_count_is_two_or_four_and_historical_group_path_is_unreachable(monkeypatch):
    b=backend(monkeypatch,base.bundle(),[])
    with pytest.raises(ValueError,match='2 or 4'):b._generate_context({'modality':'F'},3,Path('.'))
    with pytest.raises(ValueError,match='DISABLED'):b.update(None,None,None)
    assert backend_module.SharedRolloutBackend.transaction_snapshot is backend_module.ClassMarginAuxBackend.transaction_snapshot
    assert backend_module.SharedRolloutBackend.transaction_validate is backend_module.ClassMarginAuxBackend.transaction_validate


def test_backend_source_reuses_pinned_helpers_without_editing_historical_trainers():
    source=Path(backend_module.__file__).read_text()
    for name in ('f32_response_logps_observed','capture_stream','reset_effective_eos','F32GenerationModelView','frozen_loss_node'):assert name in source
    assert 'aux_loss' not in source and 'auxiliary_gradient' not in source and 'objective.ALPHA' not in source       # archived margin auxiliary is never invoked
    assert not any(n=='mlx' or n.startswith('mlx.') for n in sys.modules)


# ------------------------------------------------------------------ matched F32 evaluation entry
def test_eval_budget_scales_the_matched_budget_and_reuses_its_functions():
    from mri_rl import shared_rollout_gspo_eval_v1 as ev
    from mri_rl import paired_z045_matched_f32_eval_v1 as matched
    three=ev.eval_budget(3);two=ev.eval_budget(2)
    assert (three['greedy_generations'],three['candidate_score_API_calls'],three['maximum_generated_tokens'],three['teacherforced_input_positions'])==(120,360,30720,153000)
    assert (three['generation_language_model_calls'],three['vision_embedding_calls'],three['checkpoints'])==(30960,480,3)
    assert all(two[k]==matched.budget()[k] for k in ('greedy_generations','candidate_score_API_calls','maximum_generated_tokens','teacherforced_input_positions'))
    counter=ev.EvalCounter(3);counter.charge('greedy_generations',120)
    with pytest.raises(ValueError,match='budget exceeded'):counter.charge('greedy_generations')
    source=Path(ev.__file__).read_text()
    for forbidden in ('generate_record','greedy_generate','score_record','NativeGSPOBackend('.replace('F32','')):
        assert forbidden not in source.replace('F32NativeGSPOBackend','')
    assert 'F32NativeGSPOBackend' in source and 'matched.run_checkpoint' in source and ev.matched.run_checkpoint is matched.run_checkpoint
    from mri_rl.paired_z045_f32_backend_v1 import F32NativeGSPOBackend
    for method in ('generate_record','greedy_generate','score_record'):
        with pytest.raises(ValueError,match='F32_EVALUATION_NOT_QUALIFIED'):getattr(F32NativeGSPOBackend,method)(None)


def test_eval_binds_committed_arm_terminals_and_aliases_inactive_arm_to_parent(tmp_path):
    from mri_rl import shared_rollout_gspo_eval_v1 as ev
    from mri_rl.uniform_balance_continuation_gate_v1 import parent_anchor
    base.Fake.instances=[];base.Fake.accuracy={'F':[1,1,1,1],'T':[0,0,0,0]}
    base.launch(tmp_path,tmp_path/'run')
    parent=tmp_path/'adam60'/'boundary_0008';anchor=parent_anchor(parent,base.old.digest(parent/'receipt.json'))
    a=ev.arm_checkpoint(tmp_path/'run','A',anchor);b=ev.arm_checkpoint(tmp_path/'run','B',anchor)
    assert a['alias_of_parent'] is True and a['updates']==60 and b['alias_of_parent'] is False and b['updates']==61 and b['groups']==130
    assert a['adapter_sha256']==anchor['adapter_sha256']!=b['adapter_sha256']
    (tmp_path/'run'/'arm_B_COMMITTED.json').write_text(json.dumps({**json.loads((tmp_path/'run'/'arm_B_COMMITTED.json').read_text()),'updated':False}))
    with pytest.raises(ValueError):ev.arm_checkpoint(tmp_path/'run','B',anchor)
    with pytest.raises(ValueError):ev.arm_checkpoint(tmp_path/'run','B',{**anchor,'receipt_sha256':base.sha('foreign')})


def test_parent_only_budget_and_aliases_when_both_arms_made_no_update(tmp_path):
    from mri_rl import shared_rollout_gspo_eval_v1 as ev
    from mri_rl.uniform_balance_continuation_gate_v1 import parent_anchor
    assert ev.eval_budget(1)['greedy_generations']==40 and ev.eval_budget(1)['checkpoints']==1 and ev.eval_budget(1)['maximum_seconds']==1200
    with pytest.raises(ValueError):ev.eval_budget(0)
    base.Fake.instances=[];base.Fake.accuracy={'F':[1,1,1,1],'T':[1,1,1,1]}
    base.launch(tmp_path,tmp_path/'run')
    parent=tmp_path/'adam60'/'boundary_0008';anchor=parent_anchor(parent,base.old.digest(parent/'receipt.json'))
    arms={a:ev.arm_checkpoint(tmp_path/'run',a,anchor) for a in 'AB'}
    assert all(v['alias_of_parent'] for v in arms.values()) and 1+sum(not v['alias_of_parent'] for v in arms.values())==1


def fake_matched_prepare(tmp_path,run):
    from mri_rl.uniform_balance_continuation_gate_v1 import parent_anchor
    parent=tmp_path/'adam60'/'boundary_0008';anchor={**parent_anchor(parent,base.old.digest(parent/'receipt.json')),'path':str(parent)}
    return {'parent':anchor,'source_sha256':{'/x/matched.py':base.sha('m')},'numerical_contract_sha256':base.sha('numeric'),'output':str(tmp_path/'eval_out'),'records_PRIVATE':[],'status':'x'}


def test_evaluator_prepare_cli_binds_source_closure_and_is_idempotent_without_native(tmp_path,monkeypatch,capsys):
    from mri_rl import shared_rollout_gspo_eval_v1 as ev
    base.Fake.instances=[];base.Fake.accuracy={'F':[1,1,1,1],'T':[0,0,0,0]};base.launch(tmp_path,tmp_path/'run')
    monkeypatch.setattr(ev.matched,'prepare',lambda spec:fake_matched_prepare(tmp_path,tmp_path/'run'))
    spec={'shared_run':str(tmp_path/'run')};spec_path=tmp_path/'spec.json';spec_path.write_text(json.dumps(spec))
    before=set(sys.modules)
    result=ev.main(['--spec',str(spec_path),'--scope-output',str(tmp_path/'scope.json')])
    scope=json.loads((tmp_path/'scope.json').read_text())
    names={Path(p).name for p in scope['source_sha256']}
    assert {'paired_z045_evaluation_v1.py','answer_likelihood.py','vero_mri_reward.py','paired_z045_f32_backend_v1.py','shared_rollout_gspo_eval_v1.py','run_shared_rollout_gspo_eval_v1.py'}<=names
    assert scope['status']==core.STATUS and scope['evaluated_checkpoints']==['parent','B'] and scope['arms']['A']['alias_of_parent'] is True and scope['budget']['checkpoints']==2
    assert result['model_loads']==0 and not any(n=='mlx' or n.startswith('mlx.') for n in set(sys.modules)-before)
    assert ev.main(['--spec',str(spec_path),'--scope-output',str(tmp_path/'scope.json')])['scope_sha256']==result['scope_sha256']
    monkeypatch.setattr(ev.matched,'prepare',lambda spec:{**fake_matched_prepare(tmp_path,None),'output':'/changed'})
    with pytest.raises(ValueError,match='scope differs'):ev.main(['--spec',str(spec_path),'--scope-output',str(tmp_path/'scope.json')])
    monkeypatch.setattr(ev.matched,'prepare',lambda spec:fake_matched_prepare(tmp_path,tmp_path/'run'))
    with pytest.raises(ValueError,match='NOT_AUTHORIZED'):ev.main(['--spec',str(spec_path),'--scope-output',str(tmp_path/'scope.json'),'--execute'])


def eval_authorization(tmp_path,scope,*,status='ACTUAL_SHARED_ROLLOUT_GSPO_READONLY_QUALIFIED',blockers=(),action=None,sources=None):
    from types import SimpleNamespace as NS
    from mri_rl import shared_rollout_gspo_eval_v1 as ev
    from mri_rl import paired_z045_continuation_entry_v1 as inherited
    binding=inherited.scope_digest(scope)
    proof={'status':status,'blockers':list(blockers),'optimizer_updates':0,'source_sha256':sources or scope['source_sha256']}
    files={'proof':proof,'proof_review':{'verdict':'PASS','actual_proof_sha256':core.identity(proof),'reviewer_id':'r','reviewer_kind':'claude_code'},
           'review':{'verdict':'PASS','scope_sha256':binding,'reviewer_id':'r','reviewer_kind':'isolated_codex'},
           'human':{'authorized':True,'action':action or ev.ACTION,'scope_sha256':binding,'human_instruction_sha256':base.sha('h')}}
    paths={}
    for key,value in files.items():
        paths[key]=tmp_path/(key+'.json');paths[key].write_text(json.dumps(value))
    numeric={'actual_proof_path':str(paths['proof']),'actual_proof_sha256':inherited.digest(paths['proof']),'actual_review_path':str(paths['proof_review']),'actual_review_sha256':inherited.digest(paths['proof_review'])}
    paths['numeric']=tmp_path/'numeric.json';paths['numeric'].write_text(json.dumps(numeric))
    return NS(review=paths['review'],expected_review_sha256=inherited.digest(paths['review']),human_authorization=paths['human'],expected_human_sha256=inherited.digest(paths['human']),
              numeric_contract=paths['numeric'],expected_numeric_sha256=inherited.digest(paths['numeric']))


def test_evaluator_admission_requires_the_new_shared_qualification_not_the_old_status(tmp_path):
    from mri_rl import shared_rollout_gspo_eval_v1 as ev
    from mri_rl import shared_rollout_gspo_entry_v1 as shared_entry
    root=Path(ev.core.__file__).resolve().parents[2]
    sources={str(root/n):base.sha(n) for n in shared_entry.OWN_SOURCES}
    def build(**kw):
        scope={'status':core.STATUS,'source_sha256':sources,'output':str(tmp_path/'o')}
        probe=eval_authorization(tmp_path,scope,**kw);scope['numerical_contract_sha256']=probe.expected_numeric_sha256     # contract SHA = numeric file digest
        return eval_authorization(tmp_path,scope,**kw),scope
    good,scope=build();assert ev.authorize_eval(good,scope)['binding']
    with pytest.raises(ValueError,match='status'):ev.authorize_eval(good,{**scope,'status':'ACTUAL_F32_ONPOLICY_GSPO_READONLY_QUALIFIED'})
    for kw in ({'status':'ACTUAL_F32_ONPOLICY_GSPO_READONLY_QUALIFIED'},{'status':'PARTIAL_SHARED_ROLLOUT_GSPO_QUALIFICATION'},{'blockers':['ARM_A_NO_EFFECTIVE_FINITE_NONZERO_BACKWARD_PATH']},
               {'action':'finite_train'},{'sources':{**sources,next(iter(sources)):base.sha('edited')}}):
        bad,bad_scope=build(**kw)
        with pytest.raises(ValueError,match='prerequisite|authorization|review'):ev.authorize_eval(bad,bad_scope)


def test_eval_execution_refuses_before_native_import_without_review_human_and_numeric_proof():
    from mri_rl import shared_rollout_gspo_eval_v1 as ev
    before=set(sys.modules)
    with pytest.raises(ValueError,match='NOT_AUTHORIZED'):ev.execute(NS(),{},{})
    assert not any(n=='mlx' or n.startswith('mlx.') for n in set(sys.modules)-before)


def test_launch_scope_shape_is_exactly_what_the_admission_contract_accepts_and_is_never_self_authorizing():
    from mri_rl import shared_rollout_gspo_entry_v1 as e
    prepared={'budgets':{m:core.derive_budget(base.LENGTHS,m) for m in core.MODES},'source_sha256':{'/s.py':base.sha('s')},'input_sha256':base.sha('i'),'output':'/x'}
    for mode in core.MODES:
        scope=e.launch_scope(prepared,mode);assert scope['scope_kind']==core.SCOPE_KINDS[mode] and not any(k in scope for k in ('approved','authorized','training_authorized'))
        binding=core.scope_digest(scope)
        human={'authorized':True,'action':core.ACTIONS[mode],'scope_sha256':binding,'human_instruction_sha256':base.sha('h'+mode)}
        review={'verdict':'PASS','scope_sha256':binding,'reviewer_id':'r','reviewer_kind':'claude_code','source_sha256':scope['source_sha256'],'input_sha256':scope['input_sha256']}
        proof={'status':'ACTUAL_SHARED_ROLLOUT_GSPO_READONLY_QUALIFIED','optimizer_updates':0,'blockers':[],'source_sha256':scope['source_sha256'],'input_sha256':scope['input_sha256'],
               **{k:base.sha(k) for k in ('bundle_sha256','old_sha256','release_witness_sha256','run_identity_sha256')}}
        qualification={'proof':proof,'review':{'verdict':'PASS','actual_proof_sha256':core.identity(proof),'reviewer_id':'r','reviewer_kind':'claude_code'}} if mode=='train' else None
        assert core.authorize_launch(mode,scope,human,review,qualification=qualification,source_sha256=prepared['source_sha256'])
        with pytest.raises(ValueError):core.authorize_launch(mode,scope,human,review,qualification=qualification,source_sha256={'/s.py':base.sha('edited')})
    with pytest.raises(ValueError):e.launch_scope(prepared,'execute')


def test_entry_prepare_to_authorized_workflow_with_lightweight_fixtures(tmp_path,monkeypatch):
    import shutil
    import uuid
    from types import SimpleNamespace
    from mri_rl import shared_rollout_gspo_entry_v1 as e
    from mri_rl import paired_z045_continuation_entry_v1 as inherited
    from mri_rl import paired_z045_continuation_driver_v1 as old
    parent,sha_,rows=base.parent_fixture(tmp_path)
    calibration={'mri_sequence':{'p50':.5,'iqr':.2}};cal=tmp_path/'calibration.json';cal.write_text(json.dumps(calibration))
    runtime={'parent_checkpoint':str(parent),'parent_receipt_sha256':sha_,'calibration':str(cal),'file_sha256':{'calibration':inherited.digest(cal)}}
    (tmp_path/'runtime.json').write_text(json.dumps(runtime));(tmp_path/'schedule.json').write_text(json.dumps(rows))
    config={**old.PROFILE,'group_size':4,'max_tokens':256}
    monkeypatch.setattr(inherited,'parent_metadata',lambda path,sha:{'receipt_sha256':sha,'groups':129,'updates':60,'original_binding':{'configuration':config,
        'experiment':{'parent_experiment':{'arm':{'name':'U_calibrated','reward':'medgrpo_eq3','sampling':'uniform'},'seed':13,'calibration_sha256':inherited.scope_digest(calibration)}}}})
    monkeypatch.setattr(inherited,'source_closure',lambda spec:{'/installed/runtime.py':base.sha('runtime')})
    monkeypatch.setattr(inherited,'cache_metadata',lambda spec:{'expected_binding':{'cache':'fixture'}})
    root=Path(e.__file__).resolve().parents[2];output=root/'local_data/shared_rollout_gspo'/('pytest-'+uuid.uuid4().hex)
    images={r['id']:inherited.digest(r['images'][0]) for r in rows}
    spec={'runtime_spec':str(tmp_path/'runtime.json'),'schedule':str(tmp_path/'schedule.json'),'expected_image_sha256':images,'output':str(output),
          'file_sha256':{'runtime_spec':inherited.digest(tmp_path/'runtime.json'),'schedule':inherited.digest(tmp_path/'schedule.json')}}
    try:
        prepared=e.prepare(spec,verify_images=True)
        assert prepared['training_authorized'] is False and prepared['model_loads_this_preparation']==0 and set(prepared['budgets'])=={'qualify','train'}
        assert set(str(root/n) for n in e.OWN_SOURCES)<=set(prepared['source_sha256']) and prepared['reward_policy_sha256']==inherited.scope_digest(calibration)
        scope=e.launch_scope(prepared,'qualify');binding=core.scope_digest(scope)
        files={'scope':scope,'human':{'authorized':True,'action':core.ACTIONS['qualify'],'scope_sha256':binding,'human_instruction_sha256':base.sha('h')},
               'review':{'verdict':'PASS','scope_sha256':binding,'reviewer_id':'r','reviewer_kind':'claude_code','source_sha256':scope['source_sha256'],'input_sha256':scope['input_sha256']}}
        paths={}
        for key,value in files.items():paths[key]=tmp_path/(key+'.json');paths[key].write_text(json.dumps(value))
        args=SimpleNamespace(spec=tmp_path/'spec.json',scope=paths['scope'],human=paths['human'],review=paths['review'],expected_scope_sha256=inherited.digest(paths['scope']),
            expected_human_sha256=inherited.digest(paths['human']),expected_review_sha256=inherited.digest(paths['review']))
        args.spec.write_text(json.dumps(spec))
        assert e._authorized('qualify',args)[1]['input_sha256']==prepared['input_sha256']
        with pytest.raises(ValueError,match='already consumed'):e._authorized('qualify',args)            # one human instruction, one launch
        wrong=json.loads(paths['scope'].read_text());wrong['budget']['backward_calls']=1;paths['scope'].write_text(json.dumps(wrong));args.expected_scope_sha256=inherited.digest(paths['scope'])
        with pytest.raises(ValueError):e._authorized('qualify',args)
        broken=json.loads(json.dumps(rows));broken[0].pop('paired_z045_data_v1');(tmp_path/'schedule.json').write_text(json.dumps(broken))
        spec['file_sha256']['schedule']=inherited.digest(tmp_path/'schedule.json');(tmp_path/'spec.json').write_text(json.dumps(spec))
        with pytest.raises(ValueError,match='z045'):e.prepare(spec)
        (tmp_path/'schedule.json').write_text(json.dumps(rows));spec['file_sha256']['schedule']=inherited.digest(tmp_path/'schedule.json')
        with pytest.raises(ValueError,match='private output'):e.prepare({**spec,'output':str(tmp_path/'outside')})
    finally:
        shutil.rmtree(output,ignore_errors=True);shutil.rmtree(output.with_name(output.name+'_authorizations_PRIVATE'),ignore_errors=True)


def test_every_owned_source_exists_for_the_source_closure():
    from mri_rl import shared_rollout_gspo_entry_v1 as e
    root=Path(e.__file__).resolve().parents[2]
    assert all((root/name).is_file() for name in e.OWN_SOURCES)
