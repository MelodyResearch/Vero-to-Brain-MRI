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
        proof={'status':'ACTUAL_SHARED_ROLLOUT_GSPO_READONLY_QUALIFIED','optimizer_updates':0,'source_sha256':scope['source_sha256'],'input_sha256':scope['input_sha256']}
        qualification={'proof':proof,'review':{'verdict':'PASS','actual_proof_sha256':core.identity(proof),'reviewer_id':'r','reviewer_kind':'claude_code'}} if mode=='train' else None
        assert core.authorize_launch(mode,scope,human,review,qualification=qualification,source_sha256=prepared['source_sha256'])
        with pytest.raises(ValueError):core.authorize_launch(mode,scope,human,review,qualification=qualification,source_sha256={'/s.py':base.sha('edited')})
    with pytest.raises(ValueError):e.launch_scope(prepared,'execute')


def test_every_owned_source_exists_for_the_source_closure():
    from mri_rl import shared_rollout_gspo_entry_v1 as e
    root=Path(e.__file__).resolve().parents[2]
    assert all((root/name).is_file() for name in e.OWN_SOURCES)
