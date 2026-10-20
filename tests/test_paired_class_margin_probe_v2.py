"""Tiny CPU arithmetic/autodiff fixtures; not native qualification or training."""
import copy
import math
import types
import numpy as np
import pytest
from mri_rl import paired_class_margin_probe_v2 as p


def test_frozen_budget_is_four_vjp_eleven_TF_and_exact2868_positions():
    b=p.budget();assert b['full_TF_forwards']==11 and b['backward_calls']==4
    assert b['teacherforced_positions']==4675 and b['expected_teacherforced_positions']==2868
    c=p.Counter()
    for n in [225,228,225,228,416,416,225,228,224,225,228]:c.forward(n)
    for _ in range(4):c.backward()
    assert c.values['teacherforced_positions']==2868
    with pytest.raises(ValueError):c.forward(1)
    with pytest.raises(ValueError):c.backward()


def test_adam_preview_matches_actual_uncorrected_formula_without_mutation():
    g=np.array([1.,-2.],np.float32);m=np.array([.2,.3],np.float32);v=np.array([.4,.5],np.float32);before=[a.copy() for a in [g,m,v]]
    r=p.readonly_adam_leaf(g,m,v,learning_rate=1e-5,betas=(.9,.999),eps=1e-8,bias_correction=False,step=60,ops=np)
    expected=-np.float32(1e-5)*(.9*m+.1*g)/(np.sqrt(.999*v+.001*g*g)+1e-8)
    zero=-np.float32(1e-5)*(.9*m)/(np.sqrt(.999*v)+1e-8)
    np.testing.assert_allclose(r['delta'],expected,rtol=1e-6)
    np.testing.assert_allclose(r['baseline_delta'],zero,rtol=1e-6)
    np.testing.assert_allclose(r['incremental_delta'],expected-zero,rtol=1e-6)
    assert r['preview_step']==61
    for a,b in zip([g,m,v],before):np.testing.assert_array_equal(a,b)


def test_zero_aux_gradient_can_still_have_adam_momentum_but_increment_is_zero():
    r=p.readonly_adam_leaf(np.zeros(2),np.ones(2),np.ones(2),learning_rate=1e-5,betas=(.9,.999),eps=1e-8,bias_correction=False,step=60,ops=np)
    assert np.linalg.norm(r['delta'])>0
    np.testing.assert_array_equal(r['incremental_delta'],np.zeros(2))


def test_adam_preview_uses_supplied_beta_positions_and_rejects_unreviewed_bias_correction():
    r=p.readonly_adam_leaf(np.ones(1),np.array([2.]),np.array([3.]),learning_rate=.02,betas=(.7,.8),eps=.03,bias_correction=False,step=60,ops=np)
    np.testing.assert_allclose(r['delta'],-.02*(.7*2+.3)/(np.sqrt(.8*3+.2)+.03))
    with pytest.raises(ValueError):p.readonly_adam_leaf(np.ones(1),np.ones(1),np.ones(1),learning_rate=.02,betas=(.7,.8),eps=.03,bias_correction=True,step=60,ops=np)


def test_aux_gradient_chain_rule_and_swap_blank_aliases():
    gF=np.array([1.,2.]);gT=np.array([3.,4.]);gB=np.array([5.,6.])
    scores=[[-.1,-1.4],[-.1,-1.0]]
    out=p.auxiliary_gradient_coefficients(scores)
    assert set(out)=={'v1','v2'}
    assert out['v1'][0]==out['v1'][1]
    assert out['v2'][0]!=out['v2'][1]
    aliases=p.gradient_aliases({'F':gF,'T':gT,'BLANK':gB})
    np.testing.assert_array_equal(aliases['SWAPPED_F'],-gT)
    np.testing.assert_array_equal(aliases['SWAPPED_T'],-gF)
    np.testing.assert_array_equal(aliases['BLANK_T'],-gB)
    # v1 identical blank opposite-label gradients cancel;v2 may retain prior-only signal.
    blank=p.auxiliary_gradient_coefficients([[-.1,-1.0],[-.1,-1.0]])
    assert blank['v1'][0]-blank['v1'][1]==0
    assert blank['v2'][0]-blank['v2'][1]!=0


def test_fresh_human_authorization_and_exact_independent_review_required():
    scope={'schema':'paired-class-margin-probe-v2'};digest=p.canonical(scope)
    with pytest.raises(ValueError):p.authorize(scope,{'authorized':True,'action':'readonly_paired_signal_candidate_probe','scope_sha256':digest},{})
    assert p.authorize(scope,{'authorized':True,'action':'readonly_class_margin_v2_probe','scope_sha256':digest,'human_instruction_sha256':'a'},
        {'scope_sha256':digest,'verdict':'PASS_WITH_LIMITATIONS','reviewer_id':'isolated test reviewer','reviewer_kind':'isolated_codex'})


class Dual:
    def __init__(self,value,jac):self.value=np.asarray(value);self.jac=np.asarray(jac)
    def __getitem__(self,i):return Dual(self.value[i],self.jac[i])
    def mean(self):return Dual(self.value.mean(),self.jac.mean(axis=0))
    def __sub__(self,b):return Dual(self.value-b.value,self.jac-b.jac)


class Model(dict):
    def trainable_parameters(self):return dict(self)


@pytest.mark.parametrize('truth',[0,1])
def test_shared_vjp_is_label_only_correct_minus_wrong_two_TF_one_backward(monkeypatch,truth):
    model=Model(w=np.array([.3,-.7]));calls=[]
    def scorer(backend,tokens,inputs,*,evidence=None):
        n=len(tokens)-11;label=0 if n==2 else 1
        value=np.r_[np.full(11,-42.),np.full(n,model['w'][label])]
        jac=np.vstack([np.tile([7.,11.],(11,1)),np.tile([1.,0.] if label==0 else [0.,1.],(n,1))])
        if evidence is not None:evidence.append({'F32_fixture_only':True})
        calls.append(tokens);return Dual(value,jac)
    monkeypatch.setattr(p,'f32_response_logps_observed',scorer)
    class NN:
        @staticmethod
        def value_and_grad(model,objective):
            def fn(m):
                result=objective(m);return np.asarray(result.value),{'w':result.jac}
            return fn
    mx=types.SimpleNamespace(mean=lambda x:x.mean(),stop_gradient=lambda x:x.value.copy(),eval=lambda *a:None,
        all=np.all,isfinite=np.isfinite,float32=np.float32)
    backend=types.SimpleNamespace(model=model,mx=mx,nn=NN(),tree_flatten=lambda d:list(d.items()),tree_map=lambda fn,d:{k:fn(v) for k,v in d.items()})
    forwards=[];backs=[]
    plan={'prefix':list(range(11)),'labels':[[2,3],[4,5,6,7,8]],'label_order':['FLAIR','contrast-enhanced T1']}
    r=p.serial_margin_vjp(backend,{'input_ids':np.zeros((1,213),int)},plan,truth,before=lambda s:None,charge_forward=forwards.append,charge_backward=lambda:backs.append(1))
    assert forwards==[225,228] and backs==[1] and len(calls)==2
    assert r['mean_scores']==[.3,-.7]
    assert r['margins']['mean']==pytest.approx((1 if truth==0 else -1)*1.)
    np.testing.assert_array_equal(r['gradient']['w'],np.array([1.,-1.])*(1 if truth==0 else -1))


def test_saved_gradient_comparison_has_defined_norms_but_undefined_zero_cosine():
    r=p.compare_gradient_vectors([('a',np.array([1.,2.]),np.array([1.,2.]))])
    assert r['cosine']==pytest.approx(1.) and r['difference_l2']==0
    z=p.compare_gradient_vectors([('a',np.zeros(2),np.ones(2))])
    assert z['cosine']=='UNDEFINED' and z['left_l2']==0


def test_actual_manifest_moment_layout_parser_preserves_model_keynames_and_shapes():
    tensor=lambda shape:{'kind':'tensor','key':'x','shape':shape,'dtype':'mlx.core.float32'}
    leaf={'kind':'dict','items':[['m',tensor([2])],['v',tensor([2])]]}
    opt={'kind':'dict','items':[['step',tensor([])],['layers',{'kind':'list','items':[leaf]}]]}
    layout={'kind':'dict','items':[['optimizer',opt]]}
    assert p.manifest_moment_inventory(layout)=={'layers.0':{'shape':[2],'dtype':'mlx.core.float32'}}


@pytest.mark.parametrize('g,m,v',[(1e30,0.,1.),(float('nan'),0.,1.),(1.,float('inf'),1.),(1.,0.,-1.)])
def test_adam_preview_fails_closed_on_nonfinite_negative_or_overflow(g,m,v):
    with np.errstate(over='ignore',invalid='ignore',divide='ignore'),pytest.raises(ValueError):
        p.readonly_adam_leaf(*[np.array([x],np.float32) for x in (g,m,v)],learning_rate=1e-5,
            betas=(.9,.999),eps=1e-8,bias_correction=False,step=60,ops=np)


def test_source_closure_requires_direct_evaluator_and_rejects_byte_tamper(tmp_path):
    sources={}
    for relative in p.REQUIRED_DIRECT_PROBE_SOURCES:
        path=tmp_path/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(relative)
        sources[str(path)]=p.hash_file(path)
    assert p.validate_source_closure(sources,root=tmp_path)
    removed=str(tmp_path/'src/mri_rl/paired_z045_matched_f32_eval_v1.py')
    with pytest.raises(ValueError):p.validate_source_closure({k:v for k,v in sources.items() if k!=removed},root=tmp_path)
    (tmp_path/'src/mri_rl/paired_z045_evaluation_v1.py').write_text('tampered')
    with pytest.raises(ValueError):p.validate_source_closure(sources,root=tmp_path)


def test_gradient_comparison_fails_closed_on_scalar_overflow():
    with np.errstate(over='ignore',invalid='ignore'),pytest.raises(ValueError):
        p.compare_gradient_vectors([('a',np.array([1e300]),np.array([-1e300]))])
