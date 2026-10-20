"""Pure CPU mechanism fixtures; no native model or actual MRI response claims."""
import dataclasses
import math
import pytest
from mri_rl import paired_prompt_groups_v1 as p


def test_matched_grouping_keeps_exact_eight_responses_and_own_contexts():
    block=p.synthetic_block([1]*4,[0]*4)
    same=p.group_responses(block,'same_prompt');mixed=p.group_responses(block,'paired_prompt')
    assert [[r.response_id for r in g] for g in same]==[['F0','F1','F2','F3'],['T0','T1','T2','T3']]
    assert [[r.response_id for r in g] for g in mixed]==[['F0','F1','T0','T1'],['F2','F3','T2','T3']]
    assert {id(r) for g in same for r in g}=={id(r) for g in mixed for r in g}
    assert all(r.context.modality==r.response_id[0] for g in mixed for r in g)
    with pytest.raises(ValueError):p.group_responses(block[:-1],'paired_prompt')


def test_cross_prompt_activates_groups_but_does_not_create_correct_T_response():
    block=p.synthetic_block([1]*4,[0]*4)
    same=p.make_plan(block,'same_prompt');mixed=p.make_plan(block,'paired_prompt')
    assert same.effective_groups==0 and mixed.effective_groups==2
    assert same.response_count==mixed.response_count==8
    assert all(item.advantage>0 for item in mixed.items if item.response.context.modality=='F')
    assert all(item.advantage<0 for item in mixed.items if item.response.context.modality=='T')
    assert [r.grade.accuracy for r in block if r.context.modality=='T']==[0]*4
    assert {r.response_id:r.grade for r in block}=={item.response.response_id:item.response.grade for item in mixed.items}


def test_total_reward_variation_does_not_bypass_inherited_Adam60_correctness_gate():
    block=p.synthetic_block([0]*4,[0]*4,format_F=1,format_T=0,total_F=1.,total_T=0.)
    mixed=p.make_plan(block,'paired_prompt')
    assert mixed.reward_variable_groups==2 and mixed.effective_groups==0
    assert all(item.weight==0 for item in mixed.items)


def test_block_loss_and_gradients_normalize_eight_once_and_use_own_length():
    block=p.synthetic_block([1]*4,[0]*4,lengths_F=[1,2,3,4],lengths_T=[4,3,2,1])
    plan=p.make_plan(block,'paired_prompt');calls=[]
    def scorer(request):
        calls.append(request)
        return p.ConditionedLogps(request.response_id,request.context_sha256,request.token_sha256,request.previous_logps)
    result=p.score_plan(plan,scorer)
    assert len(calls)==8 and math.isclose(result['loss'],0.,abs_tol=1e-14)
    for item,entry in zip(plan.items,result['responses']):
        assert item.weight==1/8
        assert sum(entry['d_loss_d_token_logps'])==pytest.approx(-item.advantage/8)
        assert len(entry['d_loss_d_token_logps'])==len(item.response.token_ids)


def test_foreign_context_or_tokens_cannot_receive_another_response_logps():
    plan=p.make_plan(p.synthetic_block([1]*4,[0]*4),'paired_prompt')
    def wrong(request):return p.ConditionedLogps(request.response_id,'0'*64,request.token_sha256,request.previous_logps)
    with pytest.raises(ValueError,match='attribution'):p.score_plan(plan,wrong)
    foreign=list(p.synthetic_block([1]*4,[0]*4))
    for i in range(4,8):foreign[i]=dataclasses.replace(foreign[i],context=dataclasses.replace(foreign[i].context,checkpoint_sha256='f'*64))
    with pytest.raises(ValueError,match='checkpoint'):p.make_plan(tuple(foreign),'paired_prompt')


def test_unchanged_clipping_sign_and_logratio_cap():
    assert p.response_surrogate([0.],[0.],1.)['d_loss_d_token_logps']==[-1.]
    assert p.response_surrogate([.01],[0.],1.)['d_loss_d_token_logps']==[0.]
    assert p.response_surrogate([-.01],[0.],-1.)['d_loss_d_token_logps']==[0.]
    assert p.response_surrogate([11.],[0.],-1.)['d_loss_d_token_logps']==[0.]
    with pytest.raises(ValueError):p.response_surrogate([float('nan')],[0.],1.)


def test_more_effective_groups_can_have_zero_visual_gradient_or_harm_T():
    plan=p.make_plan(p.synthetic_block([1]*4,[0]*4),'paired_prompt')
    equal={r.response_id:[1.] for r in p.synthetic_block([1]*4,[0]*4)}
    assert p.local_descent(plan,equal)==pytest.approx([0.],abs=1e-15)
    prior={r.response_id:[2. if r.context.modality=='F' else 1.] for r in p.synthetic_block([1]*4,[0]*4)}
    direction=p.local_descent(plan,prior)[0]
    assert direction>0 and -.5*direction<0


def test_conditional_constant_reward_has_zero_expected_fixed_length_score():
    result=p.enumerate_bernoulli(.8,.2,'paired_prompt',reward_mode='prompt_constant')
    assert result['expected_descent']==pytest.approx([0.,0.],abs=1e-15)
    assert result['effective_group_probability']==pytest.approx(1.)


def test_mixed_std_weights_are_not_unbiased_balanced_reward_and_no_T_guarantee():
    result=p.enumerate_bernoulli(.3,.9,'paired_prompt')
    true=[.3*.7/2,.9*.1/2];g=result['expected_descent']
    assert abs(g[0]/true[0]-g[1]/true[1])>1e-3
    # Shared theta increases F correctness but decreases T correctness. The
    # mixed normalized field can oppose balanced reward's local gradient.
    slope=-2.346
    assert true[0]+slope*true[1]<0 and g[0]+slope*g[1]>0
    assert slope*(g[0]+slope*g[1])<0


def test_reference_surrogate_matches_unchanged_native_loss_forward_AST():
    import ast
    from pathlib import Path
    from types import SimpleNamespace as NS
    import numpy as np
    source=Path(p.__file__).with_name('mlx_continuous_gspo.py')
    tree=ast.parse(source.read_text());node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='loss_fn')
    for advantage in (-1.,0.,1.):
        for change in (-.01,0.,.01,11.):
            previous=np.array([-2.,-3.]);current=previous+change
            mx=NS(stop_gradient=lambda v:v,mean=np.mean,exp=np.exp,minimum=np.minimum,maximum=np.maximum,clip=np.clip)
            namespace={'mx':mx,'self':NS(_logps=lambda tokens:current)}
            exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),namespace)
            native=float(namespace['loss_fn'](None,[1,2],previous,advantage))
            assert p.response_surrogate(current,previous,advantage)['loss']==pytest.approx(native,rel=1e-14,abs=1e-14)


def test_constant_reward_variable_length_breaks_zero_mean_normalized_score():
    result=p.variable_length_constant_reward_example()
    assert result['expected_joint_score']==pytest.approx(0.)
    assert result['expected_mean_logp_score']==pytest.approx(.125)
    assert result['mixed_descent_F_logit']>0
    assert result['true_reward_gradient']==0


def test_inherited_Adam60_gate_scope_changes_without_within_prompt_variation():
    block=p.synthetic_block([1]*4,[0]*4)
    for grouping in ('same_prompt','paired_prompt'):
        for group in p.group_diagnostics(block,grouping):
            assert all(not values['within_condition_calibrated_accuracy_varies'] for values in group['accuracy_by_condition'].values())
    assert all(group['inherited_Adam60_gate_admitted'] for group in p.group_diagnostics(block,'paired_prompt'))


def test_reward_calibration_identity_cannot_change_between_prompts():
    block=list(p.synthetic_block([1]*4,[0]*4))
    for i in range(4,8):block[i]=dataclasses.replace(block[i],context=dataclasses.replace(block[i].context,reward_policy_sha256='e'*64))
    with pytest.raises(ValueError):p.group_responses(block,'paired_prompt')


def test_exact_native_stopgradient_AST_derivative_on_CPU_dual_numbers():
    import ast
    from pathlib import Path
    from types import SimpleNamespace as NS
    import numpy as np
    class D:
        def __init__(self,value,jac):self.v=np.asarray(value,dtype=float);self.j=np.asarray(jac,dtype=float)
        def __sub__(self,other):
            if not isinstance(other,D):other=D(other,np.zeros_like(self.j))
            return D(self.v-other.v,self.j-other.j)
        def __add__(self,other):return D(self.v+other.v,self.j+other.j)
        def __mul__(self,value):return D(self.v*value,self.j*value)
        __rmul__=__mul__
    def unary(value,f,df):return D(f(value.v),value.j*np.asarray(df(value.v))[...,None])
    def maximum(left,right):
        choose=left.v>=right.v
        return D(np.maximum(left.v,right.v),np.where(choose[...,None],left.j,right.j))
    mx=NS(stop_gradient=lambda x:D(x.v,np.zeros_like(x.j)),mean=lambda x:D(x.v.mean(),x.j.mean(axis=0)),
          exp=lambda x:unary(x,np.exp,np.exp),minimum=lambda x,b:unary(x,lambda v:np.minimum(v,b),lambda v:(v<b).astype(float)),
          maximum=maximum,clip=lambda x,l,h:unary(x,lambda v:np.clip(v,l,h),lambda v:((v>l)&(v<h)).astype(float)))
    source=Path(p.__file__).with_name('mlx_continuous_gspo.py');node=next(n for n in ast.walk(ast.parse(source.read_text())) if isinstance(n,ast.FunctionDef) and n.name=='loss_fn')
    for advantage in (-1.,1.):
        for delta in (-.01,0.,.01,11.):
            previous=np.array([-3.,-2.]);current=previous+delta
            namespace={'mx':mx,'self':NS(_logps=lambda tokens:D(current,np.eye(2)))}
            exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),namespace)
            result=namespace['loss_fn'](None,[1,2],previous,advantage)
            expected=p.response_surrogate(current,previous,advantage)
            np.testing.assert_allclose(result.j,expected['d_loss_d_token_logps'],rtol=1e-14,atol=1e-14)


def test_half_format_grade_is_retained_but_inherited_Adam60_gate_rejects_group():
    grade=p.Grade(1.,.5,1.5)
    assert grade.format==.5 and grade.total==1.5
    block=p.synthetic_block([1]*4,[0]*4,format_F=.5,format_T=1,total_F=1.5,total_T=.5)
    plan=p.make_plan(block,'paired_prompt')
    assert plan.reward_variable_groups==2 and plan.effective_groups==0
    assert all(item.weight==0 for item in plan.items)
    assert all(item.advantage>0 for item in plan.items if item.response.context.modality=='F')
    assert all(item.advantage<0 for item in plan.items if item.response.context.modality=='T')
    assert [r.grade.total for r in block]==[1.5]*4+[.5]*4


def test_inherited_Adam60_valid_subset_gate_differs_from_historical18_all_valid_gate():
    grades=[{'accuracy':1.,'format':1.,'total':1.,'overlong':0.},
            {'accuracy':0.,'format':1.,'total':0.,'overlong':0.},
            {'accuracy':1.,'format':.5,'total':1.,'overlong':0.},
            {'accuracy':0.,'format':1.,'total':0.,'overlong':0.}]
    assert p.inherited_Adam60_gate(grades) is True
    assert p.historical18_strict_gate(grades) is False
    block=list(p.synthetic_block([1,0,1,0],[1,0,1,0]))
    block[2]=dataclasses.replace(block[2],grade=p.Grade(1.,.5,1.))
    plan=p.make_plan(tuple(block),'same_prompt')
    assert plan.effective_groups==2


def test_calibrated_accuracy_keeps_raw_correctness_and_total_without_regrading():
    with pytest.raises(ValueError):p.Grade(.7,1.,9.25)
    correct=p.Grade(.7,1.,9.25,raw_accuracy=1.)
    wrong=p.Grade(.3,1.,8.75,raw_accuracy=0.)
    assert correct.accuracy==.7 and correct.raw_accuracy==1. and correct.total==9.25
    assert wrong.accuracy==.3 and wrong.raw_accuracy==0. and wrong.total==8.75
    block=tuple(dataclasses.replace(r,grade=correct if r.context.modality=='F' else wrong) for r in p.synthetic_block([1]*4,[0]*4))
    plan=p.make_plan(block,'paired_prompt');assert plan.effective_groups==2
    for group in p.group_diagnostics(block,'paired_prompt'):
        for modality,values in group['accuracy_by_condition'].items():
            assert values['calibrated_accuracy_values']==([.7] if modality=='F' else [.3])
            assert values['raw_accuracy_values']==([1.] if modality=='F' else [0.])
            assert values['within_condition_raw_correctness_varies'] is False
    with pytest.raises(ValueError):p.Grade(.7,1.,9.25,raw_accuracy=.5)
