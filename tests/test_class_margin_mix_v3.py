import numpy as np
import pytest
from mri_rl import paired_classification_mix_v3 as mix
from mri_rl import class_margin_mix_v3 as native
from mri_rl import paired_interaction_signal_v1 as v1
from mri_rl import paired_classification_signal_v2 as v2


def test_frozen_mix_loss_and_derivative_match_finite_differences():
    scores=np.array([[-.01,-1.4],[-.01,-1.3]])
    assert mix.WEIGHT==.457130 and mix.ALPHA==.1
    assert mix.loss(scores)==pytest.approx((1-mix.WEIGHT)*v1.pair_loss(scores)+mix.WEIGHT*v2.classification_loss(scores))
    analytic=mix.score_derivatives(scores)
    for i in range(2):
        for j in range(2):
            high=scores.copy();low=scores.copy();high[i,j]+=1e-5;low[i,j]-=1e-5
            assert analytic[i][j]==pytest.approx((mix.loss(high)-mix.loss(low))/2e-5,abs=1e-10)
    assert mix.differentiable_loss(scores,np)==pytest.approx(mix.loss(scores))


def test_nonfinite_or_unfrozen_coefficient_is_not_a_training_knob():
    with pytest.raises(ValueError):mix.require_fixed_weight(.5)
    with pytest.raises(ValueError):mix.loss([[np.nan,-1],[-1,-2]])


def test_joint_adam_guard_uses_total_and_discloses_momentum_increment():
    assert native.check_direction_report({'mT':.002,'mF':.003,'mDallas':.01},.024,guard=1e-6,delta_cap=.025)
    with pytest.raises(ValueError,match='CLASS_PROTECTION'):
        native.check_direction_report({'mT':.002,'mF':-.003,'mDallas':.01},.024,guard=1e-6,delta_cap=.025)
    with pytest.raises(ValueError,match='NORM'):
        native.check_direction_report({'mT':.002,'mF':.003,'mDallas':.01},.026,guard=1e-6,delta_cap=.025)


def test_current_context_gradients_are_required_not_stale_Adam60_targets():
    backend=native.MixtureClassMarginBackend.__new__(native.MixtureClassMarginBackend)
    with pytest.raises(ValueError,match='fresh current'):
        backend.require_current_targets({'step':61}, {'step':60,'targets':{}})


def test_existing_atomic_driver_is_reused_not_an_unprotected_new_loop():
    from mri_rl import class_margin_aux_v2_driver as original
    assert native.run_micro_driver is original.run
    assert issubclass(native.MixtureClassMarginBackend,native.ClassMarginAuxBackend)


def test_native_plumbing_checks_current_joint_Adam_before_single_step(monkeypatch):
    from types import SimpleNamespace as NS
    monkeypatch.setattr(native,'_state',lambda *args:{'Adam':'same','RNG':'same'})
    monkeypatch.setattr(native,'_moments',lambda value:{})
    backend=native.MixtureClassMarginBackend.__new__(native.MixtureClassMarginBackend)
    backend.reference_contexts={};backend.guard_token_plans={};backend.companions={'seq':True}
    backend.tree_map=lambda f,*trees:{k:f(*(t[k] for t in trees)) for k in trees[0]}
    backend.tree_flatten=lambda tree:list(tree.items());backend._checked_gradient=lambda g:float(np.linalg.norm(g['w']))
    backend.mx=NS(eval=lambda *args:None,clear_cache=lambda:None)
    backend.model=NS(trainable_parameters=lambda:{'w':np.ones(1)})
    backend.optimizer_state=lambda:{'step':np.array(60)};backend.before_unit=lambda name:None
    calls=[];backend.optimizer=NS(update=lambda model,g:calls.append(g),state={},learning_rate=np.array(1e-5),betas=(.9,.999),eps=1e-8,bias_correction=False)
    backend.detached_ownold=lambda *args:([],[])
    backend.gspo_gradient=lambda *args:({'w':np.array([1.])},{'loss_terms':[.2]})
    backend.auxiliary_gradient=lambda *args:({'w':np.array([.1])},{'auxiliary_loss':.4})
    backend.fresh_targets=lambda:{'step':60,'targets':{k:{'w':np.ones(1)} for k in ('mF','mT','mDallas')},'target_scope_PRIVATE':{'CPU_fixture':'current'}}
    backend.expected_image_sha256={}
    monkeypatch.setattr(native,'target_scope',lambda *args:{'CPU_fixture':'current'})
    report={'directions':{'Adam_total':{'mF':.001,'mT':.002,'mDallas':.03}},'Adam_total_delta_l2':.024}
    monkeypatch.setattr(native,'preview_tree',lambda *args,**kwargs:report)
    result=backend.transition({'id':'seq'},{},[0,0,0,0],True)
    assert result['mode']=='JOINT' and len(calls)==1 and calls[0]['w'][0]==pytest.approx(1.1)
    calls.clear();report['directions']['Adam_total']['mF']=-.001
    with pytest.raises(ValueError,match='CLASS_PROTECTION'):backend.transition({'id':'seq'},{},[0,0,0,0],True)
    assert calls==[]


def test_missing_new_readonly_evidence_cannot_configure_native_candidate():
    backend=native.MixtureClassMarginBackend.__new__(native.MixtureClassMarginBackend)
    with pytest.raises(ValueError,match='review is absent'):
        backend.configure_mixture_guards({}, {},qualification_receipt={})


def test_current_auxiliary_target_scope_never_claims_fixed_reference_identity():
    ref={name:{'row_PRIVATE':{'id':'reference-'+name,'extra_info':{'question':'generic'}} ,'image_sha256':'ref-'+name} for name in ('F_REAL','T_REAL','DALLAS_REAL')}
    pair=[{'id':'current-F','extra_info':{'question':'generic'}},{'id':'current-T','extra_info':{'question':'generic'}}]
    scope=native.target_scope(pair,ref,{'current-F':'image-F','current-T':'image-T'})
    assert scope['mF']['record_id_PRIVATE']=='current-F' and scope['mF']['mode']=='CURRENT_AUX_PAIR'
    assert scope['mT']['record_id_PRIVATE']=='current-T' and scope['mT']['image_sha256']=='image-T'
    assert scope['mDallas']['mode']=='FIXED_REFERENCE_CASE'
    fallback=native.target_scope(None,ref,{})
    assert fallback['mF']['record_id_PRIVATE']=='reference-F_REAL'


def test_direction_preview_refuses_missing_Dallas_target():
    g={'w':np.ones(1,np.float32)};moments={'w':(np.ones(1,np.float32),np.ones(1,np.float32))}
    profile={'learning_rate':1e-5,'betas':(.9,.999),'eps':1e-8,'bias_correction':False,'step':60}
    with pytest.raises(ValueError,match='three exact'):
        native.preview_tree(g,{'mF':g,'mT':g},moments,profile,ops=np)
