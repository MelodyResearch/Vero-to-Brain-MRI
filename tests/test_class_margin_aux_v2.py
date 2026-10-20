"""CPU fixtures only, never native training evidence."""
import copy
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest
from mri_rl import class_margin_aux_v2 as a


def test_inactive_has_aux_only_and_no_fake_GSPO():
    assert a.branch_mode(False,True)=='AUXILIARY_ONLY'
    assert a.branch_mode(True,True)=='JOINT'
    assert a.branch_mode(True,False)=='GSPO_ONLY'
    assert a.branch_mode(False,False)=='NO_UPDATE'
    gradient=a.combine_gradients(None,[({'w':np.array([2.,-1.])},.1)],lambda f,*trees:{k:f(*(t[k] for t in trees)) for k in trees[0]})
    np.testing.assert_array_equal(gradient['w'],[.2,-.1])


def test_joint_gradient_is_added_before_exactly_one_adam_step():
    tree_map=lambda f,*trees:{k:f(*(t[k] for t in trees)) for k in trees[0]}
    g=a.combine_gradients({'w':np.array([1.,-2.])},[({'w':np.array([2.,-3.])},.1),({'w':np.array([-1.,4.])},.2)],tree_map)
    np.testing.assert_allclose(g['w'],[1.,-1.5])
    w=np.array([.3,-.1]);m=np.array([.4,-.2]);v=np.array([.1,.2])
    def adam(grad):
        mm=.9*m+.1*grad;vv=.999*v+.001*grad*grad
        return w-1e-5*mm/(np.sqrt(vv)+1e-8),mm,vv
    for x,y in zip(adam(g['w']),adam(np.array([1.,-1.5]))):np.testing.assert_allclose(x,y,rtol=1e-15,atol=0)


def test_all_optimizer_branches_share_two_total_update_budget():
    assert a.require_update_budget(1,2,'AUXILIARY_ONLY')
    for mode in ('JOINT','AUXILIARY_ONLY','GSPO_ONLY'):
        with pytest.raises(ValueError):a.require_update_budget(2,2,mode)
    assert a.require_update_budget(2,2,'NO_UPDATE')


def test_token_reservation_includes_aux_only_even_inactive_GSPO():
    assert a.group_reservation(213,906)==4*(213+256)+906
    assert a.group_reservation(405,0)==4*(405+256)
    with pytest.raises(ValueError):a.group_reservation(214,906)


def test_unknown_authorization_fails_before_any_native_import():
    import sys
    before=set(sys.modules)
    with pytest.raises(ValueError):a.authorize_launch({}, {}, {}, {})
    assert not any(x=='mlx' or x.startswith('mlx.') for x in set(sys.modules)-before)


def cpu_fixture(tmp_path):
    file=Path(__file__).with_name('test_paired_z045_continuation_v1.py')
    spec=importlib.util.spec_from_file_location('class_aux_parent_fixture',file);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    B,parent,sha,rows=old.fixture(tmp_path)
    while len(rows)<4:rows.append(copy.deepcopy(rows[-1])|{'id':'fixture-'+str(len(rows))})
    rows=rows[:4]
    class Backend(B):
        def transaction_snapshot(self):
            result=copy.deepcopy((self.weight,self.state,self.rng.getstate()))
            if not hasattr(self,'first_snapshot'):self.first_snapshot=result
            return result
        def transaction_restore(self,snapshot):
            self.weight,self.state,rng=copy.deepcopy(snapshot);self.rng.setstate(rng)
        def rollout(self,row,index,work):
            raw=super().rollout(row,index,work);raw['group_index']=index;return raw
        def activate_saved_rollout(self,row,raw):self.calls.append(('reuse',row['id']))
        def transition(self,row,raw,advantages,active):
            super().update(row,raw,advantages)
            return {'updated':True,'mode':'JOINT' if active else 'AUXILIARY_ONLY','accepted_GSPO':active,'gradient_l2':1.}
    return Backend,parent,sha,rows


def launch_fixture(tmp_path,B,parent,sha,rows,out,**kwargs):
    from mri_rl import class_margin_aux_v2_driver as d
    return d.run(output=out,schedule=rows,bindings={'stage':'CPU_FIXTURE_ONLY'},backend_factory=B,authorize=lambda context:None,
                 parent_checkpoint=parent,parent_receipt_sha256=sha,maximum_seconds=600,
                 config={**d.old.PROFILE,'group_size':2,'max_tokens':256},row_validator=lambda row:None,**kwargs)


def test_real_driver_exact_resume_matches_weights_Adam_RNG_and_cursor(tmp_path):
    B,parent,sha,rows=cpu_fixture(tmp_path)
    full=launch_fixture(tmp_path,B,parent,sha,rows,tmp_path/'full');b=B.instances[-1]
    launch_fixture(tmp_path,B,parent,sha,rows,tmp_path/'split',max_new_groups=1)
    split=launch_fixture(tmp_path,B,parent,sha,rows,tmp_path/'split',resume=True);c=B.instances[-1]
    assert full['optimizer_updates']==split['optimizer_updates']==2
    assert split['completed_groups']==2
    np.testing.assert_array_equal(b.weight,c.weight)
    for key in ('m','v'):np.testing.assert_array_equal(b.state['weight'][key],c.state['weight'][key])
    assert b.rng.random()==c.rng.random()


def test_transition_failure_rolls_back_model_Adam_RNG_and_saved_rollout_reused(tmp_path):
    B,parent,sha,rows=cpu_fixture(tmp_path)
    class Fail(B):
        def transition(self,*args):
            self.before=self.transaction_snapshot();super().transition(*args);self.rng.random();raise RuntimeError('after optimizer simulated failure')
    with pytest.raises(RuntimeError,match='simulated failure'):launch_fixture(tmp_path,Fail,parent,sha,rows,tmp_path/'failure')
    b=Fail.instances[-1];np.testing.assert_array_equal(b.weight,b.before[0]);np.testing.assert_array_equal(b.state['weight']['m'],b.before[1]['weight']['m'])
    assert b.rng.getstate()==b.first_snapshot[2]
    assert not (tmp_path/'failure/boundary_0001').exists()
    launch_fixture(tmp_path,B,parent,sha,rows,tmp_path/'failure',resume=True,max_new_groups=1)
    assert ('reuse',rows[0]['id']) in B.instances[-1].calls
    assert not any(call[0]=='rollout' for call in B.instances[-1].calls)


def test_rename_committed_before_error_is_not_replayed(tmp_path,monkeypatch):
    from mri_rl import class_margin_aux_v2_driver as d
    B,parent,sha,rows=cpu_fixture(tmp_path);publish=d.old._publish
    def crash(*args,**kwargs):
        result=publish(*args,**kwargs)
        if result['completed_groups']==1:raise RuntimeError('after atomic rename')
        return result
    monkeypatch.setattr(d.old,'_publish',crash)
    with pytest.raises(RuntimeError,match='atomic rename'):launch_fixture(tmp_path,B,parent,sha,rows,tmp_path/'renamed',max_new_groups=1)
    assert (tmp_path/'renamed/boundary_0001/receipt.json').exists()
    monkeypatch.setattr(d.old,'_publish',publish)
    result=launch_fixture(tmp_path,B,parent,sha,rows,tmp_path/'renamed',resume=True,max_new_groups=1)
    assert result['completed_groups']==2
    assert not any(call[0]=='rollout' and call[1]==rows[0]['id'] for call in B.instances[-1].calls)


def test_actual_native_transition_combines_then_one_step_and_zero_rules(monkeypatch):
    from types import SimpleNamespace as NS
    monkeypatch.setattr(a,'_state',lambda *args:{'Adam':'unchanged','RNG':'unchanged'})
    def tree_map(f,*trees):return {key:f(*(tree[key] for tree in trees)) for key in trees[0]}
    backend=a.ClassMarginAuxBackend.__new__(a.ClassMarginAuxBackend)
    backend.tree_map=tree_map;backend.tree_flatten=lambda value:list(value.items());backend.trainable_signature=lambda:{'w':{}}
    backend.mx=NS(eval=lambda *args:None,clear_cache=lambda:None,all=np.all,isfinite=np.isfinite,sum=np.sum,float32=np.float32)
    backend.model=NS(trainable_parameters=lambda:{'w':np.array([.3,-.1])});backend.active={'genuine':'rollout'}
    backend.before_unit=lambda name:None;backend.companions={'sequence':True};calls=[]
    backend.optimizer=NS(update=lambda model,gradient:calls.append(copy.deepcopy(gradient)),state={})
    backend.detached_ownold=lambda *args:([],[])
    backend.gspo_gradient=lambda *args:({'w':np.array([1.,-2.])},{'unchanged_loss':'AST'})
    backend.auxiliary_gradient=lambda *args:({'w':np.array([.2,.5])},{'alpha_already_applied':.1})
    result=backend.transition({'id':'sequence'},{},[0,0,0,0],True)
    assert result['mode']=='JOINT' and result['optimizer_calls']==len(calls)==1
    np.testing.assert_allclose(calls[0]['w'],[1.2,-1.5])
    calls.clear();result=backend.transition({'id':'sequence'},{},[0,0,0,0],False)
    assert result['mode']=='AUXILIARY_ONLY' and result['accepted_GSPO'] is False and len(calls)==1
    calls.clear();backend.auxiliary_gradient=lambda *args:({'w':np.zeros(2)},{})
    result=backend.transition({'id':'sequence'},{},[0,0,0,0],False)
    assert result['mode']=='NO_UPDATE' and not result['updated'] and calls==[]
    backend.auxiliary_gradient=lambda *args:({'w':np.array([-1.,2.])},{})
    with pytest.raises(ValueError,match='TOTAL_GRADIENT_CANCELLATION'):backend.transition({'id':'sequence'},{},[0,0,0,0],True)
    assert calls==[]


def test_exact_forward_cap_formula_and_atomic_budget_guard():
    maximum=8*(2*(213+256-1)+2*(405+256-1))+1812
    assert maximum==19860
    compute=a.ComputeBudget();assert compute.limits['teacherforced_positions']==maximum
    for _ in range(4):compute.auxiliary_forward(225);compute.auxiliary_forward(228)
    assert compute.used['auxiliary_positions']==1812
    before=copy.deepcopy(compute.used)
    with pytest.raises(ValueError):compute.auxiliary_forward(1)
    assert compute.used==before


def test_completed_pending_generation_charged_once_on_resume(tmp_path):
    from mri_rl.two_task_training import TokenLedger
    B,parent,sha,rows=cpu_fixture(tmp_path)
    class Fail(B):
        def transition(self,*args):raise RuntimeError('before step')
    plan={'group_size':2,'max_tokens':256,'budget':{'maximum_attempted_groups':4,'text_visual_token_cap':9040,'input_tokens_min':213,'input_tokens_max':213}}
    ledger=TokenLedger(plan)
    with pytest.raises(RuntimeError):launch_fixture(tmp_path,Fail,parent,sha,rows,tmp_path/'cost',ledger=ledger,input_tokens=lambda row:213)
    used=ledger.used;restored=TokenLedger(plan)
    launch_fixture(tmp_path,B,parent,sha,rows,tmp_path/'cost',ledger=restored,input_tokens=lambda row:213,resume=True,max_new_groups=1)
    assert restored.used==used and restored.charged_groups==1


def test_unknown_train_entry_refuses_before_model_import():
    import sys
    from types import SimpleNamespace
    from mri_rl import class_margin_aux_v2_entry as e
    before=set(sys.modules)
    with pytest.raises(ValueError,match='NOT_AUTHORIZED'):e.execute(SimpleNamespace())
    assert not any(x=='mlx' or x.startswith('mlx.') for x in set(sys.modules)-before)


def test_original_main_image_path_can_differ_from_identical_authenticated_companion(tmp_path):
    from mri_rl import class_margin_aux_v2_entry as e
    first=tmp_path/'old.png';second=tmp_path/'paired.png';first.write_bytes(b'CPU_IMAGE');second.write_bytes(first.read_bytes())
    expected=e.inherited.digest(second)
    assert e.same_image_bytes(first,second,expected)
    second.write_bytes(b'CHANGED')
    with pytest.raises(ValueError):e.same_image_bytes(first,second,expected)


def admission_fixture(tmp_path):
    from mri_rl import class_margin_aux_v2_entry as e
    # Schema-valid synthetic CPU evidence: these files are NOT actual native proofs.
    names=('paired_classification_signal_v2.py','paired_class_margin_probe_v2.py','paired_z045_f32_backend_v1.py','conditional_gspo_gradient_v2.py',
           'native_probability_resolution_v1.py','rollout_precision_v2.py','mlx_continuous_gspo.py','class_margin_aux_v2.py','class_margin_aux_v2_driver.py','class_margin_aux_v2_entry.py')
    sources={str(tmp_path/name):'CPU_FIXTURE_'+name for name in names};sources['/CPU_FIXTURE/site-packages/runtime.py']='runtime'
    budget={'maximum_attempted_groups':4,'maximum_optimizer_updates':2,'maximum_seconds':1200}
    prepared={'parent':{'receipt_sha256':'Adam60_CPU_FIXTURE'},'source_sha256':sources,'budget':budget}
    probe={'parent':prepared['parent'],'source_sha256':sources,'CPU_FIXTURE_ONLY':True}
    actual={'status':'ACTUAL_MARGIN_VJP_READONLY_COMPLETE','optimizer_updates':0,'generation_calls':0,'scope_sha256':e.inherited.scope_digest(probe),'source_sha256':sources,'CPU_FIXTURE_ONLY':True}
    records={}
    def record(name,value):
        path=tmp_path/(name+'.json');path.write_text(json.dumps(value));return {'path':str(path),'sha256':e.inherited.digest(path)}
    records['readonly_scope']=record('scope',probe);records['readonly_actual']=record('actual',actual)
    records['readonly_actual_review']=record('actual_review',{'actual_proof_sha256':records['readonly_actual']['sha256'],'verdict':'PASS_WITH_LIMITATIONS','reviewer_id':'CPU_FIXTURE','reviewer_kind':'isolated_codex'})
    records['CPU_transaction_proof']=record('cpu',{'schema':'class-margin-aux-v2-cpu-transactions','status':'TESTED_CPU','native_execution':False,'tested_source_sha256':sources})
    certificate={'schema':'independent-class-margin-v2-compatibility-admission-v1','status':'ADMITTED_WITH_EXPLICIT_NATIVE_LIMITATIONS','policy':a.POLICY,
        'preparation_sha256':e.inherited.scope_digest(prepared),'training_source_sha256':sources,'training_budget':budget,
        'native_joint_update_measured':False,'native_onpolicy_backward_measured':False,'independent_reviewer_id':'CPU_FIXTURE','independent_reviewer_kind':'isolated_codex',
        'dynamic_group_gates':{'rollout_selected_vs_full_max_abs':.002,'own_old_current_max_abs':1e-5,'finite_nonzero_final_gradient':True,'single_persistent_Adam':True},**records}
    scope={'scope_kind':'FUTURE_EXACT_CLASS_MARGIN_AUX_V2_TRAIN','source_sha256':sources,'budget':budget,'admission_certificate_sha256':e.inherited.scope_digest(certificate)}
    human={'authorized':True,'action':'finite_class_margin_aux_v2_train','scope_sha256':e.inherited.scope_digest(scope),'human_instruction_sha256':'CPU_FIXTURE_NOT_PERMISSION','accepts_unmeasured_native_joint_onpolicy':True}
    review={'scope_sha256':e.inherited.scope_digest(scope),'verdict':'PASS_WITH_LIMITATIONS','reviewer_id':'CPU_FIXTURE','reviewer_kind':'isolated_codex','accepts_unmeasured_native_joint_onpolicy':True}
    return prepared,scope,certificate,human,review


def test_admission_certificate_is_distinct_from_unmeasured_training_qualification(tmp_path):
    from mri_rl import class_margin_aux_v2_entry as e
    prepared,scope,certificate,human,review=admission_fixture(tmp_path)
    assert e.validate_admission(prepared,scope,certificate)
    assert a.authorize_launch(scope,certificate,human,review)
    with pytest.raises(ValueError):a.authorize_launch(scope,certificate,{},review)
    for field,value in [('schema','FORGED'),('status','ACTUAL_CLASS_MARGIN_AUX_V2_NUMERICALLY_QUALIFIED'),('native_joint_update_measured',True)]:
        with pytest.raises(ValueError):a.authorize_launch(scope,{**certificate,field:value},human,review)
    for value in ({**certificate['training_budget'],'maximum_optimizer_updates':1},{**certificate['training_budget'],'maximum_attempted_groups':5}):
        with pytest.raises(ValueError):e.validate_admission(prepared,scope,{**certificate,'training_budget':value})
    with pytest.raises(ValueError):e.validate_admission(prepared,scope,{**certificate,'readonly_actual':{**certificate['readonly_actual'],'sha256':'FOREIGN'}})
    actual=json.loads(Path(certificate['readonly_actual']['path']).read_text());actual['status']='PARTIAL_ACTUAL_GROUP_UNINFORMATIVE'
    Path(certificate['readonly_actual']['path']).write_text(json.dumps(actual));certificate['readonly_actual']['sha256']=e.inherited.digest(certificate['readonly_actual']['path'])
    with pytest.raises(ValueError,match='old inactive'):e.validate_admission(prepared,scope,certificate)


def test_valid_limited_admission_reaches_fake_backend_factory_not_native(tmp_path):
    from mri_rl import class_margin_aux_v2_entry as e
    prepared,scope,certificate,human,review=admission_fixture(tmp_path)
    assert e.validate_admission(prepared,scope,certificate)
    B,parent,sha,rows=cpu_fixture(tmp_path)
    calls=[]
    def factory(*args):calls.append('CPU_FAKE_FACTORY');raise RuntimeError('factory reached without model')
    def authorize(context):a.authorize_launch(scope,certificate,human,review)
    from mri_rl import class_margin_aux_v2_driver as d
    with pytest.raises(RuntimeError,match='factory reached'):
        d.run(output=tmp_path/'admitted',schedule=rows,bindings={'CPU_FIXTURE':'admitted'},backend_factory=factory,authorize=authorize,
              parent_checkpoint=parent,parent_receipt_sha256=sha,config={**d.old.PROFILE,'group_size':2,'max_tokens':256},row_validator=lambda row:None)
    assert calls==['CPU_FAKE_FACTORY']


def test_native_auxiliary_gradient_coefficients_match_joint_loss_finite_difference(monkeypatch):
    from types import SimpleNamespace as NS
    from mri_rl import paired_class_margin_probe_v2 as shared
    # Two shared parameters, two image-conditioned candidate Jacobians. Prefix
    # is excluded by the shared VJP; fixture proves ALPHA applied exactly once.
    J=np.array([[[1.,.2],[-.3,.5]],[[.4,-.2],[.1,.8]]]);w=np.array([.3,-.7]);bias=np.array([[-1.,-2.],[-3.,-1.]])
    scores=bias+np.einsum('ijk,k->ij',J,w)
    calls=[]
    def vjp(backend,inputs,plan,index,**kwargs):
        calls.append(index);return {'mean_scores':scores[index].tolist(),'gradient':{'w':J[index,index]-J[index,1-index]}}
    monkeypatch.setattr(shared,'serial_margin_vjp',vjp)
    b=a.ClassMarginAuxBackend.__new__(a.ClassMarginAuxBackend)
    b.companions={'main':[{'reward_model':{'ground_truth':'FLAIR'}},{'reward_model':{'ground_truth':'contrast-enhanced T1'}}]}
    b.aux_token_plan={};b._companion_inputs=lambda row:row;b.before_unit=lambda name:None;b.compute=a.ComputeBudget()
    b.mx=NS(eval=lambda *args:None,clear_cache=lambda:None)
    b.tree_map=lambda f,*trees:{key:f(*(tree[key] for tree in trees)) for key in trees[0]};b._checked_gradient=lambda gradient:float(np.linalg.norm(gradient['w']))
    gradient,evidence=b.auxiliary_gradient({'id':'main'})
    def loss(weights):return a.objective.ALPHA*a.objective.classification_loss(bias+np.einsum('ijk,k->ij',J,weights))
    eps=1e-5;finite=np.array([(loss(w+np.eye(2)[i]*eps)-loss(w-np.eye(2)[i]*eps))/(2*eps) for i in range(2)])
    np.testing.assert_allclose(gradient['w'],finite,rtol=1e-8,atol=1e-10)
    assert calls==[0,1] and evidence['alpha']==.1


def test_restored_compute_budget_rejects_malformed_or_lost_counters():
    valid=dict(a.ComputeBudget().used)
    for key,value in [('backward_calls',-1),('backward_calls',.5),('backward_calls',True),('backward_calls',21),('full_TF_forwards',41)]:
        with pytest.raises(ValueError):a.ComputeBudget(used={**valid,key:value})
    for malformed in ({'backward_calls':-1},{**valid,'unknown_counter':0},{key:value for key,value in valid.items() if key!='backward_calls'}):
        with pytest.raises(ValueError):a.ComputeBudget(used=malformed)
    legacy={'full_TF_forwards':0,'backward_calls':0,'teacherforced_positions':0}
    assert a.ComputeBudget(used=legacy).used==valid
    with pytest.raises(ValueError):a.ComputeBudget(used={**legacy,'backward_calls':1})
    full={**valid,'backward_calls':20};restored=a.ComputeBudget(used=full)
    with pytest.raises(ValueError):restored.backward()
    assert restored.used==full
