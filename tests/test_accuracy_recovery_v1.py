"""CPU fixtures only for VERO_TO_MRI_ACCURACY_RECOVERY_V1 core: original-schema Adam52 anchor, budget, authorization, gradient archive.
Synthetic numpy parents; never a model, MLX, MRI pixel, final100 or new16 artifact."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pytest
from mri_rl import accuracy_recovery_v1 as ar
from mri_rl import continuous_rl as original
from mri_rl.optimizer_checkpoint import load_optimizer_checkpoint

ROOT=Path(ar.__file__).resolve().parents[2]
LENGTHS={'F':213,'T':213}


def sha(text):return hashlib.sha256(text.encode()).hexdigest()


# ------------------------------------------------------------------ synthetic ORIGINAL-schema boundary114/Adam52 parent
def original_parent(tmp_path,backend_cls,name='parent',completed=114,updates=52,binding=None):
    """Written by the ORIGINAL continuous_rl._publish (no absolute_* keys, no adaptive_parent)."""
    class WithConfig(backend_cls):
        def save_adapter(self,path):super().save_adapter(path);Path(path).with_name('adapter_config.json').write_text('{}')
    b=WithConfig(None,{});b.state['step']=np.array(updates,dtype=np.uint64);b.state['weight']={'m':np.array([.4,-.3]),'v':np.array([.2,.1])}
    out=tmp_path/name;out.mkdir();pending=out/'.pending';pending.mkdir()
    binding=binding or {'experiment':{'arm':{'name':'P_calibrated','reward':'medgrpo_eq3','sampling':'paced'},'seed':13,'calibration_sha256':sha('cal')},'configuration':{'group_size':4}}
    receipt=original._publish(out,pending,b,binding,completed,updates,'prior-boundary-sha',{'updated':True,'record_id':'r','schedule_record_id':'r'})
    return receipt['path'],original.digest(receipt['path']/'receipt.json')


def frozen_for(path,receipt_sha):
    r=json.loads((path/'receipt.json').read_text())
    return {**ar.FROZEN_PARENT,'receipt_sha256':receipt_sha,'adapter_sha256':r['files_sha256']['adapter.safetensors']}


@pytest.fixture
def backend_cls():
    import importlib.util
    spec=importlib.util.spec_from_file_location('base',Path(__file__).with_name('test_continuous_rl_driver.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m.NumericBackend


def test_frozen_parent_identity_is_exact_and_not_the_lost_adam60():
    assert ar.FROZEN_PARENT['adapter_sha256']=='c70b426b58191446545ba6333f7595e96f4fdb4cc689ddbaf5206e276b329db6'
    assert ar.FROZEN_PARENT['receipt_sha256']=='e0696a8186d413b2bb35034b5c5eb2568ddc0b05e1b719778de42fa491cc882a'
    assert (ar.FROZEN_PARENT['groups'],ar.FROZEN_PARENT['updates'],ar.FROZEN_PARENT['boundary'])==(114,52,'boundary_0114')
    assert ar.EXPERIMENT=='VERO_TO_MRI_ACCURACY_RECOVERY_V1' and ar.EXPERIMENT!=ar.BUNDLE_EXPERIMENT


def test_anchor_authenticates_original_schema_without_fabricating_nested_offsets(tmp_path,backend_cls):
    path,receipt_sha=original_parent(tmp_path,backend_cls);frozen=frozen_for(path,receipt_sha)
    anchor=ar.adam52_parent_anchor(path,receipt_sha,frozen=frozen)
    assert anchor['groups']==114 and anchor['updates']==52 and anchor['parent_schema']=='ORIGINAL_NON_NESTED_CONTINUOUS_RL_BOUNDARY_V1'
    assert 'local_groups' not in anchor and 'local_updates' not in anchor and 'absolute_completed_groups' not in json.loads((path/'receipt.json').read_text())
    assert anchor['receipt_sha256']==receipt_sha and anchor['adapter_sha256']==frozen['adapter_sha256']
    assert set(anchor['required_files_sha256'])=={'adapter.safetensors','adapter_config.json','optimizer/manifest.json','optimizer/state.safetensors'}
    # the existing optimizer-state loader accepts the anchor's original binding unchanged (no nested rewrite)
    b=backend_cls(None,{})
    state,rng,_=load_optimizer_checkpoint(path/'optimizer',expected_bindings={'run':anchor['original_binding'],'adapter_sha256':anchor['adapter_sha256'],'trainable_signature':b.trainable_signature()},load_arrays=b.load_arrays)
    assert int(state['step'])==52


@pytest.mark.parametrize('case',['wrong_frozen_receipt','nested_adaptive_parent','absolute_counters','wrong_dir','groups','updates','missing_file','tampered_file','manifest_updates','manifest_binding','symlink','schema'])
def test_anchor_rejects_every_deviation(tmp_path,backend_cls,case):
    path,receipt_sha=original_parent(tmp_path,backend_cls);frozen=frozen_for(path,receipt_sha)
    receipt_path=path/'receipt.json';receipt=json.loads(receipt_path.read_text())
    def rewrite(edit):
        edit(receipt);receipt_path.write_text(json.dumps(receipt,sort_keys=True))
        return original.digest(receipt_path)
    if case=='wrong_frozen_receipt':frozen={**frozen,'receipt_sha256':sha('other')};receipt_sha=sha('other')
    elif case=='nested_adaptive_parent':receipt_sha=rewrite(lambda r:r['binding']['experiment'].__setitem__('adaptive_parent',{'groups':105,'updates':48}))
    elif case=='absolute_counters':receipt_sha=rewrite(lambda r:r.update(absolute_completed_groups=114,absolute_optimizer_updates=52))
    elif case=='wrong_dir':path=path.rename(path.with_name('boundary_0113'))
    elif case=='groups':receipt_sha=rewrite(lambda r:r.__setitem__('completed_groups',113))
    elif case=='updates':receipt_sha=rewrite(lambda r:r.__setitem__('optimizer_updates',51))
    elif case=='schema':receipt_sha=rewrite(lambda r:r.__setitem__('schema','other'))
    elif case=='missing_file':(path/'adapter_config.json').unlink()
    elif case=='tampered_file':(path/'optimizer/state.safetensors').write_bytes(b'tampered')
    elif case=='manifest_updates':
        m=json.loads((path/'optimizer/manifest.json').read_text());m['optimizer_updates']=53;(path/'optimizer/manifest.json').write_text(json.dumps(m))
    elif case=='manifest_binding':
        m=json.loads((path/'optimizer/manifest.json').read_text());m['bindings']['run']={'x':1};(path/'optimizer/manifest.json').write_text(json.dumps(m))
    elif case=='symlink':
        (path/'x').write_text('x');(path/'adapter_alias').symlink_to(path/'x')
        (path/'adapter.safetensors').unlink();(path/'adapter.safetensors').symlink_to(path/'x')
    with pytest.raises(ValueError):ar.adam52_parent_anchor(path,receipt_sha,frozen=frozen)


def test_default_frozen_identity_refuses_a_synthetic_parent(tmp_path,backend_cls):
    path,receipt_sha=original_parent(tmp_path,backend_cls)
    with pytest.raises(ValueError,match='frozen'):ar.adam52_parent_anchor(path,receipt_sha)


def test_nested_adam60_shape_is_not_accepted_as_adam52(tmp_path,backend_cls):
    b=backend_cls(None,{});b.state['step']=np.array(60,dtype=np.uint64);b.state['weight']={'m':np.array([.4,-.3]),'v':np.array([.2,.1])}
    out=tmp_path/'adam60';out.mkdir();pending=out/'.pending';pending.mkdir()
    from mri_rl import paired_z045_continuation_driver_v1 as old
    r=old._publish(out,pending,b,{'experiment':{'adaptive_parent':{'groups':121,'updates':56}}},8,4,'prior',{'updated':True})
    with pytest.raises(ValueError):ar.adam52_parent_anchor(r['path'],old.digest(r['path']/'receipt.json'),frozen={**ar.FROZEN_PARENT,'receipt_sha256':old.digest(r['path']/'receipt.json')})


def test_parent_disk_snapshot_detects_any_change_and_extra_file(tmp_path,backend_cls):
    path,_=original_parent(tmp_path,backend_cls);before=ar.parent_disk_snapshot(path);ar.require_parent_unchanged(before,path)
    (path/'adapter_config.json').write_text('{"x":1}')
    with pytest.raises(ValueError,match='Parent'):ar.require_parent_unchanged(before,path)
    path2,_=original_parent(tmp_path,backend_cls,name='parent2');before2=ar.parent_disk_snapshot(path2);(path2/'new.txt').write_text('x')
    with pytest.raises(ValueError,match='Parent'):ar.require_parent_unchanged(before2,path2)


# ------------------------------------------------------------------ budget
def test_budget_is_shared_f4t4_one_step_per_arm_with_integrated_preflight_and_no_qualification_mode():
    from mri_rl import shared_rollout_gspo_core_v1 as shared
    b=ar.derive_budget(LENGTHS);s=shared.derive_budget(LENGTHS,'train')
    assert b['experiment']==ar.EXPERIMENT and b['optimizer_updates']==2 and b['optimizer_updates_per_arm']==1 and b['new_generation_per_arm']==0
    assert all(b[k]==s[k] for k in shared.CHARGEABLE) and b['responses']==8 and b['backward_calls']==16 and b['generated_response_tokens']==8*256
    assert b['integrated_preflight']['forwards']==0 and b['integrated_preflight']['generation']==0 and b['integrated_preflight']['optimizer_updates']==0
    assert b['gradient_archive_before_each_step'] is True and b['automatic_retry'] is False and b['simultaneous_models']==1
    assert ar.counter_limits(b)=={k:s[k] for k in shared.CHARGEABLE}
    assert not hasattr(ar,'MODES') or ar.MODES==('run',)
    with pytest.raises(ValueError):ar.derive_budget({'F':213})


# ------------------------------------------------------------------ authorization
def source_closure():return {str(ROOT/name):ar.file_sha256(ROOT/name) for name in ar.OWN_SOURCES}


def scope_fixture(parent=None,**edit):
    scope={'scope_kind':ar.SCOPE_KIND,'experiment':ar.EXPERIMENT,'execution_subtype':ar.EXECUTION_SUBTYPE,'numerical_policy':ar.POLICY,'recipe_label':ar.RECIPE_LABEL,
           'implemented_by':'claude_code','budget':ar.derive_budget(LENGTHS),'source_sha256':source_closure(),'input_sha256':sha('input'),
           'parent':parent or {'receipt_sha256':ar.FROZEN_PARENT['receipt_sha256'],'adapter_sha256':ar.FROZEN_PARENT['adapter_sha256']},'output':'/x/local_data/run','invocation':{'resume':False}}
    scope.update(edit);return scope


def signed(scope,**edit):
    binding=ar.scope_digest(scope)
    human={'authorized':True,'action':ar.ACTION,'scope_sha256':binding,'human_instruction_sha256':sha('fresh human instruction')}
    review={'verdict':'PASS','scope_sha256':binding,'reviewer_id':'codex-1','reviewer_kind':'isolated_codex','source_sha256':scope['source_sha256'],'input_sha256':scope['input_sha256']}
    return human,review


def ledger(tmp_path):return ar.consume_once(tmp_path/'ledger')


def test_authorization_issues_a_scope_bound_token_and_is_consumed_once(tmp_path):
    scope=scope_fixture();human,review=signed(scope);consumed=ledger(tmp_path)
    token=ar.authorize_launch(scope,human,review,consumed=consumed)
    assert isinstance(token,ar.Authorization) and token.scope_sha256==ar.scope_digest(scope) and token.input_sha256==scope['input_sha256']
    with pytest.raises(ValueError,match='already consumed'):ar.authorize_launch(scope,human,review,consumed=consumed)


def test_authorization_token_cannot_be_constructed_directly():
    with pytest.raises(ValueError):ar.Authorization(object(),scope_fixture())
    with pytest.raises(TypeError):ar.authorize_launch(scope_fixture(),*signed(scope_fixture()))     # consumed ledger is mandatory, not optional


@pytest.mark.parametrize('case',['flag_top','flag_nested','kind','experiment','budget','parent','human_action','human_scope','human_false','human_instruction','review_scope','review_verdict','review_kind_same_as_implementer','review_source','review_input','source_changed','own_source_missing','mode_key'])
def test_authorization_is_fail_closed(tmp_path,case):
    scope=scope_fixture();human,review=signed(scope)
    if case=='flag_top':scope['approved']=True
    elif case=='flag_nested':scope['invocation']['training_authorized']=True
    elif case=='kind':scope['scope_kind']='PREPARATION_ONLY'
    elif case=='experiment':scope['experiment']='SHARED_ROLLOUT_BATCHED_ONE_STEP_GSPO_MECHANISM_CONTROL'
    elif case=='budget':scope['budget']={**scope['budget'],'optimizer_updates':4}
    elif case=='parent':scope['parent']={'receipt_sha256':sha('adam60'),'adapter_sha256':sha('adam60 adapter')}
    elif case=='human_action':human['action']='shared_rollout_gspo_one_step_train'
    elif case=='human_scope':human['scope_sha256']=sha('other')
    elif case=='human_false':human['authorized']='true'
    elif case=='human_instruction':human['human_instruction_sha256']='x'
    elif case=='review_scope':review['scope_sha256']=sha('other')
    elif case=='review_verdict':review['verdict']='FAIL'
    elif case=='review_kind_same_as_implementer':review['reviewer_kind']='claude_code'
    elif case=='review_source':review['source_sha256']={**scope['source_sha256'],'/x.py':sha('x')}
    elif case=='review_input':review['input_sha256']=sha('other')
    elif case=='source_changed':scope['source_sha256']={**scope['source_sha256'],next(iter(scope['source_sha256'])):sha('stale')}
    elif case=='own_source_missing':scope['source_sha256']={k:v for k,v in scope['source_sha256'].items() if not k.endswith('accuracy_recovery_driver_v1.py')}
    elif case=='mode_key':scope['mode']='qualify'
    if case not in ('human_action','human_scope','human_false','human_instruction','review_scope','review_verdict','review_kind_same_as_implementer','review_source','review_input'):
        human,review=signed(scope)
        if case=='source_changed':review['source_sha256']=scope['source_sha256']
    with pytest.raises(ValueError):ar.authorize_launch(scope,human,review,consumed=ledger(tmp_path))
    assert not list((tmp_path/'ledger').glob('authorization_consumed_*')) if (tmp_path/'ledger').exists() else True    # a refused authorization consumes nothing


def test_authorization_check_rehashes_the_whole_closure_each_phase(tmp_path):
    copy_dir=tmp_path/'src';copy_dir.mkdir();target=copy_dir/'one.py';target.write_text('a=1\n')
    scope=scope_fixture(source_sha256={**source_closure(),str(target):ar.file_sha256(target)});human,review=signed(scope)
    token=ar.authorize_launch(scope,human,review,consumed=ledger(tmp_path));token.check('load')
    target.write_text('a=2\n')
    with pytest.raises(ValueError,match='source'):token.check('rollout')


def test_no_flag_or_no_mlx_import_and_cpu_prepare_modules_import_without_mlx():
    assert not any(n=='mlx' or n.startswith('mlx.') for n in sys.modules)


# ------------------------------------------------------------------ gradient archive (full arrays BEFORE the step; independent recompute)
class Planned:
    def __init__(self,ids,advantages):
        class Item:
            def __init__(s,i):s.response=type('R',(),{'response_id':i})();s.advantage=advantages.get(i,0.);s.weight=.125 if i in advantages else 0.
        self.arm='A';self.admitted_groups=1 if ids else 0;self.admitted_response_ids=tuple(ids);self.plan=type('P',(),{'items':[Item(i) for i in ('F0','F1','F2','F3','T0','T1','T2','T3')]})()


class Backend:
    def save_arrays(self,path,arrays):
        with Path(path).open('wb') as handle:np.savez(handle,**arrays)
    def load_arrays(self,path):
        with np.load(path,allow_pickle=False) as data:return {k:data[k].copy() for k in data.files}


PROFILE={'optimizer':{'name':'Adam','learning_rate':1e-5,'betas':[.9,.999],'eps':1e-8,'bias_correction':False}}


def archive(tmp_path,arm,gradient,ids=('F0','F1'),advantages=None):
    anchor={'receipt_sha256':sha('r'),'adapter_sha256':sha('a'),'optimizer_manifest_sha256':sha('m'),'optimizer_state_sha256':sha('s'),'groups':114,'updates':52}
    planned=Planned(ids if gradient is not None else (),advantages or {'F0':1.,'F1':-1.})
    decision={'update':gradient is not None,'reason':'ADMITTED_FINITE_NONZERO_GRADIENT' if gradient is not None else 'NO_ADMITTED_GROUP'}
    norm=ar.gradient_l2({k:np.asarray(v) for k,v in ar._flatten(gradient)}) if gradient is not None else 0.
    return ar.archive_gradient(backend=Backend(),directory=tmp_path/'gradients',arm=arm,planned=planned,gradient=gradient,evidence={'gradient_l2':norm,'weighted_loss':.2,'backward_calls':len(ids)},
        bundle_sha256=sha('bundle'),parent_identity=sha('identity'),anchor=anchor,parent_path='/private/parent',profile=PROFILE,decision=decision,
        audit={'transaction_validate_unchanged_parameters':True,'adam_digest_equals_parent':True,'rng_digest_equals_parent':True})


def test_active_arm_gradient_arrays_are_saved_with_hashes_and_recompute_exactly(tmp_path):
    gradient={'language_model':{'q':{'lora_a':np.array([[1.,2.],[3.,4.]],dtype=np.float32),'lora_b':np.array([[.5,-.5]],dtype=np.float32)}},'blocks':[np.array([1.,0.],dtype=np.float32)]}
    meta=archive(tmp_path,'A',gradient)
    stored=json.loads((tmp_path/'gradients/arm_A_gradient_meta_PRIVATE.json').read_text())
    assert stored==json.loads(json.dumps(meta)) and stored['tensors_sha256']==ar.file_sha256(tmp_path/'gradients'/stored['tensors_file']) and stored['tensor_count']==3
    assert stored['advantages']['F0']==1. and stored['advantages']['T0'] is None and stored['weights']['F0']==.125 and stored['parent']['updates']==52
    flat=ar.load_gradient_archive(tmp_path/'gradients/arm_A_gradient_meta_PRIVATE.json',Backend().load_arrays)
    assert set(flat)=={'language_model.q.lora_a','language_model.q.lora_b','blocks.0'}
    np.testing.assert_array_equal(flat['language_model.q.lora_a'],gradient['language_model']['q']['lora_a'])
    assert ar.gradient_l2(flat)==pytest.approx(np.sqrt(1+4+9+16+.25+.25+1))


def test_inactive_arm_gradient_is_explicit_none_without_invented_arrays_or_advantages(tmp_path):
    meta=archive(tmp_path,'B',None)
    assert meta['gradient'] is None and meta['tensors_file'] is None and meta['tensors_sha256'] is None and meta['reason']=='NO_ADMITTED_GROUP'
    assert all(v is None for v in meta['advantages'].values()) and all(v==0. for v in meta['weights'].values())
    assert not list((tmp_path/'gradients').glob('*.safetensors'))
    assert ar.load_gradient_archive(tmp_path/'gradients/arm_B_gradient_meta_PRIVATE.json',Backend().load_arrays) is None
    with pytest.raises(ValueError,match='update'):                       # a gradient-less record cannot coexist with an update decision
        ar.archive_gradient(backend=Backend(),directory=tmp_path/'g2',arm='B',planned=Planned(('F0',),{'F0':1.}),gradient=None,evidence={},bundle_sha256=sha('b'),parent_identity=sha('i'),
            anchor={'receipt_sha256':sha('r'),'adapter_sha256':sha('a'),'optimizer_manifest_sha256':sha('m'),'optimizer_state_sha256':sha('s'),'groups':114,'updates':52},parent_path='/p',profile=PROFILE,decision={'update':True,'reason':'x'},audit={})


def test_archive_is_never_overwritten_and_tampering_is_detected(tmp_path):
    gradient={'w':np.array([1.,2.],dtype=np.float32)};archive(tmp_path,'A',gradient)
    with pytest.raises(FileExistsError):archive(tmp_path,'A',gradient)
    path=tmp_path/'gradients/arm_A_gradient_PRIVATE.safetensors';path.write_bytes(path.read_bytes()+b'x')
    with pytest.raises(ValueError,match='hash'):ar.load_gradient_archive(tmp_path/'gradients/arm_A_gradient_meta_PRIVATE.json',Backend().load_arrays)


def test_cosine_and_adam_delta_recompute_independently(tmp_path):
    ga={'w':np.array([1.,0.],dtype=np.float32)};gb={'w':np.array([1.,1.],dtype=np.float32)}
    archive(tmp_path,'A',ga);archive(tmp_path,'B',gb,ids=('F0','F1'))
    fa=ar.load_gradient_archive(tmp_path/'gradients/arm_A_gradient_meta_PRIVATE.json',Backend().load_arrays);fb=ar.load_gradient_archive(tmp_path/'gradients/arm_B_gradient_meta_PRIVATE.json',Backend().load_arrays)
    assert ar.cosine(fa,fb)==pytest.approx(1/np.sqrt(2)) and ar.cosine(fa,None) is None
    state={'weight':{'m':np.array([.4,-.3]),'v':np.array([.2,.1])},'step':np.array(52,dtype=np.uint64)}
    flat_state={'w.m':np.array([.4,-.3]),'w.v':np.array([.2,.1])}
    delta=ar.adam_delta(flat_state,fa,PROFILE['optimizer'])
    m=.9*np.array([.4,-.3])+.1*np.array([1.,0.]);v=.999*np.array([.2,.1])+.001*np.array([1.,0.])**2
    np.testing.assert_allclose(delta['w'],-1e-5*m/(np.sqrt(v)+1e-8),rtol=1e-6)
    assert ar.adam_delta(flat_state,None,PROFILE['optimizer']) is None
    with pytest.raises(ValueError,match='bias'):ar.adam_delta(flat_state,fa,{**PROFILE['optimizer'],'bias_correction':True})
