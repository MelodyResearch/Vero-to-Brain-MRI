"""CPU driver/entry fixtures for VERO_TO_MRI_ACCURACY_RECOVERY_V1 (fake numpy model; original-schema synthetic Adam52 parent). Never native or MRI evidence."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import uuid
import numpy as np
import pytest
from mri_rl import accuracy_recovery_v1 as ar
from mri_rl import accuracy_recovery_driver_v1 as drv
from mri_rl import accuracy_recovery_entry_v1 as entry
from mri_rl import continuous_rl as original
from mri_rl.optimizer_checkpoint import load_optimizer_checkpoint


def load(name):
    spec=importlib.util.spec_from_file_location(name[:-3],Path(__file__).with_name(name));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


f=load('test_shared_rollout_gspo_v1.py')        # Fake numpy backend, grader, rows
c=load('test_accuracy_recovery_v1.py')          # scope/signature helpers
c2=load('test_accuracy_recovery_round2_v1.py')
ROOT=Path(ar.__file__).resolve().parents[2]
FAULT=(ValueError,RuntimeError,AssertionError)


class Tree(f.Fake):
    """Fake backend whose gradient is a one-leaf tree so flat names line up with the Adam state ('weight.m'/'weight.v')."""
    out=None
    def arm_gradient(self,planned,own_old):
        self.current_arm=planned.arm;total,evidence=super().arm_gradient(planned,own_old);return ({'weight':total} if total is not None else None),evidence
    def apply_update(self,gradient):
        if Tree.out is not None:                # this arm's full archive must already be durable BEFORE the optimizer call
            directory=Tree.out/'gradients_PRIVATE';meta=json.loads((directory/f'arm_{self.current_arm}_gradient_meta_PRIVATE.json').read_text())
            assert meta['gradient'] is not None and (directory/meta['tensors_file']).exists() and ar.file_sha256(directory/meta['tensors_file'])==meta['tensors_sha256'],'gradient archive missing before step'
        return super().apply_update(gradient['weight'])


@pytest.fixture(autouse=True)
def reset():
    f.Fake.instances=[];f.Fake.rollouts=0;f.Fake.log=None;f.Fake.accuracy=f.ACCURACY;f.Fake.fail_update_arm=None;f.Fake.identities=[];Tree.out=None


def world(tmp_path):
    if (tmp_path/'adam52').exists():
        parent=tmp_path/'adam52/boundary_0114';return parent,original.digest(parent/'receipt.json'),json.loads((tmp_path/'rows.json').read_text())
    parent,receipt=c.original_parent(tmp_path,f.Fake,name='adam52')
    _,_,rows=f.parent_fixture(tmp_path)
    f.Fake.instances=[];(tmp_path/'rows.json').write_text(json.dumps(rows));return parent,receipt,rows


class Env:
    counter=0


def issue(tmp_path,parent,receipt,extra_sources=None,input_sha=None):
    frozen=c.frozen_for(parent,receipt);closure={**c.source_closure(),**(extra_sources or {})}
    scope=c.scope_fixture(parent={'receipt_sha256':receipt,'adapter_sha256':frozen['adapter_sha256']},source_sha256=closure,input_sha256=input_sha or c.sha('input'))
    human,review=c.signed(scope);Env.counter+=1;human['human_instruction_sha256']=c.sha(f'instruction {Env.counter} {uuid.uuid4().hex}')
    return ar.authorize_launch(scope,human,review,consumed=ar.consume_once(tmp_path/f'ledger{Env.counter}'),frozen=frozen),frozen


def launch(tmp_path,out,cls=Tree,resume=False,phase_gate=None,token=None,extra_sources=None):
    parent,receipt,rows=world(tmp_path)
    issued,frozen=issue(tmp_path,parent,receipt,extra_sources)
    token=token or issued;Tree.out=Path(out) if issubclass(cls,Tree) else None
    return drv.run(output=out,schedule=rows,bindings={'stage':'fixture','scope_sha256':issued.scope_sha256,'source_sha256':issued.source_sha256,'input_sha256':issued.input_sha256},
        backend_factory=cls,authorization=token,phase_gate=phase_gate or (lambda context:None),parent_checkpoint=parent,parent_receipt_sha256=receipt,grader=f.grader,
        reward_policy_sha256=c.sha('calibration'),budget=ar.derive_budget(c.LENGTHS),resume=resume,frozen=frozen)


def test_integrated_run_restores_authentic_original_parent_preflights_rolls_out_once_and_updates_each_arm_once(tmp_path):
    result=launch(tmp_path,tmp_path/'run');b=f.Fake.instances[-1];out=tmp_path/'run'
    assert result['status']=='SCHEDULE_COMPLETE' and result['optimizer_updates']==2 and f.Fake.rollouts==1 and len(f.Fake.instances)==1
    assert b.events.index('restore')<b.events.index('rollout')<b.events.index('score_old')<b.events.index('gradient_A')<b.events.index('gradient_B') and b.events.count('update')==2
    pre=json.loads((out/'preflight_PRIVATE.json').read_text())
    assert pre['adam_step']==52 and pre['forwards']==pre['generation']==pre['backward']==pre['optimizer_updates']==0 and 'NOT_A_QUALIFICATION' in pre['scope']
    assert f.Fake.identities[0]==f.Fake.identities[1]
    parent,receipt,_=world(tmp_path)
    for arm in 'AB':
        r=json.loads((out/f'arm_{arm}/boundary_0001/receipt.json').read_text())
        assert r['updated'] and r['optimizer_updates']==1 and r['absolute_optimizer_updates']==53 and r['absolute_completed_groups']==115 and r['binding']['experiment']['experiment']==ar.EXPERIMENT
        assert r['own_old_parity_max_abs']<=.002 and r['gradient_archive_tensors_sha256']
    np.testing.assert_array_equal(b.weight,np.load(parent/'adapter.safetensors')['weight'])
    done=json.loads((out/'complete_SAFE.json').read_text());assert done['final_state_equals_parent'] and done['parent_disk_unchanged'] and done['recipe']==ar.RECIPE_LABEL


def test_full_gradient_archives_are_durable_before_each_step_and_recompute_norm_cosine_and_adam_delta(tmp_path):
    launch(tmp_path,tmp_path/'run');out=tmp_path/'run';parent,receipt,_=world(tmp_path);bk=f.Fake(None,{})
    flat={a:ar.load_gradient_archive(out/f'gradients_PRIVATE/arm_{a}_gradient_meta_PRIVATE.json',bk.load_arrays) for a in 'AB'}
    for a in 'AB':
        meta=json.loads((out/f'gradients_PRIVATE/arm_{a}_gradient_meta_PRIVATE.json').read_text())
        assert ar.gradient_l2(flat[a])==pytest.approx(meta['gradient_l2_reported']) and meta['parent']['updates']==52 and meta['adam_step_before_update']==52
        assert meta['parent']['optimizer_state_sha256']==original.digest(parent/'optimizer/state.safetensors')
        assert set(meta['admitted_response_ids'])=={k for k,v in meta['advantages'].items() if v is not None}
    assert ar.cosine(flat['A'],flat['B']) is not None and -1<=ar.cosine(flat['A'],flat['B'])<=1 and not np.array_equal(flat['A']['weight'],flat['B']['weight'])
    state,_,_=load_optimizer_checkpoint(parent/'optimizer',expected_bindings={'run':json.loads((parent/'receipt.json').read_text())['binding'],'adapter_sha256':original.digest(parent/'adapter.safetensors'),
        'trainable_signature':bk.trainable_signature()},load_arrays=bk.load_arrays)
    flat_state={'weight.m':state['weight']['m'],'weight.v':state['weight']['v']};parent_weight=np.load(parent/'adapter.safetensors')['weight']
    for a in 'AB':
        delta=ar.adam_delta(flat_state,flat[a],{'name':'Adam','learning_rate':1e-5,'betas':[.9,.999],'eps':1e-8,'bias_correction':False})
        np.testing.assert_allclose(np.load(out/f'arm_{a}/boundary_0001/adapter.safetensors')['weight']-parent_weight,delta['weight'],rtol=1e-6,atol=1e-15)


def test_inactive_arm_has_explicit_none_gradient_no_update_no_resample_while_other_arm_updates(tmp_path):
    f.Fake.accuracy={'F':[1,1,1,1],'T':[0,0,0,0]}                  # same-prompt groups uniform; paired groups informative
    result=launch(tmp_path,tmp_path/'run');out=tmp_path/'run';parent,_,_=world(tmp_path)
    assert result['optimizer_updates']==1 and f.Fake.rollouts==1
    a=json.loads((out/'gradients_PRIVATE/arm_A_gradient_meta_PRIVATE.json').read_text())
    assert a['gradient'] is None and a['tensors_file'] is None and all(v is None for v in a['advantages'].values()) and not (out/'gradients_PRIVATE/arm_A_gradient_PRIVATE.safetensors').exists()
    receipt=json.loads((out/'arm_A/boundary_0001/receipt.json').read_text());assert receipt['updated'] is False and receipt['update_decision']['reason']=='NO_ADMITTED_GROUP'
    assert receipt['files_sha256']['adapter.safetensors']==original.digest(parent/'adapter.safetensors') or receipt['optimizer_updates']==0
    assert json.loads((out/'gradients_PRIVATE/arm_B_gradient_meta_PRIVATE.json').read_text())['tensors_file']


def test_own_old_parity_point_zero_zero_two_blocks_every_gradient_and_update(tmp_path):
    class Drift(Tree):
        def score_old(self,bundle):return {k:[v+.01 for v in vs] for k,vs in super().score_old(bundle).items()}
    with pytest.raises(ValueError,match='exceeds .002'):launch(tmp_path,tmp_path/'run',cls=Drift)
    b=f.Fake.instances[-1];out=tmp_path/'run'
    assert not any(e.startswith('gradient') or e=='update' for e in b.events) and not list(out.glob('arm_*')) and not (out/'gradients_PRIVATE').exists() and (out/'bundle_COMMITTED.json').exists()


@pytest.mark.parametrize('kind',['Adam','RNG'])
def test_backward_cannot_hide_adam_or_rng_mutation_by_restore(tmp_path,kind):
    class Corrupt(Tree):
        def arm_gradient(self,planned,old_logps):
            result=super().arm_gradient(planned,old_logps)
            self.state['weight']['m']=self.state['weight']['m']+.01 if kind=='Adam' else self.rng.random() and self.state['weight']['m']
            return result
    with pytest.raises(FAULT,match='pre-restore audit'):launch(tmp_path,tmp_path/'run',cls=Corrupt)
    assert not list((tmp_path/'run').glob('arm_*_COMMITTED.json')) and not (tmp_path/'run/gradients_PRIVATE').exists()


def test_failure_inside_the_optimizer_call_keeps_the_archive_rolls_back_and_is_never_retried(tmp_path):
    f.Fake.fail_update_arm=1
    with pytest.raises(RuntimeError,match='simulated failure'):launch(tmp_path,tmp_path/'run')
    out=tmp_path/'run';assert (out/'gradients_PRIVATE/arm_A_gradient_meta_PRIVATE.json').exists() and not (out/'arm_A/boundary_0001').exists()
    stop=json.loads(next(out.glob('arm_A_*_STOP_PRIVATE.json')).read_text());assert stop['rolled_back_model_Adam_RNG'] is True and stop['parent_disk_unchanged'] is True
    f.Fake.fail_update_arm=None
    with pytest.raises(RuntimeError,match='INCOMPLETE_ARM_A'):launch(tmp_path,out,resume=True)


# ------------------------------------------------------------------ authorization is not a skippable callback
def test_driver_rejects_callbacks_foreign_tokens_and_builds_no_backend(tmp_path):
    with pytest.raises(ValueError,match='Authorization token'):launch(tmp_path,tmp_path/'r1',token=lambda context:None)
    parent,receipt,_=world(tmp_path);other,_=issue(tmp_path,parent,receipt,input_sha=c.sha('different input'))
    with pytest.raises(ValueError,match='different scope'):launch(tmp_path,tmp_path/'r2',token=other)
    assert not f.Fake.instances
    with pytest.raises(TypeError):drv.run(output=tmp_path/'r3',schedule=[],bindings={},backend_factory=Tree,authorize=lambda context:None,parent_checkpoint=parent,parent_receipt_sha256=receipt,
        grader=f.grader,reward_policy_sha256=c.sha('c'),budget=ar.derive_budget(c.LENGTHS))


def test_resource_gate_cannot_replace_but_can_refuse_and_precedes_the_backend(tmp_path):
    def deny(context):raise PermissionError('resource/admission gate denied')
    with pytest.raises(PermissionError):launch(tmp_path,tmp_path/'run',phase_gate=deny)
    assert not f.Fake.instances and f.Fake.rollouts==0


def test_reviewed_source_change_between_phases_stops_before_any_step(tmp_path):
    src=tmp_path/'extra_source.py';src.write_text('x=1\n')
    class Edits(Tree):
        def generate_shared_rollout(self,rows,workdir):
            result=super().generate_shared_rollout(rows,workdir);src.write_text('x=2\n');return result
    with pytest.raises(ValueError,match='Reviewed source changed'):launch(tmp_path,tmp_path/'run',cls=Edits,extra_sources={str(src):ar.file_sha256(src)})
    assert not list((tmp_path/'run').glob('arm_*')) and f.Fake.rollouts==1


def test_parent_checkpoint_disk_is_immutable_across_the_run(tmp_path):
    parent,_,_=world(tmp_path)
    class Touch(Tree):
        def generate_shared_rollout(self,rows,workdir):
            result=super().generate_shared_rollout(rows,workdir);(parent/'adapter_config.json').write_text('{"mutated":true}');return result
    with pytest.raises(ValueError,match='Parent checkpoint disk changed'):launch(tmp_path,tmp_path/'run',cls=Touch)
    assert not list((tmp_path/'run').glob('arm_*_COMMITTED.json')) and not (tmp_path/'run/complete_SAFE.json').exists()


def test_parent_mutation_during_the_last_arm_is_caught_by_the_end_of_run_check(tmp_path):
    parent,_,_=world(tmp_path)
    class LateTouch(Tree):
        def apply_update(self,gradient):
            result=super().apply_update(gradient)
            if self.events.count('update')==2:(parent/'adapter_config.json').write_text('{"late":true}')
            return result
    with pytest.raises(ValueError,match='Parent checkpoint disk changed'):launch(tmp_path,tmp_path/'run',cls=LateTouch)
    assert not (tmp_path/'run/complete_SAFE.json').exists()


def test_completed_run_resumes_without_a_backend_and_existing_output_needs_resume(tmp_path):
    launch(tmp_path,tmp_path/'run');built=len(f.Fake.instances)
    with pytest.raises(FileExistsError):launch(tmp_path,tmp_path/'run')
    assert launch(tmp_path,tmp_path/'run',resume=True)['status']=='SCHEDULE_COMPLETE' and len(f.Fake.instances)==built and f.Fake.rollouts==1


def test_preflight_record_is_stable_on_resume_and_tampering_is_refused(tmp_path):
    def stop_after_preflight(context):
        if context['phase']=='rollout':raise PermissionError('stop before rollout')
    with pytest.raises(PermissionError):launch(tmp_path,tmp_path/'run',phase_gate=stop_after_preflight)
    path=tmp_path/'run/preflight_PRIVATE.json';record=json.loads(path.read_text());assert f.Fake.rollouts==0 and record['adapter_readback_sha256']
    record['rng_digest']='0'*64;path.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='Preflight differs'):launch(tmp_path,tmp_path/'run',resume=True)


# ------------------------------------------------------------------ entry / CPU prepare
def test_modules_import_and_prepare_run_without_mlx():
    code=("import sys;sys.path.insert(0,'src');import mri_rl.accuracy_recovery_v1,mri_rl.accuracy_recovery_driver_v1,mri_rl.accuracy_recovery_entry_v1 as e;"
          "p=e.prepare({'runtime_spec':'/nonexistent'});assert p['blockers'] and not any(n=='mlx' or n.startswith('mlx.') for n in sys.modules);print('OK')")
    done=subprocess.run([sys.executable,'-I','-c',code],cwd=ROOT,capture_output=True,text=True);assert done.returncode==0 and 'OK' in done.stdout,done.stderr


def test_prepare_with_pending_runtime_reports_blockers_emits_no_scope_and_no_model_pass(tmp_path):
    prepared=entry.prepare({'runtime_spec':str(tmp_path/'missing.json')})
    assert prepared['blockers']==['RUNTIME_SPEC_NOT_PRESENT']
    assert prepared['model_PASS'] is False and prepared['training_authorized'] is False and prepared['model_loads_this_preparation']==0 and prepared['required_next_steps']
    with pytest.raises(ValueError,match='pending'):entry.launch_scope(prepared)
    (tmp_path/'runtime.json').write_text(json.dumps({'parent_checkpoint':str(tmp_path/'absent_parent')}))
    partial=entry.prepare({'runtime_spec':str(tmp_path/'runtime.json')});assert any(b.startswith('RUNTIME_KEYS_NOT_BOUND') for b in partial['blockers'])


def test_cli_prepare_writes_actionable_blocked_preparation_under_the_canonical_root_only(tmp_path,capsys):
    R,W=c2.make_repo(tmp_path);base=R/'local_data/accuracy_recovery_v1';base.mkdir(parents=True);out=base/'prep.json'
    (tmp_path/'spec.json').write_text(json.dumps({'runtime_spec':str(tmp_path/'none.json')}))
    entry.main(['prepare','--spec',str(tmp_path/'spec.json'),'--output',str(out)],start=W)
    printed=json.loads(capsys.readouterr().out);assert printed['status']=='PREPARED_BLOCKED_PENDING_INPUTS' and printed['model_PASS'] is False and printed['model_loads']==0
    assert out.exists() and not list(base.glob('*_scope.json'))
    for bad in (tmp_path/'elsewhere.json',W/'local_data/accuracy_recovery_v1/prep.json'):
        with pytest.raises(ValueError):entry.main(['prepare','--spec',str(tmp_path/'spec.json'),'--output',str(bad)],start=W)


@pytest.mark.parametrize('text',['/data/final100/outputs.json','/data/new16/images/x.png','/x/HOLDOUT16/labels.csv'])
def test_forbidden_final100_and_new16_inputs_are_refused_by_name(text):
    with pytest.raises(ValueError,match='Forbidden input'):entry.prepare({'runtime_spec':'/nonexistent','schedule':text})


def test_synthetic_parent_is_refused_by_the_entry_default_frozen_identity_and_run_needs_artifacts(tmp_path):
    parent,receipt,rows=world(tmp_path)
    (tmp_path/'runtime.json').write_text(json.dumps({'parent_checkpoint':str(parent),'parent_receipt_sha256':receipt,'calibration':str(tmp_path/'cal.json'),'cache_dir':str(tmp_path),'cache_certificate':str(tmp_path/'cal.json'),
        'cache_certificate_sha256':'0'*64,'model':str(tmp_path),'contract':'c','global_lock':str(tmp_path/'lock'),'runtime_source_sha256':{'/x.py':'0'*64}}))
    (tmp_path/'cal.json').write_text('{}')
    spec={'runtime_spec':str(tmp_path/'runtime.json'),'schedule':str(tmp_path/'rows.json'),'output':str(ROOT/'local_data/accuracy_recovery_v1/x'),'expected_image_sha256':{},'expected_parent_arm':{'name':'P_calibrated'}}
    with pytest.raises(ValueError,match='frozen Adam52'):entry.prepare(spec)
    from types import SimpleNamespace as NS
    with pytest.raises(ValueError,match='NOT_AUTHORIZED'):entry.execute(NS(spec=tmp_path/'spec.json'))
    assert not any(n=='mlx' or n.startswith('mlx.') for n in sys.modules)
