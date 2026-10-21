"""Codex findings 1, 2 and 4: gA,gB,cosine BEFORE the first optimizer step; transaction validation after backward; honest preflight."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest
from mri_rl import accuracy_recovery_v1 as ar
from mri_rl.optimizer_checkpoint import load_optimizer_checkpoint


def load(name):
    spec=importlib.util.spec_from_file_location(name[:-3],Path(__file__).with_name(name));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


d=load('test_accuracy_recovery_driver_v1.py')
f,c,Tree,launch,world=d.f,d.c,d.Tree,d.launch,d.world
FAULT=(ValueError,RuntimeError,AssertionError)
reset=d.reset


def test_both_gradients_and_the_comparison_are_durable_before_the_first_optimizer_step(tmp_path):
    seen=[]
    class Watch(Tree):
        def apply_update(self,gradient):
            out=Tree.out;seen.append(((out/'comparison_PRIVATE.json').exists(),all((out/f'gradients_PRIVATE/arm_{a}_gradient_meta_PRIVATE.json').exists() for a in 'AB'),
                                     self.events.count('gradient_A'),self.events.count('gradient_B')))
            assert gradient['weight'] is not self.computed['weight'],'step must use the gradient read back from disk, not the in-memory tree'
            return super().apply_update(gradient)
        def arm_gradient(self,planned,own_old):
            result=super().arm_gradient(planned,own_old);self.computed=result[0];return result
    launch(tmp_path,tmp_path/'run',cls=Watch);b=f.Fake.instances[-1]
    assert seen==[(True,True,1,1),(True,True,1,1)] and b.events.index('gradient_B')<b.events.index('update')
    comparison=json.loads((tmp_path/'run/comparison_PRIVATE.json').read_text())
    assert comparison['cosine'] is not None and comparison['norms']['A']>0 and comparison['norms']['B']>0 and comparison['cosine_reason'] is None
    assert comparison['bundle_sha256']==json.loads((tmp_path/'run/bundle_COMMITTED.json').read_text())['bundle_sha256']


def test_if_the_B_gradient_fails_there_are_zero_updates_and_no_boundaries(tmp_path):
    class BFails(Tree):
        def arm_gradient(self,planned,own_old):
            if planned.arm=='B':raise RuntimeError('simulated B backward failure')
            return super().arm_gradient(planned,own_old)
    with pytest.raises(RuntimeError,match='simulated B backward'):launch(tmp_path,tmp_path/'run',cls=BFails)
    b=f.Fake.instances[-1];out=tmp_path/'run'
    assert b.events.count('update')==0 and not list(out.glob('arm_*')) and not list(out.glob('arm_*_COMMITTED.json')) and not (out/'comparison_PRIVATE.json').exists()
    assert (out/'gradients_PRIVATE/arm_A_gradient_meta_PRIVATE.json').exists() and not (out/'gradients_PRIVATE/arm_B_gradient_meta_PRIVATE.json').exists()
    assert list(out.glob('gradient_B_*_STOP_PRIVATE.json'))
    with pytest.raises(RuntimeError,match='INCOMPLETE_GRADIENT_B'):launch(tmp_path,out,resume=True)       # never retried automatically


def test_secret_trainable_weight_mutation_during_backward_is_caught_before_any_restore_or_update(tmp_path):
    class Sneaky(Tree):
        def arm_gradient(self,planned,own_old):
            result=super().arm_gradient(planned,own_old);self.weight=self.weight+1e-3;return result      # Adam/RNG untouched: only transaction_validate can see this
    with pytest.raises(FAULT):launch(tmp_path,tmp_path/'run',cls=Sneaky)
    b=f.Fake.instances[-1];out=tmp_path/'run'
    assert b.events.count('update')==0 and not list(out.glob('arm_*_COMMITTED.json')) and not (out/'gradients_PRIVATE').exists()
    stop=json.loads(next(out.glob('gradient_A_*_STOP_PRIVATE.json')).read_text());assert stop['parent_disk_unchanged'] is True


def test_adam_recurrence_and_delta_recompute_from_parent_state_gradient_archive_and_saved_after_state(tmp_path):
    launch(tmp_path,tmp_path/'run');out=tmp_path/'run';parent,receipt,_=world(tmp_path);bk=f.Fake(None,{})
    def flat_state(path):
        r=json.loads((path/'receipt.json').read_text())
        state,_,_=load_optimizer_checkpoint(path/'optimizer',expected_bindings={'run':r['binding'],'adapter_sha256':r['files_sha256']['adapter.safetensors'],'trainable_signature':bk.trainable_signature()},load_arrays=bk.load_arrays)
        return {'step':state['step'],'weight.m':state['weight']['m'],'weight.v':state['weight']['v']}
    before=flat_state(parent);parent_weight=np.load(parent/'adapter.safetensors')['weight']
    for arm in 'AB':
        archive=ar.load_gradient_archive(out/f'gradients_PRIVATE/arm_{arm}_gradient_meta_PRIVATE.json',bk.load_arrays)
        nxt=ar.adam_next_state(before,archive,{'name':'Adam','learning_rate':1e-5,'betas':[.9,.999],'eps':1e-8,'bias_correction':False})
        terminal=out/f'arm_{arm}/boundary_0001';after=flat_state(terminal)
        assert ar.compare_adam_after(nxt,after) is True and int(after['step'])==53
        np.testing.assert_allclose(np.load(terminal/'adapter.safetensors')['weight']-parent_weight,nxt['delta']['weight'],rtol=1e-6,atol=1e-15)
        receipt_a=json.loads((terminal/'receipt.json').read_text());assert receipt_a['adam_recurrence']['step_before']==52 and receipt_a['adam_recurrence']['optimizer_profile']['bias_correction'] is False


def test_inactive_arm_in_the_two_phase_flow_has_none_gradient_none_cosine_and_no_momentum_only_step(tmp_path):
    f.Fake.accuracy={'F':[1,1,1,1],'T':[0,0,0,0]}
    result=launch(tmp_path,tmp_path/'run');out=tmp_path/'run'
    comparison=json.loads((out/'comparison_PRIVATE.json').read_text())
    assert comparison['cosine'] is None and comparison['norms']['A'] is None and 'A:NO_ADMITTED_GROUP' in comparison['cosine_reason'] and result['optimizer_updates']==1
    a=json.loads((out/'arm_A/boundary_0001/receipt.json').read_text());assert a['updated'] is False and a['optimizer_before_fingerprint']==a['optimizer_after_fingerprint']
    assert f.Fake.instances[-1].events.count('update')==1


def test_preflight_is_honestly_labelled_state_only_with_full_state_identity_and_defers_numerics_to_own_old_parity(tmp_path):
    launch(tmp_path,tmp_path/'run');pre=json.loads((tmp_path/'run/preflight_PRIVATE.json').read_text())
    assert pre['kind']=='STATE_ONLY_RESTORE_PREFLIGHT_NO_NUMERICAL_PROBABILITY_CHECK' and pre['full_state_identity_equals_parent'] is True
    assert 'OWN_OLD_PARITY' in pre['numerical_check'] and pre['forwards']==0 and pre['model_numerical_PASS'] is False
    old=json.loads((tmp_path/'run/old_PRIVATE.json').read_text());assert max(v['max_abs_error'] for v in old['rollout_vs_own_context_parity'].values())<=.002
