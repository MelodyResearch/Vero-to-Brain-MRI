"""Strict saved SFT boundary validation and native optimizer/RNG restoration.

The first boundary predates this consumer. Its immutable producer snapshot is
bound by independent review; migration preserves the recorded Adam defaults.
"""
from __future__ import annotations
import json
import math
import random
from pathlib import Path
from mri_rl.experiment_contract import canonical_hash, sha256_file
from mri_rl.optimizer_checkpoint import load_optimizer_checkpoint

OPTIMIZER={'name':'Adam','lr':1e-5,'bias_correction':False}
OPTIMIZER_SEMANTICS={**OPTIMIZER,'betas':[0.9,0.999],'eps':1e-8,
                     'scheduler':'constant','weight_decay':0.0}


def inspect_boundary(parent: Path, *, contract_id: str, inventory: list[dict],
                     starting_adapter: Path) -> dict:
    parent=Path(parent).resolve();adapter=Path(starting_adapter).resolve()
    receipt=json.loads((parent/'receipt.json').read_text())
    count=receipt.get('optimizer_steps')
    if type(count) is not int or not 1<=count<28:
        raise ValueError('Invalid completed SFT boundary count')
    if (adapter!=parent/'after_adapters.safetensors' or
        receipt.get('status')!='SFT_ENGINEERING_CONTROL' or
        receipt.get('experiment_contract_id')!=contract_id or
        receipt.get('checkpoint_reloaded') is not True or
        receipt.get('after_adapter_sha256')!=sha256_file(adapter) or
        receipt.get('adapter_config_sha256')!=sha256_file(parent/'adapter_config.json') or
        receipt.get('schedule_sha256')!=canonical_hash(inventory) or
        receipt.get('executed_prefix_sha256')!=canonical_hash(inventory[:4*count]) or
        receipt.get('executed_prefix_examples')!=4*count):
        raise ValueError('Parent SFT identity or completed prefix mismatch')
    examples=receipt.get('examples',[])
    losses=receipt.get('loss_terms',[]);grads=receipt.get('gradient_l2_by_step',[])
    if (len(examples)!=4*count or len(losses)!=4*count or len(grads)!=count or
        any(not isinstance(x,(int,float)) or not math.isfinite(x) for x in losses+grads) or
        any(x<=0 for x in grads)):
        raise ValueError('Incomplete finite SFT boundary history')
    for actual,expected in zip(examples,inventory):
        for key in ('step','category','record_id','patient_key','image_sha256','target_label'):
            if actual.get(key)!=expected[key]:raise ValueError('Executed SFT history differs from frozen schedule')
    reload_path=parent/'reload_inference_receipt.json'
    if receipt.get('reload_inference_receipt_sha256')!=sha256_file(reload_path):
        raise ValueError('Parent reload receipt changed')
    reload=json.loads(reload_path.read_text())
    if (reload.get('loaded_adapter_sha256')!=sha256_file(adapter) or
        reload.get('exact_adapter_tensor_match') is not True or
        reload.get('actual_model_generation') is not True or reload.get('contract_id')!=contract_id):
        raise ValueError('Parent reload evidence mismatch')
    state=parent/'checkpoints'/f'state_{count:03d}'
    manifest=json.loads((state/'manifest.json').read_text());bindings=manifest['bindings']
    if (manifest.get('optimizer_updates')!=count or manifest.get('completed_groups')!=count or
        bindings.get('adapter_sha256')!=sha256_file(adapter) or
        bindings.get('contract_id')!=contract_id or bindings.get('schedule_sha256')!=canonical_hash(inventory) or
        bindings.get('sampler_position_examples')!=4*count or bindings.get('optimizer')!=OPTIMIZER or
        manifest.get('tensors_sha256')!=sha256_file(state/'state.safetensors')):
        raise ValueError('Saved optimizer boundary identity/count/cursor mismatch')
    return {'completed_steps':count,'state_directory':state,'bindings':bindings,'receipt':receipt,
            'manifest_sha256':sha256_file(state/'manifest.json'),
            'receipt_sha256':sha256_file(parent/'receipt.json')}


def restore_native_state(state_directory: Path, *, optimizer, expected_bindings: dict, mx):
    state,rng,manifest=load_optimizer_checkpoint(state_directory,expected_bindings=expected_bindings,
                                                load_arrays=lambda p:mx.load(str(p)))
    if set(rng)!={'mlx','python'}:raise ValueError('SFT requires exactly its MLX and Python RNG inventory')
    if expected_bindings.get('backend')!=mx.__version__:
        raise ValueError('Resume backend changed')
    if (list(optimizer.betas)!=[0.9,0.999] or optimizer.eps!=1e-8 or optimizer.bias_correction is not False or
        not math.isclose(float(state['learning_rate'].item()),1e-5,rel_tol=1e-7)):
        raise ValueError('Restored Adam semantics or learning rate differ')
    if len(mx.random.state)!=len(rng['mlx']):raise ValueError('MLX RNG inventory changed')
    from mlx.utils import tree_flatten
    for name,value in tree_flatten(state):
        if not bool(mx.all(mx.isfinite(value)).item()):raise ValueError('Nonfinite optimizer state')
        if name.endswith('.v') and not bool(mx.all(value>=0).item()):raise ValueError('Negative Adam second moment')
    optimizer.state=state
    mx.eval(optimizer.state)
    actual=dict(tree_flatten(optimizer.state));expected=dict(tree_flatten(state))
    if set(actual)!=set(expected) or not all(bool(mx.array_equal(v,actual[k]).item()) for k,v in expected.items()):
        raise ValueError('Native optimizer state assignment changed saved tensors')
    random.setstate(rng['python'])
    for index,value in enumerate(rng['mlx']):mx.random.state[index][:]=value
    mx.eval(mx.random.state)
    if random.getstate()!=rng['python'] or not all(bool(mx.array_equal(v,mx.random.state[i]).item()) for i,v in enumerate(rng['mlx'])):
        raise ValueError('RNG restoration mismatch')
    return manifest


def verify_resume_evidence(dynamic, *, contract, inventory, starting_adapter, configuration, budget, private_root):
    parent=Path(configuration.get('resume_from','')).resolve()
    if not parent.is_relative_to(private_root.resolve()):raise ValueError('Resume boundary must remain private')
    info=inspect_boundary(parent,contract_id=contract.identity(),inventory=inventory,starting_adapter=starting_adapter)
    start=info['completed_steps'];end=start+budget.get('optimizer_steps',0)
    if (dynamic.get('kind')!='supervised_resume_v1' or dynamic.get('contract_id')!=contract.identity() or
        dynamic.get('parent_run')!=str(parent) or dynamic.get('parent_receipt_sha256')!=info['receipt_sha256'] or
        dynamic.get('parent_state_manifest_sha256')!=info['manifest_sha256'] or
        configuration.get('resume_receipt_sha256')!=info['receipt_sha256'] or
        configuration.get('resume_manifest_sha256')!=info['manifest_sha256'] or
        configuration.get('runner')!='mlx_sft_warmstart' or configuration.get('seed')!=7 or
        configuration.get('schedule_policy')!='balanced_sequence_v2' or
        configuration.get('optimizer_semantics')!=OPTIMIZER_SEMANTICS or
        configuration.get('schedule_sha256')!=canonical_hash(inventory) or
        configuration.get('stage_start_step')!=start or configuration.get('stage_end_step')!=end or
        configuration.get('executed_prefix_examples')!=4*end or
        configuration.get('executed_prefix_sha256')!=canonical_hash(inventory[:4*end]) or
        configuration.get('stage_examples_sha256')!=canonical_hash(inventory[4*start:4*end]) or
        budget.get('optimizer_steps') not in (1,2) or budget.get('max_groups')!=budget.get('optimizer_steps') or
        budget.get('rollouts_per_group')!=4 or end>28):
        raise ValueError('Resume evidence/configuration/stage budget mismatch')
    from mri_rl.sft_evidence import bound_file
    approval=bound_file(dynamic['parent_approval'],private_root)
    parent_approval=json.loads(approval.read_text())
    if (sha256_file(approval)!=info['receipt'].get('review_sha256') or
        parent_approval.get('verdict') not in ('PASS','PASS_WITH_LIMITATIONS') or
        parent_approval.get('contract_id')!=contract.identity()):
        raise ValueError('Parent producer approval mismatch')


def completed_prefix(inventory, completed_steps):
    return {'executed_prefix_examples':4*completed_steps,
            'executed_prefix_sha256':canonical_hash(inventory[:4*completed_steps])}
