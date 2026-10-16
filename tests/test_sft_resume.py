import copy
import json
import os
import random
from pathlib import Path
import pytest
from mri_rl.experiment_contract import canonical_hash, sha256_file
from mri_rl import sft_resume as module


def test_boundary_metadata_rejects_cursor_identity_and_partial_history(tmp_path):
    # Native optimizer bytes are verified by optimizer_checkpoint separately.
    row={'step':0,'category':'a','record_id':'r','patient_key':'p','image_sha256':'i','target_label':'yes','question_sha256':'q'}
    inventory=[{**row,'record_id':str(i)} for i in range(112)]
    parent=tmp_path/'run';parent.mkdir();adapter=parent/'after_adapters.safetensors';adapter.write_bytes(b'weights')
    (parent/'adapter_config.json').write_text('{}')
    state=parent/'checkpoints/state_001';state.mkdir(parents=True)
    binding={'adapter_sha256':sha256_file(adapter),'contract_id':'contract','schedule_sha256':canonical_hash(inventory),'sampler_position_examples':4,'optimizer':module.OPTIMIZER,'backend':'x','trainable_signature':{'w':{'shape':[2],'dtype':'float32'}}}
    manifest={'bindings':binding,'optimizer_updates':1,'completed_groups':1,'tensors_sha256':sha256_file(adapter)}
    (state/'state.safetensors').write_bytes(b'weights');(state/'manifest.json').write_text(json.dumps(manifest))
    reload={'loaded_adapter_sha256':sha256_file(adapter),'exact_adapter_tensor_match':True,'actual_model_generation':True,'adapter_tensor_count':1,'contract_id':'contract'}
    (parent/'reload_inference_receipt.json').write_text(json.dumps(reload))
    receipt={'status':'SFT_ENGINEERING_CONTROL','experiment_contract_id':'contract','optimizer_steps':1,'checkpoint_reloaded':True,'after_adapter_sha256':sha256_file(adapter),'adapter_config_sha256':sha256_file(parent/'adapter_config.json'),'schedule_sha256':canonical_hash(inventory),'executed_prefix_sha256':canonical_hash(inventory[:4]),'executed_prefix_examples':4,'examples':[dict(r) for r in inventory[:4]],'loss_terms':[1.]*4,'gradient_l2_by_step':[2.],'reload_inference_receipt_sha256':sha256_file(parent/'reload_inference_receipt.json')}
    (parent/'receipt.json').write_text(json.dumps(receipt))
    info=module.inspect_boundary(parent,contract_id='contract',inventory=inventory,starting_adapter=adapter)
    assert info['completed_steps']==1
    for key,bad in [('sampler_position_examples',8),('contract_id','wrong'),('adapter_sha256','wrong')]:
        broken=copy.deepcopy(manifest);broken['bindings'][key]=bad;(state/'manifest.json').write_text(json.dumps(broken))
        with pytest.raises(ValueError):module.inspect_boundary(parent,contract_id='contract',inventory=inventory,starting_adapter=adapter)
    (state/'manifest.json').write_text(json.dumps(manifest))
    receipt['examples'].pop();(parent/'receipt.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError):module.inspect_boundary(parent,contract_id='contract',inventory=inventory,starting_adapter=adapter)


@pytest.mark.skipif(os.environ.get('MRI_TEST_MLX_CPU')!='1',reason='Explicit tiny native CPU fixture only')
def test_native_training_restore_matches_uninterrupted(tmp_path):
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    from mri_rl.optimizer_checkpoint import save_optimizer_checkpoint
    mx.set_default_device(mx.cpu)
    mx.random.seed(17);random.seed(27)
    class Tiny(nn.Module):
        def __init__(self):super().__init__();self.linear=nn.Linear(2,1)
        def __call__(self,x):return self.linear(x)
    model=Tiny();opt=optim.Adam(learning_rate=1e-5,bias_correction=False)
    def update(m,o,cursor):
        x=mx.random.uniform(shape=(2,2))+random.random()+cursor
        loss,grad=nn.value_and_grad(m,lambda m:mx.mean((m(x)-1)**2))(m)
        mx.eval(loss,grad);o.update(m,grad);mx.eval(m.parameters(),o.state)
        return float(loss.item()),dict(tree_flatten(grad)),cursor+4
    _,_,cursor=update(model,opt,0)
    saved={k:mx.array(v) for k,v in tree_flatten(model.trainable_parameters())};mx.eval(saved)
    mx.save_safetensors(str(tmp_path/'adapter.safetensors'),saved)
    signature={k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in saved.items()}
    bindings={'adapter_sha256':sha256_file(tmp_path/'adapter.safetensors'),'contract_id':'fixture','schedule_sha256':'frozen','sampler_position_examples':cursor,'optimizer':module.OPTIMIZER,'backend':mx.__version__,'trainable_signature':signature}
    rng={'mlx':[mx.array(v) for v in mx.random.state],'python':random.getstate()};mx.eval(rng['mlx'])
    save_optimizer_checkpoint(tmp_path/'state',optimizer_state=opt.state,rng_state=rng,bindings=bindings,completed_groups=1,optimizer_updates=1,save_arrays=lambda p,a:mx.save_safetensors(str(p),a))
    loss1,grad1,cursor1=update(model,opt,cursor);next1=mx.random.uniform(shape=(4,));mx.eval(next1);py1=random.random()
    restored=Tiny();restored.load_weights(str(tmp_path/'adapter.safetensors'));mx.eval(restored.parameters())
    opt2=optim.Adam(learning_rate=1e-5,bias_correction=False)
    module.restore_native_state(tmp_path/'state',optimizer=opt2,expected_bindings=bindings,mx=mx)
    loss2,grad2,cursor2=update(restored,opt2,cursor);next2=mx.random.uniform(shape=(4,));mx.eval(next2);py2=random.random()
    assert loss1==loss2 and cursor1==cursor2==8 and py1==py2
    assert bool(mx.array_equal(next1,next2).item())
    for a,b in [(model.parameters(),restored.parameters()),(opt.state,opt2.state),(grad1,grad2)]:
        aa=dict(tree_flatten(a));bb=dict(tree_flatten(b));assert set(aa)==set(bb)
        for k,v in aa.items():assert v.dtype==bb[k].dtype and bool(mx.array_equal(v,bb[k]).item()),k
    assert int(opt2.step.item())==2


def test_intermediate_checkpoint_prefix_is_completed_not_planned():
    inventory=[{'id':i} for i in range(112)]
    assert module.completed_prefix(inventory,2)=={'executed_prefix_examples':8,'executed_prefix_sha256':canonical_hash(inventory[:8])}
    assert module.completed_prefix(inventory,2)!=module.completed_prefix(inventory,3)
