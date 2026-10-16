"""Opt-in CPU-native integration; synthetic tiny model, never a Qwen load.

Run only in an explicitly reserved native window with MRI_RUN_NATIVE_RL_TEST=1.
Default collection does not import MLX. This tests real GSPO gradients, Adam,
safetensors and RNG under the same continuous driver, not MRI understanding.
"""
import os
import pytest

pytestmark=pytest.mark.skipif(os.environ.get('MRI_RUN_NATIVE_RL_TEST')!='1',reason='Requires reserved native CPU window')


def test_native_gspo_continuous_matches_resume_and_skip(tmp_path):
    import hashlib
    import json
    import random
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten,tree_map
    from mri_rl.continuous_rl import run,digest
    from mri_rl.mlx_continuous_gspo import NativeGSPOBackend

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__();self.weight=mx.array([[.1,.3,-.2],[.4,-.1,.2],[.2,.1,-.3]],dtype=mx.float32)
        def __call__(self,ids,pixels,mask,**kwargs):
            from types import SimpleNamespace
            return SimpleNamespace(logits=self.weight[ids])

    class TinyNative(NativeGSPOBackend):
        def __init__(self,adapter,config):
            self.mx=mx;self.nn=nn;self.tree_flatten=tree_flatten;self.tree_map=tree_map
            self.model=Tiny();self.processor=None;self.active=None;self.config=config
            self.before_unit=lambda name:None
            if adapter:self.model.update(mx.load(str(adapter)))
            self.model.eval();self.optimizer=optim.Adam(learning_rate=1e-5,betas=[.9,.999],eps=1e-8,bias_correction=False)
            mx.random.seed(7);random.seed(7)
        def save_adapter(self,path):
            self.mx.eval(self.model.trainable_parameters())
            self.mx.save_safetensors(str(path),dict(tree_flatten(self.model.trainable_parameters())))
        def rollout(self,row,index,workdir):
            # Exercise both restored RNGs without choosing labels from eval outcomes.
            draw=float(mx.random.uniform().item());py=random.random()
            labels=['FLAIR','contrast-enhanced T1'] if index!=1 else ['FLAIR','FLAIR']
            raw={'record_id':row['id'],'image_sha256':digest(row['images'][0]),
                'synthetic_fixture':True,'rng_draws':[draw,py],
                'responses':[{'text':'<think>Fixture.</think><answer>\\boxed{'+label+'}</answer>',
                              'token_ids':[j+1],'finish_reason':'stop'} for j,label in enumerate(labels)]}
            self.active={'record_id':row['id'],'inputs':{'input_ids':mx.array([[0,1]],dtype=mx.int32)},
                'prompt_length':2,'raw_sha256':hashlib.sha256(json.dumps(raw,sort_keys=True).encode()).hexdigest()}
            return raw

    image=tmp_path/'synthetic.png';image.write_bytes(b'SYNTHETIC NOT MRI')
    rows=[{'id':f'fixture-{i}','data_source':'OpenNeuro_ds007045','ability':'mri_sequence','images':[str(image)],
       'reward_model':{'ground_truth':'FLAIR'},'extra_info':{'reward_type':'mri_sequence','internal_split':'train',
       'patient_lineage':'BIDS_SUBJECT_VERIFIED','patient_key':f'synthetic-{i}','question':'Fixture'}} for i in range(3)]
    common=dict(schedule=rows,bindings={'scope':'synthetic_native_fixture'},backend_factory=TinyNative,authorize=lambda _:None)
    full=run(output=tmp_path/'full',max_new_groups=3,**common)
    run(output=tmp_path/'split',max_new_groups=1,**common)
    middle=run(output=tmp_path/'split',max_new_groups=1,resume=True,**common)
    assert middle['optimizer_updates']==1 and middle['completed_groups']==2
    split=run(output=tmp_path/'split',max_new_groups=1,resume=True,**common)
    assert full['optimizer_updates']==split['optimizer_updates']==2
    for filename in ['adapter.safetensors','optimizer/state.safetensors']:
        a=mx.load(str(tmp_path/'full/boundary_0003'/filename));b=mx.load(str(tmp_path/'split/boundary_0003'/filename))
        assert set(a)==set(b)
        assert all(bool(mx.array_equal(v,b[k]).item()) for k,v in a.items())
    for i in [1,2,3]:
        left=json.loads((tmp_path/f'full/boundary_{i:04d}/rollout_PRIVATE.json').read_text())
        right=json.loads((tmp_path/f'split/boundary_{i:04d}/rollout_PRIVATE.json').read_text())
        assert left==right
    assert digest(tmp_path/'split/boundary_0001/adapter.safetensors')==digest(tmp_path/'split/boundary_0002/adapter.safetensors')
