"""Lazy native adapter for continuous_rl; no CLI or automatic launch approval.

The caller must first validate the exact independent review/contract and resource
scope through continuous_rl.run(authorize=...), and supply before_unit for finer
resource checks. This is the historical MLX GSPO-token diagnostic, with persistent
Adam, not official veRL parity. Native Qwen execution remains separately gated.
"""
from __future__ import annotations
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
from mri_rl.continuous_rl import PROFILE, require, digest
from mri_rl.paced_resources import PACED_POLICY


class NativeGSPOBackend:
    def __init__(self, adapter, config, *, model_path, contract, seed, before_unit):
        require(callable(before_unit), 'Explicit resource authorizer required')
        require(adapter is not None and Path(adapter).is_file(), 'Exact starting adapter required')
        before_unit('native_model_load')  # Must precede even the runtime imports.
        self.before_unit=before_unit;self.config=config;self.contract=contract
        self.active=None;self.model=None
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
        from mlx.utils import tree_flatten, tree_map
        from mlx_vlm import generate, load
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.trainer.common.model import prepare_model_for_training
        from mlx_vlm.trainer.common.utils import save_trainable_weights, enable_gradient_checkpointing
        from mlx_vlm.utils import prepare_inputs, should_add_special_tokens
        self.mx=mx;self.nn=nn;self.tree_flatten=tree_flatten;self.tree_map=tree_map
        self.generate=generate;self.template=apply_chat_template;self.prepare_inputs=prepare_inputs
        self.special_tokens=should_add_special_tokens;self.save_weights=save_trainable_weights
        os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
        mx.set_memory_limit(PACED_POLICY['memory_guideline_gib'] * 1024**3)
        mx.set_cache_limit(PACED_POLICY['cache_limit_mib'] * 1024**2)
        random.seed(seed);mx.random.seed(seed)
        try:
            self.model,self.processor=load(str(model_path),lazy=True)
            self.model=prepare_model_for_training(self.model,train_type='lora',lora_rank=4,lora_alpha=8,
                quantization_bits=4,quantization_group_size=64,checkpoint_path=Path(adapter),verbose=False)
            from mri_rl.likelihood_loading import materialize_serial
            # Same serial_parameter_v2 discipline as the successful read-only loader, not a bulk eval.
            materialize_serial(tree_flatten(self.model.parameters()),evaluate=mx.eval,collect=gc.collect,
                clear=mx.clear_cache,observe=before_unit,observe_every=64)
            gc.collect();mx.clear_cache()
            before_unit('quantized_model_ready')
            enable_gradient_checkpointing(self.model.language_model.layers)
            saved=mx.load(str(adapter));actual=dict(tree_flatten(self.model.trainable_parameters()))
            require(set(saved)==set(actual) and all(bool(mx.array_equal(v,actual[k]).item()) for k,v in saved.items()),
                    'Loaded adapter tensors differ from completed boundary')
            self.model.eval()
            # Exactly once per live backend; skips never call update().
            self.optimizer=optim.Adam(learning_rate=1e-5,betas=[.9,.999],eps=1e-8,bias_correction=False)
            mx.eval(self.optimizer.state)
        except BaseException:
            self.close();raise

    def optimizer_state(self):
        self.mx.eval(self.optimizer.state);return self.optimizer.state

    def rng_state(self):
        self.mx.eval(self.mx.random.state)
        # MLX exposes a mutable _RandomState container, not a list/tuple.
        # Snapshot concrete arrays, as the existing SFT producer does.
        return {'mlx':[self.mx.array(value) for value in self.mx.random.state],
                'python':random.getstate()}

    def restore(self,state,rng):
        mx=self.mx
        require(set(rng)=={'mlx','python'} and len(rng['mlx'])==len(mx.random.state),'Native RNG inventory changed')
        require(math.isclose(float(state['learning_rate'].item()),1e-5,rel_tol=1e-7),'Restored Adam learning rate differs')
        for name,value in self.tree_flatten(state):
            require(bool(mx.all(mx.isfinite(value)).item()),'Nonfinite optimizer state')
            if name.endswith('.v'):require(bool(mx.all(value>=0).item()),'Negative Adam second moment')
        self.optimizer.state=state;mx.eval(self.optimizer.state)
        actual=dict(self.tree_flatten(self.optimizer.state));expected=dict(self.tree_flatten(state))
        require(set(actual)==set(expected) and all(bool(mx.array_equal(v,actual[k]).item()) for k,v in expected.items()),
                'Native state assignment changed saved tensors')
        random.setstate(rng['python'])
        for index,value in enumerate(rng['mlx']):mx.random.state[index][:]=value
        mx.eval(mx.random.state)
        require(random.getstate()==rng['python'] and all(bool(mx.array_equal(v,mx.random.state[i]).item()) for i,v in enumerate(rng['mlx'])),
                'Native RNG restore mismatch')

    def trainable_signature(self):
        return {key:{'shape':list(value.shape),'dtype':str(value.dtype)}
                for key,value in self.tree_flatten(self.model.trainable_parameters())}

    def validate_state(self):
        mx=self.mx
        parameters=self.tree_flatten(self.model.trainable_parameters())
        state=self.tree_flatten(self.optimizer_state())
        return {'parameters_finite':all(bool(mx.all(mx.isfinite(value)).item()) for _,value in parameters),
                'optimizer_finite':all(bool(mx.all(mx.isfinite(value)).item()) for _,value in state),
                'second_moments_nonnegative':all(bool(mx.all(value>=0).item()) for name,value in state if name.endswith('.v')),
                'parameter_tensor_count':len(parameters),'optimizer_tensor_count':len(state)}

    def save_arrays(self,path,arrays):
        self.mx.eval(arrays);self.mx.save_safetensors(str(path),arrays)

    def load_arrays(self,path):return self.mx.load(str(path))

    def save_adapter(self,path):
        self.mx.eval(self.model.trainable_parameters());self.save_weights(self.model,Path(path))

    def rollout(self,row,index,workdir):
        mx=self.mx;self.active=None;self.model.eval()
        self.before_unit(f'rollout_group_{index}')
        image=row['images'][0]
        messages=[{'role':'system','content':self.contract.system_instruction},
                  {'role':'user','content':row['extra_info']['question']}]
        prompt=self.template(self.processor,self.model.config,messages,add_generation_prompt=True,
                             num_images=1,enable_thinking=self.contract.enable_thinking)
        require(prompt.count('<|image_pad|>')==1 and prompt.endswith(self.contract.assistant_prefix),
                'Native image placeholder/assistant prefix mismatch')
        inputs=self.prepare_inputs(self.processor,images=[image],prompts=prompt,
            image_token_index=getattr(self.model.config,'image_token_index',None),
            add_special_tokens=self.special_tokens(self.model.config.model_type,self.processor))
        length=int(inputs['input_ids'].shape[-1]);require(0<length<=2048,'Prompt exceeds bounded profile')
        self.active={'record_id':row['id'],'inputs':inputs,'prompt_length':length}
        raw={'record_id':row['id'],'group_index':index,'image_sha256':digest(image),
             'rendered_prompt':prompt,'prompt_token_ids':inputs['input_ids'].tolist(),
             'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'responses':[]}
        for number in range(self.config['group_size']):
            self.before_unit(f'rollout_{index}_{number}')
            result=self.generate(self.model,self.processor,prompt,image=[image],max_tokens=self.config['max_tokens'],
                temperature=1.,top_p=1.,top_k=0,min_p=0.,repetition_penalty=None,verbose=False)
            tokens=list(result.token_ids or [])
            require(tokens and result.prompt_tokens==length,'Rollout/log-probability prompt mismatch')
            raw['responses'].append({'text':result.text,'token_ids':tokens,
                'generation_tokens':getattr(result,'generation_tokens',None),
                'finish_reason':getattr(result,'finish_reason',None)})
            # Keep completed responses if generation or a subsequent resource gate fails.
            partial=Path(workdir)/'rollout_progress.partial.json'
            with partial.open('w') as handle:
                json.dump(raw,handle,sort_keys=True,allow_nan=False);handle.flush();os.fsync(handle.fileno())
            os.replace(partial,Path(workdir)/'rollout_progress_PRIVATE.json')
            mx.clear_cache()
        # Prevent update callers from substituting a different response set.
        self.active['raw_sha256']=hashlib.sha256(json.dumps(raw,sort_keys=True).encode()).hexdigest()
        return raw

    def after_rollout(self):
        # Driver has persisted the complete raw group before calling this gate.
        self.before_unit('post_generation')

    def _logps(self,tokens):
        mx=self.mx;inputs=self.active['inputs'];prompt=inputs['input_ids']
        ids=mx.concatenate((prompt,mx.array(tokens,dtype=prompt.dtype)[None,:]),axis=1)
        kwargs={k:v for k,v in inputs.items() if k not in {'input_ids','attention_mask','pixel_values'}}
        logits=self.model(ids[:,:-1],inputs.get('pixel_values'),mx.ones_like(ids[:,:-1]),**kwargs).logits.astype(mx.float32)
        labels=ids[:,1:];require(logits.shape[1]==labels.shape[1],'Logit/label alignment mismatch')
        selected=mx.take_along_axis(self.nn.log_softmax(logits,axis=-1),labels[...,None],axis=-1)[...,0]
        result=selected[0,self.active['prompt_length']-1:]
        require(result.shape[0]==len(tokens),'Response likelihood length mismatch')
        return result

    def update(self,row,raw,advantages):
        mx=self.mx;self.model.eval()
        require(self.active is not None and self.active['record_id']==row['id'] and
                self.active['raw_sha256']==hashlib.sha256(json.dumps(raw,sort_keys=True).encode()).hexdigest(),
                'Update differs from current on-policy rollout')
        require(len(advantages)==len(raw['responses'])==self.config['group_size'],'Group update inventory mismatch')
        self.before_unit('old_current_log_probabilities')
        old=[];errors=[]
        for response in raw['responses']:
            values=mx.stop_gradient(self._logps(response['token_ids']));mx.eval(values);old.append(values)
            current=self._logps(response['token_ids']);mx.eval(current)
            error=float(mx.max(mx.abs(values-current)).item());errors.append(error)
            require(math.isfinite(error) and error<=1e-5,'Old/current alignment failed before update')
            mx.clear_cache()
        def loss_fn(model,tokens,previous,advantage):
            fresh=self._logps(tokens)
            ratio=mx.exp(mx.minimum(fresh-mx.stop_gradient(fresh)+mx.stop_gradient(mx.mean(fresh-previous)),10.))
            return mx.mean(mx.maximum(-advantage*ratio,-advantage*mx.clip(ratio,1-.0003,1+.0004)))
        loss_grad=self.nn.value_and_grad(self.model,loss_fn);summed=None;losses=[]
        for response,previous,advantage in zip(raw['responses'],old,advantages):
            self.before_unit('group_response_gradient')
            loss,grad=loss_grad(self.model,response['token_ids'],previous,advantage);mx.eval(loss,grad)
            value=float(loss.item());require(math.isfinite(value),'Nonfinite group loss');losses.append(value)
            scaled=self.tree_map(lambda x:x/self.config['group_size'],grad)
            summed=scaled if summed is None else self.tree_map(lambda a,b:a+b,summed,scaled)
            mx.eval(summed);mx.clear_cache()
        norm=math.sqrt(sum(float(mx.sum(g.astype(mx.float32)**2).item()) for _,g in self.tree_flatten(summed)))
        require(math.isfinite(norm) and norm>0,'No finite nonzero group gradient')
        self.before_unit('optimizer_update')
        self.optimizer.update(self.model,summed);mx.eval(self.model.trainable_parameters(),self.optimizer.state)
        self.active=None;mx.clear_cache()
        return {'loss_terms':losses,'gradient_l2':norm,'old_current_logp_max_abs_errors':errors,
                'optimizer_profile':'persistent Adam 1e-5; no bias correction; no scheduler change',
                'checkpoint_reload':'Required at next backend construction; not separately inferred'}

    def greedy_generate(self,row,expected_image_sha256):
        """Deterministic reload-check generation from the loaded adapter; no gradient or state change.

        The frozen image hash is checked immediately after the resource callback (which may
        pause/mutate the world) and before generation, then again after generation."""
        require(isinstance(expected_image_sha256,str) and len(expected_image_sha256)==64,'Frozen image hash required')
        self.model.eval();self.before_unit('reload_greedy_generation')
        require(digest(row['images'][0])==expected_image_sha256,'Frozen image changed after resource callback')
        messages=[{'role':'system','content':self.contract.system_instruction},
                  {'role':'user','content':row['extra_info']['question']}]
        prompt=self.template(self.processor,self.model.config,messages,add_generation_prompt=True,
                             num_images=1,enable_thinking=self.contract.enable_thinking)
        require(prompt.count('<|image_pad|>')==1 and prompt.endswith(self.contract.assistant_prefix),
                'Native image placeholder/assistant prefix mismatch')
        result=self.generate(self.model,self.processor,prompt,image=[row['images'][0]],
            max_tokens=self.config['max_tokens'],temperature=0.,verbose=False)
        require(digest(row['images'][0])==expected_image_sha256,'Frozen image changed during generation')
        self.mx.clear_cache()
        return {'record_id':row['id'],'text':result.text,'token_ids':list(result.token_ids or []),
                'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'image_sha256':expected_image_sha256}

    def generate_record(self,record,expected_image_sha256,max_tokens):
        """Greedy dev generation with frozen_eval's prompt rendering (enable_thinking=False).

        Image hash is rechecked after the resource callback, before generation, and after it."""
        self.model.eval();self.before_unit('dev_'+record['record_id'])
        image=record.get('image')
        require((digest(image)==expected_image_sha256) if image else expected_image_sha256 is None,
                'Frozen image changed after resource callback')
        prompt=self.template(self.processor,self.model.config,record['prompt'],
                             num_images=1 if image else 0,enable_thinking=False)
        placeholders=prompt.count('<|image_pad|>')
        require(placeholders==(1 if image else 0),'Image placeholder count mismatch')
        result=self.generate(self.model,self.processor,prompt,image=[str(image)] if image else None,
            max_tokens=max_tokens,temperature=0,verbose=False)
        require((digest(image)==expected_image_sha256) if image else True,'Frozen image changed during generation')
        self.mx.clear_cache()
        tokens=list(result.token_ids or [])
        return {'response':result.text,'token_ids':tokens,'image_placeholder_count':placeholders,
                'generation_tokens':getattr(result,'generation_tokens',len(tokens)),
                'finish_reason':getattr(result,'finish_reason',None),
                'hit_max_tokens':len(tokens)>=max_tokens}

    def close(self):
        self.active=None;self.model=None;self.processor=None
        if hasattr(self,'optimizer'):del self.optimizer
        gc.collect()
        if hasattr(self,'mx'):self.mx.clear_cache()


def backend_factory(*,model_path,contract,seed,before_unit):
    """Inject into continuous_rl.run only behind independently reviewed authorize.

    Native callers bind this source, backend versions, model/contract files and
    initial adapter hash into run(bindings=...). No preflight is supplied here.
    """
    return lambda adapter,config:NativeGSPOBackend(adapter,config,model_path=model_path,
        contract=contract,seed=seed,before_unit=before_unit)
