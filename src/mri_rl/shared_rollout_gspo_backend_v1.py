"""Lazy-runtime mixed-context F32 rollout/scorer/gradient backend; IMPLEMENTED_CPU_NOT_NATIVE_QUALIFIED.

No MLX import at module import; mlx/mlx_vlm are reached only through the inherited loader (after
the resource gate) and function-local imports. The historical single-context path
(`self.active`, K4-hardcoded rollout, group update) is DISABLED here; every response is
generated, scored and differentiated under its OWN prompt/image context, one context's
prepared inputs resident at a time, with no cached vision graph. The unchanged native
GSPO `loss_fn` AST is compiled per response. Trainable weights are only attached through the
inherited adapter load and the Adam60 full state restored by the driver; weights are never
reinitialized. A future independent native qualification is mandatory; this is not it.
"""
from __future__ import annotations
import ast
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
from types import SimpleNamespace
from mri_rl import shared_rollout_gspo_core_v1 as core
from mri_rl.paired_z045_f32_backend_v1 import (F32NativeGSPOBackend,F32GenerationModelView,ObservedFP32Processor,capture_stream,
                                               reset_effective_eos,f32_response_logps_observed,require_prompt_cost)
from mri_rl.class_margin_aux_v2 import ClassMarginAuxBackend
from mri_rl.paired_z045_f32_qualification_v1 import frozen_loss_node

require=core.require
STATUS=core.STATUS
SAMPLING={'max_tokens':core.MAX_TOKENS,'temperature':1.,'top_p':1.,'top_k':0,'min_p':0.,'repetition_penalty':None}


class CountingProcessor(ObservedFP32Processor):
    """Same F32 materialization barrier as the inherited rollout; charges every sampler draw first."""
    def __init__(self,mx,charge):super().__init__(mx,None);self.charge=charge
    def __call__(self,tokens,logits):
        self.charge('sampler_draws');return super().__call__(tokens,logits)


class SharedRolloutBackend(F32NativeGSPOBackend):
    # Existing transaction APIs, unchanged (full 1942-parameter / Adam / Python+MLX RNG snapshot and verification).
    transaction_snapshot=ClassMarginAuxBackend.transaction_snapshot
    transaction_validate=ClassMarginAuxBackend.transaction_validate

    def rollout(self,*args,**kwargs):raise ValueError('HISTORICAL_SINGLE_CONTEXT_ROLLOUT_DISABLED: use generate_shared_rollout')
    def update(self,*args,**kwargs):raise ValueError('HISTORICAL_GROUP_UPDATE_DISABLED: use arm_gradient/apply_update')
    def _logps(self,tokens):raise ValueError('SINGLE_ACTIVE_CONTEXT_SCORER_DISABLED: mixed contexts are scored per response context')

    # ------------------------------------------------------------ configuration / state audit
    def configure_shared(self,*,counters,rows,parent,binding_sha256,expected_image_sha256):
        self.counters=counters;self.rows=rows;self.shared_parent=parent;self.binding_sha256=binding_sha256
        self.expected_image_sha256=expected_image_sha256;self._active_modality=None;self._active=None;self._recorded=None;self._arm_updates=0

    def bind_bundle(self,bundle):
        """Record the immutable per-context input fingerprints every later activation must reproduce."""
        self._recorded={m:bundle.raw['contexts'][m] for m in ('F','T')}

    def transaction_restore(self,snapshot):
        ClassMarginAuxBackend.transaction_restore(self,snapshot);self._arm_updates=0
        self._active=None;self._active_modality=None

    def rng_digest(self):
        from mri_rl import early_native_conditioning_probe_v1 as utility
        state=self.rng_state();arrays={f'mlx_{i}':v for i,v in enumerate(state['mlx'])}
        return core.identity({'mlx':utility._fingerprints(self,arrays),'python':hashlib.sha256(repr(state['python']).encode()).hexdigest()})

    def optimizer_digest(self):
        from mri_rl import early_native_conditioning_probe_v1 as utility
        from mri_rl.paired_z045_f32_qualification_v1 import _optimizer
        return core.identity(_optimizer(self,utility))

    # ------------------------------------------------------------ one context at a time
    def _verify_image(self,modality):
        from mri_rl.paired_z045_continuation_entry_v1 import verify_images
        verify_images([self.rows[modality]],self.expected_image_sha256)

    def _prepare_context(self,modality):
        """Prepare THIS context only (its own prompt, pixels, grid, position metadata); no cached features."""
        from mlx_vlm.generate.dispatch import _prepare_generation_inputs
        from mri_rl.native_gspo_path_probe_v1 import strict_tensor_identity
        from mri_rl import early_native_conditioning_probe_v1 as utility
        mx=self.mx;row=self.rows[modality];self.before_unit(f'shared_context_prepare_{modality}');self._verify_image(modality)
        prompt=self.template(self.processor,self.model.config,[{'role':'system','content':self.contract.system_instruction},
            {'role':'user','content':row['extra_info']['question']}],add_generation_prompt=True,num_images=1,enable_thinking=self.contract.enable_thinking)
        require(prompt.count('<|image_pad|>')==1 and prompt.endswith(self.contract.assistant_prefix),'Native prompt placeholder/prefix differs')
        inputs=self.prepare_inputs(self.processor,images=row['images'],prompts=prompt,image_token_index=getattr(self.model.config,'image_token_index',None),
            add_special_tokens=self.special_tokens(self.model.config.model_type,self.processor))
        ids,pixels,mask,extra=_prepare_generation_inputs(self.model,self.processor,prompt,row['images'],None,None,dict(SAMPLING))
        prepared={'input_ids':ids,'pixel_values':pixels,'attention_mask':mask,**extra}
        fingerprints=strict_tensor_identity(self,prepared,inputs,utility)
        length=int(ids.shape[-1]);require_prompt_cost(row['ability'],length)
        require(0<length<=2048 and bool(mx.all(mask==1).item()),'Bounded unpadded native prompt required')
        self._verify_image(modality)
        return {'modality':modality,'prompt':prompt,'inputs':inputs,'prompt_ids':ids.tolist(),'extra':extra,'prompt_length':length,
                'fingerprints':fingerprints,'fingerprints_sha256':core.identity(fingerprints)}

    def activate(self,modality):
        """Release the other context, then prepare+verify this one. Never reuse one context's inputs for another."""
        require(modality in ('F','T') and getattr(self,'rows',None) is not None,'Configured F/T context required')
        if self._active_modality==modality:return self._active
        self._active=None;self._active_modality=None;gc.collect();self.mx.clear_cache()
        context=self._prepare_context(modality)
        if self._recorded is not None:
            record=self._recorded[modality]
            require(context['fingerprints_sha256']==record['input_fingerprints_sha256'] and context['prompt_length']==record['prompt_length'] and
                    hashlib.sha256(context['prompt'].encode()).hexdigest()==record['prompt_text_sha256'] and
                    core.identity(context['prompt_ids'])==record['rendered_prompt_token_ids_sha256'],'Reproduced context inputs differ from the immutable bundle')
        self._active=context;self._active_modality=modality;return context

    def _context_logps(self,context,tokens,counter):
        """Own-context F32 full-TF selected logps; charges exactly one scorer API forward before running it."""
        P=context['prompt_length']
        self.counters.charge_many({counter:1,'scorer_vision_embedding_calls':1,'scorer_language_model_calls':1,'teacherforced_input_positions':P+len(tokens)-1})
        self.before_unit('shared_rollout_'+counter);self._verify_image(context['modality'])
        return f32_response_logps_observed(self,tokens,context['inputs'])

    # ------------------------------------------------------------ F4 + T4 rollout, once
    def generate_shared_rollout(self,rows,workdir):
        self.rows=rows;captured={'raw':{},'contexts':{}}
        for modality in ('F','T'):
            context=self.activate(modality);captured['raw'][modality]=self._generate_context(context,4,Path(workdir))
            captured['contexts'][modality]={'prompt_text_sha256':hashlib.sha256(context['prompt'].encode()).hexdigest(),
                'image_sha256':captured['raw'][modality]['image_sha256'],'rendered_prompt_token_ids_sha256':core.identity(context['prompt_ids']),
                'input_fingerprints_sha256':context['fingerprints_sha256'],'prompt_length':context['prompt_length']}
        self._active=None;self._active_modality=None;gc.collect();self.mx.clear_cache()
        return captured

    def _generate_context(self,context,count,workdir):
        """Minimal count-aware copy of the pinned F32 rollout: 2 or 4 responses, no `self.active`."""
        require(count in (2,4),'Count-aware rollout supports 2 or 4 responses')
        from mlx_vlm.generate.dispatch import stream_generate
        from mri_rl.native_gspo_path_probe_v1 import strict_tensor_identity
        from mri_rl import early_native_conditioning_probe_v1 as utility
        from mri_rl.continuous_rl import digest
        mx=self.mx;modality=context['modality'];row=self.rows[modality]
        require(self.config['max_tokens']==core.MAX_TOKENS,'Frozen 256-token cap required');self.model.eval();self.before_unit(f'shared_rollout_context_{modality}')
        raw={'record_id':row['id'],'modality':modality,'image_sha256':digest(row['images'][0]),'rendered_prompt':context['prompt'],'prompt_token_ids':context['prompt_ids'],
             'prompt_sha256':hashlib.sha256(context['prompt'].encode()).hexdigest(),'responses':[],'numerical_policy':core.POLICY,
             'prepared_inputs_fingerprints_PRIVATE':context['fingerprints']}
        for number in range(count):
            evidence=[];observed=[]
            def observe(actual_ids,actual_pixels,actual_mask,kwargs):
                actual={'input_ids':actual_ids,'pixel_values':actual_pixels,'attention_mask':actual_mask,**{k:kwargs[k] for k in context['extra']}}
                observed.append(strict_tensor_identity(self,actual,context['inputs'],utility))
            def generation_before(name):
                self.before_unit(name)
                if name=='f32_generation_LM_call':self.counters.charge('generation_language_model_calls')
                if name=='f32_generation_vision_embeddings':self.counters.charge('generation_vision_embedding_calls');self._verify_image(modality)
            view=F32GenerationModelView(self.model,mx,generation_before,evidence,observe)
            tokenizer=self.processor.tokenizer;eos_ids=reset_effective_eos(tokenizer,self.model.config.eos_token_id)
            iterator=stream_generate(view,self.processor,context['prompt'],image=row['images'],max_tokens=core.MAX_TOKENS,temperature=1.,top_p=1.,top_k=0,min_p=0.,
                repetition_penalty=None,verbose=False,logits_processors=[CountingProcessor(mx,self.counters.charge)])
            response=capture_stream(iterator,prompt_tokens=context['prompt_length'],max_tokens=core.MAX_TOKENS,eos_ids=eos_ids,before=self.before_unit)
            self.counters.charge('generated_response_tokens',len(response['token_ids']));self._verify_image(modality)
            require(len(observed)==1 and bool(evidence),'Exactly one actual generation image preparation required')
            clean=getattr(self.processor,'clean_output',None)
            if callable(clean):response['text']=clean(response['text'])
            response['generation_dtype_cache_evidence_PRIVATE']=evidence;response['effective_eos_ids_PRIVATE']=sorted(eos_ids)
            raw['responses'].append(response)
            partial=Path(workdir)/f'rollout_progress_{modality}.partial.json'
            with partial.open('w') as handle:json.dump(raw,handle,sort_keys=True,allow_nan=False);handle.flush();os.fsync(handle.fileno())
            os.replace(partial,Path(workdir)/f'rollout_progress_{modality}_PRIVATE.json')
            del iterator,view,evidence,observed;gc.collect();mx.clear_cache()
        return raw

    # ------------------------------------------------------------ scoring / gradients
    def score_old(self,bundle):
        """Detached own-context old logps for all eight responses under the exact parent, computed once."""
        mx=self.mx;self.bind_bundle(bundle);self.model.eval();result={}
        for item in bundle.raw['responses']:
            context=self.activate(item['modality'])
            value=mx.stop_gradient(self._context_logps(context,item['token_ids'],'detached_old_forwards'));mx.eval(value)
            require(bool(mx.all(mx.isfinite(value)).item()),'Nonfinite own-context old logp')
            result[item['response_id']]=value.tolist();del value;gc.collect();mx.clear_cache()
        self._active=None;self._active_modality=None
        return result

    def _checked_gradient(self,gradient):
        mx=self.mx;flat=dict(self.tree_flatten(gradient));signature=self.trainable_signature()
        require(set(flat)==set(signature) and all(bool(mx.all(mx.isfinite(v)).item()) for v in flat.values()),'Complete finite trainable gradient required')
        return math.sqrt(sum(float(mx.sum(v.astype(mx.float32)**2).item()) for v in flat.values()))

    def arm_gradient(self,planned,own_old):
        """Serial per-response autograd of the UNCHANGED native GSPO loss; sum with weight 1/8; no optimizer here."""
        mx=self.mx;self.model.eval();node=frozen_loss_node()
        def one(item):
            response=item.response;context=self.activate(response.context.modality);fresh=[]
            previous=mx.array(own_old[response.response_id],dtype=mx.float32)
            def score(tokens):
                value=self._context_logps(context,tokens,'gradient_forwards');fresh.append(mx.stop_gradient(value));return value
            namespace={'mx':mx,'self':SimpleNamespace(_logps=score)}
            exec(compile(ast.Module(body=[node],type_ignores=[]),'<unchanged-native-GSPO-loss-AST>','exec'),namespace)
            self.counters.charge('backward_calls');self.before_unit('shared_rollout_response_gradient')
            loss,gradient=self.nn.value_and_grad(self.model,namespace['loss_fn'])(self.model,list(response.token_ids),previous,item.advantage)
            mx.eval(loss,gradient,fresh);require(len(fresh)==1,'One fresh own-context scorer per GSPO backward required')
            error=float(mx.max(mx.abs(fresh[0]-previous)).item());self._checked_gradient(gradient)
            result={'loss':float(loss.item()),'gradient':gradient,'old_current_max_abs':error}
            del fresh,namespace,loss;return result
        def accumulate(left,right):
            total=self.tree_map(lambda a,b:a+b,left,right);mx.eval(total);return total
        def scale(tree,weight):
            scaled=self.tree_map(lambda g:g.astype(mx.float32)*weight,tree);mx.eval(scaled);gc.collect();mx.clear_cache();return scaled
        total,evidence=core.weighted_gradient_sum(planned.plan,one,scale,accumulate)
        self._active=None;self._active_modality=None
        if total is None:return None,{**evidence,'gradient_l2':0.,'native_loss_AST_sha256':hashlib.sha256(ast.dump(node).encode()).hexdigest()}
        mx.eval(total)
        return total,{**evidence,'gradient_l2':self._checked_gradient(total),'native_loss_AST_sha256':hashlib.sha256(ast.dump(node).encode()).hexdigest()}

    def apply_update(self,gradient):
        """The ONE persistent-Adam optimizer call of this arm."""
        require(self._arm_updates==0,'A second optimizer update in one arm is forbidden')
        norm=self._checked_gradient(gradient);require(norm>0,'Zero gradient cannot update')
        self.before_unit('shared_rollout_one_optimizer_update');self._arm_updates=1
        self.optimizer.update(self.model,gradient);self.mx.eval(self.model.trainable_parameters(),self.optimizer.state);self.mx.clear_cache()
        return {'optimizer_calls':1,'gradient_l2':norm,'optimizer_profile':'persistent Adam 1e-5, no bias correction, scheduler unchanged'}


def backend_factory(*,model_path,contract,seed,before_unit,cache=None):
    """Inject into the driver only behind the independently reviewed admission gate."""
    return lambda adapter,config:SharedRolloutBackend(adapter,config,model_path=model_path,contract=contract,seed=seed,before_unit=before_unit,cache=cache)
