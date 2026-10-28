"""Versioned F32 activation GSPO backend; IMPLEMENTED_CPU_NOT_NATIVE_QUALIFIED.

No MLX import at module import. The inherited loader/Adam/state/GSPO update remain
unchanged. Delegating generation views cast functional activations, never model
weights or registered structure. Installed AR sampling/chunking/stopping and
streaming detokenizer are reused. A future independently reviewed native
qualification is mandatory; CPU mocks are not numerical-policy qualification.
"""
from __future__ import annotations
import gc
import hashlib
import json
import math
import os
from pathlib import Path
from mri_rl.mlx_continuous_gspo import NativeGSPOBackend
from mri_rl.continuous_rl import require,digest
from mri_rl.rollout_precision_v2 import append_fp32_logits_processor
from mri_rl.conditional_gspo_gradient_v2 import f32_activation_response_logps

STATUS='IMPLEMENTED_CPU_NOT_NATIVE_QUALIFIED'
POLICY='F32_ACTIVATION_GENERATION_SCORER_GSPO_V1'


def require_prompt_cost(task,length):
    require(task in ('mri_sequence','mri_visit_correspondence') and length=={'mri_sequence':213,'mri_visit_correspondence':405}[task], 'Actual prompt length differs from fixed ledger input cost')
    return length


class ObservedFP32Processor:
    """Last AR processor; same F32 materialization barrier with/without capture."""
    def __init__(self,mx,observer=None):self.mx=mx;self.observer=observer
    def __call__(self,tokens,logits):
        result=logits.astype(self.mx.float32);self.mx.eval(result)
        if self.observer is not None:self.observer(result)
        return result


class VocabularyWriter:
    """PRIVATE row stream; one CPU vocabulary row resident, no retained model graph."""
    def __init__(self,path):self.path=Path(path);self.handle=self.path.open('xb');self.rows=0;self.vocab=None
    def __call__(self,logits):
        import numpy as np
        row=np.asarray(logits,dtype=np.float32)
        require(row.ndim==2 and row.shape[0]==1 and np.isfinite(row).all(),'Finite F32 generation vocabulary required')
        if self.vocab is None:self.vocab=int(row.shape[1])
        require(row.shape[1]==self.vocab and self.rows<257,'Generation vocabulary shape/call budget changed')
        self.handle.write(row.astype('<f4',copy=False).tobytes());self.rows+=1
    def close(self):
        if not self.handle.closed:self.handle.flush();os.fsync(self.handle.fileno());self.handle.close()
    def finalize(self,returned):
        self.close();require(self.rows==len(returned)+1,'Installed AR one-token lookahead count differs')
        return {'path_PRIVATE':str(self.path),'sha256':digest(self.path),'rows_including_unreturned_lookahead':self.rows,
                'returned_rows':len(returned),'vocabulary':self.vocab,'dtype':'F32_LE','lookahead_rows':1}


class FeatureView:
    def __init__(self,features):self._features=features
    def to_dict(self):return dict(self._features)
    def __getattr__(self,key):
        if key in self._features:return self._features[key]
        raise AttributeError(key)


class F32LanguageView:
    def __init__(self,language,mx,before,evidence):
        self._language=language;self._mx=mx;self._before=before;self._evidence=evidence
    def __getattr__(self,key):return getattr(self._language,key)
    def __call__(self,*args,**kwargs):
        mx=self._mx;self._before('f32_generation_LM_call')
        ids=args[0] if args else kwargs.get('inputs')
        embeds=kwargs.get('inputs_embeds')
        if embeds is None:
            require(ids is not None,'F32 generation requires explicit token input')
            embeds=self._language.model.embed_tokens(ids)
        kwargs={**kwargs,'inputs_embeds':embeds.astype(mx.float32)}
        if kwargs.get('deepstack_visual_embeds') is not None:
            kwargs['deepstack_visual_embeds']=[v.astype(mx.float32) for v in kwargs['deepstack_visual_embeds']]
        result=self._language(*args,**kwargs)
        require(result.logits.dtype==mx.float32,'F32 generation logits endpoint downcast')
        cache=[]
        for layer in kwargs.get('cache') or []:
            keys,values=layer.state
            require(keys.dtype==values.dtype==mx.float32,'F32 generation KV cache endpoint downcast')
            cache.append({'keys_dtype':str(keys.dtype),'values_dtype':str(values.dtype),'offset':int(layer.offset)})
        self._evidence.append({'inputs_embeds_dtype':str(kwargs['inputs_embeds'].dtype),
            'logits_dtype':str(result.logits.dtype),'input_tokens':int(kwargs['inputs_embeds'].shape[1]),'cache':cache})
        return result


class F32GenerationModelView(dict):
    """Shallow mapping view preserves wired_limit's parameter-tree inventory.

    nn.Module is a dict subclass. Values remain the exact original module/weight
    objects. Attribute calls delegate to the real model except the two functional
    generation interfaces below; no setter/monkeypatch touches the real model.
    """
    def __init__(self,model,mx,before,evidence,observe_inputs=None):
        super().__init__(model if isinstance(model,dict) else {})
        self._model=model;self._mx=mx;self._before=before;self._evidence=evidence;self._observe_inputs=observe_inputs
        self._language_view=F32LanguageView(model.language_model,mx,before,evidence)
    def __getattr__(self,key):return getattr(self._model,key)
    @property
    def language_model(self):return self._language_view
    def get_input_embeddings(self,input_ids,pixel_values,mask=None,**kwargs):
        self._before('f32_generation_vision_embeddings')
        if self._observe_inputs:self._observe_inputs(input_ids,pixel_values,mask,kwargs)
        original=self._model.get_input_embeddings(input_ids,pixel_values,mask=mask,**kwargs)
        from mri_rl.native_probability_resolution_v1 import activation_control
        result=activation_control(original.to_dict(),self._mx.float32)
        require(result['inputs_embeds'].dtype==self._mx.float32 and all(v.dtype==self._mx.float32 for v in result.get('deepstack_visual_embeds') or []),'F32 generation image-feature conversion failed')
        return FeatureView(result)


def generation_alignment(sampled,full):
    require(len(sampled)==len(full)>0 and all(math.isfinite(float(v)) for v in [*sampled,*full]),'Finite complete generated/scorer vectors required')
    error=max(abs(float(a)-float(b)) for a,b in zip(sampled,full))
    require(error<=.002,'Actual rollout/scorer selected logp mismatch exceeds .002')
    return {'max_abs_error':error,'tolerance':.002,'positions':len(full)}


def reset_effective_eos(tokenizer,configured):
    tokenizer.stopping_criteria.reset(configured)
    values=tokenizer.stopping_criteria.eos_token_ids
    require(values and all(type(v) is int and v>=0 for v in values),'Installed effective EOS set unavailable')
    return set(values)


def capture_stream(iterator,*,prompt_tokens,max_tokens,eos_ids,before):
    """Capture installed stream, including its final EOS/duplicate-last receipt."""
    values={};pieces=[];final=None
    try:
        for step in iterator:
            before('f32_stream_capture')
            n=step.generation_tokens;token=step.token
            require(step.prompt_tokens==prompt_tokens and type(n) is int and 1<=n<=max_tokens and type(token) is int,'Stream prompt/count/token mismatch')
            lp=step.logprobs
            require(len(lp.shape)==1 and 0<=token<lp.shape[0],'Stream full vocabulary shape differs')
            selected=float(lp[token].item());require(math.isfinite(selected) and selected<=0,'Nonfinite/positive sampled logp')
            if n in values:require(values[n]==(token,selected),'Final duplicate token/logp differs')
            else:require(n==len(values)+1,'Shifted streaming token position');values[n]=(token,selected)
            pieces.append(step.text)
            finish=getattr(step,'finish_reason',None)
            if finish is not None:
                ids=[values[i][0] for i in range(1,len(values)+1)]
                require(finish in ('stop','length') and step.token_ids==ids and
                        not any(t in eos_ids for t in ids[:-1]) and
                        ((finish=='stop' and ids[-1] in eos_ids) or (finish=='length' and len(ids)==max_tokens and ids[-1] not in eos_ids)),'EOS/termination receipt differs')
                final={'text':''.join(pieces),'token_ids':ids,'raw_selected_logps':[values[i][1] for i in range(1,len(values)+1)],
                    'generation_tokens':len(ids),'finish_reason':finish,'EOS_coverage':'OBSERVED_AND_SCORED' if finish=='stop' else 'NOT_OBSERVED_LENGTH_LIMIT'}
                break
    finally:
        close=getattr(iterator,'close',None)
        if callable(close):close()
    require(final is not None,'Missing complete final stream receipt')
    return final


def f32_response_logps_observed(backend, tokens, inputs, *, evidence=None, vocab_observer=None):
    """Functional Task1 F32 full-TF path; model/weight dtypes stay unchanged.

    Recompute image features inside each call; no cached-image-feature shortcut.
    As in accepted Task1 full route, normalize selected response-logit rows with
    actual backend.nn.log_softmax. Different saved response lengths are disclosed
    in the B3 scope; only the first 17-token B2 instance was KV/full qualified.
    """
    from mri_rl.native_probability_resolution_v1 import activation_control
    mx = backend.mx; prompt = inputs['input_ids']; P = int(prompt.shape[-1])
    ids = mx.concatenate((prompt, mx.array(tokens, dtype=prompt.dtype)[None, :]), axis=1)[:, :-1]
    extra = {k: v for k, v in inputs.items() if k not in ('input_ids', 'pixel_values', 'attention_mask')}
    require(not {'cache', 'cached_image_features', 'image_features', 'inputs_embeds'} & set(extra), 'Unreviewed feature/cache injection')
    features = backend.model.get_input_embeddings(ids, inputs['pixel_values'], **extra).to_dict()
    features = activation_control({k: v for k, v in features.items() if v is not None}, mx.float32)
    output = backend.model.language_model(ids, mask=mx.ones_like(ids), pixel_values=inputs['pixel_values'], **{**extra, **features})
    require(output.logits.dtype == mx.float32 and features['inputs_embeds'].dtype == mx.float32 and
            all(v.dtype == mx.float32 for v in features.get('deepstack_visual_embeds', [])), 'F32 full scorer endpoint/input dtype differs')
    require(features['position_ids'].shape[-1] == ids.shape[-1], 'Full scorer position context length differs')
    if evidence is not None:
        evidence.append({'prompt_tokens': P, 'response_tokens': len(tokens), 'full_input_tokens': int(ids.shape[-1]),
            'inputs_embeds_dtype': str(features['inputs_embeds'].dtype), 'logits_dtype': str(output.logits.dtype),
            'deepstack_dtypes': [str(v.dtype) for v in features.get('deepstack_visual_embeds', [])], 'cache': 'NONE_FULL_CAUSAL'})
    logits = output.logits[:, P-1:P-1+len(tokens), :].astype(mx.float32)
    if vocab_observer is not None: vocab_observer(mx.stop_gradient(logits))
    result = mx.take_along_axis(backend.nn.log_softmax(logits, axis=-1), mx.array(tokens)[None, :, None], axis=-1)[0, :, 0]
    require(result.shape[0] == len(tokens), 'F32 response scorer alignment differs')
    return result


class F32NativeGSPOBackend(NativeGSPOBackend):
    """Native loader/optimizer/state inherited; new functional numerical stage."""
    def generate_record(self,*args,**kwargs):raise ValueError('F32_EVALUATION_NOT_QUALIFIED: separate matched F32 eval integration required')
    def greedy_generate(self,*args,**kwargs):raise ValueError('F32_EVALUATION_NOT_QUALIFIED: separate matched F32 eval integration required')
    def score_record(self,*args,**kwargs):raise ValueError('F32_EVALUATION_NOT_QUALIFIED: separate matched F32 eval integration required')

    def _logps(self,tokens):
        require(self.active is not None,'No active exact on-policy group')
        return f32_response_logps_observed(self,tokens,self.active['inputs'],vocab_observer=self.active.get('vocab_observer'))

    def rollout(self,row,index,workdir):
        from mri_rl.paired_z045_continuation_entry_v1 import verify_images
        from mri_rl.native_gspo_path_probe_v1 import strict_tensor_identity
        from mri_rl import early_native_conditioning_probe_v1 as utility
        from mlx_vlm.generate.dispatch import stream_generate,_prepare_generation_inputs
        mx=self.mx;self.active=None;self.model.eval();self.before_unit('f32_rollout_group_'+str(index))
        verify_images([row],getattr(self,'expected_image_sha256',{}))
        require(self.config['group_size']==4 and self.config['max_tokens']==256,'FrozenK4/256 required')
        prompt=self.template(self.processor,self.model.config,[{'role':'system','content':self.contract.system_instruction},
            {'role':'user','content':row['extra_info']['question']}],add_generation_prompt=True,num_images=1,enable_thinking=self.contract.enable_thinking)
        require(prompt.count('<|image_pad|>')==1 and prompt.endswith(self.contract.assistant_prefix),'Native prompt placeholder/prefix differs')
        inputs=self.prepare_inputs(self.processor,images=row['images'],prompts=prompt,image_token_index=getattr(self.model.config,'image_token_index',None),add_special_tokens=self.special_tokens(self.model.config.model_type,self.processor))
        ids,pixels,mask,extra=_prepare_generation_inputs(self.model,self.processor,prompt,row['images'],None,None,
            {'max_tokens':256,'temperature':1.,'top_p':1.,'top_k':0,'min_p':0.,'repetition_penalty':None})
        prepared={'input_ids':ids,'pixel_values':pixels,'attention_mask':mask,**extra}
        input_fingerprints=strict_tensor_identity(self,prepared,inputs,utility)
        length=int(ids.shape[-1]);require_prompt_cost(row['ability'],length);require(0<length<=2048 and bool(mx.all(mask==1).item()),'Bounded unpadded native prompt required')
        self.active={'record_id':row['id'],'inputs':inputs,'prompt_length':length}
        raw={'record_id':row['id'],'group_index':index,'image_sha256':digest(row['images'][0]),'rendered_prompt':prompt,
             'prompt_token_ids':ids.tolist(),'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'responses':[],
             'numerical_policy':POLICY,'prepared_inputs_fingerprints_PRIVATE':input_fingerprints}
        for number in range(4):
            evidence=[];observed=[]
            def observe(actual_ids,actual_pixels,actual_mask,kwargs):
                actual={'input_ids':actual_ids,'pixel_values':actual_pixels,'attention_mask':actual_mask,
                        **{k:kwargs[k] for k in extra}}
                observed.append(strict_tensor_identity(self,actual,inputs,utility))
            def generation_before(name):
                self.before_unit(name)
                if name=='f32_generation_vision_embeddings':verify_images([row],self.expected_image_sha256)
            view=F32GenerationModelView(self.model,mx,generation_before,evidence,observe)
            tokenizer=self.processor.tokenizer
            # Same default setup performed by installed generate() before stream.
            eos_ids=reset_effective_eos(tokenizer,self.model.config.eos_token_id)
            writer=VocabularyWriter(Path(workdir)/f'response_{number}_generation_vocab_PRIVATE.bin') if getattr(self,'qualification_vocab_capture',False) else None
            iterator=stream_generate(view,self.processor,prompt,image=row['images'],max_tokens=256,temperature=1.,top_p=1.,top_k=0,min_p=0.,
                repetition_penalty=None,verbose=False,logits_processors=[ObservedFP32Processor(mx,writer)])
            try:response=capture_stream(iterator,prompt_tokens=length,max_tokens=256,eos_ids=eos_ids,before=self.before_unit)
            finally:
                if writer is not None:writer.close()
            if writer is not None:response['generation_full_vocab_PRIVATE']=writer.finalize(response['token_ids'])
            verify_images([row],self.expected_image_sha256)
            require(len(observed)==1 and bool(evidence),'Exactly one actual generation image preparation required')
            clean=getattr(self.processor,'clean_output',None)
            if callable(clean):response['text']=clean(response['text'])
            response['generation_dtype_cache_evidence_PRIVATE']=evidence
            response['effective_eos_ids_PRIVATE']=sorted(eos_ids)
            raw['responses'].append(response)
            partial=Path(workdir)/'rollout_progress.partial.json'
            with partial.open('w') as handle:json.dump(raw,handle,sort_keys=True,allow_nan=False);handle.flush();os.fsync(handle.fileno())
            os.replace(partial,Path(workdir)/'rollout_progress_PRIVATE.json')
            del iterator,view,evidence,observed;gc.collect();mx.clear_cache()
        self.active['raw_sha256']=hashlib.sha256(json.dumps(raw,sort_keys=True).encode()).hexdigest()
        return raw

    def update(self,row,raw,advantages):
        # This gate precedes the inherited loss and its single optimizer update.
        alignments=[]
        for response in raw['responses']:
            self.before_unit('f32_rollout_vs_full_logps')
            full=self._logps(response['token_ids']);self.mx.eval(full)
            alignments.append(generation_alignment(response['raw_selected_logps'],full.tolist()))
        result=super().update(row,raw,advantages)
        return {**result,'rollout_full_selected_logp_alignment':alignments,'numerical_policy':POLICY}
