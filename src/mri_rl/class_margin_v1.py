"""Read-only candidate margins along saved native GSPO gradients.

No loader, generation, optimizer, or loss modification. Native functions require
an independently reviewed caller. Primary scores normalize label length; they
are not calibrated posteriors or probabilities of completed answer strings.
"""
from __future__ import annotations
import gc
import math
import re
from pathlib import Path
from mri_rl.conditional_gspo_gradient_v2 import require


def budget():
    return {'margin_backward_calls': 14, 'language_model_calls': 33,
            'vision_embedding_calls': 16, 'generation_calls': 0,
            'optimizer_updates': 0, 'maximum_seconds': 1700,
            'simultaneous_image_autograd_graphs': 1, 'model_loads': 1}


def margin_scores(correct, wrong):
    require(len(correct)>0 and len(wrong)>0 and all(math.isfinite(v) for v in [*correct,*wrong]), 'Finite nonempty label spans required')
    return {'mean': sum(correct)/len(correct)-sum(wrong)/len(wrong),
            'sum': sum(correct)-sum(wrong), 'first': correct[0]-wrong[0]}


def margin_directions(correct, wrong, ncorrect, nwrong, *, first=None):
    require(ncorrect>0 and nwrong>0 and all(math.isfinite(v) for v in (correct,wrong)), 'Finite class derivatives required')
    if ncorrect == nwrong == 1: first = correct-wrong
    require(first is not None and math.isfinite(first), 'Multi-token first direction requires separate shared-prefix gradient')
    return {'mean': correct-wrong, 'sum': ncorrect*correct-nwrong*wrong, 'first': first}


def saved_prefix(text, truth):
    # Deliberately no normalization/salvage of escaped labels or formatting.
    match = re.fullmatch(r'(<think>[^<]*</think><answer>\\boxed\{)('+re.escape(truth)+r')\}</answer>', text)
    require(match is not None, 'Saved response0 must contain exact strict canonical truth')
    return match.group(1)


def score_alignment(actual, previous, tolerance=.002):
    require(len(actual)==len(previous)>0 and all(math.isfinite(v) for v in [*actual,*previous]), 'Label alignment length/nonfinite failure')
    error=max(abs(a-b) for a,b in zip(actual,previous))
    require(error<=tolerance, 'Same causal-prefix score alignment exceeds frozen .002')
    return {'within_tolerance': True, 'max_abs_error': error, 'tolerance': tolerance,
            'gradient_parity': 'NOT_ESTABLISHED_BY_SCORE_AGREEMENT'}


def directional_dot(chunks):
    import numpy as np
    dot=gg=hh=0.; count=0
    for name,shape,g,h in chunks:
        g=np.asarray(g,dtype=np.float64).ravel(); h=np.asarray(h,dtype=np.float64).ravel()
        require(g.shape==h.shape and g.size==math.prod(shape) and np.isfinite(g).all() and np.isfinite(h).all(), 'Gradient inventory/nonfinite failure')
        dot+=float(g@h); gg+=float(g@g); hh+=float(h@h); count+=1
    require(count>0,'No gradient tensors')
    return {'derivative_along_saved_descent': -dot, 'dot': dot, 'saved_group_l2': math.sqrt(gg),
            'candidate_gradient_l2': math.sqrt(hh), 'tensors': count}


def token_plan(group, encode):
    from mri_rl.answer_likelihood import PREFIX, SUFFIX
    task=group['row']['ability']; truth=group['row']['reward_model']['ground_truth']
    classes={'mri_sequence': ('FLAIR','contrast-enhanced T1'), 'mri_visit_correspondence': ('same','different')}
    require(task in classes and truth in classes[task] and group['row']['extra_info']['internal_split']=='train','Fixed strict TRAIN class mapping required')
    wrong=next(c for c in classes[task] if c!=truth)
    historical=saved_prefix(group['raw']['responses'][0]['text'],truth)
    contexts=[]
    for name,prefix in [('COMMON_ANSWER_PREFIX',PREFIX),('SAVED_RESPONSE0_PREFIX',historical)]:
        existing=next((c for c in contexts if c['prefix']==prefix),None)
        if existing: existing['aliases'].append(name); continue
        ids=list(encode(prefix)); labels={}
        for role,label in [('correct',truth),('wrong',wrong)]:
            tokens=list(encode(label)); require(tokens and list(encode(prefix+label+SUFFIX))==ids+tokens+list(encode(SUFFIX)), 'Tokenizer boundary overlaps label/prefix/suffix')
            labels[role]={'text':label,'token_ids':tokens}
        require(labels['correct']['token_ids'][0]!=labels['wrong']['token_ids'][0], 'First-token contrast not identifiable')
        contexts.append({'name':name,'aliases':[name],'prefix':prefix,'prefix_ids':ids,'labels':labels})
    return {'task':task,'truth':truth,'contexts':contexts}


def _state(backend,utility):
    from mri_rl.optimizer_checkpoint import pack_tree
    layout,arrays=pack_tree({'optimizer':backend.optimizer_state(),'rng':backend.rng_state()})
    return {'layout':layout,'arrays':utility._fingerprints(backend,arrays)}


def _forward(backend,inputs,context,label_ids,first_pair,evidence):
    """One F32 full scorer, retaining only the requested label or shared first row."""
    from mri_rl.native_probability_resolution_v1 import activation_control
    mx=backend.mx; prompt=inputs['input_ids']; P=int(prompt.shape[-1]); prefix=context['prefix_ids']
    continuation=prefix+(label_ids[:-1] if first_pair is None else [])
    ids=mx.concatenate((prompt,mx.array(continuation,dtype=prompt.dtype)[None,:]),axis=1)
    extra={k:v for k,v in inputs.items() if k not in ('input_ids','pixel_values','attention_mask')}
    require(not {'cache','cached_image_features','image_features','inputs_embeds'} & set(extra),'Feature/cache injection forbidden')
    features=backend.model.get_input_embeddings(ids,inputs['pixel_values'],**extra).to_dict()
    features=activation_control({k:v for k,v in features.items() if v is not None},mx.float32)
    output=backend.model.language_model(ids,mask=mx.ones_like(ids),pixel_values=inputs['pixel_values'],**{**extra,**features})
    require(output.logits.dtype==mx.float32 and features['inputs_embeds'].dtype==mx.float32 and all(v.dtype==mx.float32 for v in features.get('deepstack_visual_embeds',[])), 'F32 scorer endpoint/features required')
    require(features['position_ids'].shape[-1]==ids.shape[-1], 'Candidate full context positions differ')
    start=P+len(prefix)-1
    logits=output.logits[0,start:start+(len(label_ids) if first_pair is None else 1),:]
    lp=backend.nn.log_softmax(logits.astype(mx.float32),axis=-1)
    values=(mx.take_along_axis(lp,mx.array(label_ids)[:,None],axis=-1)[:,0] if first_pair is None else lp[0,mx.array(first_pair)])
    evidence.append({'prompt_length':P,'prefix_length':len(prefix),'label_length':len(label_ids), 'full_input_length':int(ids.shape[-1]),'logits_dtype':str(output.logits.dtype),'embedding_dtype':str(features['inputs_embeds'].dtype),'deepstack_dtypes':[str(v.dtype) for v in features.get('deepstack_visual_embeds',[])], 'objective':'LABEL_MEAN' if first_pair is None else 'SHARED_PREFIX_FIRST_CONTRAST'})
    return values


def qualification_gate(comparison, direct_max, dtype):
    require(dtype and comparison['max_full_vocab_normalized_logp_delta'] <= .002 and math.isfinite(direct_max) and direct_max <= .002, 'Dallas F32 KV/full fixed .002 gate failed')
    return True


def dallas_qualification(backend,group,inputs,out,utility,before,charge,make_cache):
    """One prebound Dallas saved response, no sequence re-run or generation."""
    import numpy as np
    from mri_rl import native_probability_resolution_v1 as q
    mx=backend.mx; out=Path(out); out.mkdir(exist_ok=False)
    require(group['row']['ability']=='mri_visit_correspondence','Dallas only qualification')
    tokens=group['raw']['responses'][0]['token_ids']; ids=inputs['input_ids']; P=int(ids.shape[-1])
    require(P==405 and len(tokens)==17,'Predetermined Dallas P405/T17 required')
    full=mx.concatenate((ids,mx.array(tokens,dtype=ids.dtype)[None,:]),axis=1)[:,:-1]
    extra={k:v for k,v in inputs.items() if k not in ('input_ids','pixel_values','attention_mask')}
    before('Dallas_prompt_features');charge('vision_embedding_calls')
    prompt={k:v for k,v in backend.model.get_input_embeddings(ids,inputs['pixel_values'],mask=inputs['attention_mask'],**extra).to_dict().items() if v is not None}; mx.eval(prompt)
    before('Dallas_full_features');charge('vision_embedding_calls')
    features={k:v for k,v in backend.model.get_input_embeddings(full,inputs['pixel_values'],**extra).to_dict().items() if v is not None}; mx.eval(features)
    require(q.feature_fingerprints(backend,utility,prompt)==q.feature_fingerprints(backend,utility,q.prefix_features(features,P)), 'Dallas prompt/full feature prefix differs')
    embeds=backend.model.language_model.model.embed_tokens(mx.array(tokens[:-1],dtype=ids.dtype)[None,:])
    require(bool(mx.array_equal(embeds,features['inputs_embeds'][:,P:]).item()), 'Dallas saved suffix embedding differs')
    controlled=q.activation_control(features,mx.float32)
    require(bool(mx.array_equal(controlled['inputs_embeds'],features['inputs_embeds'].astype(mx.float32)).item()) and all(bool(mx.array_equal(a,b.astype(mx.float32)).item()) for a,b in zip(controlled.get('deepstack_visual_embeds',[]),features.get('deepstack_visual_embeds',[]))), 'Dallas F32 value preservation failure')
    utility.durable(out/'context_PRIVATE.json',{'P':P,'T':len(tokens),'prompt_features':q.feature_fingerprints(backend,utility,prompt),'prefix_features':q.feature_fingerprints(backend,utility,q.prefix_features(features,P)),'position_matrix_PRIVATE':features['position_ids'].tolist(),'rope_delta_PRIVATE':features['rope_deltas'].tolist()})
    kv=q._kv_route(backend,utility,inputs,q.prefix_features(controlled,P),controlled,tokens,q.ROUTES[2],out,before,charge,make_cache)
    tf=q._full_route(backend,utility,full,controlled,inputs['pixel_values'],extra,tokens,P,'DALLAS_F32_FULL_TF',out,before,charge)
    a=q.load_f32_arrays(out/(q.ROUTES[2]+'_vocab_PRIVATE.safetensors')); b=q.load_f32_arrays(out/'DALLAS_F32_FULL_TF_vocab_PRIVATE.safetensors')
    cmp=q.distribution_comparison(a['raw_logits_F32'],b['raw_logits_F32'],tokens)
    direct=float(np.max(np.abs(a['stable_logps_F32'].astype(np.float64)-b['frozen_scorer_logps_F32'].astype(np.float64))))
    rows=kv['per_position_dtype_PRIVATE']; caches=[c for event in kv['causal_cache_trace_PRIVATE'] for c in (event.get('cache') or [])]
    dtype=all(r['logits_dtype']==str(mx.float32) for r in rows) and tf['logits_dtype']==str(mx.float32) and bool(caches) and all(c['keys_dtype']==c['values_dtype']==str(mx.float32) for c in caches)
    selected_direct=max(abs(float(a['stable_logps_F32'][i,t])-float(b['frozen_scorer_logps_F32'][i,t])) for i,t in enumerate(tokens))
    result={'comparison':cmp,'direct_AR_vs_centered_full_max_abs':direct,'direct_AR_vs_centered_full_selected_max_abs':selected_direct,'actual_F32_endpoints_cache':dtype,'tolerance':.002}
    utility.durable(out/'comparison_PRIVATE.json',result)
    qualification_gate(cmp,direct,dtype)
    return {k:v for k,v in result.items() if k!='comparison'} | {'comparison':{k:v for k,v in cmp.items() if not k.endswith('_PRIVATE')}}


def run_group(backend,group,plan,inputs,saved_dir,out,utility,before,charge,*,qualify=False,make_cache=None):
    from mri_rl.native_gspo_path_probe_v1 import label_support
    mx=backend.mx; out=Path(out); out.mkdir(exist_ok=False); saved_dir=Path(saved_dir)
    state=_state(backend,utility); weights=utility._weights(backend); full=utility._fingerprints(backend,dict(backend.tree_flatten(backend.model.parameters())))
    language=backend.model.language_model; metadata=(language._rope_deltas,language._position_ids)
    def integrity(): require(_state(backend,utility)==state and utility._weights(backend)==weights,'Margin changed Adam/RNG/adapter')
    records=[]; qualification=None
    try:
        require(len(full)==1942 and group['row']['extra_info']['internal_split']=='train','Full1942/TRAIN required')
        if qualify:
            qualification=dallas_qualification(backend,group,inputs,out/'Dallas_qualification_PRIVATE',utility,before,charge,make_cache);integrity()
        def decode(tokens): return backend.processor.tokenizer.decode(tokens[:-1] if tokens and tokens[-1] in (151643,151645) else tokens)
        response=group['raw']['responses'][0]; support=label_support(response['token_ids'],decode,group['row']['ability'])['mask']
        positions=[i for i,v in enumerate(support) if v]
        import json
        for condition in ('REAL','ZERO_NORMALIZED_PIXELS'):
            current=inputs if condition=='REAL' else {**inputs,'pixel_values':mx.zeros_like(inputs['pixel_values'])}; mx.eval(current)
            if condition!='REAL': utility.validate_condition_fingerprints(utility._fingerprints(backend,inputs),utility._fingerprints(backend,current))
            for context in plan['contexts']:
                values={}; paths={}; evidence=[]
                objectives=[(role,label['token_ids'],None) for role,label in context['labels'].items()]
                counts=[len(context['labels'][role]['token_ids']) for role in ('correct','wrong')]
                if counts!=[1,1]: objectives.append(('first_contrast',[],[context['labels'][r]['token_ids'][0] for r in ('correct','wrong')]))
                for role,tokens,pair in objectives:
                    captured=[]
                    def objective(model):
                        value=_forward(backend,current,context,tokens,pair,evidence); captured.append(mx.stop_gradient(value))
                        return mx.mean(value) if pair is None else value[0]-value[1]
                    before('margin_'+condition+'_'+context['name']+'_'+role)
                    for key in ('margin_backward_calls','language_model_calls','vision_embedding_calls'):charge(key)
                    vg=backend.nn.value_and_grad(backend.model,objective); score,grad=vg(backend.model);mx.eval(score,grad,captured)
                    require(len(captured)==1 and math.isfinite(float(score.item())),'One finite native margin forward required')
                    flat=dict(backend.tree_flatten(grad));require(set(flat)==set(weights) and all(bool(mx.all(mx.isfinite(v)).item()) for v in flat.values()),'Finite504 native margin gradient required')
                    path=out/(condition+'_'+context['name']+'_'+role+'_gradient_PRIVATE.safetensors')
                    backend.save_arrays(path,{k:v.astype(mx.float32) for k,v in flat.items()});utility._fsync(path)
                    values[role]=captured[0].tolist();paths[role]=str(path)
                    del grad,flat,score,vg,captured,objective;gc.collect();mx.clear_cache();integrity()
                align={}
                if 'SAVED_RESPONSE0_PREFIX' in context['aliases']:
                    old=json.loads((saved_dir/(condition+'_old_PRIVATE.json')).read_text())[0]
                    require([response['token_ids'][i] for i in positions]==context['labels']['correct']['token_ids'],'Saved strict label mask not identical to new canonical span')
                    align['saved_B3_label_logps']=score_alignment(values['correct'],[old[i] for i in positions])
                if 'first_contrast' in values:
                    align['shared_first_correct']=score_alignment([values['first_contrast'][0]],[values['correct'][0]])
                    align['shared_first_wrong']=score_alignment([values['first_contrast'][1]],[values['wrong'][0]])
                directions={}
                for saved_condition in ('REAL','ZERO_NORMALIZED_PIXELS'):
                    group_path=saved_dir/(saved_condition+'_FULL_group_gradient_PRIVATE.safetensors')
                    dots={role:directional_dot(utility._gradient_chunks(group_path,path)) for role,path in paths.items()}
                    require(all(d['tensors']==504 for d in dots.values()),'Saved/new504gradient inventory required')
                    directions[saved_condition]={'correct':dots['correct'],'wrong':dots['wrong'],'margins':margin_directions(dots['correct']['derivative_along_saved_descent'],dots['wrong']['derivative_along_saved_descent'],*counts,first=dots.get('first_contrast',{}).get('derivative_along_saved_descent'))}
                record={'context':context['name'],'aliases':context['aliases'],'image_condition':condition,'scores':margin_scores(values['correct'],values['wrong']),'class_logps_PRIVATE':values,'directions':directions,'alignment':align,'scorer_context_dtype':evidence,'new_gradient_paths_PRIVATE':paths}
                utility.durable(out/(condition+'_'+context['name']+'_PRIVATE.json'),record);records.append(record);integrity()
        return {'records_PRIVATE':records,'Dallas_qualification':qualification,'saved_B3_response0_reference':json.loads((saved_dir/'measurements_SAFE.json').read_text())['directions']['responses'][0], 'saved_reference_denominator':len(response['token_ids']), 'reference_interpretation':'Saved B3 label-supported derivative uses full response T denominator, not new label mean; score agreement does not establish gradient parity.'}
    finally:
        try:
            after=utility._fingerprints(backend,dict(backend.tree_flatten(backend.model.parameters()))); after_state=_state(backend,utility)
            utility.durable(out/'exit_integrity_PRIVATE.json',{'all_parameters_before':full,'all_parameters_after':after,'state_before':state,'state_after':after_state,'unchanged':full==after and state==after_state})
            require(full==after and state==after_state,'Margin full-model/Adam/RNG exit integrity failure')
        finally:
            language._rope_deltas,language._position_ids=metadata;gc.collect();mx.clear_cache()
