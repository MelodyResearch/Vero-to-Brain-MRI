"""Independent CPU source/selected-logp analysis; no native imports or launches."""
from __future__ import annotations
import __future__
import argparse
import ast
import functools
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np


def sha(path):
    with Path(path).open('rb') as handle: return hashlib.file_digest(handle, 'sha256').hexdigest()


def read(path): return json.loads(Path(path).read_text())


def extract(path, name, *, parent=None):
    tree = ast.parse(Path(path).read_text())
    if parent is not None: tree = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == parent)
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)


def execute_ast(nodes, namespace):
    tree = ast.Module(body=nodes, type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), '<CPU-extracted-source>', 'exec', flags=__future__.annotations.compiler_flag), namespace)
    return namespace


def selected_summary(left, right):
    assert len(left) == len(right) > 0 and all(math.isfinite(x) for x in left+right)
    delta = [b-a for a, b in zip(left, right)]
    mean = math.fsum(delta)/len(delta)
    return {'positions':len(delta), 'max_abs_delta':max(map(abs, delta)),
            'mean_abs_delta':math.fsum(map(abs, delta))/len(delta), 'mean_signed_delta':mean,
            'hypothetical_exp_mean_selected_ratio':math.exp(mean),
            'positions_exceeding_002':sum(abs(x)>.002 for x in delta),
            'distribution_equivalence':'NOT_IDENTIFIABLE_FROM_SELECTED_ONLY',
            'not_a_vocabulary_offset_test':'Different response positions have different causal contexts; one selected token per context is not a vocabulary vector.'}


def common_shift_fixture():
    old=[-.5,-1.5]; z=math.log1p(math.exp(-1)); reference=[-z,-1-z]
    def normalize(values):
        peak=max(values); weights=[math.exp(x-peak) for x in values]; total=math.fsum(weights)
        return [x/total for x in weights]
    a,b=normalize(old),normalize(reference)
    return {'fixture':'Synthetic exact-BF16 logits64/63; not patient data or native kernel execution',
            'old_probability_sum':math.fsum(math.exp(x) for x in old), 'old_normalized':a,
            'reference_normalized':b,'max_categorical_delta':max(abs(x-y) for x,y in zip(a,b)),
            'implication':'Absolute returned-logp error need not alter categorical probabilities.'}


def check_logps_indexing(path, prompt, response):
    """Execute actual _logps AST with identity-marker logits, not a copied slice."""
    node=extract(path,'_logps',parent='NativeGSPOBackend'); captures={}; P=len(prompt[0]); R=len(response)
    class Marker:
        def __init__(self,length):self.shape=(1,length,200000)
        def astype(self,dtype):return self
    def model(ids,pixels,mask,**kwargs):
        captures.update(ids=ids,pixels=pixels,mask=mask,kwargs=kwargs)
        return SimpleNamespace(logits=Marker(ids.shape[-1]))
    def take(logits,labels,axis):
        assert axis == -1 and labels.shape[-1] == 1
        return np.arange(logits.shape[1])[None,:,None]*1000000+labels
    mx=SimpleNamespace(concatenate=np.concatenate,array=np.array,ones_like=np.ones_like,
                       float32=np.float32,take_along_axis=take)
    pixels=object();inputs={'input_ids':np.array(prompt,dtype=np.int64),'pixel_values':pixels,'attention_mask':np.ones((1,P),dtype=np.int64),'image_grid_thw':np.array([[1,28,22]])}
    backend=SimpleNamespace(mx=mx,nn=SimpleNamespace(log_softmax=lambda x,axis:x),model=model,
                            active={'inputs':inputs,'prompt_length':P})
    ns=execute_ast([node],{'require':lambda ok,msg: (_ for _ in ()).throw(AssertionError(msg)) if not ok else None})
    result=ns['_logps'](backend,response)
    assert result.tolist()==[(P-1+i)*1000000+token for i,token in enumerate(response)]
    assert captures['ids'].tolist()==[prompt[0]+response[:-1]]
    assert captures['pixels'] is pixels and np.all(captures['mask']==1)
    assert set(captures['kwargs'])=={'image_grid_thw'}
    return {'positions_verified':R,'first_causal_logit_position':P-1,'last_causal_logit_position':P+R-2,
            'full_TF_input_length':P+R-1,'all_input_tokens_and_masks_equal':True,
            'claim':'CPU identity-marker execution of actual indexing AST; not native attention/logit computation.'}


def check_cache_update(path,P,R):
    tree=ast.parse(Path(path).read_text()); original=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='KVCache')
    body=[n for n in original.body if isinstance(n,ast.Assign) or isinstance(n,ast.FunctionDef) and n.name in {'__init__','update_and_fetch'}]
    node=ast.ClassDef(name='KVCache',bases=[],keywords=[],body=body,decorator_list=[],type_params=[])
    ns=execute_ast([node],{'mx':np}); cache=ns['KVCache']();offsets=[]
    for i in range(R):
        start,end=(0,P) if i==0 else (P+i-1,P+i)
        values=np.arange(start,end,dtype=np.float32).reshape(1,1,end-start,1)
        keys,_=cache.update_and_fetch(values,values+1000)
        assert cache.offset==P+i and keys.flatten().tolist()==list(range(P+i))
        offsets.append(cache.offset)
    return {'offsets':offsets,'no_missing_or_duplicate_prefix_values':True,
            'claim':'NumPy execution of actual buffer/cursor AST; not native storage-precision or attention parity.'}


def check_sampler_input(path):
    """Execute actual neutral-parameter sampler AST against a nondrawing stub."""
    captured=[]
    def categorical(value): captured.append(value.copy()); return np.array(0)
    def forbidden(*args,**kwargs): raise AssertionError('Unexpected neutral-sampling filter')
    functions=[extract(path,name) for name in ('clamp_temperature','categorical_sampling','make_sampler')]
    ns=execute_ast(functions,{'mx':SimpleNamespace(random=SimpleNamespace(categorical=categorical,state=[]),compile=lambda function,**kwargs:function),
        'partial':functools.partial,
        'MIN_SAMPLING_TEMPERATURE':.01,
        **{name:forbidden for name in ('apply_top_n_sigma','apply_p_less','apply_typical_p','apply_top_p','apply_min_p','apply_xtc','apply_top_k')}})
    sampler=ns['make_sampler'](temp=1.,top_p=1.,min_p=0.,top_k=0,top_n_sigma=0.,p_less=False,typical_p=1.)
    values=np.array([[-.5,-1.5]],dtype=np.float32);sampler(values)
    assert len(captured)==1 and np.array_equal(values,captured[0]) and captured[0].dtype==values.dtype
    return {'parameters':{'temperature':1.,'top_p':1.,'min_p':0.,'top_k':0,'top_n_sigma':0.,'p_less':False,'typical_p':1.},
            'categorical_argument_equals_input_at_temperature1':True,'additional_filters_called':0,
            'native_random_draws':0,'native_cast_semantics':'NOT_EXECUTED; source adds no explicit cast here; CPU compile decorator is an identity stub',
            'distribution_scope':'F64 renormalization describes mathematical categorical probabilities of recorded scores, not empirical finite-precision random draws or sampled frequencies.'}


def check_positions(language_path,cache_path,prompt,response,grid,config):
    node=extract(language_path,'get_rope_index',parent='LanguageModel')
    ns=execute_ast([node],{'mx':np}); fn=ns['get_rope_index']
    cfg=SimpleNamespace(**{k:config[k] for k in ('image_token_id','video_token_id','vision_start_token_id')},vision_config=SimpleNamespace(spatial_merge_size=config['vision_config']['spatial_merge_size']))
    self=SimpleNamespace(config=cfg);p=np.array(prompt,dtype=np.int64);g=np.array(grid,dtype=np.int64);P=p.shape[-1];R=len(response)
    full=np.concatenate((p,np.array([response[:-1]],dtype=p.dtype)),axis=1)
    pp,pd=fn(self,p,g,attention_mask=np.ones_like(p));fp,fd=fn(self,full,g,attention_mask=np.ones_like(full))
    without_mask,_=fn(self,full,g)
    assert np.array_equal(fp,without_mask) and np.array_equal(pp,fp[:,:,:P]) and np.array_equal(pd,fd)
    for i in range(1,R): assert np.all(fp[:,0,P+i-1]==P+i-1+pd[0,0])
    causal=execute_ast([extract(cache_path,'create_causal_mask')],{'mx':np})['create_causal_mask']
    fm=causal(P+R-1);pm=causal(P)
    assert np.array_equal(fm[:P,:P],pm)
    for i in range(R):
        row=P-1+i
        assert np.all(fm[row,:row+1]) and not np.any(fm[row,row+1:])
        assert np.all(causal(1,offset=row))
    return {'prompt_length':P,'response_length':R,'grid_value_sha_authenticated':True,
            'full_prefix_and_prefill_positions_equal':True,'full_and_prompt_rope_delta_equal':True,
            'implicit_decode_positions_match_full_for_16_inputs':True,
            'all_ones_mask_equals_no_padding_positions':True,'causal_allowed_contexts_equal':True,
            'rope_delta':int(pd[0,0]),'claim':'Actual get_rope_index/create_causal_mask AST executed with NumPy; real native cache/position tensors were not persisted in old run.'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--old-worktree',type=Path,required=True);parser.add_argument('--package-root',type=Path,required=True);parser.add_argument('--reviewer-id',required=True);parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();old=a.old_worktree;pkg=a.package_root;W=Path(__file__).resolve().parents[1]
    base=old/'local_data/native_gspo_path_probe_v1';recordpath=base/'run_v3_PRIVATE/group_0_PRIVATE/B2_response_0_PRIVATE.json';scopepath=base/'scope_v3_PRIVATE.json';identitypath=base/'run_v3_PRIVATE/group_0_PRIVATE/input_identity_PRIVATE.json'
    assert sha(recordpath)=='c27e74c3c5001de4d5059911e2700507114900cc1ef70ef56652ba71eb0d9a1b'
    assert sha(scopepath)=='c6f52c5c86ddcea5d3d0dca1b98894e9ba3af7dd1ae201917fc4efb242a0d115'
    scope,record,identity=read(scopepath),read(recordpath),read(identitypath)
    configpath=Path(scope['model_path'])/'config.json';assert sha(configpath)==scope['bound_artifacts'][str(configpath)];config=read(configpath)
    assert identity['generation']==identity['scorer'];gridpin=identity['generation']['image_grid_thw'];pixels=identity['generation']['pixel_values']
    matches=[]; patches=pixels['shape'][0]
    for h in range(2,patches+1,2):
        if patches%h:continue
        width=patches//h
        if width%2:continue
        grid=np.array([[1,h,width]],dtype=np.int64)
        if hashlib.sha256(grid.tobytes()).hexdigest()==gridpin['sha256']:matches.append(grid.tolist())
    assert len(matches)==1
    prompt=scope['groups_PRIVATE'][0]['raw']['prompt_token_ids'];response=record['token_ids']
    assert response==scope['groups_PRIVATE'][0]['raw']['responses'][0]['token_ids']
    paths={'frozen_backend':W/'src/mri_rl/mlx_continuous_gspo.py','old_forced_KV_helper':old/'src/mri_rl/native_gspo_path_probe_v1.py',
           'ar':pkg/'mlx_vlm/generate/ar.py','dispatch':pkg/'mlx_vlm/generate/dispatch.py','common':pkg/'mlx_vlm/generate/common.py',
           'qwen_language':pkg/'mlx_vlm/models/qwen3_vl/language.py','qwen_model':pkg/'mlx_vlm/models/qwen3_vl/qwen3_vl.py',
           'cache':pkg/'mlx_vlm/models/cache.py','attention_base':pkg/'mlx_vlm/models/base.py','quantized_linear':pkg/'mlx/nn/layers/quantized.py',
           'sampler':pkg/'mlx_vlm/sample_utils.py','categorical_API':pkg/'mlx/core/random.pyi'}
    for key in ('ar','dispatch','qwen_language','qwen_model','cache'):
        assert sha(paths[key])==scope['source_sha256'][str(paths[key])]
    assert sha(paths['frozen_backend'])==scope['source_sha256'][str(old/'src/mri_rl/mlx_continuous_gspo.py')]
    assert sha(paths['old_forced_KV_helper'])==scope['source_sha256'][str(paths['old_forced_KV_helper'])]
    kv=record['incremental_KV_conditional_logps'];n,s,t=kv['native_dtype_selected_logps'],kv['fp32_stable_selected_logps'],record['full_teacherforce_conditional_logps']
    numeric={'native_to_TF':selected_summary(n,t),'stable_to_TF':selected_summary(s,t),'native_to_stable_same_KV':selected_summary(n,s)}
    for i in range(17):assert math.isclose(t[i]-n[i],(s[i]-n[i])+(t[i]-s[i]),abs_tol=1e-15,rel_tol=0)
    alignment=check_logps_indexing(paths['frozen_backend'],prompt,response)
    positions=check_positions(paths['qwen_language'],paths['cache'],prompt,response,matches[0],config)
    cursor=check_cache_update(paths['cache'],len(prompt[0]),len(response))
    sampler=check_sampler_input(paths['sampler'])
    common=ast.parse(paths['common'].read_text());default=next(n.value.value for n in common.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='DEFAULT_PREFILL_STEP_SIZE' for t in n.targets))
    ns=execute_ast([extract(paths['common'],'_policy_enabled'),extract(paths['common'],'_chunked_prefill_enabled')],{})
    enabled=ns['_chunked_prefill_enabled'](SimpleNamespace(language_model=SimpleNamespace()),draft_model=None)
    assert default==2048 and enabled
    rollout=ast.unparse(extract(paths['frozen_backend'],'rollout',parent='NativeGSPOBackend'))
    assert 'prefill_step_size' not in rollout and 'draft_model' not in rollout
    out={'schema':'saved-native-probability-independent-source-analysis-v1','reviewer_kind':'isolated_codex','reviewer_id':a.reviewer_id,'status':'CPU_ANALYSIS_COMPLETED_NATIVE_RESIDUAL_NOT_RESOLVED',
        'evidence_tier':'SAVED_SELECTED_LOGPS_AND_CPU_EXECUTED_PINNED_AST','native_execution_approved':False,'training_readiness':'NO',
        'mission_alignment':'Isolate concrete policy/scorer path differences before interpreting Vero MRI GSPO learning; no new model or optimizer work.',
        'evidence_sha256':{'saved17record':sha(recordpath),'old_scope':sha(scopepath),'prepared_tensor_identity':sha(identitypath),'old_external_exit':sha(base/'actual_exit_v3_PRIVATE.json'),'old_closed_stdout':sha(base/'native_v3_stdout_PRIVATE.log'),'config_metadata':sha(configpath)},
        'source_sha256':{k:sha(p) for k,p in paths.items()},'analysis_source_sha256':sha(__file__),
        'selected_logp_arithmetic':numeric,'decomposition_positions_verified':17,'synthetic_common_offset':common_shift_fixture(),
        'source_alignment_check':alignment,'CPU_actual_RoPE_mask_AST_check':positions,'CPU_actual_KV_buffer_AST_check':cursor,'CPU_actual_neutral_sampler_AST_check':sampler,
        'confirmed_source_route_difference':{'installed_default_prefill_step_size':default,'ordinary_no_draft_policy_enabled':enabled,'source_derived_actual_AR_prefill_lengths':[212,1],'old_direct_probe_prefill_lengths':[213],'old_probe_not_exact_AR_prefill_schedule':True,'historical_runtime_prefill_metadata':'NOT_SAVED','fresh_native_runtime_policy_verification_required':True},
        'source_interpretation':[
            'Off-by-one is NOT_FOUND in actual _logps identity-marker execution for all17 saved targets. Hash-authenticated grid and actual get_rope_index AST give identical prefix/full positions and decode offset+delta; buffer/cursor and causal-mask AST checks likewise agree. These checks do not prove native kernels executed that state.',
            'A concrete implementation-level comparison mismatch is CONFIRMED in inspected current source: ordinary AR default2048 chunks213 prompt tokens as212+1, whereas old forced-KV probe used213 in one prefill. This is a versioned diagnostic route mismatch, not an established historical training defect or proven explanation of .009733.',
            'Source Qwen projections/output head use shape-dependent quantized matmul and attention calls fast SDPA with different query lengths in full and incremental paths. Output logits are BF16 in saved evidence; F32 normalization cannot undo upstream rounding. Attention/GEMM precision or shape effects are plausible, not localized or proven kernel causes.',
            'Frozen GSPO update recomputes old logps with the same F32 full-TF scorer as fresh; it does not consume saved generation KV logps. Hypothetical exp(mean selected differences) above are not recorded historical importance ratios or evidence that clipping actually occurred.',
            'Only one selected vocabulary element per changing causal context was saved. Complete distribution equality, KL/JS, top-k, vocabulary-centered shift and gradient impact are NOT_IDENTIFIABLE from these vectors. The synthetic64/63 common-offset fixture does not establish actual distribution equivalence.'
        ],
        'bounded_falsifiers_for_separate_native_scope':[
            'One existing saved TRAIN response/checkpoint only; no generation/backward/update. Compare original direct213 prefill, source-derived ordinary212+1 prefill and full TF using identical conditioning. Save actual cache offsets/position tensors and policy flags, not only assumed counts.',
            'At each of17 contexts persist private full-vocabulary logits/logps or sufficient distribution statistics: common-shift-centered differences, normalized probabilities, KL/JS, top-k and selected score. Keep .002 gate unchanged and distinguish returned-logp error from categorical difference.',
            'Precision control must retain prefill schedule, complete position/rope/mask metadata and equal represented embedding/deepstack values. Verify full future positions equal implicit offset+delta and shared feature prefixes equal before upcast; otherwise attribute change to a combined functional path intervention, not precision alone.',
            'Only if numerical gate passes and a separately reviewed actual scope admits it may B3 run. No native launch, automatic retry, tolerance relaxation or training permission is created by this CPU analysis.'
        ],
        'execution':{'model_or_optimizer_loads':0,'native_imports':0,'images_opened':0,'native_forwards':0,'backwards':0,'generated_tokens':0,'optimizer_updates':0,'final100_access':False},
        'limitations':['Grid integers were recovered from a finite factor enumeration and authenticated against saved raw-int64 SHA; no pixels were decoded.','NumPy source-AST replay validates indexing/metadata algebra, not MLX storage, fused attention, mutable runtime state or gradient parity.','common/base/quantized-linear source hashes are current inspected dependencies; their digests were not all included in the old25-file scope. Runtime/historical identity should not be invented.','No MRI accuracy/generalization or unique root-cause claim.']}
    assert a.output.resolve().is_relative_to(W/'results')
    with a.output.open('x') as handle:json.dump(out,handle,indent=2,sort_keys=True,allow_nan=False);handle.write('\n');handle.flush();os.fsync(handle.fileno())
    print(json.dumps({'status':out['status'],'output_sha256':sha(a.output),'native_execution_approved':False}))


if __name__=='__main__':main()
