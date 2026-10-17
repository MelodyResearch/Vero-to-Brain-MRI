
"""Small source-bound read-only C40 likelihood probe; NOT_RUN until reviewed.
Reuses native loader/cache/resource gates and exact existing likelihood functions.
"""
from __future__ import annotations
import argparse,copy,fcntl,gc,hashlib,json,os
from pathlib import Path
from types import FunctionType,SimpleNamespace
from mri_rl import answer_likelihood as al
from mri_rl import paced_answer_likelihood as old
from mri_rl.c40_visual_controls import LABELS,bound_candidate_functions,select_probe_rows
from mri_rl.two_task_pool import sha,canonical_sha,require
from mri_rl.experiment_contract import _check_review,load_contract

ROOT=Path(__file__).resolve().parents[2]
PLAN_SHA='732260275ca977ccce1b59ea5856701f77a54b0150899acfc17aee2a6484b0ab'
MANIFEST_SHA='f47169577b285d03e21dafe904244d77be83c10edcf730f3b72a7a129cd1bd01'
U7_PROOF_SHA='39b1a6bd0d9064323a4b62222de1f3539f7bdf1b8d70d0dc8555bc9589ab7b89'
B=ROOT/'local_data/c_two_task_pool_v1'
MANIFEST=B/'tier_c_v1/remaining20_bundle_v1/manifest_READY_REQUIRES_REVIEW_PRIVATE.json'
U7_PROOF=ROOT/'results/U_RAW_SEED7_FINAL_NUMERICAL_INDEPENDENT_VERIFICATION.json'

def load(path):return json.loads(Path(path).read_text())

def numeric_validator(task):
    require(task in LABELS,'Unknown likelihood task')
    partition,compare=bound_candidate_functions(LABELS[task])
    proxy=SimpleNamespace(LABELS=LABELS[task],partition_target=partition,summarize_tokens=al.summarize_tokens,compare_candidates=compare)
    fn=old.validate_numeric_record
    return FunctionType(fn.__code__,{**fn.__globals__,'al':proxy},fn.__name__,fn.__defaults__,fn.__closure__)

def task_comparisons(task,comparisons,*,to_legacy=False):
    """Names only; preserve exact existing comparison arithmetic."""
    require(task in LABELS,'Unknown comparison task')
    if task=='mri_sequence':return dict(comparisons)
    old_suffix,new_suffix=('same_minus_different','t1_minus_flair') if to_legacy else ('t1_minus_flair','same_minus_different')
    return {key.replace(old_suffix,new_suffix):value for key,value in comparisons.items()}

def validate_numeric_route(row,tokenizer=None):
    validate_record_route(row)
    legacy={**row,'comparisons':task_comparisons(row['category'],row['comparisons'],to_legacy=True)}
    numeric_validator(row['category'])(legacy,tokenizer)

def validate_record_route(row):
    task=row.get('category');require(task in LABELS and row.get('candidate_labels')==list(LABELS[task]),'Candidate order or task changed')
    if row.get('condition')=='no_image':
        require(row.get('image') is None and row.get('image_sha256') is None and row.get('visible_target_PRIVATE') is None,'No-image has no visual target')
    else:
        require(row.get('condition')=='real_image' and row.get('image') and row.get('image_sha256') and row.get('visible_target_PRIVATE') in LABELS[task],'Invalid real-image reference')

def require_training_done(cursor,required_ids):
    require(len(required_ids)==20 and len(set(required_ids))==20,'Exact original20 plus adoptedU7 required')
    require(not cursor.get('halted'),'Original training queue halted')
    for key in [*required_ids,'rl:U_raw:7']:
        entry=cursor.get('stages',{}).get(key,{})
        require(entry.get('state')=='DONE' and entry.get('summary',{}).get('full_budget_terminal_verified') is True,'Original21training not terminal')

def authenticate_training(cursor_path,expected_sha):
    require(sha(cursor_path)==expected_sha and sha(MANIFEST)==MANIFEST_SHA,'Exact terminal cursor/manifest digest required')
    manifest=load(MANIFEST);cursor=load(cursor_path);require_training_done(cursor,[x['id'] for x in manifest['stages']])
    require(sha(U7_PROOF)==U7_PROOF_SHA and load(U7_PROOF)['verdict']=='PASS_WITH_LIMITATIONS','Adopted actual fullU7 proof changed')
    for spec in manifest['stages']:
        summary=cursor['stages'][spec['id']]['summary'];exit_path=Path(spec['exit_receipt'])
        require(sha(exit_path)==summary['exit_receipt_sha256'],'Completed exit witness changed')
        witness=load(exit_path)
        require(witness['exit_code']==0 and witness['stage_id']==spec['id'] and witness['raw_log_sha256']==sha(spec['raw_log']),'Actual terminal child/log missing')
        require(witness['review_sha256']==spec['review_sha256']==sha(spec['review']) and witness['launcher_sha256']==spec['launcher_sha256']==sha(spec['launcher']),'Actual completed launch scope changed')
        boundary=Path(spec['output'])/f"boundary_{summary['completed_groups']:04d}"
        require(sha(boundary/'receipt.json')==summary['boundary_receipt_sha256'] and sha(boundary/'adapter.safetensors')==summary['adapter_sha256'],'Actual terminal checkpoint changed')
        require(summary['native_terminal_status'] in {'TOKEN_BUDGET_EXHAUSTED','SCHEDULE_COMPLETE'},'Pilot is not full training terminal')
    return cursor

def source_hashes():
    from mri_rl.continuous_rl_dev_eval import source_hashes as native_sources
    result=native_sources()
    for name in ('answer_likelihood.py','paced_answer_likelihood.py','c40_visual_controls.py','c40_visual_probe.py'):
        result[name]=sha(Path(__file__).with_name(name))
    native=(ROOT/'../../work/mlx-venv/lib/python3.13/site-packages/mlx_vlm').resolve()
    for name in ('utils.py','tokenizer_utils.py','prompt_utils.py','generate/dispatch.py','generate/ar.py','generate/common.py','trainer/common/model.py','models/qwen3_vl/qwen3_vl.py','models/qwen3_vl/vision.py','models/qwen3_vl/language.py'):
        result['installed_mlx_vlm/'+name]=sha(native/name)
    result['entry_script']=sha(ROOT/'scripts/run_c40_qwen_visual_probe.py')
    return result

def validate_plan(path):
    require(sha(path)==PLAN_SHA,'Frozen score-independent probe plan changed')
    plan=load(path)
    require(plan['weight_updates']==0 and plan['likelihood']['prefix']==al.PREFIX and plan['likelihood']['suffix']==al.SUFFIX and plan['likelihood']['common_prefix_tolerance']==.002,'Likelihood prefix/scoring contract changed')
    require(plan['greedy']=={'max_tokens':256,'temperature':0,'new_generations_per_checkpoint':2,'existing_real_image_generations_reused_per_checkpoint':8},'Generation contract changed')
    require(sha(plan['source_manifest'])==plan['source_manifest_sha256'] and sha(plan['prompt_contract_path'])==plan['prompt_contract_sha256'],'Source/contract bytes changed')
    for name,digest in plan['source_sha256'].items():require(sha(Path(__file__).with_name(name))==digest,'Frozen numerical/data source changed')
    records=plan['records_PRIVATE'];require(len(records)==10 and len({r['record_id'] for r in records})==10,'Finite10-record scope required')
    rows=[json.loads(x) for x in Path(plan['source_manifest']).read_text().splitlines()]
    chosen={r['id']:r for r in select_probe_rows(rows)};contract=load(plan['prompt_contract_path'])
    real=[r for r in records if r['condition']=='real_image'];require({r['record_id'] for r in real}==set(chosen),'Score-independent subset changed')
    for record in records:
        validate_record_route(record)
        if record['image']:
            original=chosen[record['record_id']]
            require(record['image']==original['images'][0] and sha(record['image'])==record['image_sha256'] and record['visible_target_PRIVATE']==original['reward_model']['ground_truth'],'Frozen source pixels/target differ')
        question=next(r['extra_info']['question'] for r in chosen.values() if r['ability']==record['category'])
        require(record['prompt']==[{'role':'system','content':contract['system_instruction']},{'role':'user','content':question}],'Native system/question differs')
    return plan

def saved_predictions(checkpoint):
    if checkpoint=='B28':
        path=B/'evaluation_v1/C40_B28_baseline_independent_scoring_PRIVATE.json';expected='32ce2cd88a9d316e94f75177eb8c6c36e33bcd351fe4028ed9e88aa3c26dda8f';key='normalized_response_components_PRIVATE'
    else:
        path=B/'tier_c_v1/U_raw_seed7_final_C40_release_v1/final_C40_scoring_PRIVATE.json';expected='de73178d59d5894d7ddfe4494f05dd785aa35211905b9457e9a1757e1f8b059e';key='normalized_new_outputs_PRIVATE'
    require(sha(path)==expected,'Authenticated saved real-image generations changed')
    return load(path)[key],expected

def prepare_scope(plan_path,checkpoint,output,cursor_path,expected_cursor):
    plan=validate_plan(plan_path);require(checkpoint in plan['checkpoints'],'Unknown fixed checkpoint')
    require(Path(output).resolve().is_relative_to(ROOT/'local_data'),'Private output required')
    adapter=plan['checkpoints'][checkpoint]
    require(sha(adapter['path'])==adapter['sha256'] and sha(Path(adapter['path']).with_name('adapter_config.json'))==adapter['adapter_config_sha256'],'Frozen exact checkpoint/config changed')
    require(sha(cursor_path)==expected_cursor,'Externally captured cursor digest differs')
    cursor=load(cursor_path);manifest=load(MANIFEST);ready=True
    try:require_training_done(cursor,[s['id'] for s in manifest['stages']])
    except ValueError:ready=False
    sources=source_hashes();identity=canonical_sha({'plan':PLAN_SHA,'checkpoint':checkpoint,'adapter':adapter,'sources':sources})
    # Reuse the original durable-prefix and numerical validator code with task routing.
    prefix_fn=old.completed_prefix
    namespace={**prefix_fn.__globals__,'validate_numeric_record':validate_numeric_route}
    prefix=FunctionType(prefix_fn.__code__,namespace,prefix_fn.__name__,prefix_fn.__defaults__,prefix_fn.__closure__)
    completed=prefix(Path(output),plan['records_PRIVATE'],identity)
    _,saved_sha=saved_predictions(checkpoint)
    scope={'execution_tier':'B','kind':'C40_visual_likelihood_v1','plan_sha256':PLAN_SHA,'checkpoint':checkpoint,'adapter':adapter,'run_identity':identity,
        'sources':sources,'derived_cache':plan['derived_cache'],'output':str(Path(output).resolve()),
        'existing_output_sha256':sha(output) if Path(output).exists() else None,'completed_records':completed,
        'new_record_ids':[r['record_id'] for r in plan['records_PRIVATE'][completed:]],'forwards_per_record':3,'max_new_forwards':3*(10-completed),
        'new_greedy_generations':sum(r['condition']=='no_image' for r in plan['records_PRIVATE'][completed:]),
        'saved_real_image_outputs_sha256':saved_sha,'weight_updates':0,'quantization_bits':4,'quantization_group_size':64,'lora_rank':4,'lora_alpha':8,
        'fixed_prefix':al.PREFIX,'fixed_suffix':al.SUFFIX,'candidate_order':{k:list(v) for k,v in LABELS.items()},'prefix_tolerance':.002,
        'original21training_ready':ready,'terminal_cursor_sha256':expected_cursor,'original20manifest_sha256':MANIFEST_SHA,'U7_full_proof_sha256':U7_PROOF_SHA}
    return plan,scope

def score_record(backend,record,index):
    """Same existing causal gather, target segmentation and third-prefix oracle."""
    mx=backend.mx;nn=backend.nn;model=backend.model;processor=backend.processor;task=record['category']
    partition_target,compare=bound_candidate_functions(LABELS[task]);labels=LABELS[task]
    image=record['image'];backend.before_unit(f'probe_record_{index}')
    require(not image or sha(image)==record['image_sha256'],'Reviewed image changed')
    prompt=backend.template(processor,model.config,record['prompt'],num_images=int(image is not None),enable_thinking=False,add_generation_prompt=True)
    require(prompt.count('<|image_pad|>')==int(image is not None) and prompt.endswith('<|im_start|>assistant\n'),'Native prefix/placeholder mismatch')
    special=backend.special_tokens(model.config.model_type,processor)
    prepared=backend.prepare_inputs(processor,images=[image] if image else None,prompts=prompt,image_token_index=getattr(model.config,'image_token_index',None),add_special_tokens=special)
    prompt_ids=prepared['input_ids'];P=int(prompt_ids.shape[-1]);pixels=prepared.get('pixel_values')
    require(prompt_ids.shape[0]==1 and 1<=P<=2048 and bool(mx.all(prepared['attention_mask']==1).item()),'Prompt budget/padding mismatch')
    kwargs={k:v for k,v in prepared.items() if k not in ('input_ids','attention_mask','pixel_values')}
    prompt_list=prompt_ids[0].tolist();visual=prompt_list.count(151655);grid=prepared['image_grid_thw'].tolist() if image else None
    if image:require(pixels is not None and visual==sum(t*h*w//4 for t,h,w in grid) and visual>0,'Visual conditioning mismatch')
    else:require(pixels is None and visual==0,'No-image has visual tensors')
    candidates={};first=[];prefix_scores=[];parts={}
    for label_index,label in enumerate(labels):
        backend.before_unit(f'probe_candidate_{index}_{label_index}')
        part=partition_target(processor.tokenizer,label,prompt_length=P);parts[label]=part
        enc=lambda text:list(processor.tokenizer.encode(text,add_special_tokens=False))
        require(enc(prompt+al.PREFIX+label+al.SUFFIX)==enc(prompt)+part['target_ids'],'Prompt/target boundary mismatch')
        target=mx.array(part['target_ids'],dtype=prompt_ids.dtype);ids=mx.concatenate((prompt_ids,target[None,:]),axis=1);inputs=ids[:,:-1]
        logits=model(inputs,pixels,mx.ones_like(inputs),**kwargs).logits;require(logits.shape[1]==inputs.shape[1],'Causal logits length mismatch')
        lp=nn.log_softmax(logits[:,P-1:,:].astype(mx.float32),axis=-1)
        chosen=mx.take_along_axis(lp,target[None,:,None],axis=-1)[0,:,0]
        n=len(part['prefix_ids']);label_first=[partition_target(processor.tokenizer,l,prompt_length=P)['label_ids'][0] for l in labels]
        require(len(set(label_first))==2,'First candidate tokens unexpectedly identical')
        first_lp=lp[0,n,mx.array(label_first)];mx.eval(chosen,first_lp)
        values=chosen.tolist();first.append(first_lp.tolist());prefix_scores.append(values[:n])
        candidates[label]={**al.summarize_tokens(values,part),'partition':part,'target_token_logps':values,'first_label_candidate_logps':first[-1]}
        del logits,lp,chosen,first_lp,ids,inputs,target;gc.collect();mx.clear_cache()
    backend.before_unit(f'probe_prefix_oracle_{index}')
    prefix=mx.array(parts[labels[0]]['prefix_ids'],dtype=prompt_ids.dtype);oracle_ids=mx.concatenate((prompt_ids,prefix[None,:]),axis=1)
    oracle_logits=model(oracle_ids,pixels,mx.ones_like(oracle_ids),**kwargs).logits
    oracle_lp=nn.log_softmax(oracle_logits[:,-1,:].astype(mx.float32),axis=-1)[0,mx.array(label_first)]
    mx.eval(oracle_lp);oracle=oracle_lp.tolist()
    errors=[abs(a-b) for a,b in zip(prefix_scores[0],prefix_scores[1])]+[abs(a-b) for row in first for a,b in zip(row,oracle)]
    require(max(errors)<=.002,'Independent prefix oracle disagrees')
    result={**record,'prompt_ids':prompt_list,'rendered_prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'image_grid_thw':grid,
        'image_token_count':visual,'add_special_tokens':special,'candidate_scores':candidates,
        'comparisons':task_comparisons(task,compare(candidates)),'candidate_order':list(labels),
        'margin_definition':'T1 minus FLAIR' if task=='mri_sequence' else 'same minus different',
        'prefix_oracle_first_logps':oracle,'prefix_max_disagreement':max(errors),'weight_updates':0}
    del oracle_logits,oracle_lp,oracle_ids,prefix,prepared,pixels,kwargs,prompt_ids;gc.collect();mx.clear_cache()
    require(not image or sha(image)==record['image_sha256'],'Reviewed image changed during scoring')
    return result

def execute(plan_path,checkpoint,output,cursor_path,expected_cursor,review,review_sha):
    plan,scope=prepare_scope(plan_path,checkpoint,output,cursor_path,expected_cursor)
    require(scope['original21training_ready'],'Original21training must finish before this diagnostic')
    _check_review(review_verdict_path=Path(review),expected_review_sha256=review_sha,scope=scope)
    if scope['completed_records']==10:return {'status':'ACTUAL_PROBE_ALREADY_COMPLETE','checkpoint':checkpoint,'completed':10,'output_sha256':sha(output),'weight_updates':0}
    # Reuse original queue lock: no new heavyweight job can overlap that queue or
    # another copy of this probe. Existing resource gate also checks other jobs.
    with (Path(cursor_path).parent/'remaining20.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        authenticate_training(cursor_path,expected_cursor)
        require((sha(output) if Path(output).exists() else None)==scope['existing_output_sha256'],'Output changed after review')
        from mri_rl.paced_resources import PacedGate
        from mri_rl.mlx_continuous_gspo import NativeGSPOBackend,is_loading_unit
        from mri_rl.likelihood_cache import verify_cache_identity
        from mri_rl.paced_frozen_eval import sync_directory,generation_metadata
        Path(output).parent.mkdir(parents=True,exist_ok=True)
        gate=PacedGate(Path(output).with_suffix('.resources.jsonl'))
        cache={'dir':plan['derived_cache']['cache_dir']};certificate=ROOT/'results/EXACT_QUANT_CACHE_CERTIFICATE.json'
        gate.before_unit('probe_before_native_import',loading=True)
        cert=load(certificate);model=Path(plan['model_path'])
        artifacts={str(model/name):sha(model/name) for name in set(cert['non_weight_files_identical'])|{'config.json'}}
        require(verify_cache_identity(cache['dir'],certificate,plan['derived_cache']['certificate_sha256'],artifacts)==plan['derived_cache'],'Exact certified cache differs')
        def before(name):gate.before_unit(name,loading=is_loading_unit(name,cache))
        backend=None;completed=scope['completed_records'];saved,_=saved_predictions(checkpoint);out=Path(output);out.parent.mkdir(parents=True,exist_ok=True)
        try:
            backend=NativeGSPOBackend(plan['checkpoints'][checkpoint]['path'],{'group_size':2,'max_tokens':256},model_path=model,contract=load_contract(Path(plan['prompt_contract_path'])),seed=7,before_unit=before,cache=cache)
            with out.open('a' if out.exists() else 'x') as handle:
                sync_directory(out.parent)
                for index,record in enumerate(plan['records_PRIVATE'][completed:],completed):
                    result=score_record(backend,record,index)
                    if record['condition']=='no_image':
                        generated=backend.generate_record(record,None,256);result.update(generated);result['new_greedy_generation']=True
                    else:
                        result['response']=saved[record['source_record_id']]['response'];result['new_greedy_generation']=False
                    result.update(run_identity=scope['run_identity'],checkpoint=checkpoint,adapter_sha256=plan['checkpoints'][checkpoint]['sha256'])
                    validate_numeric_route(result,backend.processor.tokenizer)
                    handle.write(json.dumps(result,allow_nan=False)+'\n');handle.flush();os.fsync(handle.fileno());completed+=1
                    print(json.dumps({'checkpoint':checkpoint,'completed':completed,'total':10,'weight_updates':0}),flush=True)
                    gc.collect();backend.mx.clear_cache();before('probe_record_completed')
            disk=backend.mx.load(plan['checkpoints'][checkpoint]['path']);actual=dict(backend.tree_flatten(backend.model.trainable_parameters()))
            require(set(disk)==set(actual) and all(bool(backend.mx.array_equal(v,actual[k]).item()) for k,v in disk.items()),'Read-only adapter integrity failed')
            require(int(backend.optimizer_state()['step'].item())==0,'Read-only probe mutated optimizer')
        finally:
            if backend is not None:backend.close()
        return {'status':'ACTUAL_PROBE_COMPLETED' if completed==10 else 'INCOMPLETE','checkpoint':checkpoint,'completed':completed,'output_sha256':sha(out),'weight_updates':0}

def main():
    p=argparse.ArgumentParser()
    for name in ('plan','output','queue-cursor'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--checkpoint',choices=['B28','U_raw_seed7'],required=True);p.add_argument('--expected-queue-sha256',required=True)
    p.add_argument('--scope-output',type=Path);p.add_argument('--review-verdict',type=Path);p.add_argument('--expected-review-sha256');p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    if not a.execute:
        plan,scope=prepare_scope(a.plan,a.checkpoint,a.output,a.queue_cursor,a.expected_queue_sha256)
        require(a.scope_output is not None and a.scope_output.resolve().is_relative_to(ROOT/'local_data'),'Private preview scope output required')
        require(not a.scope_output.exists() or load(a.scope_output)==scope,'Preview scope changed; preserve previous file')
        if not a.scope_output.exists():
            with a.scope_output.open('x') as handle:handle.write(json.dumps(scope,indent=2,sort_keys=True)+'\n');handle.flush();os.fsync(handle.fileno())
        print(json.dumps({'status':'SCOPE_PREVIEW_ONLY_NOT_RUN','original21training_ready':scope['original21training_ready'],'scope_sha256':canonical_sha(scope),'scope_file_sha256':sha(a.scope_output),'pending_records':len(scope['new_record_ids'])}))
    else:
        require(a.review_verdict and a.expected_review_sha256,'Independent exact review required before model load')
        print(json.dumps(execute(a.plan,a.checkpoint,a.output,a.queue_cursor,a.expected_queue_sha256,a.review_verdict,a.expected_review_sha256),sort_keys=True))
if __name__=='__main__':main()
