"""Separate matched F32 parent/terminal evaluator: CPU_IMPLEMENTED, NOT_RUN.

Frozen trainer/backend/qualifier are untouched. Native execution requires exact
terminal metadata, actual numeric qualification, new human authorization and an
independent exact scope review. Swap aliases reuse authenticated real outputs.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import time
from mri_rl import paired_z045_continuation_entry_v1 as entry
from mri_rl.paired_z045_evaluation_v1 import LABELS,require
from mri_rl.answer_likelihood import PREFIX

STATUS='IMPLEMENTED_CPU_NOT_NATIVE_QUALIFIED'


def budget():
    return {'checkpoints':2,'greedy_generations':80,'candidate_score_API_calls':240,'maximum_generated_tokens':20480,
        'teacherforced_input_positions':102000,'generation_language_model_calls':20640,'vision_embedding_calls':320,
        'optimizer_updates':0,'maximum_seconds':2400,'simultaneous_models':1,'simultaneous_image_graphs':1}


class Counter:
    def __init__(self):self.values={}
    def charge(self,name,value=1):
        self.values[name]=self.values.get(name,0)+value
        require(type(value) is int and value>=0 and self.values[name]<=budget()[name],'Frozen evaluation budget exceeded: '+name)


def call_receipt(counter,before):
    names={'maximum_generated_tokens':'generated_tokens','greedy_generations':'greedy_calls'}
    return {names.get(k,k):v-before.get(k,0) for k,v in counter.values.items()}


def common_prefix_alignment(vectors):
    require(len(vectors)==2 and len(vectors[0])==len(vectors[1])>0 and all(math.isfinite(float(v)) for values in vectors for v in values),'Two finite identical prefix spans required')
    error=max(abs(float(a)-float(b)) for a,b in zip(*vectors));require(error<=.002,'Full candidate common-prefix shape agreement exceeds .002')
    return {'max_abs_error':error,'tolerance':.002}


def greedy_settings():return {'max_tokens':256,'temperature':0.,'top_p':1.,'top_k':0,'min_p':0.,'repetition_penalty':None,'verbose':False}


def compile_records(evaluation):
    records=evaluation['records_PRIVATE'];real=[r for r in records if r['condition']=='real_image'];aliases=[r for r in records if r['condition']!='real_image']
    require(len(real)==40 and len(aliases)==8 and len({r['id'] for r in records})==48,'Exact40 real and8 unique swap aliases required')
    lookup={r['id']:r for r in real}
    require(sum(r['row_PRIVATE']['ability']=='mri_sequence' for r in real)==24 and sum(r['row_PRIVATE']['ability']=='mri_visit_correspondence' for r in real)==16,'Exact24 sequence/16 Dallas required')
    result={}
    for alias in aliases:
        reuse=lookup.get(alias.get('reuse_real_id_PRIVATE'));original=lookup.get(alias.get('original_real_id_PRIVATE'))
        require(reuse and original and alias['condition']=='same_patient_same_level_opposite_modality','Explicit real-output alias required')
        row=alias['row_PRIVATE'];other=reuse['row_PRIVATE'];source=original['row_PRIVATE']
        require({k:v for k,v in row.items() if k!='id'}=={k:v for k,v in other.items() if k!='id'} and alias['image_sha256']==reuse['image_sha256'], 'Alias inputs differ from reused real output')
        require(other['ability']==source['ability']=='mri_sequence' and other['extra_info']['patient_key']==source['extra_info']['patient_key'] and
                other['extra_info']['level']==source['extra_info']['level']==45 and other['reward_model']['ground_truth']!=source['reward_model']['ground_truth'],'Alias is not fixed samepatient/z045 opposite modality')
        result[alias['id']]={'reuse_real_id_PRIVATE':reuse['id'],'original_real_id_PRIVATE':original['id'],'image_sha256':alias['image_sha256']}
    require(evaluation['candidate_prefix']==PREFIX and {k:tuple(v) for k,v in evaluation['label_order'].items()}==LABELS,'Frozen candidate prefix/class order changed')
    return {'real':real,'aliases':result}


def alias_outputs(real,aliases):return {key:real[value['reuse_real_id_PRIVATE']] for key,value in aliases.items()}


def label_scores(logps,n):
    require(type(n) is int and 0<n<=len(logps) and all(math.isfinite(float(v)) for v in logps),'Strict finite label span required')
    values=[float(v) for v in logps[-n:]]
    return {'mean':sum(values)/n,'sum':sum(values),'first':values[0],'token_count':n,'label_logps_PRIVATE':values}


def prefix_alignment(candidate,oracle):
    require(len(candidate)==len(oracle)==2 and all(math.isfinite(float(v)) for v in [*candidate,*oracle]),'Two aligned first-token scores required')
    error=max(abs(float(a)-float(b)) for a,b in zip(candidate,oracle));require(error<=.002,'Independent prefix oracle mismatch exceeds .002')
    return {'max_abs_error':error,'tolerance':.002,'first_logps_PRIVATE':list(oracle)}


def score_input_positions(prompt,prefix,label):
    require(type(prompt) is int and 0<prompt<=405 and 0<prefix<=16 and 0<label<=5,'Frozen teacherforce shape budget differs')
    result=prompt+prefix+label-1;require(result<=425,'Teacherforce425 cap exceeded');return result


def token_plan(task,prefix,encode):
    require(task in LABELS and prefix==PREFIX,'Frozen task/prefix required')
    prefix_ids=list(encode(prefix));labels=[]
    for label in LABELS[task]:
        ids=list(encode(label));require(list(encode(prefix+label))==prefix_ids+ids and ids,'Class boundary overlaps prefix')
        score_input_positions(405,len(prefix_ids),len(ids));labels.append(ids)
    require(labels[0][0]!=labels[1][0],'First-token oracle cannot distinguish canonical labels')
    return {'prefix':prefix_ids,'labels':labels,'label_order':list(LABELS[task])}


def require_terminal(terminal,parent_sha):
    require(terminal is not None and terminal.get('receipt_sha256') and terminal['receipt_sha256']!=parent_sha,'Exact completed terminal checkpoint required; unknown terminal cannot execute')
    return True


def terminal_metadata(spec,parent):
    if not spec.get('terminal_checkpoint'):return None
    path=Path(spec['terminal_checkpoint']);require(entry.digest(path/'receipt.json')==spec['terminal_receipt_sha256'],'Terminal receipt changed')
    r=json.loads((path/'receipt.json').read_text());m=json.loads((path/'optimizer/manifest.json').read_text())
    require(entry.digest(path/'optimizer/manifest.json')==r['files_sha256']['optimizer/manifest.json'],'Terminal manifest changed')
    require(r['schema']=='continuous-rl-boundary-v1' and 1<=r['completed_groups']<=32 and 0<=r['optimizer_updates']<=8 and
            r['absolute_completed_groups']==129+r['completed_groups'] and r['absolute_optimizer_updates']==60+r['optimizer_updates'] and
            path.name==f"boundary_{r['completed_groups']:04d}",'Terminal finite-stage counters differ')
    require(r['binding']['experiment']['stage']=='paired_z045_continuation_v1' and r['binding']['experiment']['adaptive_parent']['receipt_sha256']==parent['receipt_sha256'] and
            m['bindings']['run']==r['binding'] and m['optimizer_updates']==r['absolute_optimizer_updates'] and
            m['bindings']['adapter_sha256']==r['files_sha256']['adapter.safetensors'],'Terminal parent/Adam/source binding differs')
    cs_path=Path(spec['continuation_spec']);require(entry.digest(cs_path)==spec['continuation_spec_sha256'],'Exact continuation spec changed')
    cs=json.loads(cs_path.read_text());schedule_path=Path(cs['candidate_schedule'])
    require(entry.digest(schedule_path)==cs['file_sha256']['candidate_schedule'],'Exact continuation schedule artifact changed')
    schedule=json.loads(schedule_path.read_text())
    if isinstance(schedule,dict):schedule=schedule['schedule']
    expected_schedule=entry.scope_digest(schedule);experiment=r['binding']['experiment']
    require(experiment.get('spec_sha256')==entry.stage_spec_binding(cs) and
            experiment.get('schedule_sha256')==r['binding'].get('schedule_sha256')==expected_schedule and
            r['binding'].get('configuration')=={**entry.driver.PROFILE,'group_size':4,'max_tokens':256},
            'Terminal does not match exact continuation spec/schedule/configuration')
    proofpath=Path(spec['terminal_completion']);require(entry.digest(proofpath)==spec['terminal_completion_sha256'],'Completed-terminal evidence changed')
    proof=json.loads(proofpath.read_text());require(proof['status'] in ('UPDATE_BUDGET_REACHED','SCHEDULE_COMPLETE','WALL_BUDGET_REACHED','TOKEN_BUDGET_EXHAUSTED') and
        proof['completed_groups']==r['completed_groups'] and proof['optimizer_updates']==r['optimizer_updates'] and Path(proof['boundary']).resolve()==path.resolve(),'Terminal is not exact first budget endpoint')
    return {'path':str(path.resolve()),'receipt_sha256':spec['terminal_receipt_sha256'],'adapter_sha256':r['files_sha256']['adapter.safetensors'],
        'optimizer_manifest_sha256':r['files_sha256']['optimizer/manifest.json'],'optimizer_state_sha256':r['files_sha256']['optimizer/state.safetensors'],
        'groups':r['absolute_completed_groups'],'updates':r['absolute_optimizer_updates'],'array_bytes_verified':False}


def prepare(spec):
    from tokenizers import Tokenizer
    cp=Path(spec['continuation_spec']);require(entry.digest(cp)==spec['continuation_spec_sha256'],'Frozen continuation spec changed')
    cs=json.loads(cp.read_text());require(entry.digest(spec['evaluation'])==spec['evaluation_sha256'],'Frozen evaluationv3 changed')
    evaluation=json.loads(Path(spec['evaluation']).read_text());plan=compile_records(evaluation)
    parent=entry.parent_metadata(cs['parent_checkpoint'],cs['parent_receipt_sha256']);parent['path']=cs['parent_checkpoint']
    terminal=terminal_metadata(spec,parent)
    tok=Tokenizer.from_file(str(Path(cs['model'])/'tokenizer.json'));encode=lambda text:tok.encode(text,add_special_tokens=False).ids
    tokens={task:token_plan(task,PREFIX,encode) for task in LABELS}
    expected={r['id']:r['image_sha256'] for r in plan['real']};entry.verify_images([r['row_PRIVATE'] for r in plan['real']],expected)
    output=Path(spec['output']).resolve();require(output.is_relative_to(entry.ROOT/'local_data'),'Fresh private evaluator output required')
    sources={**entry.source_closure(cs),str(Path(__file__)):entry.digest(__file__),str(entry.ROOT/'src/mri_rl/paired_z045_evaluation_v1.py'):entry.digest(entry.ROOT/'src/mri_rl/paired_z045_evaluation_v1.py')}
    return {'schema':'paired-z045-matched-F32-evaluator-v1','status':STATUS,'scope_ready_for_native':False,
        'terminal_binding_status':'EXACT_TERMINAL_METADATA_BOUND' if terminal else 'PENDING_EXACT_COMPLETED_TERMINAL',
        'parent':parent,'terminal':terminal,'records_PRIVATE':plan['real'],'aliases_PRIVATE':plan['aliases'],'token_plans_PRIVATE':tokens,
        'expected_image_sha256':expected,'source_sha256':sources,'evaluation_sha256':spec['evaluation_sha256'],
        'continuation_spec_sha256':spec['continuation_spec_sha256'],'numerical_contract_sha256':spec.get('numerical_contract_sha256'),'derived_cache':entry.cache_metadata(cs)['expected_binding'],
        'budget':budget(),'output':str(output),'policy':'F32_ACTIVATION_GENERATION_SCORER_GSPO_V1',
        'decoding':greedy_settings(),'candidate':'Strict canonical label spans only; mean primary length-normalized score, sum jointspan and first nexttoken secondary. Prefix/suffix/EOS excluded.',
        'numeric_gate':'Separate prefix-only same-forward vector oracle first scores vs each full candidate first score<=.002; matched actual F32 generation/scorer/backward prerequisite required, no BF16 fallback.',
        'state':'Full1942 parameters+Adam unchanged before any RNG restoration; capture actual RNG then restore exact checkpoint Python/MLX state; release before next checkpoint.',
        'scope_limitation':'CPU metadata preparation, no model/array payload reads. Parent and exact first-budget terminal only; no independent-generalization/clinical/training-gain claim.',
        'teacherforce_budget_basis':'At most240 scorer API calls×425 inputpositions=102000, conditioned on actualP<=405/prefix<=16/class<=5; no extra alias calls.',
        'resource_contract':'One model serially, AC/existingglobalflock/PacedGate,2400s load-inclusive operation-start checks before/after every gate; atomiccall/release may overrun.',
        'model_loads_this_preparation':0,'optimizer_updates':0,'training_authorized':False}


def prepared_inputs(backend,row):
    from mlx_vlm.generate.dispatch import _prepare_generation_inputs
    from mri_rl.native_gspo_path_probe_v1 import strict_tensor_identity
    from mri_rl import early_native_conditioning_probe_v1 as utility
    from mri_rl.paired_z045_f32_backend_v1 import require_prompt_cost
    prompt=backend.template(backend.processor,backend.model.config,[{'role':'system','content':backend.contract.system_instruction},
        {'role':'user','content':row['extra_info']['question']}],num_images=1,add_generation_prompt=True,enable_thinking=backend.contract.enable_thinking)
    ids,pixels,mask,extra=_prepare_generation_inputs(backend.model,backend.processor,prompt,row['images'],None,None,greedy_settings())
    actual={'input_ids':ids,'pixel_values':pixels,'attention_mask':mask,**extra}
    scorer=backend.prepare_inputs(backend.processor,images=row['images'],prompts=prompt,image_token_index=backend.model.config.image_token_index,
        add_special_tokens=backend.special_tokens(backend.model.config.model_type,backend.processor))
    strict_tensor_identity(backend,actual,scorer,utility);require_prompt_cost(row['ability'],int(ids.shape[-1]))
    require(bool(backend.mx.all(mask==1).item()),'Unpadded exact eval context required')
    return prompt,scorer


def run_row(backend,record,tokens,counter,before,expected):
    from mlx_vlm.generate.dispatch import stream_generate
    from mri_rl.paired_z045_f32_backend_v1 import F32GenerationModelView,ObservedFP32Processor,capture_stream,reset_effective_eos,f32_response_logps_observed
    from mri_rl.native_gspo_path_probe_v1 import strict_tensor_identity
    from mri_rl import early_native_conditioning_probe_v1 as utility
    row=record['row_PRIVATE'];mx=backend.mx;before('matched_eval_row');entry.verify_images([row],expected)
    prompt,inputs=prepared_inputs(backend,row);P=int(inputs['input_ids'].shape[-1]);evidence=[];observed=[]
    def observe(ids,pixels,mask,kwargs):
        extra={k:kwargs[k] for k in inputs if k not in ('input_ids','pixel_values','attention_mask')}
        observed.append(strict_tensor_identity(backend,{'input_ids':ids,'pixel_values':pixels,'attention_mask':mask,**extra},inputs,utility))
    def generation_before(name):
        before(name)
        if name=='f32_generation_LM_call':counter.charge('generation_language_model_calls')
        if name=='f32_generation_vision_embeddings':counter.charge('vision_embedding_calls');entry.verify_images([row],expected)
    view=F32GenerationModelView(backend.model,mx,generation_before,evidence,observe)
    eos=reset_effective_eos(backend.processor.tokenizer,backend.model.config.eos_token_id)
    before('matched_greedy');counter.charge('greedy_generations')
    response=capture_stream(stream_generate(view,backend.processor,prompt,image=row['images'],**greedy_settings(),logits_processors=[ObservedFP32Processor(mx)]),
        prompt_tokens=P,max_tokens=256,eos_ids=eos,before=before)
    require(len(observed)==1,'One actual greedy input preparation required');counter.charge('maximum_generated_tokens',len(response['token_ids']))
    clean=getattr(backend.processor,'clean_output',None)
    if callable(clean):response['text']=clean(response['text'])
    response['generation_dtype_cache_evidence_PRIVATE']=evidence
    response['token_ids_PRIVATE']=response.pop('token_ids');response['raw_selected_logps_PRIVATE']=response.pop('raw_selected_logps')
    entry.verify_images([row],expected);del view;gc.collect();mx.clear_cache()
    values=[];prefix_vectors=[];score_positions=0;prefix=tokens['prefix']
    def score(target,capture=None):
        nonlocal score_positions
        positions=score_input_positions(P,len(prefix),len(target)-len(prefix))
        before('matched_candidate_score');entry.verify_images([row],expected)
        counter.charge('candidate_score_API_calls');counter.charge('vision_embedding_calls');counter.charge('teacherforced_input_positions',positions);score_positions+=positions
        value=f32_response_logps_observed(backend,target,inputs,vocab_observer=capture)
        mx.eval(value);require(bool(mx.all(mx.isfinite(value)).item()),'Finite candidate scores required');return value.tolist()
    for label,ids in zip(tokens['label_order'],tokens['labels']):
        logps=score(prefix+ids);prefix_vectors.append(logps[:len(prefix)]);values.append({'label':label,**label_scores(logps,len(ids))});gc.collect();mx.clear_cache()
    common_prefix=common_prefix_alignment(prefix_vectors)
    capture=[];score(prefix+[tokens['labels'][0][0]],lambda logits:capture.append(logits));require(len(capture)==1,'One shared prefix oracle required')
    mx.eval(capture);oracle=backend.nn.log_softmax(capture[0][0,-1,:].astype(mx.float32),axis=-1)
    first=[float(oracle[ids[0]].item()) for ids in tokens['labels']]
    alignment=prefix_alignment([v['first'] for v in values],first);alignment['full_candidate_common_prefix']=common_prefix;del capture,oracle;gc.collect();mx.clear_cache()
    return {'id':record['id'],'task':row['ability'],'truth':row['reward_model']['ground_truth'],'image_sha256':record['image_sha256'],
        'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'greedy':response,
        'candidates':{'labels':tokens['label_order'],'per_label':values,'prefix_oracle':alignment},
        'costs':{'greedy_calls':1,'candidate_score_API_calls':3,'generated_tokens':response['generation_tokens'],'teacherforced_input_positions':score_positions}}


def audit_before_restore(parameters_before,parameters_after,adam_before,adam_after):
    require(parameters_before==parameters_after and adam_before==adam_after,'Readonly evaluation mutated weights/Adam before RNG restore')
    return True


def run_checkpoint(backend,scope,label,checkpoint,out,counter,before):
    from mri_rl import early_native_conditioning_probe_v1 as utility
    from mri_rl.paired_z045_f32_qualification_v1 import _state,_optimizer
    out=Path(out);out.mkdir();initial_state=backend.optimizer_state();initial_rng=backend.rng_state();state_before=_state(backend,utility);adam_before=_optimizer(backend,utility)
    params_before=utility._fingerprints(backend,dict(backend.tree_flatten(backend.model.parameters())));require(len(params_before)==1942,'Full1942 inventory required')
    rows=[];started=time.monotonic();counts_before=dict(counter.values)
    try:
        for record in scope['records_PRIVATE']:
            result=run_row(backend,record,scope['token_plans_PRIVATE'][record['row_PRIVATE']['ability']],counter,before,scope['expected_image_sha256'])
            utility.durable(out/(record['id']+'_PRIVATE.json'),result);rows.append(result)
        return {'checkpoint_label':label,'checkpoint_receipt_sha256':checkpoint['receipt_sha256'],'rows':rows,'aliases':scope['aliases_PRIVATE'],
            'numerical_contract_sha256':scope['numerical_contract_sha256'],'wall_seconds':time.monotonic()-started,'actual_calls':call_receipt(counter,counts_before),'model_parameters_Adam_unchanged':True,'RNG_restored':True}
    finally:
        after=utility._fingerprints(backend,dict(backend.tree_flatten(backend.model.parameters())));adam_after=_optimizer(backend,utility);pre_restore=_state(backend,utility)
        utility.durable(out/'before_restore_integrity_PRIVATE.json',{'parameters_before':params_before,'parameters_after':after,'Adam_before':adam_before,'Adam_after':adam_after,'state_initial':state_before,'actual_state_before_restore':pre_restore,'unchanged':params_before==after and adam_before==adam_after})
        audit_before_restore(params_before,after,adam_before,adam_after)
        backend.restore(initial_state,initial_rng);restored=_state(backend,utility)
        utility.durable(out/'restored_state_PRIVATE.json',{'initial':state_before,'restored':restored,'unchanged':state_before==restored});require(state_before==restored,'Evaluation RNG/Adam restoration failed')


def execute(args,spec,scope):
    import fcntl
    from mri_rl import early_native_conditioning_probe_v1 as utility
    require_terminal(scope['terminal'],scope['parent']['receipt_sha256'])
    require(args.review and args.expected_review_sha256 and args.human_authorization and args.expected_human_sha256 and args.numeric_contract and args.expected_numeric_sha256,'Exact new evaluation review/human/actual numeric proof required')
    for path,expected in ((args.review,args.expected_review_sha256),(args.human_authorization,args.expected_human_sha256),(args.numeric_contract,args.expected_numeric_sha256)):require(entry.digest(path)==expected,'Eval authorization evidence changed')
    require(scope['numerical_contract_sha256']==args.expected_numeric_sha256,'Exact matched numerical contract file SHA missing/changed')
    review=json.loads(args.review.read_text());human=json.loads(args.human_authorization.read_text());numeric=json.loads(args.numeric_contract.read_text());binding=entry.scope_digest(scope)
    require(review.get('verdict') in ('PASS','PASS_WITH_LIMITATIONS') and review.get('scope_sha256')==binding and review.get('reviewer_id') and review.get('reviewer_kind') in ('isolated_codex','claude_code'),'Independent exact eval review missing')
    require(human.get('authorized') is True and human.get('action')=='readonly_matched_f32_evaluation' and human.get('scope_sha256')==binding,'New human matched evaluation authorization missing')
    require(numeric.get('actual_proof_path') and numeric.get('actual_review_path'),'Actual native F32 prerequisite missing')
    for key in ('actual_proof','actual_review'):require(entry.digest(numeric[key+'_path'])==numeric[key+'_sha256'],'Actual numeric artifact changed')
    actual=json.loads(Path(numeric['actual_proof_path']).read_text());actual_review=json.loads(Path(numeric['actual_review_path']).read_text())
    require(actual.get('status')=='ACTUAL_F32_ONPOLICY_GSPO_READONLY_QUALIFIED' and actual_review.get('verdict') in ('PASS','PASS_WITH_LIMITATIONS') and
        actual_review.get('actual_proof_sha256')==numeric['actual_proof_sha256'] and actual_review.get('reviewer_id') and actual_review.get('reviewer_kind') in ('isolated_codex','claude_code') and all(scope['source_sha256'].get(p)==d for p,d in actual['source_sha256'].items()),'Actual complete same-source F32 prerequisite not qualified')
    cs=json.loads(Path(spec['continuation_spec']).read_text());out=Path(scope['output']);require(not out.exists(),'Fresh private matched evaluation output required')
    def verify():
        for path,expected in scope['source_sha256'].items():require(entry.digest(path)==expected,'Matched eval source changed')
        require(prepare(spec)==scope,'Matched eval data/checkpoint/source metadata changed')
    verify()
    from mri_rl.paced_resources import PacedGate
    from mri_rl.continuous_rl_pilot import cache_binding_for
    from mri_rl.optimizer_checkpoint import load_optimizer_checkpoint
    from mri_rl.uniform_balance_continuation_gate_v1 import parent_anchor
    from mri_rl.experiment_contract import load_contract
    from mri_rl.paired_z045_f32_backend_v1 import F32NativeGSPOBackend
    from mri_rl.mlx_continuous_gspo import is_loading_unit
    from types import SimpleNamespace
    started=time.monotonic();counter=Counter();results=[]
    with Path(cs['global_lock']).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);out.mkdir(parents=True);gate=PacedGate(out/'resources_PRIVATE.jsonl');cache={'dir':cs['cache_dir']}
        derived=cache_binding_for(SimpleNamespace(cache_dir=Path(cs['cache_dir']),cache_certificate=Path(cs['cache_certificate']),cache_certificate_sha256=cs['cache_certificate_sha256'],model=Path(cs['model'])))
        require(derived==scope['derived_cache'],'Matched evaluator certified cache changed')
        def before(name):
            require(time.monotonic()-started<=2400,'Matched eval operation-start deadline exhausted');gate.before_unit(name,loading=is_loading_unit(name,cache));require(time.monotonic()-started<=2400,'Matched eval deadline exhausted during HOLD')
            if name=='native_model_load':entry.verify_images([r['row_PRIVATE'] for r in scope['records_PRIVATE']],scope['expected_image_sha256'])
        for label in ('parent','terminal'):
            checkpoint=scope[label];directory=out/(label+'_PRIVATE');directory.mkdir();resident=[None]
            def run():
                anchor=parent_anchor(checkpoint['path'],checkpoint['receipt_sha256'])
                resident[0]=F32NativeGSPOBackend(Path(checkpoint['path'])/'adapter.safetensors',{**entry.driver.PROFILE,'group_size':4,'max_tokens':256},model_path=Path(cs['model']),contract=load_contract(Path(cs['contract'])),seed=13,before_unit=before,cache=cache)
                backend=resident[0];state,rng,_=load_optimizer_checkpoint(Path(checkpoint['path'])/'optimizer',expected_bindings={'run':anchor['original_binding'],'adapter_sha256':anchor['adapter_sha256'],'trainable_signature':backend.trainable_signature()},load_arrays=backend.load_arrays)
                backend.restore(state,rng);state=rng=None
                return run_checkpoint(backend,scope,label,checkpoint,directory/'rows_PRIVATE',counter,before)
            results.append(entry.run_with_release(run,resident,directory));utility.durable(directory/'complete_PRIVATE.json',results[-1]);resident[0]=None;gc.collect();verify()
    utility.durable(out/'complete_PRIVATE.json',{'status':'ACTUAL_MATCHED_F32_PARENT_TERMINAL_EVALUATION_COMPLETE','scope_sha256':binding,'checkpoints':results,'actual_calls':counter.values,'optimizer_updates':0})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--spec',type=Path,required=True);parser.add_argument('--scope-output',type=Path,required=True);parser.add_argument('--execute',action='store_true')
    for name in ('review','human-authorization','numeric-contract'):parser.add_argument('--'+name,type=Path)
    for name in ('expected-review-sha256','expected-human-sha256','expected-numeric-sha256'):parser.add_argument('--'+name)
    args=parser.parse_args();spec=json.loads(args.spec.read_text());scope=prepare(spec);args.scope_output.parent.mkdir(parents=True,exist_ok=True)
    if args.scope_output.exists():require(json.loads(args.scope_output.read_text())==scope,'Frozen evaluator scope differs')
    else:entry.driver.write_json(args.scope_output,scope)
    if args.execute:execute(args,spec,scope)
    else:print(json.dumps({'status':STATUS,'scope_sha256':entry.scope_digest(scope),'terminal_binding_status':scope['terminal_binding_status'],'model_loads':0}))

if __name__=='__main__':main()
