"""Fixed four-context canonical candidate signal probe; CPU_PREPARED_NOT_APPROVED.

No generation, gradients, optimizer update or historical qualification replay.
Only new independently reviewed/human-authorized execution may construct MLX.
"""
from __future__ import annotations
import copy
import gc
import hashlib
import json
import math
import os
from dataclasses import asdict
from pathlib import Path
import time
from mri_rl import paired_z045_continuation_entry_v1 as entry
from mri_rl.paired_z045_f32_backend_v1 import f32_response_logps_observed
from mri_rl.paired_z045_matched_f32_eval_v1 import token_plan,label_scores,common_prefix_alignment,prefix_alignment,score_input_positions
from mri_rl.paired_z045_evaluation_v1 import LABELS
from mri_rl.answer_likelihood import PREFIX
from mri_rl.continuous_rl import require
from mri_rl.paced_resources import PacedGate, PACED_POLICY, decisions

ROOT=Path(__file__).resolve().parents[2]
PRIOR_SCOPE_SHA='865aeda5a772c9a917370bacddfd84309ac51ac225cd52732e0ee0e2f56f3511'
canonical=entry.scope_digest
POWER_EXCEPTION={
    'scope':'paired_signal_candidate_probe_v1',
    'minimum_battery_percent':60,
    'battery_threshold_applies_on_AC_too':True,
    'approved_prior_scope_file_sha256':'818a4469a842ea46145c1134f382e34bcacb96f92aa9b938eb8ec08deddf3c92',
    'human_source':'Current direct human authorization reported by root: one12-forward readonly probe; local battery>=60 exception and necessary exact amendment review permitted.',
    'global_training_gate_changed':False,
}


def probe_power_decisions(sample,*,loading,supervisor_reasons=()):
    """Only this approved probe may remove the AC reason; never alter a sample."""
    action,reasons=decisions(sample,loading=loading,supervisor_reasons=supervisor_reasons)
    if action=='STOP':return action,reasons
    battery=sample.battery_percent
    verified=(isinstance(battery,(int,float)) and not isinstance(battery,bool)
              and math.isfinite(battery) and 60<=battery<=100)
    if not verified or sample.power_source not in ('AC','BATTERY'):
        return 'STOP',reasons+['probe_battery_below60_or_unverified']
    if sample.power_source=='AC':return action,reasons
    reasons=[reason for reason in reasons if reason!='power_not_verified_AC']
    return ('HOLD' if reasons else 'CONTINUE'),reasons


class ProbeBatteryPacedGate(PacedGate):
    """Local power-only exception; original supervisor sees and records raw samples."""
    def before_unit(self,name,loading=False):
        for attempt in range(PACED_POLICY['wait_rechecks']):
            sample=self.sampler();level,reasons=self.policy.observe(sample)
            action,reasons=probe_power_decisions(sample,loading=loading,supervisor_reasons=reasons)
            event={'unit_id':name,'action':action,'supervisor_level':level,'reasons':reasons,
                   'sample':asdict(sample),'power_exception':POWER_EXCEPTION}
            with self.path.open('a') as f:
                f.write(json.dumps(event,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
            if action=='CONTINUE':return sample
            if action=='STOP':raise RuntimeError('Paced workload stopped at safe boundary: '+','.join(reasons))
            print('Resource hold at '+name+': '+','.join(reasons),flush=True)
            self.sleep(PACED_POLICY['wait_seconds'])
        raise RuntimeError('Resource hold persists; completed checkpoints preserved, no automatic restart')


def budget():
    return {'contexts':4,'full_TF_forwards':12,'teacherforced_positions':5100,'vision_TF_forwards':9,'text_only_TF_forwards':3,
        'model_loads':1,'maximum_seconds':600,'generation_calls':0,'backward_calls':0,'optimizer_updates':0,'simultaneous_image_graphs':1}


class Counter:
    def __init__(self):self.values={'full_TF_forwards':0,'teacherforced_positions':0}
    def charge(self,positions):
        require(type(positions) is int and 0<positions<=425 and self.values['full_TF_forwards']+1<=12 and self.values['teacherforced_positions']+positions<=5100,'Fixed12-forward/5100-position budget exceeded')
        self.values['full_TF_forwards']+=1;self.values['teacherforced_positions']+=positions


def fixed_contexts(prior,pool):
    require(len(prior['rows_PRIVATE'])==2,'Original exactly two qualification context rows required')
    seq=next(r for r in prior['rows_PRIVATE'] if r['ability']=='mri_sequence');Dallas=next(r for r in prior['rows_PRIVATE'] if r['ability']=='mri_visit_correspondence')
    source=next(r for r in pool if r['id']==seq['source_record_id'])
    require(source['extra_info']['patient_key']==seq['extra_info']['patient_key'] and source['paired_z045_data_v1']['level']==45 and source['reward_model']==seq['reward_model'] and source['images']==seq['images'],'Original sequence row is not the fixed56pool source')
    others=[r for r in pool if r['extra_info']['patient_key']==source['extra_info']['patient_key'] and r['paired_z045_data_v1']['level']==45 and r['reward_model']['ground_truth']!=source['reward_model']['ground_truth']]
    require(len(others)==1,'Exactly one same-person/z045 opposite-modality source required');other=others[0]
    require(other['extra_info']['question']==seq['extra_info']['question'] and other['reward_model']['ground_truth'] in LABELS['mri_sequence'],'Opposite input changed question/canonical task')
    def item(name,row,sha,truth):return {'name':name,'row_PRIVATE':copy.deepcopy(row),'image_sha256':sha,'source_truth':truth,'no_image':False}
    contexts=[item('SEQUENCE_REAL',seq,prior['expected_image_sha256'][seq['id']],seq['reward_model']['ground_truth']),
        item('SEQUENCE_OPPOSITE_SAME_PERSON_Z045',other,other['rc_data_v1']['image_sha256'],other['reward_model']['ground_truth'])]
    null=item('SEQUENCE_NO_IMAGE_NULL',seq,None,None);null['row_PRIVATE']['id']=seq['id']+'__NO_IMAGE_NULL';null['row_PRIVATE']['images']=[];null['no_image']=True
    null['interpretation']='Text-only same question prior; no displayed visual truth and no fabricated correct-class claim.'
    contexts.append(null);contexts.append(item('DALLAS_REAL',Dallas,prior['expected_image_sha256'][Dallas['id']],Dallas['reward_model']['ground_truth']))
    return contexts


def normalize_inputs(values,*,no_image):
    result={k:v for k,v in values.items() if v is not None}
    result['pixel_values']=values.get('pixel_values')
    if no_image:require(result['pixel_values'] is None and not any(k in result for k in ('image_grid_thw','video_grid_thw','pixel_values_videos')),'NULL must contain no pixels/grid; zero-pixel image is not a no-image control')
    else:require(result['pixel_values'] is not None and result.get('image_grid_thw') is not None,'Real MRI image/grid required')
    return result


def prompt_guard(task,length,no_image):
    require(task in LABELS and type(length) is int and ((task=='mri_sequence' and 0<length<213) if no_image else length=={'mri_sequence':213,'mri_visit_correspondence':405}[task]),'Actual prompt shape differs from frozen image/text context contract')
    return length


def margins(values,labels,truth):
    delta={key:values[0][key]-values[1][key] for key in ('mean','sum','first')}
    return {'canonical_label0_minus_label1':delta,'source_truth_minus_alternative':{k:(v if labels.index(truth)==0 else -v) for k,v in delta.items()} if truth in labels else None}


def authorize(scope,human,review):
    binding=canonical(scope)
    require(human.get('authorized') is True and human.get('action')=='readonly_paired_signal_candidate_probe' and human.get('scope_sha256')==binding,'New explicit human candidate-probe authorization required; prior one-shot permission consumed')
    require(review.get('scope_sha256')==binding and review.get('verdict') in ('PASS','PASS_WITH_LIMITATIONS') and review.get('reviewer_id') and review.get('reviewer_kind') in ('isolated_codex','claude_code'),'New independent exact candidate-probe review required')
    return True


def state_unchanged(before,after):
    require(before==after,'Readonly probe changed full parameters/Adam/Python/MLX RNG before restoration')
    return True


def verify_context(context):
    row=context['row_PRIVATE']
    if context['no_image']:require(row['images']==[] and context['image_sha256'] is None,'Genuine NULL context changed')
    else:require(len(row['images'])==1 and entry.digest(row['images'][0])==context['image_sha256'],'Frozen context PNG bytes changed')


def runtime_artifact_bindings(runtime):
    result={}
    for key,expected in runtime['file_sha256'].items():
        require(entry.digest(runtime[key])==expected,'Pinned runtime input changed: '+key)
        result[str(runtime[key])]=expected
    return result


def prepare(spec):
    from tokenizers import Tokenizer
    for key,expected in spec['file_sha256'].items():require(entry.digest(spec[key])==expected,'Pinned input source changed: '+key)
    require(entry.digest(spec['prior_scope'])==PRIOR_SCOPE_SHA,'Original image-bound fixed qualification scope required')
    prior=json.loads(Path(spec['prior_scope']).read_text());runtime=json.loads(Path(spec['runtime_spec']).read_text())
    runtime_inputs=runtime_artifact_bindings(runtime)
    require(spec.get('power_exception')==POWER_EXCEPTION,'Exact current human local battery amendment binding required')
    require(entry.digest(spec['sequence_pool'])==runtime['file_sha256']['sequence56'],'Original frozen56 pool changed')
    pool=[json.loads(line) for line in Path(spec['sequence_pool']).read_text().splitlines() if line.strip()];require(len(pool)==56,'Exactly existing56 rows required')
    contexts=fixed_contexts(prior,pool)
    membership={r['record_id_PRIVATE']:r['PNG_sha256'] for r in json.loads(Path(runtime['data_membership']).read_text())}
    require(entry.digest(runtime['data_membership'])==runtime['file_sha256']['data_membership'],'Existing membership binding changed')
    for context in contexts:
        verify_context(context)
        if context['name']=='SEQUENCE_OPPOSITE_SAME_PERSON_Z045':require(membership[context['row_PRIVATE']['id']]==context['image_sha256'],'Opposite PNG not membership authenticated')
    tokpath=Path(runtime['model'])/'tokenizer.json';tok=Tokenizer.from_file(str(tokpath));encode=lambda text:tok.encode(text,add_special_tokens=False).ids
    plans={task:token_plan(task,PREFIX,encode) for task in LABELS}
    sources={**entry.source_closure(runtime),str(ROOT/'src/mri_rl/paired_z045_matched_f32_eval_v1.py'):entry.digest(ROOT/'src/mri_rl/paired_z045_matched_f32_eval_v1.py')}
    sources[str(spec['runtime_spec'])]=entry.digest(spec['runtime_spec']);sources[str(runtime['contract'])]=runtime['file_sha256']['contract']
    for path in (Path(__file__),ROOT/'scripts/run_paired_signal_candidate_probe_v1.py',ROOT/'tests/test_paired_signal_candidate_probe_v1.py'):sources[str(path)]=entry.digest(path)
    for path,expected in prior['source_sha256'].items():
        candidate=Path(path)
        if candidate.is_relative_to(Path(spec['old_worktree'])/'src/mri_rl'):
            local=ROOT/'src/mri_rl'/candidate.name;require(entry.digest(local)==expected,'Relocated frozen prior helper differs: '+str(local))
    derived=entry.cache_metadata(runtime)['expected_binding'];require(derived==prior['derived_cache'],'Original certified parent cache binding changed')
    output=Path(spec['output']).resolve();require(output.is_relative_to(ROOT/'local_data/paired_signal_candidate_probe_v1'),'New isolated private output required')
    return {'schema':'paired-signal-canonical-candidate-probe-v1','status':'CPU_PREPARED_NOT_APPROVED','training_authorized':False,'human_permission_present':False,'native_approved':False,
        'contexts_PRIVATE':contexts,'token_plans_PRIVATE':plans,'parent':entry.parent_metadata(runtime['parent_checkpoint'],runtime['parent_receipt_sha256']),
        'runtime_spec_sha256':entry.digest(spec['runtime_spec']),'source_sha256':sources,'bound_artifact_sha256':{**{spec[k]:v for k,v in spec['file_sha256'].items()},**runtime_inputs},
        'derived_cache':derived,'budget':budget(),'output':str(output),
        'numeric_contract':'Existing functional F32 full-TF scorer only; two canonical labels and independent same-prefix first-vector oracle percontext. Full-candidate common-prefix and oracle .002 gates unchanged. No KV, stochastic rollout, B2/B3 or backward requalification.',
        'candidate_semantics':'Strict label token spans only; mean primary length-normalized score, sum joint-label span and first nexttoken secondary. No EOS/suffix/prefix in candidate scores.',
        'control':'Same person/level opposite acquisition image; genuine same-question text-only NULL (pixels/grid absent), not zero-image substitution. NULL has no visual correctness label.',
        'state_contract':'All1942 parameters+full Adam+Python/MLX RNG unchanged before any restore; actualstate fingerprints durable, then restore same checkpoint RNG/state and verify; release<128MiB.',
        'power_exception':dict(POWER_EXCEPTION),
        'resources':'Verified battery60..100 on AC and BATTERY; true BATTERY allowed for this probe only/existingglobalflock/localPacedGate/originalResourcePolicy/nootherheavy; one load/one imagegraph;600s operation-start checked before+after resourceholds; atomiccall/release canoverrun; no automatic retry.',
        'mission_alignment':{'tag':'ENGINEERING_UNBLOCK','evidence_tier':'PROPOSED_NOT_RUN','question':'Measure canonical class preference and image-conditioned signal under fixed Adam60, separating same-person modality pair and text prior.','training_claim':False},
        'limitations':['Current direct human approved prior candidate scope and this minimal local battery amendment; exact amended-scope authorization record and independent review still required before execution; no training permission.','Twelve teacherforces only, no learning/greedy-generation/generalization claim.','Three real images and one textNULL context are fixed engineering diagnostics.'],
        'model_loads_this_preparation':0,'generation_calls':0,'backward_calls':0,'optimizer_updates':0}


def prepare_inputs(backend,context,utility):
    from mlx_vlm.generate.dispatch import _prepare_generation_inputs
    row=context['row_PRIVATE'];null=context['no_image'];images=None if null else row['images'];mx=backend.mx
    prompt=backend.template(backend.processor,backend.model.config,[{'role':'system','content':backend.contract.system_instruction},{'role':'user','content':row['extra_info']['question']}],num_images=0 if null else 1,add_generation_prompt=True,enable_thinking=backend.contract.enable_thinking)
    require(prompt.count('<|image_pad|>')==(0 if null else 1) and prompt.endswith(backend.contract.assistant_prefix),'Exact original prompt/NULL placeholder count differs')
    ids,pixels,mask,extra=_prepare_generation_inputs(backend.model,backend.processor,prompt,images,None,None,{})
    actual=normalize_inputs({'input_ids':ids,'pixel_values':pixels,'attention_mask':mask,**extra},no_image=null)
    scorer=normalize_inputs(backend.prepare_inputs(backend.processor,images=images,prompts=prompt,image_token_index=backend.model.config.image_token_index,add_special_tokens=backend.special_tokens(backend.model.config.model_type,backend.processor)),no_image=null)
    a,b=utility._fingerprints(backend,actual),utility._fingerprints(backend,scorer)
    require(a==b and bool(mx.all(mask==1).item()),'Actual inputIDs/pixels/grid/mask/metadata differ across prepare paths')
    prompt_guard(row['ability'],int(ids.shape[-1]),null)
    return prompt,scorer,a


def score_context(backend,context,inputs,plans,counter,before,scorer=f32_response_logps_observed):
    mx=backend.mx;prefix=plans['prefix'];P=int(inputs['input_ids'].shape[-1]);values=[];prefixes=[];evidence=[]
    def score(tokens,capture=None):
        positions=score_input_positions(P,len(prefix),len(tokens)-len(prefix));before('candidate_full_TF');verify_context(context);counter.charge(positions)
        value=scorer(backend,tokens,inputs,evidence=evidence,vocab_observer=capture);mx.eval(value);require(bool(mx.all(mx.isfinite(value)).item()),'Nonfinite canonical candidate logps');return value.tolist()
    for label,ids in zip(plans['label_order'],plans['labels']):
        result=score(prefix+ids);prefixes.append(result[:len(prefix)]);values.append({'label':label,**label_scores(result,len(ids))});gc.collect();mx.clear_cache()
    common=common_prefix_alignment(prefixes);captured=[]
    score(prefix+[plans['labels'][0][0]],lambda value:captured.append(value));require(len(captured)==1,'One independent prefix oracle capture required')
    mx.eval(captured);oracle=backend.nn.log_softmax(captured[0][0,-1,:].astype(mx.float32),axis=-1)
    first=[float(oracle[ids[0]].item()) for ids in plans['labels']];alignment=prefix_alignment([v['first'] for v in values],first)
    del captured,oracle;gc.collect();mx.clear_cache()
    return {'context':context['name'],'task':context['row_PRIVATE']['ability'],'source_truth':context['source_truth'],'image_sha256':context['image_sha256'],
        'per_label':values,'prefix_oracle':{**alignment,'full_candidate_common_prefix':common},'margins':margins(values,plans['label_order'],context['source_truth']),'dtype_context_evidence':evidence}


def run_resident(backend,scope,out,counter,before):
    from mri_rl import early_native_conditioning_probe_v1 as utility
    from mri_rl.paired_z045_f32_qualification_v1 import _state
    out=Path(out);params=utility._fingerprints(backend,dict(backend.tree_flatten(backend.model.parameters())));state=_state(backend,utility)
    require(len(params)==1942,'Full1942 model parameter inventory required');initial_adam=backend.optimizer_state();initial_rng=backend.rng_state();results=[]
    language=backend.model.language_model;metadata=(language._rope_deltas,language._position_ids)
    try:
        for context in scope['contexts_PRIVATE']:
            before('context_prepare');verify_context(context);prompt,inputs,fingerprints=prepare_inputs(backend,context,utility)
            result=score_context(backend,context,inputs,scope['token_plans_PRIVATE'][context['row_PRIVATE']['ability']],counter,before)
            result.update({'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'inputs_fingerprints_PRIVATE':fingerprints})
            utility.durable(out/(context['name']+'_PRIVATE.json'),result);results.append(result)
            state_unchanged(state,_state(backend,utility));inputs=None;gc.collect();backend.mx.clear_cache()
        require(counter.values['full_TF_forwards']==12,'Exactly12 fixed-context forwards required')
        return {'status':'ACTUAL_FIXED_FOUR_CONTEXT_CANDIDATE_PROBE_COMPLETE','contexts':results,'actual_calls':counter.values,'generation_calls':0,'backward_calls':0,'optimizer_updates':0}
    finally:
        after_params=utility._fingerprints(backend,dict(backend.tree_flatten(backend.model.parameters())));after_state=_state(backend,utility)
        utility.durable(out/'before_restore_integrity_PRIVATE.json',{'all_parameters_before':params,'all_parameters_after':after_params,'state_before':state,'state_after':after_state,'unchanged':params==after_params and state==after_state})
        state_unchanged({'parameters':params,'state':state},{'parameters':after_params,'state':after_state})
        backend.restore(initial_adam,initial_rng);restored=_state(backend,utility)
        utility.durable(out/'restored_state_PRIVATE.json',{'initial':state,'restored':restored,'unchanged':state==restored});state_unchanged(state,restored)
        language._rope_deltas,language._position_ids=metadata


def execute(spec,scope,*,human_path,human_sha,review_path,review_sha):
    import fcntl,importlib.metadata
    require(entry.digest(human_path)==human_sha and entry.digest(review_path)==review_sha,'Exact new permission/review files changed')
    authorize(scope,json.loads(Path(human_path).read_text()),json.loads(Path(review_path).read_text()))
    require(prepare(spec)==scope,'Frozen probe metadata/source changed');out=Path(scope['output']);require(not out.exists(),'Fresh one-shot private output required')
    from mri_rl.mlx_continuous_gspo import is_loading_unit
    from mri_rl.paired_z045_f32_backend_v1 import F32NativeGSPOBackend
    from mri_rl.uniform_balance_continuation_gate_v1 import parent_anchor
    from mri_rl.continuous_rl_pilot import cache_binding_for
    from mri_rl.optimizer_checkpoint import load_optimizer_checkpoint
    from mri_rl.experiment_contract import load_contract
    from types import SimpleNamespace
    runtime=json.loads(Path(spec['runtime_spec']).read_text());started=time.monotonic();counter=Counter();resident=[None]
    with Path(runtime['global_lock']).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);require(prepare(spec)==scope,'Probe bindings changed while acquiring lock');out.mkdir(parents=True)
        gate=ProbeBatteryPacedGate(out/'resources_PRIVATE.jsonl');cache={'dir':runtime['cache_dir']}
        def before(name):
            require(time.monotonic()-started<=600,'Candidate-probe operation-start deadline exhausted');gate.before_unit(name,loading=is_loading_unit(name,cache));require(time.monotonic()-started<=600,'Probe deadline exhausted during resource HOLD')
            if name=='native_model_load':
                runtime_artifact_bindings(runtime)
                for context in scope['contexts_PRIVATE']:verify_context(context)
        def run():
            require(importlib.metadata.version('mlx')=='0.32.4' and importlib.metadata.version('mlx-vlm')=='0.7.8','Frozen installed native versions changed')
            derived=cache_binding_for(SimpleNamespace(cache_dir=Path(runtime['cache_dir']),cache_certificate=Path(runtime['cache_certificate']),cache_certificate_sha256=runtime['cache_certificate_sha256'],model=Path(runtime['model'])))
            require(derived==scope['derived_cache'],'Certified Adam60 base cache changed')
            anchor=parent_anchor(runtime['parent_checkpoint'],runtime['parent_receipt_sha256'])
            resident[0]=F32NativeGSPOBackend(Path(runtime['parent_checkpoint'])/'adapter.safetensors',{**entry.driver.PROFILE,'group_size':4,'max_tokens':256},model_path=Path(runtime['model']),contract=load_contract(Path(runtime['contract'])),seed=13,before_unit=before,cache=cache)
            backend=resident[0];adam,rng,_=load_optimizer_checkpoint(Path(runtime['parent_checkpoint'])/'optimizer',expected_bindings={'run':anchor['original_binding'],'adapter_sha256':anchor['adapter_sha256'],'trainable_signature':backend.trainable_signature()},load_arrays=backend.load_arrays)
            backend.restore(adam,rng);require(int(backend.optimizer_state()['step'].item())==60,'Exact restored Adam60 required');adam=rng=None
            return run_resident(backend,scope,out,counter,before)
        result=entry.run_with_release(run,resident,out)
        require(prepare(spec)==scope,'Probe source/input/checkpoint bindings changed during execution')
        result.update({'scope_sha256':canonical(scope),'source_sha256':scope['source_sha256'],'wall_seconds':time.monotonic()-started})
        entry.driver.write_json(out/'complete_PRIVATE.json',result)
