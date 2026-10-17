"""Frozen train-only read-only likelihood probe, with independently bound stages."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time
from mri_rl import answer_likelihood as al
from mri_rl.frozen_eval import sha256
from mri_rl.experiment_contract import canonical_hash
from mri_rl.paced_resources import PacedGate, PACED_POLICY
from mri_rl.likelihood_battery import BatteryDiagnosticGate, BATTERY_POLICY
from mri_rl.paced_frozen_eval import generation_metadata, sync_directory


def validate_numeric_record(row, tokenizer=None):
    try:
        scores=row['candidate_scores'];P=len(row['prompt_ids']);oracle=row['prefix_oracle_first_logps']
        if set(scores)!=set(al.LABELS) or P<1 or len(oracle)!=2:raise ValueError()
        prefixes=[];errors=[]
        for label in al.LABELS:
            s=scores[label];part=s['partition'];values=s['target_token_logps']
            if part['target_ids']!=part['prefix_ids']+part['label_ids']+part['suffix_ids']:raise ValueError()
            if tokenizer is not None and part!=al.partition_target(tokenizer,label,prompt_length=P):raise ValueError()
            n=len(part['prefix_ids']);a=len(part['label_ids'])
            if part['label_logit_positions']!=list(range(P+n-1,P+n+a-1)):raise ValueError()
            recomputed=al.summarize_tokens(values,part)
            if any(type(s.get(k)) not in (int,float) or not math.isfinite(s[k]) or not math.isclose(s[k],v,abs_tol=1e-10,rel_tol=1e-10) for k,v in recomputed.items()):raise ValueError()
            first=s['first_label_candidate_logps']
            if len(first)!=2 or any(not math.isfinite(v) or v>1e-6 for v in first+oracle):raise ValueError()
            errors.extend(abs(x-y) for x,y in zip(first,oracle));prefixes.append(values[:n])
        if len(prefixes[0])!=len(prefixes[1]):raise ValueError()
        errors.extend(abs(x-y) for x,y in zip(*prefixes))
        if max(errors)>.002 or not math.isclose(row['prefix_max_disagreement'],max(errors),abs_tol=1e-10):raise ValueError()
        if row['comparisons']!=al.compare_candidates(scores):raise ValueError()
    except (KeyError,ValueError,TypeError,ZeroDivisionError,IndexError):
        raise ValueError('Invalid numeric likelihood receipt') from None


def completed_prefix(path, records, identity, tokenizer=None, adoption=None):
    if not path.exists():return 0
    raw=path.read_bytes()
    if raw and not raw.endswith(b'\n'):raise ValueError('Torn output; preserve and audit before resume')
    if adoption is not None and not raw.startswith(adoption['bytes']):raise ValueError('Adopted prefix bytes changed')
    rows=[json.loads(x) for x in raw.splitlines()]
    if len(rows)>len(records):raise ValueError('Too many output records')
    for index,(actual,expected) in enumerate(zip(rows,records)):
        expected_identity=adoption['identities'][index] if adoption is not None and index<len(adoption['identities']) else identity
        if (actual.get('record_id')!=expected['record_id'] or actual.get('run_identity')!=expected_identity or
            actual.get('image_sha256')!=expected['image_sha256'] or not isinstance(actual.get('response'),str)
            or not actual.get('candidate_scores') or not actual.get('comparisons')):
            raise ValueError('Existing output provenance or numeric evidence mismatch')
        for k,v in expected.items():
            if actual.get(k)!=v:raise ValueError('Existing frozen record field mismatch')
        validate_numeric_record(actual,tokenizer)
    return len(rows)


def verify_review(path,digest,scope):
    if sha256(path)!=digest:raise ValueError('Independent approval digest mismatch')
    r=json.loads(path.read_text())
    if (r.get('kind')!='answer_likelihood_v1' or r.get('verdict') not in ('PASS','PASS_WITH_LIMITATIONS')
        or not r.get('reviewer') or r.get('scope')!=scope):raise ValueError('Independent approval scope mismatch')


# Measured cache footprint 4.7756 GiB + adapter 0.041 GiB + 2 GiB critical reserve (replaces the generic 8 GiB load heuristic on the cache path only).
CACHE_LOAD_FLOOR_GIB=4.7756+0.0407+2.0


def prepare_plan(contract_path, checkpoint, output, max_new_records, battery_diagnostic=False, resident=False, adoption_path=None, cache=None):
    private=Path(__file__).resolve().parents[2]/'local_data'
    if not contract_path.resolve().is_relative_to(private) or not output.resolve().is_relative_to(private):
        raise ValueError('Ignored private contract/output paths required')
    c=json.loads(contract_path.read_text())
    if (c['version']!='training_answer_likelihood_v1' or c['patients']!=28 or c['heldout_used'] is not False
        or c['weight_updates']!=0 or c['fixed_prefix']!=al.PREFIX or c['fixed_suffix']!=al.SUFFIX
        or c['labels']!=list(al.LABELS) or c['generation']!={'max_tokens':256,'temperature':0}):raise ValueError('Wrong frozen experiment')
    if type(max_new_records) is not int or not 1<=max_new_records<=8:raise ValueError('Finite1–8 record scope required')
    if checkpoint not in ('A0','A28','B28'):raise ValueError('Unknown checkpoint')
    for key in ('train','schedule'):
        if sha256(Path(c[key+'_path']))!=c[key+'_sha256']:raise ValueError('Frozen training source changed')
    for path,digest in c['model_artifacts'].items():
        if sha256(Path(path))!=digest:raise ValueError('Model/tokenizer artifact changed')
    for spec in c['checkpoints'].values():
        p=Path(spec['path'])
        if sha256(p)!=spec['sha256'] or sha256(p.parent/'adapter_config.json')!=spec['adapter_config_sha256']:raise ValueError('Checkpoint changed')
    rows=[json.loads(x) for x in Path(c['train_path']).read_text().splitlines()]
    schedule=json.loads(Path(c['schedule_path']).read_text())['rows'];patients={x['patient_key'] for x in schedule}
    pairs=al.training_pairs(rows,patients,c['level']); records=c['records']
    if len(records)!=57 or len(patients)!=28:raise ValueError('Complete28 pair manifest required')
    from mri_rl.paced_frozen_eval import SYSTEM_PROMPT
    image_records=records[:2]+records[3:]
    for actual,source in zip(image_records,pairs):
        image=source['images'][0];q=source['extra_info']['question']
        prompt=[{'role':'system','content':SYSTEM_PROMPT},{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':q}]}]
        if (actual['record_id']!=source['id'] or actual['patient_key']!=source['extra_info']['patient_key']
            or actual['image']!=image or actual['image_sha256']!=sha256(Path(image))
            or actual['visible_label']!=source['reward_model']['ground_truth'] or actual['prompt']!=prompt):
            raise ValueError('Train-only image/label/prompt binding failed')
    noimage=records[2];q=pairs[0]['extra_info']['question']
    if (noimage['record_id']!='no_image_unique' or noimage['image'] is not None or noimage['image_sha256'] is not None
        or noimage['prompt']!=[{'role':'system','content':SYSTEM_PROMPT},{'role':'user','content':[{'type':'text','text':q}]}]):raise ValueError('Unique no-image prompt changed')
    sources={name:sha256(Path(__file__).with_name(name+'.py')) for name in ('likelihood_adoption','likelihood_resident','likelihood_loading','likelihood_battery','paced_answer_likelihood','answer_likelihood','paced_resources','resource_supervisor','frozen_eval','paced_frozen_eval','experiment_contract')}
    if cache:sources['likelihood_cache']=sha256(Path(__file__).with_name('likelihood_cache.py'))
    cache_binding=None
    if cache:
        from mri_rl.likelihood_cache import verify_cache_identity
        cache_binding=verify_cache_identity(cache['dir'],cache['certificate'],cache['certificate_sha256'],c['model_artifacts'])
    # Installed backend files are pinned before native imports.
    dist=importlib.metadata.distribution('mlx-vlm')
    backend={}
    for name in ('utils.py','prompt_utils.py','generate/dispatch.py','generate/ar.py','generate/common.py','trainer/common/model.py','models/qwen3_vl/qwen3_vl.py','models/qwen3_vl/vision.py','models/qwen3_vl/language.py'):
        p=Path(dist.locate_file('mlx_vlm/'+name))
        if not p.is_file():raise ValueError('Required installed backend source missing: '+name)
        backend[name]=sha256(p)
    identity=canonical_hash({'contract':sha256(contract_path),'checkpoint':checkpoint,'sources':sources,'backend':backend,'battery_diagnostic':battery_diagnostic,**({'derived_cache':cache_binding} if cache_binding else {})})
    from tokenizers import Tokenizer
    frozen_tokenizer=Tokenizer.from_file(str(Path(c['model'])/'tokenizer.json'))
    class RawTokenizer:
        def encode(self,text,add_special_tokens=False):return frozen_tokenizer.encode(text,add_special_tokens=add_special_tokens).ids
    completed=0;pending=[]
    scope={'runner':'answer_likelihood_v1','contract_sha256':sha256(contract_path),'checkpoint':checkpoint,
        'adapter':c['checkpoints'][checkpoint],'run_identity':identity,'source_hashes':sources,'backend_sources':backend,
        'output':str(output.resolve()),'existing_output_sha256':sha256(output) if output.exists() else None,
        'completed_records':completed,'new_record_ids':[r['record_id'] for r in pending],
        'max_new_records':max_new_records,'forwards_per_record':3,'generations_per_record':1,
        'max_generation_tokens':256,'temperature':0,'prefix_tolerance':c['common_prefix_tolerance'],
        'model_materialization':'serial_parameter_v2_exact_derived_cache' if cache_binding else 'serial_parameter_v2','derived_cache':cache_binding,'paced_policy':PACED_POLICY,'battery_exception':BATTERY_POLICY if battery_diagnostic else None,'lora_rank':4,'lora_alpha':8,'quantization_bits':4,'group_size':64,'weight_updates':0,'resident':resident}
    adoption=None
    if adoption_path is not None:
        from mri_rl.likelihood_adoption import verify_adoption
        adoption=verify_adoption(adoption_path,scope)
        if not output.exists():raise ValueError('Adopted output must already contain original prefix bytes')
    completed=completed_prefix(output,records,identity,RawTokenizer(),adoption)
    pending=records[completed:completed+max_new_records]
    scope.update(completed_records=completed,new_record_ids=[r['record_id'] for r in pending],
        adopted_prefix=adoption['binding'] if adoption else None)
    return c,pending,scope


def run(contract_path,checkpoint,output,approval,approval_digest,max_new_records,battery_diagnostic=False,
        resident=False,resident_session=None,adoption_path=None,cache=None):
    if resident and (resident_session is None or not Path(resident_session).resolve().is_relative_to(Path(__file__).resolve().parents[2]/'local_data')):
        raise ValueError('Resident execution requires a new ignored private session directory')
    c,pending,scope=prepare_plan(contract_path,checkpoint,output,max_new_records,battery_diagnostic,resident,adoption_path,cache)
    verify_review(approval,approval_digest,scope)
    if not pending:return {'status':'COMPLETED','new_records':0}
    output.parent.mkdir(parents=True,exist_ok=True)
    gate=(BatteryDiagnosticGate(output.with_suffix('.resources.jsonl'),load_floor_gib=cache['load_floor_gib']) if (battery_diagnostic and cache) else (BatteryDiagnosticGate if battery_diagnostic else PacedGate)(output.with_suffix('.resources.jsonl')));gate.before_unit('before_runtime_import',loading=True)
    os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np
    from mlx.utils import tree_flatten
    from mlx_vlm import load,generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.trainer.common.model import prepare_model_for_training
    from mlx_vlm.utils import prepare_inputs,should_add_special_tokens
    mx.set_memory_limit(PACED_POLICY['memory_guideline_gib']*1024**3);mx.set_cache_limit(PACED_POLICY['cache_limit_mib']*1024**2)
    gate.before_unit('model_load',loading=True)
    model=processor=None
    resident_control=None;chunks=None
    try:
        adapter=Path(c['checkpoints'][checkpoint]['path'])
        from mri_rl.likelihood_loading import materialize_serial
        if cache:
            # Exact derived cache: installed loader builds the 344 quantized modules from stored tensors (no BF16 staging);
            # every loaded base parameter is compared with the certified cache bytes BEFORE the adapter is attached.
            model,processor=load(cache['dir'],lazy=True)
            materialize_serial(tree_flatten(model.parameters()),evaluate=mx.eval,collect=gc.collect,clear=mx.clear_cache,
                observe=lambda name:gate.before_unit(name),observe_every=64)
            from mri_rl.likelihood_cache import verify_loaded_parameters
            def to_bytes(v):
                a=np.asarray(v.view(mx.uint16) if v.dtype==mx.bfloat16 else v)
                return a.tobytes(),('BF16' if v.dtype==mx.bfloat16 else 'U32' if v.dtype==mx.uint32 else str(v.dtype)),a.shape
            if verify_loaded_parameters(tree_flatten(model.parameters()),cache['dir'],to_bytes)!=1438:raise ValueError('Cache parameter inventory mismatch')
            gate.before_unit('cache_parameters_verified')
            model=prepare_model_for_training(model,train_type='lora',lora_rank=4,lora_alpha=8,quantization_bits=None,quantization_group_size=64,checkpoint_path=adapter,verbose=False)
        else:
            model,processor=load(c['model'],lazy=True)
            model=prepare_model_for_training(model,train_type='lora',lora_rank=4,lora_alpha=8,quantization_bits=4,quantization_group_size=64,checkpoint_path=adapter,verbose=False)
        materialize_serial(tree_flatten(model.parameters()),evaluate=mx.eval,collect=gc.collect,clear=mx.clear_cache,
            observe=lambda name:gate.before_unit(name),observe_every=64)
        model.eval();gc.collect();mx.clear_cache();gate.before_unit('quantized_model_ready')
        def assert_weights():
            disk=mx.load(str(adapter));actual=dict(tree_flatten(model.trainable_parameters()))
            if len(actual)!=504 or set(disk)!=set(actual) or any(v.shape!=actual[k].shape or v.dtype!=actual[k].dtype or not bool(mx.array_equal(v,actual[k]).item()) for k,v in disk.items()):raise ValueError('Actual adapter reload/read-only integrity failed')
        assert_weights()
        def fingerprint(value):
            if value is None:return None
            mx.eval(value);a=np.asarray(value)
            return {'shape':list(a.shape),'dtype':str(a.dtype),'sha256':hashlib.sha256(a.tobytes()).hexdigest()}
        completed=scope['completed_records']
        if (sha256(output) if output.exists() else None)!=scope['existing_output_sha256']:raise ValueError('Output changed after review')
        if resident:
            from mri_rl.likelihood_resident import ResidentSession
            def next_plan():
                next_contract,next_pending,next_scope=prepare_plan(contract_path,checkpoint,output,max_new_records,battery_diagnostic,True,adoption_path,cache)
                if next_contract!=c:raise ValueError('Frozen resident contract changed')
                return next_pending,next_scope
            resident_control=ResidentSession(records=c['records'],initial_pending=pending,initial_scope=scope,
                approval_path=approval,approval_sha256=approval_digest,session_dir=resident_session,
                plan_next=next_plan,gate=gate,verify_adapter=assert_weights,verify_approval=verify_review)
            chunks=iter(resident_control)
        else:
            chunks=iter([{'pending':pending,'scope':scope}])
        for chunk in chunks:
            pending=chunk['pending'];scope=chunk['scope']
            with output.open('a') as handle:
                sync_directory(output.parent)
                for record in pending:
                    rid=record['record_id'];image=record['image'];gate.before_unit('before_record:'+rid)
                    if image and sha256(Path(image))!=record['image_sha256']:raise ValueError('Pixel changed after review')
                    prompt=apply_chat_template(processor,model.config,record['prompt'],num_images=int(image is not None),enable_thinking=False,add_generation_prompt=True)
                    if prompt.count('<|image_pad|>')!=int(image is not None) or not prompt.endswith('<|im_start|>assistant\n'):raise ValueError('Native prefix/visual placeholder mismatch')
                    special=should_add_special_tokens(model.config.model_type,processor)
                    prepared=prepare_inputs(processor,images=[image] if image else None,prompts=prompt,
                        image_token_index=getattr(model.config,'image_token_index',None),add_special_tokens=special)
                    prompt_ids=prepared['input_ids'];P=int(prompt_ids.shape[-1]);pixels=prepared.get('pixel_values')
                    if prompt_ids.shape[0]!=1 or not 1<=P<=2048 or not bool(mx.all(prepared['attention_mask']==1).item()):raise ValueError('Prompt budget/padding mismatch')
                    kwargs={k:v for k,v in prepared.items() if k not in ('input_ids','attention_mask','pixel_values')}
                    prompt_list=prompt_ids[0].tolist();image_token_count=prompt_list.count(151655)
                    if image:
                        grid=prepared['image_grid_thw'].tolist();expected_images=sum(t*h*w//4 for t,h,w in grid)
                        if pixels is None or image_token_count!=expected_images or not image_token_count:raise ValueError('Visual token/grid conditioning mismatch')
                    elif pixels is not None or image_token_count:raise ValueError('No-image path contains visual tensors')
                    metadata={k:fingerprint(v) for k,v in prepared.items() if hasattr(v,'shape')}
                    candidates={};first=[];prefix_scores=[];parts={}
                    for label in al.LABELS:
                        gate.before_unit('candidate:'+rid+':'+label)
                        partition=al.partition_target(processor.tokenizer,label,prompt_length=P);parts[label]=partition
                        # Verify textual prompt-target boundary independently of expanded image tokens.
                        enc=lambda text:list(processor.tokenizer.encode(text,add_special_tokens=False))
                        if enc(prompt+al.PREFIX+label+al.SUFFIX)!=enc(prompt)+partition['target_ids']:raise ValueError('Native prompt/target token boundary mismatch')
                        target=mx.array(partition['target_ids'],dtype=prompt_ids.dtype)
                        ids=mx.concatenate((prompt_ids,target[None,:]),axis=1);inputs=ids[:,:-1]
                        logits=model(inputs,pixels,mx.ones_like(inputs),**kwargs).logits
                        if logits.shape[1]!=inputs.shape[1]:raise ValueError('Causal logits length mismatch')
                        lp=nn.log_softmax(logits[:,P-1:,:].astype(mx.float32),axis=-1)
                        chosen=mx.take_along_axis(lp,target[None,:,None],axis=-1)[0,:,0]
                        n=len(partition['prefix_ids']);label_first=[al.partition_target(processor.tokenizer,l,prompt_length=P)['label_ids'][0] for l in al.LABELS]
                        first_lp=lp[0,n,mx.array(label_first)];mx.eval(chosen,first_lp)
                        values=chosen.tolist();first.append(first_lp.tolist());prefix_scores.append(values[:n])
                        candidates[label]={**al.summarize_tokens(values,partition),'partition':partition,'target_token_logps':values,'first_label_candidate_logps':first[-1]}
                        del logits,lp,chosen,first_lp,ids,inputs,target;gc.collect();mx.clear_cache()
                    gate.before_unit('prefix_oracle:'+rid)
                    prefix=mx.array(parts[al.LABELS[0]]['prefix_ids'],dtype=prompt_ids.dtype)
                    oracle_ids=mx.concatenate((prompt_ids,prefix[None,:]),axis=1)
                    oracle_logits=model(oracle_ids,pixels,mx.ones_like(oracle_ids),**kwargs).logits
                    oracle_lp=nn.log_softmax(oracle_logits[:,-1,:].astype(mx.float32),axis=-1)[0,mx.array(label_first)]
                    mx.eval(oracle_lp);oracle=oracle_lp.tolist()
                    errors=[abs(a-b) for a,b in zip(prefix_scores[0],prefix_scores[1])]+[abs(a-b) for row in first for a,b in zip(row,oracle)]
                    maxerror=max(errors)
                    if maxerror>c['common_prefix_tolerance']:raise ValueError('Independent prefix forward disagrees with candidate causal scores')
                    del oracle_logits,oracle_lp,oracle_ids,prefix,prepared,pixels,kwargs,prompt_ids;gc.collect();mx.clear_cache()
                    gate.before_unit('generation:'+rid)
                    response=generate(model,processor,prompt,image=[image] if image else None,max_tokens=256,temperature=0,verbose=False)
                    result={**record,'run_identity':scope['run_identity'],'checkpoint':checkpoint,'adapter_sha256':c['checkpoints'][checkpoint]['sha256'],
                        'prompt_ids':prompt_list,'prepared_metadata':metadata,'image_token_count':image_token_count,'add_special_tokens':special,
                        'rendered_prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'candidate_scores':candidates,
                        'comparisons':al.compare_candidates(candidates),'prefix_oracle_first_logps':oracle,'prefix_max_disagreement':maxerror,
                        'response':response.text,**generation_metadata(response),'weight_updates':0}
                    validate_numeric_record(result,processor.tokenizer)
                    handle.write(json.dumps(result,allow_nan=False)+'\n');handle.flush();os.fsync(handle.fileno());completed+=1
                    print(json.dumps({'completed':completed,'record_id':rid,'checkpoint':checkpoint}),flush=True)
                    del response;gc.collect();mx.clear_cache();gate.before_unit('after_record:'+rid);time.sleep(2)
        if resident_control is None:assert_weights()
    finally:
        if chunks is not None and hasattr(chunks,'close'):chunks.close()
        del model,processor;gc.collect();mx.clear_cache();gate.before_unit('model_released')
    return {'status':resident_control.status if resident_control is not None else ('COMPLETED' if completed==57 else 'STAGE_COMPLETED'),'total_records':completed,'checkpoint':checkpoint,'weight_updates':0}


def main():
    p=argparse.ArgumentParser();p.add_argument('--contract',type=Path,required=True);p.add_argument('--checkpoint',choices=['A0','A28','B28'],required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--max-new-records',type=int,default=2)
    p.add_argument('--adopt-prefix',type=Path);p.add_argument('--resident',action='store_true');p.add_argument('--resident-session',type=Path)
    p.add_argument('--cache-dir',type=Path);p.add_argument('--cache-certificate',type=Path);p.add_argument('--cache-certificate-sha256');p.add_argument('--battery-diagnostic',action='store_true');p.add_argument('--approval',type=Path);p.add_argument('--approval-sha256');p.add_argument('--execute',action='store_true');a=p.parse_args()
    args=(a.contract.resolve(),a.checkpoint,a.output.resolve())
    cache=None
    if a.cache_dir:
        if not (a.cache_certificate and a.cache_certificate_sha256):p.error('Cache needs --cache-certificate and --cache-certificate-sha256')
        cache={'dir':str(a.cache_dir.resolve()),'certificate':str(a.cache_certificate.resolve()),'certificate_sha256':a.cache_certificate_sha256,'load_floor_gib':CACHE_LOAD_FLOOR_GIB}
    if not a.execute:print(json.dumps(prepare_plan(*args,a.max_new_records,a.battery_diagnostic,a.resident,a.adopt_prefix,cache)[2],indent=2));return
    if not a.approval or not a.approval_sha256:p.error('Independent scope approval required')
    print(json.dumps(run(*args,a.approval,a.approval_sha256,a.max_new_records,a.battery_diagnostic,a.resident,a.resident_session,a.adopt_prefix,cache)))

if __name__=='__main__':main()
