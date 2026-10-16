"""Reviewed, bounded B28-only matched development inference; no eager model import.

This wrapper leaves the historical evaluator and training source unchanged.
Each launch generates at most four previously unfinished records from the entire
frozen54-record manifest. It cannot authorize itself or declare MRI learning.
"""
from __future__ import annotations
import argparse
from collections import Counter
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time

from mri_rl.frozen_eval import adapter_metadata, prepare_resume, sha256
from mri_rl.experiment_contract import canonical_hash, load_contract, resolve_artifact_paths
from mri_rl.paced_resources import PACED_POLICY, PacedGate

MATCHED_MANIFEST_SHA256='9e5194cc808ad27592549b3fab21e31fd6ba0a9582ca76fac75928b589b9734f'
B_SCHEDULE_SHA256='2e5d6bcb55cfb949035986fa8e65debdf13013badb04c275d195fd96be661f67'
ORIGINAL_ADAPTER_SHA256='502084c8a1394fdffd765741b9b0fbfcb6720268dfe7671c20d7a67dc5f86088'
SYSTEM_PROMPT='Respond with one short observation in <think> and one boxed final answer in <answer>.'
CATEGORY_COUNTS={'mri_sequence':24,'mri_tumor_presence':12,'mri_enhancing_presence':12,'mri_tumor_grid':6}
BASELINE_ADAPTER_CONFIG_SHA256='032becabb7cbb6d2af6f3451eb127e5f944551d42eb51eaee3e8d75b839b2f6a'
BASELINES={
    'sft0_matched_native_outputs_PRIVATE.jsonl':{
        'sha256':'578e523bbbedae6f221ddc5732817e15df2b2f257d4bc23a4fb77d4d85afde59',
        'adapter_checkpoint_sha256':ORIGINAL_ADAPTER_SHA256},
    'sft28_matched_native_outputs_PRIVATE.jsonl':{
        'sha256':'c65f5e4343f8f9bc3ee6d77c71c5a87dbd794b7280f5c7c85cf937d515142d9e',
        'adapter_checkpoint_sha256':'244c26fd5ce2e3b02b46703d4f46a90608f2ef1a868eeee682ee4fe48a58ed73'},
}


def sync_directory(path):
    fd=os.open(path,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def validate_completion_claim(receipt):
    if (type(receipt.get('optimizer_steps')) is not int or receipt['optimizer_steps']!=28 or
        receipt.get('completed_control_28steps') is not True or receipt.get('status')!='SFT_ENGINEERING_CONTROL' or
        receipt.get('checkpoint_reloaded') is not True or receipt.get('schedule_policy')!='balanced_sequence_v2' or
        receipt.get('schedule_sha256')!=B_SCHEDULE_SHA256 or receipt.get('executed_prefix_examples')!=112 or
        receipt.get('executed_prefix_sha256')!=B_SCHEDULE_SHA256 or
        receipt.get('original_starting_adapter_sha256')!=ORIGINAL_ADAPTER_SHA256 or
        receipt.get('parameters_finite') is not True or receipt.get('adapter_tensor_count')!=504 or
        type(receipt.get('changed_tensors')) is not int or receipt['changed_tensors']<=0):
        raise ValueError('Completed, verified balanced B28 required; partial training cannot enter matched28 evaluation')
    losses=receipt.get('loss_terms',[]);grads=receipt.get('gradient_l2_by_step',[])
    if (len(receipt.get('examples',[]))!=112 or len(losses)!=112 or len(grads)!=28 or
        any(type(x) not in (int,float) or not math.isfinite(x) for x in losses+grads) or any(x<=0 for x in grads)):
        raise ValueError('Incomplete/nonfinite B28 update history')


def validate_b28(run,contract_id):
    receipt_path=run/'receipt.json';receipt=json.loads(receipt_path.read_text());validate_completion_claim(receipt)
    adapter=run/'after_adapters.safetensors';meta=adapter_metadata(adapter)
    if (receipt.get('experiment_contract_id')!=contract_id or receipt.get('after_adapter_sha256')!=meta['adapter_checkpoint_sha256'] or
        receipt.get('adapter_config_sha256')!=meta['adapter_config_sha256']):raise ValueError('B28 contract/adapter identity mismatch')
    inventory=json.loads((run/'schedule_PRIVATE.json').read_text())['rows']
    if len(inventory)!=112 or canonical_hash(inventory)!=B_SCHEDULE_SHA256:raise ValueError('Frozen B schedule mismatch')
    for actual,expected in zip(receipt['examples'],inventory):
        if any(actual.get(k)!=expected[k] for k in ('step','category','record_id','patient_key','image_sha256','target_label')):
            raise ValueError('B28 executed examples differ from schedule')
    reload_path=run/'reload_inference_receipt.json';reload=json.loads(reload_path.read_text())
    if (sha256(reload_path)!=receipt.get('reload_inference_receipt_sha256') or reload.get('loaded_adapter_sha256')!=meta['adapter_checkpoint_sha256'] or
        reload.get('exact_adapter_tensor_match') is not True or reload.get('adapter_tensor_count')!=504 or
        reload.get('actual_model_generation') is not True or reload.get('contract_id')!=contract_id):raise ValueError('B28 reload evidence differs')
    state=run/'checkpoints/state_028';manifest=json.loads((state/'manifest.json').read_text());bindings=manifest['bindings']
    if (manifest.get('optimizer_updates')!=28 or manifest.get('completed_groups')!=28 or
        bindings.get('adapter_sha256')!=meta['adapter_checkpoint_sha256'] or bindings.get('contract_id')!=contract_id or
        bindings.get('schedule_sha256')!=B_SCHEDULE_SHA256 or bindings.get('sampler_position_examples')!=112 or
        manifest.get('tensors_sha256')!=sha256(state/'state.safetensors')):raise ValueError('Final optimizer checkpoint mismatch')
    return adapter,{**meta,'b28_receipt_sha256':sha256(receipt_path),'b28_state_manifest_sha256':sha256(state/'manifest.json')}


def generation_metadata(response):
    tokens=getattr(response,'generation_tokens',None);peak=getattr(response,'peak_memory',None)
    prompt=getattr(response,'prompt_tokens',None);total=getattr(response,'total_tokens',None)
    finish=getattr(response,'finish_reason',None)
    if (type(tokens) is not int or tokens<0 or type(prompt) is not int or prompt<0 or type(total) is not int or total<0 or
        type(peak) not in (int,float) or not math.isfinite(peak) or peak<0 or finish not in (None,'stop','length')):
        raise ValueError('Backend generation measurements missing or invalid')
    return {'generation_tokens':tokens,'prompt_tokens':prompt,'total_tokens':total,'finish_reason':finish,
            'truncated':finish=='length' if finish is not None else None,
            'generation_peak_memory_gb':peak,'generation_peak_memory_units':'decimal GB (backend process peak, not per-record allocation delta)'}


def bounded_pending(records,completed,max_new_records):
    ids=[r['record_id'] for r in records]
    if type(max_new_records) is not int or not 1<=max_new_records<=4:raise ValueError('Finite1–4 new-record budget required')
    if len(set(ids))!=len(ids) or not set(completed)<=set(ids):raise ValueError('Unknown/duplicate evaluation record')
    return [r for r in records if r['record_id'] not in completed][:max_new_records]


def resume_completed(output,records,expected,*,resume):
    if output.exists() and not resume:raise FileExistsError('Existing predictions require explicit resume; never overwrite')
    completed=prepare_resume(output,records,expected) if resume else set()
    if output.exists():sync_directory(output.parent)
    return completed


def verify_approval(path,expected_sha,scope):
    if sha256(path)!=expected_sha:raise ValueError('Independent approval checksum mismatch')
    review=json.loads(path.read_text())
    if (review.get('kind')!='paced_frozen_eval_v1' or review.get('verdict') not in ('PASS','PASS_WITH_LIMITATIONS') or
        not isinstance(review.get('reviewer'),str) or not review['reviewer'].strip() or review.get('scope')!=scope):
        raise ValueError('Independent approval does not cover exact evaluation scope')


def verify_historical_baselines(directory,records,expected):
    """Freeze pixel/prompt/profile comparability against both actual A0/A28 outputs."""
    if expected['adapter_config_sha256']!=BASELINE_ADAPTER_CONFIG_SHA256:
        raise ValueError('Historical baseline adapter configuration differs from B28')
    current={r['record_id']:r for r in records}
    if len(current)!=54 or len(records)!=54:raise ValueError('Historical baseline comparison requires54 unique IDs')
    images={key:sha256(Path(row['image'])) for key,row in current.items()};digests={}
    for name,spec in BASELINES.items():
        path=directory/name
        if not path.is_file() or sha256(path)!=spec['sha256']:raise ValueError('Immutable historical baseline digest mismatch: '+name)
        rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        keyed={r.get('record_id'):r for r in rows}
        if len(rows)!=54 or len(keyed)!=54 or set(keyed)!=set(current):raise ValueError('Historical baseline IDs/counts differ')
        for key,row in keyed.items():
            source=current[key]
            if (row.get('category')!=source['category'] or row.get('prompt')!=source['prompt'] or
                row.get('image_sha256')!=images[key]):raise ValueError('Historical baseline prompt/category/pixel mismatch')
            for field in ('manifest_sha256','model_path','model_config_sha256','model_template_sha256','template_override_sha256','max_tokens','temperature'):
                if row.get(field)!=expected[field]:raise ValueError('Historical baseline profile mismatch: '+field)
            if (row.get('image_placeholder_count')!=1 or row.get('adapter_verified_parameter_tensors')!=504 or
                row.get('adapter_checkpoint_sha256')!=spec['adapter_checkpoint_sha256'] or
                row.get('adapter_config_sha256')!=BASELINE_ADAPTER_CONFIG_SHA256 or not isinstance(row.get('response'),str)):
                raise ValueError('Historical baseline adapter provenance or output completeness mismatch')
        digests[name]=spec['sha256']
    return digests


def prepare_plan(model_path,manifest_path,output_path,b28_run,contract_path,*,max_new_records=4,resume=False):
    private=Path(__file__).resolve().parents[2]/'local_data'
    for path in (manifest_path,output_path,b28_run,contract_path):
        if not path.resolve().is_relative_to(private):raise ValueError('Private experiment paths required')
    contract=load_contract(contract_path);adapter,bmeta=validate_b28(b28_run,contract.identity())
    if contract.system_instruction!=SYSTEM_PROMPT or contract.generation!={'temperature':0.0,'max_tokens':256}:
        raise ValueError('Matched system/decoding contract differs')
    paths=resolve_artifact_paths(contract_path)
    model_files={}
    for name,filename,field in [('model_config','config.json','model_config_sha256'),('processor_config','preprocessor_config.json','processor_config_sha256'),
                               ('tokenizer_config','tokenizer_config.json','tokenizer_config_sha256'),('chat_template','chat_template.json','chat_template_sha256')]:
        path=model_path/filename
        if paths[name].resolve()!=path.resolve() or sha256(path)!=getattr(contract,field):raise ValueError('Frozen model/processor path or configuration differs')
        model_files[field]=sha256(path)
    if sha256(manifest_path)!=MATCHED_MANIFEST_SHA256:raise ValueError('Exact matched54 development manifest required')
    records=[json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
    if len(records)!=54 or dict(Counter(r['category'] for r in records))!=CATEGORY_COUNTS:raise ValueError('Development manifest category/count mismatch')
    for record in records:
        if (not record.get('image') or not Path(record['image']).is_file() or
            not isinstance(record.get('prompt'),list) or not record['prompt'] or
            record['prompt'][0]!={'role':'system','content':SYSTEM_PROMPT}):raise ValueError('Matched system prompt or image missing')
    sources={name:sha256(Path(__file__).with_name(name+'.py')) for name in ('paced_frozen_eval','frozen_eval','paced_resources','resource_supervisor','experiment_contract')}
    expected={'model_path':str(model_path.resolve()),'model_config_sha256':model_files['model_config_sha256'],
        'model_template_sha256':model_files['chat_template_sha256'],'processor_config_sha256':model_files['processor_config_sha256'],
        'tokenizer_config_sha256':model_files['tokenizer_config_sha256'],'template_override_sha256':None,
        'manifest_sha256':MATCHED_MANIFEST_SHA256,'max_tokens':256,'temperature':0,**bmeta,
        'image_inventory_sha256':canonical_hash([{'record_id':r['record_id'],'sha256':sha256(Path(r['image']))} for r in records]),
        'paced_eval_sources_sha256':canonical_hash(sources),'experiment_contract_id':contract.identity()}
    expected['historical_baseline_sha256']=verify_historical_baselines(manifest_path.parent,records,expected)
    output_path.parent.mkdir(parents=True,exist_ok=True)
    completed=resume_completed(output_path,records,expected,resume=resume)
    pending=bounded_pending(records,completed,max_new_records)
    scope={'runner':'paced_frozen_eval_v1','expected':expected,'source_hashes':sources,'output':str(output_path.resolve()),
        'existing_output_sha256':sha256(output_path) if output_path.exists() else None,'completed_records':len(completed),
        'new_record_ids':[r['record_id'] for r in pending],
        'new_image_sha256':{r['record_id']:sha256(Path(r['image'])) for r in pending},'max_new_records':max_new_records,'max_tokens':256,
        'paced_policy':PACED_POLICY,'adapter':str(adapter.resolve()),'quantization_bits':4,'quantization_group_size':64,
        'lora_rank':4,'lora_alpha':8,'model_loading':'lazy_then_restore_quantized_adapter_then_materialize',
        'enable_thinking':False,'add_generation_prompt':True}
    return records,pending,expected,scope,adapter


def run(model_path,manifest_path,output_path,b28_run,contract_path,approval_path,approval_sha256,*,max_new_records=4,resume=False):
    records,pending,expected,scope,adapter=prepare_plan(model_path,manifest_path,output_path,b28_run,contract_path,max_new_records=max_new_records,resume=resume)
    verify_approval(approval_path,approval_sha256,scope)
    if not pending:return {'status':'INFERENCE_COMPLETED','new_records':0,'total_records':54}
    gate=PacedGate(output_path.with_suffix('.resources.jsonl'));gate.before_unit('before_runtime_import',loading=True)
    os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_vlm import load,generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.trainer.common.model import prepare_model_for_training
    mx.set_memory_limit(PACED_POLICY['memory_guideline_gib']*1024**3);mx.set_cache_limit(PACED_POLICY['cache_limit_mib']*1024**2)
    gate.before_unit('model_load',loading=True)
    model,processor=load(str(model_path),lazy=True)
    model=prepare_model_for_training(model,train_type='lora',lora_rank=4,lora_alpha=8,quantization_bits=4,quantization_group_size=64,checkpoint_path=adapter,verbose=False)
    mx.eval(model.parameters());gc.collect();mx.clear_cache();gate.before_unit('quantized_model_ready');model.eval()
    disk=mx.load(str(adapter));loaded=dict(tree_flatten(model.trainable_parameters()))
    if set(disk)!=set(loaded) or len(loaded)!=504 or not all(v.shape==loaded[k].shape and v.dtype==loaded[k].dtype and bool(mx.array_equal(v,loaded[k]).item()) for k,v in disk.items()):
        raise ValueError('Exact504-tensor adapter reload failed')
    if sha256(adapter)!=expected['adapter_checkpoint_sha256']:raise ValueError('Adapter changed during loading')
    del disk,loaded;gc.collect();mx.clear_cache()
    existing_hash=sha256(output_path) if output_path.exists() else None
    if existing_hash!=scope['existing_output_sha256']:raise ValueError('Predictions changed after approval')
    with output_path.open('a') as out:
        sync_directory(output_path.parent)
        for record in pending:
            gate.before_unit('before_generation:'+record['record_id'])
            prompt=apply_chat_template(processor,model.config,record['prompt'],num_images=1,enable_thinking=False,add_generation_prompt=True)
            if prompt.count('<|image_pad|>')!=1 or not prompt.endswith('<|im_start|>assistant\n'):raise ValueError('Native image/assistant prefix differs from matched contract')
            image=Path(record['image']);image_hash=sha256(image)
            if image_hash!=scope['new_image_sha256'][record['record_id']]:raise ValueError('Image changed after review')
            started=time.monotonic()
            response=generate(model,processor,prompt,image=[str(image)],max_tokens=256,temperature=0,verbose=False)
            if not isinstance(response.text,str):raise ValueError('Backend returned no text response')
            result={'record_id':record['record_id'],'category':record['category'],'image_sha256':image_hash,**expected,
                'adapter_verified_parameter_tensors':504,'prompt':record['prompt'],'image_placeholder_count':1,
                'response':response.text,'elapsed_seconds':round(time.monotonic()-started,3),'python':platform.python_version(),
                'rendered_prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'review_sha256':approval_sha256,
                **generation_metadata(response),'post_generation_mlx_peak_bytes':mx.get_peak_memory()}
            # Preserve actual output before a post-generation HOLD/STOP can exit.
            out.write(json.dumps(result,ensure_ascii=False)+'\n');out.flush();os.fsync(out.fileno())
            del response;gc.collect();mx.clear_cache()
            gate.before_unit('generation_completed:'+record['record_id'])
            time.sleep(PACED_POLICY['example_pause_seconds'])
            print('completed '+record['record_id'],flush=True)
    del model,processor;gc.collect();mx.clear_cache();gate.before_unit('model_released')
    return {'status':'INFERENCE_COMPLETED' if scope['completed_records']+len(pending)==54 else 'INFERENCE_STAGE_COMPLETED',
            'new_records':len(pending),'total_records':scope['completed_records']+len(pending),'manifest_records':54}


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--b28-run',type=Path,required=True);p.add_argument('--contract',type=Path,required=True)
    p.add_argument('--max-new-records',type=int,default=4);p.add_argument('--resume',action='store_true');p.add_argument('--plan-only',action='store_true')
    p.add_argument('--approval',type=Path);p.add_argument('--approval-sha256');a=p.parse_args()
    args=(a.model.resolve(),a.manifest.resolve(),a.output.resolve(),a.b28_run.resolve(),a.contract.resolve())
    if a.plan_only:print(json.dumps(prepare_plan(*args,max_new_records=a.max_new_records,resume=a.resume)[3],indent=2));return
    if not a.approval or not a.approval_sha256:p.error('Independent --approval and --approval-sha256 are required')
    print(json.dumps(run(*args,a.approval,a.approval_sha256,max_new_records=a.max_new_records,resume=a.resume)))


if __name__=='__main__':main()
