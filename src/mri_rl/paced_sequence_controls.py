"""Bounded, reviewed B28 sequence controls after completed matched54 inference.

No model imports until frozen data, fullB28, primary completion and independent
scope approval pass. This does not replace the primary evaluator or its files.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import time
from types import SimpleNamespace
from mri_rl import paced_frozen_eval as pe

HISTORICAL_ADAPTER='244c26fd5ce2e3b02b46703d4f46a90608f2ef1a868eeee682ee4fe48a58ed73'
GOLD_SHA256='0630ec7c91ed67f1059294422f8eb937435546bbf446155d187365862fda8961'
CONTROLS={
    'no_image':{'manifest':'internal_dev_manifest_sft28_sequence_no_image_PRIVATE.jsonl',
        'manifest_sha256':'420c0dc5a152d3dc3e9d5e1505525f038537bf8b06a245fdc03452fa369061d4',
        'baseline':'sft28_sequence_no_image_outputs_PRIVATE.jsonl',
        'baseline_sha256':'607888eddd082c5d04e9320134362099bb5dc6a411c58bb5534caefce11835d8',
        'suffix':'__no_image','images':0},
    'swap':{'manifest':'internal_dev_manifest_sft28_sequence_swap_PRIVATE.jsonl',
        'manifest_sha256':'c2f709242f25d5a3d3621bfd1f0f40f59d6ffcf6320613bfcf363d4d375aad11',
        'baseline':'sft28_sequence_swap_outputs_PRIVATE.jsonl',
        'baseline_sha256':'6249117b8543277b4858a5b76497c81314be5e896446bb2a20af919a3e32f179',
        'suffix':'__same_slice_sequence_swap','images':1},
}


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def derived_controls(original,mode):
    if mode not in CONTROLS:raise ValueError('Unknown sequence control mode')
    sequence={r['record_id']:r for r in original if r['category']=='mri_sequence'};result=[]
    if len(sequence)!=sum(r['category']=='mri_sequence' for r in original):raise ValueError('Duplicate sequence IDs')
    for key,row in sequence.items():
        prompt=json.loads(json.dumps(row['prompt']))
        parts=[part for message in prompt if isinstance(message['content'],list) for part in message['content'] if part.get('type')=='image']
        if len(parts)!=1 or parts[0].get('image')!=row['image']:raise ValueError('Expected one matching source image')
        if mode=='no_image':
            image=None
            for message in prompt:
                if isinstance(message['content'],list):message['content']=[part for part in message['content'] if part.get('type')!='image']
        else:
            if key.endswith('_flair_sequence'):other=key.removesuffix('_flair_sequence')+'_t1ce_sequence'
            elif key.endswith('_t1ce_sequence'):other=key.removesuffix('_t1ce_sequence')+'_flair_sequence'
            else:raise ValueError('Unrecognized sequence pair identity')
            if other not in sequence:raise ValueError('Same-patient/slice opposite sequence missing')
            image=sequence[other]['image']
            if image==row['image']:raise ValueError('Swap does not change image')
            parts[0]['image']=image
        result.append({'record_id':key+CONTROLS[mode]['suffix'],'category':'mri_sequence','image':image,'prompt':prompt})
    return result


def verify_control_baseline(path,records,expected,spec,*,expected_count=24):
    if pe.sha256(path)!=spec['baseline_sha256']:raise ValueError('Immutable control baseline digest mismatch')
    source={r['record_id']:r for r in records};baseline=read_rows(path);keyed={r.get('record_id'):r for r in baseline}
    if len(records)!=expected_count or len(source)!=expected_count or len(baseline)!=expected_count or len(keyed)!=expected_count or set(keyed)!=set(source):
        raise ValueError('Control baseline count/IDs mismatch')
    for key,row in keyed.items():
        record=source[key];image_hash=pe.sha256(Path(record['image'])) if record.get('image') else None
        if row.get('category')!='mri_sequence' or row.get('prompt')!=record['prompt'] or row.get('image_sha256')!=image_hash:
            raise ValueError('Control baseline prompt/category/pixels mismatch')
        for field in ('manifest_sha256','model_path','model_config_sha256','model_template_sha256','template_override_sha256','max_tokens','temperature','adapter_config_sha256'):
            if row.get(field)!=expected[field]:raise ValueError('Control baseline model/decoding profile mismatch: '+field)
        if (row.get('adapter_checkpoint_sha256')!=HISTORICAL_ADAPTER or row.get('adapter_verified_parameter_tensors')!=504 or
            row.get('image_placeholder_count')!=spec['images'] or not isinstance(row.get('response'),str)):
            raise ValueError('Control baseline adapter/output provenance mismatch')
    return spec['baseline_sha256']


def require_completed_primary(records,outputs,expected):
    source={r['record_id']:r for r in records};keyed={r.get('record_id'):r for r in outputs}
    if len(records)!=54 or len(source)!=54 or len(outputs)!=54 or len(keyed)!=54 or set(source)!=set(keyed):
        raise ValueError('All54 matched primary records must be completed before controls')
    for key,row in keyed.items():
        record=source[key]
        if (any(row.get(k)!=v for k,v in expected.items()) or row.get('category')!=record['category'] or row.get('prompt')!=record['prompt'] or
            row.get('image_sha256')!=pe.sha256(Path(record['image'])) or row.get('adapter_verified_parameter_tensors')!=504 or
            row.get('image_placeholder_count')!=1 or not isinstance(row.get('response'),str)):
            raise ValueError('Primary completion identity/output mismatch')
        pe.generation_metadata(SimpleNamespace(generation_tokens=row.get('generation_tokens'),prompt_tokens=row.get('prompt_tokens'),
            total_tokens=row.get('total_tokens'),finish_reason=row.get('finish_reason'),peak_memory=row.get('generation_peak_memory_gb')))


def verify_review(path,digest,scope):
    if pe.sha256(path)!=digest:raise ValueError('Independent control review checksum mismatch')
    review=json.loads(path.read_text())
    if (review.get('kind')!='paced_sequence_controls_v1' or review.get('verdict') not in ('PASS','PASS_WITH_LIMITATIONS') or
        not isinstance(review.get('reviewer'),str) or not review['reviewer'].strip() or review.get('scope')!=scope):
        raise ValueError('Review does not cover exact sequence-control scope')


def prepare_plan(model,primary_manifest,primary_outputs,output,b28_run,contract,mode,*,max_new_records=4,resume=False):
    if mode not in CONTROLS:raise ValueError('Unknown control mode')
    private=Path(__file__).resolve().parents[2]/'local_data'
    if not output.resolve().is_relative_to(private) or not primary_outputs.is_file():raise ValueError('Private outputs and completed primary file required')
    # Reuse the unchanged primary verifier for genuine B28, exact model/manifest,
    # historical pixels and complete immutable prediction-resume identities.
    primary,pending,common,primary_scope,adapter=pe.prepare_plan(model,primary_manifest,primary_outputs,b28_run,contract,max_new_records=4,resume=True)
    if pending:raise ValueError('Matched54 evaluation incomplete; controls cannot start')
    require_completed_primary(primary,read_rows(primary_outputs),common)
    spec=CONTROLS[mode];manifest=primary_manifest.parent/spec['manifest']
    if pe.sha256(manifest)!=spec['manifest_sha256']:raise ValueError('Frozen24-control manifest checksum mismatch')
    records=read_rows(manifest)
    if records!=derived_controls(primary,mode) or len(records)!=24:raise ValueError('Control transformation differs from frozen same-patient/slice source')
    gold=primary_manifest.parent/'internal_dev_gold_sft_train_matched_PRIVATE.jsonl'
    if pe.sha256(gold)!=GOLD_SHA256:raise ValueError('Private source label provenance changed')
    sources={'paced_sequence_controls':pe.sha256(Path(__file__)),'paced_frozen_eval':pe.sha256(Path(pe.__file__))}
    expected={**common,'manifest_sha256':spec['manifest_sha256'],'control_mode':mode,
        'primary_output_sha256':pe.sha256(primary_outputs),'control_gold_sha256':GOLD_SHA256,
        'sequence_control_sources_sha256':pe.canonical_hash(sources),
        'image_inventory_sha256':pe.canonical_hash([{'record_id':r['record_id'],'sha256':pe.sha256(Path(r['image'])) if r.get('image') else None} for r in records])}
    expected['historical_control_baseline_sha256']=verify_control_baseline(primary_manifest.parent/spec['baseline'],records,expected,spec)
    output.parent.mkdir(parents=True,exist_ok=True);completed=pe.resume_completed(output,records,expected,resume=resume)
    new=pe.bounded_pending(records,completed,max_new_records)
    scope={'runner':'paced_sequence_controls_v1','mode':mode,'expected':expected,'source_hashes':sources,
        'primary_eval_source_hashes':primary_scope['source_hashes'],'output':str(output.resolve()),
        'existing_output_sha256':pe.sha256(output) if output.exists() else None,'completed_records':len(completed),
        'new_record_ids':[r['record_id'] for r in new],
        'new_image_sha256':{r['record_id']:pe.sha256(Path(r['image'])) if r.get('image') else None for r in new},
        'max_new_records':max_new_records,'max_tokens':256,'image_placeholders':spec['images'],'adapter':str(adapter.resolve()),
        'paced_policy':pe.PACED_POLICY,'quantization_bits':4,'quantization_group_size':64,'lora_rank':4,'lora_alpha':8,
        'enable_thinking':False,'add_generation_prompt':True,
        'target_rule':'original acquisition label without pixels (prior diagnostic)' if mode=='no_image' else 'visible opposite-sequence label of same patient and slice'}
    return new,expected,scope,adapter


def run(model_path,primary_manifest,primary_outputs,output,b28_run,contract,mode,review,review_sha,*,max_new_records=4,resume=False):
    pending,expected,scope,adapter=prepare_plan(model_path,primary_manifest,primary_outputs,output,b28_run,contract,mode,max_new_records=max_new_records,resume=resume)
    verify_review(review,review_sha,scope)
    if not pending:return {'status':'INFERENCE_COMPLETED','mode':mode,'new_records':0,'total_records':24}
    gate=pe.PacedGate(output.with_suffix('.resources.jsonl'));gate.before_unit('before_runtime_import',loading=True)
    os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_vlm import load,generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.trainer.common.model import prepare_model_for_training
    mx.set_memory_limit(pe.PACED_POLICY['memory_guideline_gib']*1024**3);mx.set_cache_limit(pe.PACED_POLICY['cache_limit_mib']*1024**2)
    gate.before_unit('model_load',loading=True);model,processor=load(str(model_path),lazy=True)
    model=prepare_model_for_training(model,train_type='lora',lora_rank=4,lora_alpha=8,quantization_bits=4,quantization_group_size=64,checkpoint_path=adapter,verbose=False)
    mx.eval(model.parameters());gc.collect();mx.clear_cache();gate.before_unit('quantized_model_ready');model.eval()
    disk=mx.load(str(adapter));loaded=dict(tree_flatten(model.trainable_parameters()))
    if (set(disk)!=set(loaded) or len(loaded)!=504 or pe.sha256(adapter)!=expected['adapter_checkpoint_sha256'] or
        not all(v.shape==loaded[k].shape and v.dtype==loaded[k].dtype and bool(mx.array_equal(v,loaded[k]).item()) for k,v in disk.items())):
        raise ValueError('Exact504-tensor control adapter reload failed')
    del disk,loaded;gc.collect();mx.clear_cache()
    if (pe.sha256(output) if output.exists() else None)!=scope['existing_output_sha256']:raise ValueError('Control outputs changed after approval')
    with output.open('a') as out:
        pe.sync_directory(output.parent)
        for record in pending:
            gate.before_unit('before_generation:'+record['record_id']);n=scope['image_placeholders']
            prompt=apply_chat_template(processor,model.config,record['prompt'],num_images=n,enable_thinking=False,add_generation_prompt=True)
            if prompt.count('<|image_pad|>')!=n or not prompt.endswith('<|im_start|>assistant\n'):raise ValueError('Control image/assistant prefix mismatch')
            image=Path(record['image']) if record.get('image') else None;image_hash=pe.sha256(image) if image else None
            if image_hash!=scope['new_image_sha256'][record['record_id']]:raise ValueError('Control pixels changed after review')
            started=time.monotonic();response=generate(model,processor,prompt,image=[str(image)] if image else None,max_tokens=256,temperature=0,verbose=False)
            if not isinstance(response.text,str):raise ValueError('Control response is not text')
            result={'record_id':record['record_id'],'category':'mri_sequence','image_sha256':image_hash,**expected,
                'adapter_verified_parameter_tensors':504,'prompt':record['prompt'],'image_placeholder_count':n,'response':response.text,
                'elapsed_seconds':round(time.monotonic()-started,3),'python':platform.python_version(),
                'rendered_prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'review_sha256':review_sha,
                **pe.generation_metadata(response),'post_generation_mlx_peak_bytes':mx.get_peak_memory()}
            out.write(json.dumps(result,ensure_ascii=False)+'\n');out.flush();os.fsync(out.fileno())
            del response;gc.collect();mx.clear_cache();gate.before_unit('generation_completed:'+record['record_id'])
            time.sleep(pe.PACED_POLICY['example_pause_seconds']);print('completed '+record['record_id'],flush=True)
    del model,processor;gc.collect();mx.clear_cache();gate.before_unit('model_released')
    total=scope['completed_records']+len(pending)
    return {'status':'INFERENCE_COMPLETED' if total==24 else 'INFERENCE_STAGE_COMPLETED','mode':mode,'new_records':len(pending),'total_records':total}


def main():
    p=argparse.ArgumentParser()
    for name in ('model','primary-manifest','primary-outputs','output','b28-run','contract'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--mode',choices=CONTROLS,required=True);p.add_argument('--max-new-records',type=int,default=4);p.add_argument('--resume',action='store_true')
    p.add_argument('--plan-only',action='store_true');p.add_argument('--approval',type=Path);p.add_argument('--approval-sha256');a=p.parse_args()
    args=tuple(p.resolve() for p in (a.model,a.primary_manifest,a.primary_outputs,a.output,a.b28_run,a.contract))+(a.mode,)
    if a.plan_only:print(json.dumps(prepare_plan(*args,max_new_records=a.max_new_records,resume=a.resume)[2],indent=2));return
    if not a.approval or not a.approval_sha256:p.error('Independent approval and digest required')
    print(json.dumps(run(*args,a.approval,a.approval_sha256,max_new_records=a.max_new_records,resume=a.resume)))


if __name__=='__main__':main()
